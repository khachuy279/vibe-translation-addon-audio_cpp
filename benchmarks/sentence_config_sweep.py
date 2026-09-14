"""SentenceConfig Parameter Sweep & Evaluation Benchmark.

Systematically measures the trade-offs of:
- Stability-Split (split_on_stability, stability_duration_sec, stability_threshold_polls)
- VAD-Paced Soft Boundary (max_duration_sec, max_duration_grace_sec, boundary_candidate_silence_ms, max_duration_require_silence)
- Commit Gating (min_words_to_commit, max_chars) with fixed min_words_to_emit_final=1.

Audio testbed:
- Ingress reference stream: debug_audio/22d54213-6418-4985-86db-0d1fa892f2dc/00_ingress_stream.wav
- Golden reference transcript: debug_audio/22d54213-6418-4985-86db-0d1fa892f2dc/00_ingress_stream.txt

Usage:
    python -m benchmarks.sentence_config_sweep --baseline
    python -m benchmarks.sentence_config_sweep --sweep-stability
    python -m benchmarks.sentence_config_sweep --sweep-boundary
    python -m benchmarks.sentence_config_sweep --sweep-gating
    python -m benchmarks.sentence_config_sweep --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import SentenceConfig, config
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.ingress_benchmark import DEFAULT_TXT, DEFAULT_WAV, TracedTranscribeEngine, parse_ground_truth
from benchmarks.ja_text import JaScore, normalize_ja, score_ja
from benchmarks.simulator import StreamingAudioSimulator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentence_config_sweep")


@dataclass
class SentenceSweepResult:
    condition_id: str
    description: str
    params: Dict[str, Any]
    cer: float
    cer_itn: float
    num_commits: int
    commit_methods: Dict[str, int]
    avg_turn_duration_sec: float
    avg_chars_per_turn: float
    emergency_cut_rate_pct: float
    sim_wall_sec: float
    hyp_text_sample: str


async def run_sentence_condition(
    wav_path: Path,
    turns: List[Dict[str, Any]],
    condition_id: str,
    description: str,
    sentence_overrides: Dict[str, Any],
    speed: float = 0.0,
    model_name: str = "qwen3-asr-1.7b",
) -> SentenceSweepResult:
    """Run one streaming condition on the reference stream with custom SentenceConfig."""
    full_ref_text = "".join(t["norm_text"] for t in turns)

    # Base configs
    config.asr.active_model = model_name
    config.asr.language = "ja"

    # Pre-warm model
    ASRModelManager().ensure_model(model_name)

    # Build active SentenceConfig
    base_sc = config.sentence.model_copy()
    for k, v in sentence_overrides.items():
        if hasattr(base_sc, k):
            setattr(base_sc, k, v)

    engine = TracedTranscribeEngine(session_id=f"sweep_{condition_id}")
    engine.set_language("ja")
    # Apply exact sentence config
    engine.sentence_config = base_sc
    engine._segmenter = SentenceSegmenter(
        max_chars=base_sc.max_chars,
        max_duration_sec=base_sc.max_duration_sec,
        min_words_to_commit=base_sc.min_words_to_commit,
        min_words_to_emit_final=base_sc.min_words_to_emit_final,
        split_on_stability=base_sc.split_on_stability,
        stability_duration_sec=base_sc.stability_duration_sec,
        stability_threshold_polls=base_sc.stability_threshold_polls,
    )

    # Use tuned VAD champion profile
    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=config.vad.vad_engine,
        threshold=config.vad.threshold,
        silence_duration_ms=config.vad.silence_duration_ms,
        hangover_ms=config.vad.hangover_ms,
        pre_speech_buffer_ms=config.vad.pre_speech_buffer_ms,
        enabled=True,
        on_speech_chunk=engine.feed_audio,
        on_speech_start=engine.on_speech_start,
        on_speech_end=engine.on_speech_end,
    )

    sim = StreamingAudioSimulator(audio_source=wav_path, chunk_ms=64, speed=speed, trailing_silence_sec=2.0)

    async def _consume():
        try:
            async for _ in engine.stream_tokens():
                pass
        except asyncio.CancelledError:
            pass

    t0 = time.perf_counter()
    consumer = asyncio.create_task(_consume())
    try:
        for chunk in sim.iter_chunks():
            vad.feed_chunk(chunk.pcm_bytes)
            if speed > 0:
                await asyncio.sleep((chunk.duration_ms / 1000.0) / speed)
    finally:
        await engine.cleanup()
        await asyncio.sleep(0.3)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

    wall_sec = time.perf_counter() - t0

    committed_texts = [c["text"] for c in engine.commits if c.get("text")]
    hyp_full_text = "".join(normalize_ja(t) for t in committed_texts)
    score: JaScore = score_ja(full_ref_text, hyp_full_text)

    # Commit attribution
    methods: Dict[str, int] = {}
    turn_durations = []
    for c in engine.commits:
        r = c.get("reason", "UNKNOWN")
        methods[r] = methods.get(r, 0) + 1
        dur = c.get("media_end", 0.0) - c.get("media_start", 0.0)
        if dur > 0:
            turn_durations.append(dur)

    num_commits = len(committed_texts)
    avg_dur = float(np.mean(turn_durations)) if turn_durations else 0.0
    avg_chars = float(np.mean([len(t) for t in committed_texts])) if committed_texts else 0.0
    emerg_count = methods.get("MAX_DURATION_EMERGENCY", 0)
    emerg_rate = (emerg_count / max(1, num_commits)) * 100.0

    hyp_sample = " | ".join(committed_texts[:3]) if committed_texts else ""

    result = SentenceSweepResult(
        condition_id=condition_id,
        description=description,
        params=sentence_overrides,
        cer=round(score.cer * 100.0, 2) if score.cer is not None else 100.0,
        cer_itn=round(score.cer_itn * 100.0, 2) if score.cer_itn is not None else 100.0,
        num_commits=num_commits,
        commit_methods=methods,
        avg_turn_duration_sec=round(avg_dur, 2),
        avg_chars_per_turn=round(avg_chars, 1),
        emergency_cut_rate_pct=round(emerg_rate, 2),
        sim_wall_sec=round(wall_sec, 2),
        hyp_text_sample=hyp_sample,
    )
    return result


def print_result_table(results: List[SentenceSweepResult], title: str = "SentenceConfig Evaluation Results"):
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    header = (
        f"{'Condition':<18} | {'CER (%)':<8} | {'ITN CER':<8} | {'Commits':<8} | "
        f"{'Avg Dur':<8} | {'Avg Chars':<9} | {'Emerg %':<8} | {'Commit Methods'}"
    )
    print(header)
    print(f"{'-' * 95}")
    for r in results:
        methods_str = ", ".join(f"{k}:{v}" for k, v in sorted(r.commit_methods.items()))
        row = (
            f"{r.condition_id:<18} | {r.cer:<8.2f} | {r.cer_itn:<8.2f} | {r.num_commits:<8} | "
            f"{r.avg_turn_duration_sec:<8.2f}s| {r.avg_chars_per_turn:<9.1f} | {r.emergency_cut_rate_pct:<8.1f} | {methods_str}"
        )
        print(row)
    print(f"{'=' * 95}\n")


def generate_markdown_report(results: List[SentenceSweepResult], out_path: Path):
    """Generate Markdown evaluation report with Pareto analysis."""
    lines = [
        "# Báo Cáo Phân Tích & Tối Ưu Thông Số SentenceConfig",
        "",
        "## 1. Tổng Quan & Thiết Lập Đo Lường",
        "- **Audio kiểm thử:** `debug_audio/.../00_ingress_stream.wav` (đối thoại tiếng Nhật thực tế đa người nói).",
        "- **Reference chuẩn:** `00_ingress_stream.txt` (golden reference từ công cụ online).",
        f"- **Mô hình ASR:** `{config.asr.active_model}`.",
        f"- **VAD Engine:** `{config.vad.vad_engine}` (Threshold={config.vad.threshold}, Silence={config.vad.silence_duration_ms}ms, Hangover={config.vad.hangover_ms}ms).",
        "- **Ngưỡng cố định:** `min_words_to_emit_final = 1` (giữ trọn các lượt thoại ngắn tự nhiên).",
        "",
        "## 2. Bảng Kết Quả Chi Tiết",
        "",
        "| Condition | CER (Strict) | CER (ITN) | Commits | Avg Duration | Avg Chars/Turn | Emerg Cut % | Commit Breakdown |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in results:
        methods_str = "<br>".join(f"`{k}`: {v}" for k, v in sorted(r.commit_methods.items()))
        lines.append(
            f"| **{r.condition_id}**<br>*{r.description}* | {r.cer:.2f}% | {r.cer_itn:.2f}% | "
            f"{r.num_commits} | {r.avg_turn_duration_sec:.2f}s | {r.avg_chars_per_turn:.1f} | "
            f"{r.emergency_cut_rate_pct:.1f}% | {methods_str} |"
        )

    lines.extend([
        "",
        "## 3. Phân Tích Kỹ Thuật (Trade-off Matrix)",
        "",
        "### A. Cơ Chế Stability Split (`split_on_stability`, `stability_duration_sec`)",
        "- Khi `split_on_stability=False`: Hệ thống hoàn toàn phụ thuộc vào khoảng lặng VAD (`VAD_SILENCE`) để ngắt câu.",
        "- Khi `split_on_stability=True`: Cho phép chốt tiền tố (`STABLE_PREFIX`) khi người nói ngưng nghỉ nhưng chưa hết câu VAD, giúp hiển thị bản dịch sớm hơn.",
        "",
        "### B. VAD-Paced Soft Boundary (`max_duration_sec`, `boundary_candidate_silence_ms`)",
        "- `max_duration_sec`: Giới hạn an toàn ngăn chặn tình trạng một lượt thoại độc thoại kéo dài vô hạn.",
        "- `boundary_candidate_silence_ms`: Chờ khoảng lặng tự nhiên trong thời gian ân hạn (`grace_sec`) thay vì cắt ngang giữa chừng một từ (`MAX_DURATION_EMERGENCY`).",
        "",
        "### C. Gating Kích Thước (`min_words_to_commit`, `max_chars`)",
        "- `min_words_to_commit=1` cùng `min_words_to_emit_final=1` đảm bảo cả preview lẫn final đều nhạy với các phản hồi ngắn của người Nhật.",
        "",
        "## 4. Kết Luận & Đề Xuất Điểm Ngọt (Sweet Spot)",
    ])

    best_result = min(results, key=lambda x: x.cer)
    lines.extend([
        f"- **Cấu hình tối ưu nhất về CER:** `{best_result.condition_id}` với CER = **{best_result.cer:.2f}%** (ITN: {best_result.cer_itn:.2f}%).",
        f"- **Đặc trưng phân đoạn:** {best_result.num_commits} commits, độ dài trung bình {best_result.avg_chars_per_turn:.1f} ký tự/câu, tỷ lệ cắt khẩn cấp {best_result.emergency_cut_rate_pct:.1f}%.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📊 Report generated at: {out_path}")


async def main():
    parser = argparse.ArgumentParser(description="SentenceConfig Parameter Sweep Benchmark")
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="Input WAV audio")
    parser.add_argument("--txt", type=Path, default=DEFAULT_TXT, help="Ground truth TXT")
    parser.add_argument("--baseline", action="store_true", help="Run production baseline only")
    parser.add_argument("--sweep-stability", action="store_true", help="Sweep stability split parameters")
    parser.add_argument("--sweep-boundary", action="store_true", help="Sweep VAD-paced soft boundary parameters")
    parser.add_argument("--sweep-gating", action="store_true", help="Sweep commit gating and max_chars")
    parser.add_argument("--all", action="store_true", help="Run full parameter sweep matrix")
    parser.add_argument("--out-json", type=Path, default=REPO / "report" / "sentence_config_sweep.json")
    parser.add_argument("--out-md", type=Path, default=REPO / "report" / "sentence_config_sweep_report.md")

    args = parser.parse_args()

    if not args.wav.exists():
        print(f"❌ Audio file not found: {args.wav}")
        return
    if not args.txt.exists():
        print(f"❌ Ground truth file not found: {args.txt}")
        return

    turns = parse_ground_truth(args.txt)
    print(f"Loaded {len(turns)} turns from {args.txt.name}")

    conditions: List[Dict[str, Any]] = []

    # 1. Baseline
    conditions.append({
        "id": "BASELINE",
        "desc": "Current production config (stability=2.0s/2p, max_dur=15s, grace=2s, probe=80ms, min_w=1)",
        "params": {
            "max_duration_sec": 15.0,
            "max_duration_grace_sec": 2.0,
            "boundary_candidate_silence_ms": 80,
            "max_duration_require_silence": True,
            "split_on_stability": True,
            "stability_duration_sec": 2.0,
            "stability_threshold_polls": 2,
            "max_chars": 150,
            "min_words_to_commit": 1,
            "min_words_to_emit_final": 1,
        },
    })

    # 2. Stability Split Sweep
    if args.sweep_stability or args.all:
        conditions.extend([
            {
                "id": "STAB_OFF",
                "desc": "split_on_stability=False (Pure VAD silence commits only)",
                "params": {"split_on_stability": False, "min_words_to_emit_final": 1},
            },
            {
                "id": "STAB_FAST_1.0s",
                "desc": "stability_duration_sec=1.0s, polls=2 (Fast prefix emit)",
                "params": {"split_on_stability": True, "stability_duration_sec": 1.0, "stability_threshold_polls": 2, "min_words_to_emit_final": 1},
            },
            {
                "id": "STAB_MID_1.5s",
                "desc": "stability_duration_sec=1.5s, polls=2 (Balanced prefix emit)",
                "params": {"split_on_stability": True, "stability_duration_sec": 1.5, "stability_threshold_polls": 2, "min_words_to_emit_final": 1},
            },
            {
                "id": "STAB_CONSERV_2.5s",
                "desc": "stability_duration_sec=2.5s, polls=3 (Conservative prefix emit)",
                "params": {"split_on_stability": True, "stability_duration_sec": 2.5, "stability_threshold_polls": 3, "min_words_to_emit_final": 1},
            },
        ])

    # 3. VAD-Paced Boundary Sweep
    if args.sweep_boundary or args.all:
        conditions.extend([
            {
                "id": "BOUND_TIGHT_10s",
                "desc": "max_duration_sec=10.0s, grace=2.0s, probe=80ms",
                "params": {"max_duration_sec": 10.0, "max_duration_grace_sec": 2.0, "boundary_candidate_silence_ms": 80, "min_words_to_emit_final": 1},
            },
            {
                "id": "BOUND_PROBE_50ms",
                "desc": "boundary_candidate_silence_ms=50ms (Sensitive acoustic probe)",
                "params": {"boundary_candidate_silence_ms": 50, "max_duration_sec": 15.0, "min_words_to_emit_final": 1},
            },
            {
                "id": "BOUND_PROBE_120ms",
                "desc": "boundary_candidate_silence_ms=120ms (Conservative acoustic probe)",
                "params": {"boundary_candidate_silence_ms": 120, "max_duration_sec": 15.0, "min_words_to_emit_final": 1},
            },
            {
                "id": "BOUND_HARD_CUT",
                "desc": "max_duration_require_silence=False (Direct hard cut at 15s)",
                "params": {"max_duration_require_silence": False, "max_duration_sec": 15.0, "min_words_to_emit_final": 1},
            },
        ])

    # 4. Gating & Max Chars Sweep
    if args.sweep_gating or args.all:
        conditions.extend([
            {
                "id": "GATE_WORDS_2",
                "desc": "min_words_to_commit=2, min_words_to_emit_final=1",
                "params": {"min_words_to_commit": 2, "min_words_to_emit_final": 1},
            },
            {
                "id": "GATE_WORDS_4",
                "desc": "min_words_to_commit=4, min_words_to_emit_final=1",
                "params": {"min_words_to_commit": 4, "min_words_to_emit_final": 1},
            },
            {
                "id": "CHARS_COMPACT_100",
                "desc": "max_chars=100, min_words_to_emit_final=1",
                "params": {"max_chars": 100, "min_words_to_emit_final": 1},
            },
            {
                "id": "CHARS_EXPAND_180",
                "desc": "max_chars=180, min_words_to_emit_final=1",
                "params": {"max_chars": 180, "min_words_to_emit_final": 1},
            },
        ])

    if not args.sweep_stability and not args.sweep_boundary and not args.sweep_gating and not args.all:
        print("Running Baseline only. Use --all or --sweep-* flags to test specific parameter groups.")

    results: List[SentenceSweepResult] = []

    print(f"\n🚀 Starting SentenceConfig Evaluation ({len(conditions)} conditions)...")
    for cond in conditions:
        cid = cond["id"]
        desc = cond["desc"]
        params = cond["params"]
        print(f"--> Testing [{cid}]: {desc}...")
        res = await run_sentence_condition(
            wav_path=args.wav,
            turns=turns,
            condition_id=cid,
            description=desc,
            sentence_overrides=params,
            speed=0.0,
        )
        print(f"    ✓ CER: {res.cer:.2f}% (ITN: {res.cer_itn:.2f}%) | Commits: {res.num_commits} | Methods: {res.commit_methods}")
        results.append(res)

    print_result_table(results)

    # Save outputs
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    raw_dicts = [asdict(r) for r in results]
    args.out_json.write_text(json.dumps(raw_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Raw results saved to: {args.out_json}")

    generate_markdown_report(results, args.out_md)


if __name__ == "__main__":
    asyncio.run(main())
