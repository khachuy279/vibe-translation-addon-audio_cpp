"""Ingress Session Benchmark & Quality Auditor.

Uses `00_ingress_stream.wav` and `00_ingress_stream.txt` (golden reference from online tool)
to benchmark the full streaming pipeline and test parameter optimizations.

Usage:
    python -m benchmarks.ingress_benchmark --baseline
    python -m benchmarks.ingress_benchmark --silence 500 --min-words 1
    python -m benchmarks.ingress_benchmark --sweep-silence
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.ja_text import score_ja, normalize_ja
from benchmarks.simulator import StreamingAudioSimulator

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingress_bench")

SESSION_DIR = REPO / "debug_audio" / "22d54213-6418-4985-86db-0d1fa892f2dc"
DEFAULT_WAV = SESSION_DIR / "00_ingress_stream.wav"
DEFAULT_TXT = SESSION_DIR / "00_ingress_stream.txt"


def parse_ground_truth(txt_path: Path) -> List[Dict[str, Any]]:
    """Parse speaker turns and dialogue lines from online tool ground truth."""
    raw = txt_path.read_text(encoding="utf-8")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    turns = []
    for line in lines:
        m = re.match(r"^\[(\d{2}:\d{2})\]\s*Speaker\s*(\d+):\s*(.*)$", line)
        if m:
            time_str, spk, text = m.groups()
            turns.append({
                "time": time_str,
                "speaker": int(spk),
                "text": text.strip(),
                "norm_text": normalize_ja(text),
            })
    return turns


class TracedTranscribeEngine(TranscribeEngine):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commits: List[Dict[str, Any]] = []
        self.inferences: List[Dict[str, Any]] = []
        self.dropped_commits: List[Dict[str, Any]] = []

    def _emit_final(self, text, utt_id, reason, epoch=0, media_start_time=0.0, media_end_time=0.0):
        self.commits.append({
            "utterance_id": utt_id,
            "reason": reason,
            "text": (text or "").strip(),
            "t_wall": time.time(),
            "media_start": media_start_time,
            "media_end": media_end_time,
        })
        return super()._emit_final(text, utt_id, reason, epoch=epoch, media_start_time=media_start_time, media_end_time=media_end_time)

    def _run_inference(self, pcm_float32, frame_state=None, is_commit=False, utt_id=None):
        t0 = time.perf_counter()
        out = super()._run_inference(pcm_float32, frame_state=frame_state, is_commit=is_commit, utt_id=utt_id)
        self.inferences.append({
            "is_commit": bool(is_commit),
            "audio_sec": round(len(pcm_float32) / 16000.0, 3),
            "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "text": (out or "").strip(),
        })
        return out


async def run_streaming_benchmark(
    wav_path: Path,
    turns: List[Dict[str, Any]],
    silence_ms: int = 500,
    min_words: int = 1,
    threshold: float = 0.40,
    split_on_stability: bool = True,
    speed: float = 0.0,
    model_name: str = "qwen3-asr-1.7b",
) -> Dict[str, Any]:
    full_ref_text = "".join(t["norm_text"] for t in turns)
    
    # Configure ASR & Sentence
    config.asr.active_model = model_name
    config.asr.language = "ja"
    config.vad.threshold = threshold
    config.vad.silence_duration_ms = silence_ms
    config.sentence.min_words_to_commit = min_words
    config.sentence.split_on_stability = split_on_stability
    
    ASRModelManager().ensure_model(model_name)
    
    engine = TracedTranscribeEngine(session_id=f"bench_{silence_ms}ms")
    engine.set_language("ja")
    engine.sentence_config.min_words_to_commit = min_words
    engine.sentence_config.split_on_stability = split_on_stability
    
    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=config.vad.vad_engine,
        threshold=threshold,
        silence_duration_ms=silence_ms,
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
        
    elapsed = time.perf_counter() - t0
    
    committed_texts = [c["text"] for c in engine.commits if c["text"]]
    hyp_full = "".join(normalize_ja(t) for t in committed_texts)
    score = score_ja(full_ref_text, hyp_full)
    
    return {
        "silence_ms": silence_ms,
        "min_words": min_words,
        "model": model_name,
        "cer_strict_pct": round(score.cer * 100.0, 2) if score.cer is not None else None,
        "cer_itn_pct": round(score.cer_itn * 100.0, 2) if score.cer_itn is not None else None,
        "substitutions": score.substitutions,
        "deletions": score.deletions,
        "insertions": score.insertions,
        "ref_chars": len(full_ref_text),
        "hyp_chars": len(hyp_full),
        "commit_count": len(committed_texts),
        "ground_truth_turns": len(turns),
        "commits": committed_texts,
        "elapsed_sec": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingress benchmark runner")
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    parser.add_argument("--txt", type=Path, default=DEFAULT_TXT)
    parser.add_argument("--baseline", action="store_true", help="Run with current baseline config (150ms, min_words=4)")
    parser.add_argument("--silence", type=int, default=500, help="VAD silence duration in ms")
    parser.add_argument("--min-words", type=int, default=1, help="Min words to commit")
    parser.add_argument("--threshold", type=float, default=0.40, help="VAD threshold")
    parser.add_argument("--model", type=str, default="qwen3-asr-1.7b", help="Model name")
    parser.add_argument("--no-stability", action="store_true", help="Disable stability split (only VAD silence commits)")
    parser.add_argument("--sweep-silence", action="store_true", help="Sweep silence thresholds")
    args = parser.parse_args()

    if not args.wav.exists() or not args.txt.exists():
        raise SystemExit(f"Missing files: {args.wav} or {args.txt}")

    turns = parse_ground_truth(args.txt)
    print(f"Loaded ground truth: {len(turns)} dialogue turns, {sum(len(t['norm_text']) for t in turns)} normalized chars.")

    if args.baseline:
        print("\n=== RUNNING BASELINE (silence=150ms, min_words=4, threshold=0.20) ===")
        res = asyncio.run(run_streaming_benchmark(
            args.wav, turns, silence_ms=150, min_words=4, threshold=0.20, speed=0.0, model_name=args.model
        ))
        print(f"Result: CER={res['cer_strict_pct']}% (ITN: {res['cer_itn_pct']}%) | Commits: {res['commit_count']} (GT: {res['ground_truth_turns']})")
        print(f"Subs: {res['substitutions']}, Dels: {res['deletions']}, Ins: {res['insertions']}")
        return

    if args.sweep_silence:
        print("\n=== SWEEPING SILENCE DURATIONS ===")
        silence_list = [150, 300, 450, 500, 600, 700, 800]
        for s in silence_list:
            res = asyncio.run(run_streaming_benchmark(
                args.wav, turns, silence_ms=s, min_words=1, threshold=args.threshold, speed=0.0, model_name=args.model
            ))
            print(f"Silence {s:3d}ms: CER={res['cer_strict_pct']:5.2f}% | ITN={res['cer_itn_pct']:5.2f}% | Commits={res['commit_count']:3d} | Dels={res['deletions']:2d} Ins={res['insertions']:2d}")
        return

    print(f"\n=== RUNNING BENCHMARK (silence={args.silence}ms, min_words={args.min_words}, threshold={args.threshold}, stability={not args.no_stability}) ===")
    res = asyncio.run(run_streaming_benchmark(
        args.wav, turns, silence_ms=args.silence, min_words=args.min_words, threshold=args.threshold, split_on_stability=not args.no_stability, speed=0.0, model_name=args.model
    ))
    print(f"Result: CER={res['cer_strict_pct']}% (ITN: {res['cer_itn_pct']}%) | Commits: {res['commit_count']} (GT: {res['ground_truth_turns']})")
    print(f"Subs: {res['substitutions']}, Dels: {res['deletions']}, Ins: {res['insertions']}")
    print("\n--- Committed texts ---")
    for i, c in enumerate(res['commits'], 1):
        print(f"[{i:02d}] {c}")


if __name__ == "__main__":
    main()
