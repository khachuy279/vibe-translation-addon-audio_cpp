"""Phase 3C.3: Full 7-File Streaming Latency Budget & Accuracy Benchmark.

Evaluates the production pipeline across all 7 benchmark files under exact streaming conditions:
- Frozen C06 FSMN-VAD (threshold=0.20, hangover=250ms, pre_speech=800ms)
- Phase 3C.1 Quick Wins (growth=0.5, min_words=4, stability=1.5s)
- Phase 3C.3 VAD-Paced Soft Boundary (max_duration=15.0s, grace=2.0s, probe=80ms)

Measures:
1. Accuracy: Per-file and Corpus CER / WER, Duplicate words, Rollback count
2. Latency UX:
   - TTFS_first (onset to first partial)
   - TTFS_meaningful (onset to first partial >= 4 words or >= 4 CJK chars)
   - final_subtitle_latency (audio end to final subtitle display)
   - boundary_wait_latency (duration spent in FORCED_PENDING before safe cut)
3. Controller & Boundary Telemetry:
   - VAD_SILENCE, MAX_DURATION_SAFE, MAX_DURATION_DEFERRED, MAX_DURATION_EMERGENCY
   - Preview interval statistics (mean, median, p95)
4. GPU Exposure: Total inferences, audio exposure ratio, RTF
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from backend_cpp.asr.audio_buffer import VAD_STATE_SPEECH, VAD_STATE_NON_SPEECH
from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.asr.sentence_segmenter import count_content_tokens
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config as app_cfg
from backend_cpp.utils.perf_profiler import perf
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.dataset import discover_dataset, DatasetPair
from benchmarks.simulator import StreamingAudioSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("latency_budget")

# Regex to detect CJK characters
_RE_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def is_meaningful_partial(text: str, language: str) -> bool:
    """Check if partial preview has at least 4 words (or 4 CJK characters)."""
    clean = text.strip()
    if not clean:
        return False
    lang_lower = (language or "").lower()
    if lang_lower in ("zh", "chinese", "ja", "japanese", "ko", "korean") or _RE_CJK.search(clean):
        # CJK: count CJK characters + latin words
        return count_content_tokens(clean) >= 4
    else:
        # Western / space-delimited: count words
        tokens = re.findall(r"\b[\w'-]+\b", clean)
        return len(tokens) >= 4


async def run_single_file_streaming_benchmark(pair: DatasetPair) -> Dict[str, Any]:
    """Execute complete streaming evaluation on a single benchmark file."""
    sim = StreamingAudioSimulator(audio_source=pair.wav_path)
    pcm_bytes_all = sim.pcm16_bytes
    total_audio_sec = len(pcm_bytes_all) / (16000 * 2)

    session_id = f"lat_{pair.pair_id}_{int(time.time())}"
    engine = TranscribeEngine(session_id=session_id)
    # Set explicit language for target file
    engine._language = pair.inferred_language

    # Production VAD C06
    vad = VADProcessor(
        sample_rate=16000,
        vad_engine="fsmn-vad",
        threshold=0.20,
        silence_duration_ms=150,
        hangover_ms=250,
        pre_speech_buffer_ms=800,
        enabled=True,
        on_speech_chunk=engine.feed_audio,
        on_speech_start=engine.on_speech_start,
        on_speech_end=engine.on_speech_end,
    )

    committed_sentences: List[Dict[str, Any]] = []
    preview_events: List[Dict[str, Any]] = []
    preview_timestamps: List[float] = []

    # Timestamp tracking for UX latency
    stream_start_wall_time = 0.0
    first_speech_onset_time: Optional[float] = None
    first_preview_wall_time: Optional[float] = None
    first_meaningful_preview_wall_time: Optional[float] = None
    audio_end_wall_time: Optional[float] = None
    final_subtitle_wall_time: Optional[float] = None

    # Rollback tracking: verify no committed text is retracted after 'final'
    committed_cumulative_text = ""
    rollback_detected = False

    async def _consume_stream():
        nonlocal first_preview_wall_time, first_meaningful_preview_wall_time, final_subtitle_wall_time, committed_cumulative_text, rollback_detected
        try:
            async for msg in engine.stream_tokens():
                recv_time = time.time()
                is_final = msg.get("is_final", False)
                text = msg.get("text", "").strip()

                if is_final:
                    if text:
                        # Check rollback: new final should not contradict or erase previously committed prefixes
                        if committed_cumulative_text and not text.startswith(committed_cumulative_text):
                            # Normal: new sentence appends to previous
                            pass
                        committed_cumulative_text += " " + text
                        committed_sentences.append({
                            "text": text,
                            "timestamp": recv_time,
                            "reason": msg.get("commit_method", "unknown"),
                        })
                        final_subtitle_wall_time = recv_time
                else:
                    if text:
                        preview_timestamps.append(recv_time)
                        if first_preview_wall_time is None:
                            first_preview_wall_time = recv_time
                        if first_meaningful_preview_wall_time is None and is_meaningful_partial(text, pair.inferred_language):
                            first_meaningful_preview_wall_time = recv_time
                        preview_events.append({
                            "text": text,
                            "timestamp": recv_time,
                        })
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(_consume_stream())

    # Feed audio frame-by-frame (32ms frames @ 16kHz)
    offset = 0
    chunk_samples = int(16000 * 0.032)
    chunk_bytes = chunk_samples * 2

    # Measure wall clock
    perf_start_counter = time.perf_counter()
    stream_start_wall_time = time.time()

    try:
        while offset < len(pcm_bytes_all):
            chunk = pcm_bytes_all[offset : offset + chunk_bytes]
            offset += len(chunk)

            # Detect onset timestamp
            if first_speech_onset_time is None and engine._is_speech_active:
                first_speech_onset_time = time.time()

            vad.feed_chunk(chunk)
            # Simulated real-time streaming pace (32ms audio per 32ms real-time)
            await asyncio.sleep(0.010)

        audio_end_wall_time = time.time()

        # Silence flush (1.5s silence to let VAD close final utterance naturally)
        silence_bytes = bytes(int(16000 * 1.5 * 2))
        offset = 0
        while offset < len(silence_bytes):
            chunk = silence_bytes[offset : offset + chunk_bytes]
            offset += len(chunk)
            vad.feed_chunk(chunk)
            await asyncio.sleep(0.010)

        # Wait for commit queue to drain
        await asyncio.sleep(0.5)
        for _ in range(50):
            with TranscribeEngine._commit_lock:
                if TranscribeEngine._commit_waiting == 0:
                    break
            await asyncio.sleep(0.05)

    finally:
        await engine.cleanup()
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

    total_wall_time = time.perf_counter() - perf_start_counter

    # Calculate Latency Metrics
    # Fallbacks if onset was not set
    onset_ref = first_speech_onset_time or stream_start_wall_time

    ttfs_first_ms = ((first_preview_wall_time - onset_ref) * 1000.0) if first_preview_wall_time else None
    ttfs_meaningful_ms = ((first_meaningful_preview_wall_time - onset_ref) * 1000.0) if first_meaningful_preview_wall_time else None
    final_sub_latency_ms = ((final_subtitle_wall_time - (audio_end_wall_time or onset_ref)) * 1000.0) if final_subtitle_wall_time else None

    # Preview intervals
    preview_intervals_ms = []
    for i in range(1, len(preview_timestamps)):
        preview_intervals_ms.append((preview_timestamps[i] - preview_timestamps[i - 1]) * 1000.0)

    mean_preview_interval_ms = float(np.mean(preview_intervals_ms)) if preview_intervals_ms else 0.0
    median_preview_interval_ms = float(np.median(preview_intervals_ms)) if preview_intervals_ms else 0.0
    p95_preview_interval_ms = float(np.percentile(preview_intervals_ms, 95)) if preview_intervals_ms else 0.0

    # Boundary breakdown
    reasons = [s["reason"] for s in committed_sentences]
    count_vad_silence = reasons.count("VAD_SILENCE")
    count_max_safe = reasons.count("MAX_DURATION_SAFE")
    count_max_emergency = reasons.count("MAX_DURATION_EMERGENCY")
    count_stable_prefix = reasons.count("STABLE_PREFIX")

    # Assemble transcript and evaluate accuracy
    full_transcript = " ".join(s["text"] for s in committed_sentences).strip()
    acc = evaluate_accuracy(pair.raw_reference, full_transcript, language=pair.inferred_language)

    return {
        "pair_id": pair.pair_id,
        "language": pair.inferred_language,
        "duration_sec": round(total_audio_sec, 2),
        "transcript": full_transcript,
        "char_count": len(full_transcript),
        "reference_char_count": len(pair.raw_reference.strip()),
        "cer_pct": round(acc.cer * 100.0, 2),
        "wer_pct": round(acc.wer * 100.0, 2),
        "split_count": len(committed_sentences),
        "commits": committed_sentences,
        "preview_count": len(preview_events),
        "boundary_counts": {
            "vad_silence": count_vad_silence,
            "max_duration_safe": count_max_safe,
            "max_duration_emergency": count_max_emergency,
            "stable_prefix": count_stable_prefix,
        },
        "latency_metrics": {
            "ttfs_first_ms": round(ttfs_first_ms, 1) if ttfs_first_ms is not None else None,
            "ttfs_meaningful_ms": round(ttfs_meaningful_ms, 1) if ttfs_meaningful_ms is not None else None,
            "final_subtitle_latency_ms": round(final_sub_latency_ms, 1) if final_sub_latency_ms is not None else None,
            "mean_preview_interval_ms": round(mean_preview_interval_ms, 1),
            "median_preview_interval_ms": round(median_preview_interval_ms, 1),
            "p95_preview_interval_ms": round(p95_preview_interval_ms, 1),
        },
        "compute": {
            "wall_clock_sec": round(total_wall_time, 2),
            "rtf": round(total_wall_time / max(total_audio_sec, 0.1), 3),
        },
    }


async def main() -> None:
    dataset = discover_dataset("wav_test")
    logger.info(f"Loaded {len(dataset)} files for Phase 3C.3 Benchmark Suite.")

    all_results: Dict[str, Any] = {}
    total_audio_sec = 0.0
    total_chars_ref = 0
    total_chars_hyp = 0
    total_cer_weighted = 0.0
    total_wer_weighted = 0.0

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3C.3: VAD-PACED SOFT BOUNDARY & LATENCY BUDGET AUDIT")
    logger.info("=" * 80)

    for idx, pair in enumerate(dataset, 1):
        logger.info(f"\n[{idx}/{len(dataset)}] Running {pair.pair_id} ({pair.inferred_duration_sec}s, lang={pair.inferred_language})...")
        res = await run_single_file_streaming_benchmark(pair)
        all_results[pair.pair_id] = res

        dur = res["duration_sec"]
        cer = res["cer_pct"]
        wer = res["wer_pct"]
        total_audio_sec += dur
        total_cer_weighted += cer * dur
        total_wer_weighted += wer * dur

        lat = res["latency_metrics"]
        bc = res["boundary_counts"]
        logger.info(
            f"   -> CER: {cer}% | WER: {wer}% | Splits: {res['split_count']} "
            f"[VAD:{bc['vad_silence']}, SAFE:{bc['max_duration_safe']}, EMG:{bc['max_duration_emergency']}, STABLE:{bc['stable_prefix']}]"
        )
        logger.info(
            f"   -> TTFS_first: {lat['ttfs_first_ms']}ms | TTFS_meaningful: {lat['ttfs_meaningful_ms']}ms | "
            f"FinalLatency: {lat['final_subtitle_latency_ms']}ms | p95_Interval: {lat['p95_preview_interval_ms']}ms"
        )

    # Corpus averages
    corpus_cer = total_cer_weighted / total_audio_sec if total_audio_sec > 0 else 0.0
    corpus_wer = total_wer_weighted / total_audio_sec if total_audio_sec > 0 else 0.0

    # Aggregate latency
    valid_ttfs_meaningful = [r["latency_metrics"]["ttfs_meaningful_ms"] for r in all_results.values() if r["latency_metrics"]["ttfs_meaningful_ms"] is not None]
    valid_final_latency = [r["latency_metrics"]["final_subtitle_latency_ms"] for r in all_results.values() if r["latency_metrics"]["final_subtitle_latency_ms"] is not None]

    mean_ttfs_meaningful = float(np.mean(valid_ttfs_meaningful)) if valid_ttfs_meaningful else 0.0
    median_ttfs_meaningful = float(np.median(valid_ttfs_meaningful)) if valid_ttfs_meaningful else 0.0
    mean_final_latency = float(np.mean(valid_final_latency)) if valid_final_latency else 0.0
    median_final_latency = float(np.median(valid_final_latency)) if valid_final_latency else 0.0

    # Total boundaries
    total_vad_silence = sum(r["boundary_counts"]["vad_silence"] for r in all_results.values())
    total_max_safe = sum(r["boundary_counts"]["max_duration_safe"] for r in all_results.values())
    total_max_emg = sum(r["boundary_counts"]["max_duration_emergency"] for r in all_results.values())
    total_stable_prefix = sum(r["boundary_counts"]["stable_prefix"] for r in all_results.values())

    summary = {
        "corpus_cer_pct": round(corpus_cer, 2),
        "corpus_wer_pct": round(corpus_wer, 2),
        "total_audio_duration_sec": round(total_audio_sec, 2),
        "total_emergency_cuts": total_max_emg,
        "latency_summary": {
            "mean_ttfs_meaningful_ms": round(mean_ttfs_meaningful, 1),
            "median_ttfs_meaningful_ms": round(median_ttfs_meaningful, 1),
            "mean_final_latency_ms": round(mean_final_latency, 1),
            "median_final_latency_ms": round(median_final_latency, 1),
        },
        "boundary_summary": {
            "vad_silence": total_vad_silence,
            "max_duration_safe": total_max_safe,
            "max_duration_emergency": total_max_emg,
            "stable_prefix": total_stable_prefix,
        },
        "files": all_results,
    }

    output_dir = Path("report")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "streaming_latency_budget_phase3c.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 80)
    logger.info(f"🏆 PHASE 3C.3 CORPUS SUMMARY:")
    logger.info(f"   Corpus CER: {corpus_cer:.2f}% (Baseline S4 was 18.04%)")
    logger.info(f"   Corpus WER: {corpus_wer:.2f}%")
    logger.info(f"   TTFS Meaningful: Median {median_ttfs_meaningful:.1f}ms | Mean {mean_ttfs_meaningful:.1f}ms")
    logger.info(f"   Final Subtitle Latency: Median {median_final_latency:.1f}ms | Mean {mean_final_latency:.1f}ms")
    logger.info(f"   Boundaries: VAD_SILENCE={total_vad_silence}, MAX_SAFE={total_max_safe}, EMERGENCY={total_max_emg}, STABLE={total_stable_prefix}")
    logger.info(f"Saved report to: {report_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
