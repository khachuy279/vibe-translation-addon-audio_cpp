"""Phase 3C.2: Forced Boundary Counterfactual Matrix (D0-D4) on Chinese_fast_speed_11s.

Evaluates:
- D0: max_duration_sec = 8.0s (Baseline streaming)
- D1: max_duration_sec = 10.0s
- D2: max_duration_sec = 12.0s
- D3: max_duration_sec = 15.0s
- D4: max_duration_sec = float('inf') (VAD-only natural boundaries)

Under exact streaming simulation:
- 16kHz chunking (same as WebSocket)
- C06 VAD frozen
- Phase 3C.1 Quick Wins (growth=0.5, min_words=4, stability=1.5s)
- Context stitching OFF
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from backend_cpp.asr.audio_buffer import AudioBufferManager
VAD_STATE_SPEECH = 1
VAD_STATE_SILENCE = 0
from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config as app_cfg
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.simulator import StreamingAudioSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("counterfactual_max_duration")


async def run_streaming_simulation_for_max_duration(
    wav_path: Path,
    ground_truth: str,
    max_duration_sec: float,
    condition_id: str,
) -> Dict[str, Any]:
    sim = StreamingAudioSimulator(audio_source=wav_path)
    pcm_bytes_all = sim.pcm16_bytes

    # Instantiate engine with current production Quick Wins
    session_id = f"sim_{condition_id}_{int(time.time())}"
    engine = TranscribeEngine(session_id=session_id)
    
    # Configure exact max_duration_sec
    engine.sentence_config.max_duration_sec = max_duration_sec
    engine._segmenter.max_duration_sec = max_duration_sec
    engine.sentence_config.min_words_to_commit = 4
    engine.sentence_config.stability_duration_sec = 1.5
    engine.sentence_config.split_on_stability = True
    engine._preview_min_growth_ratio = 0.5

    # C06 VAD
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
    # Consume from stream_tokens in background task
    async def _consume_stream():
        try:
            async for msg in engine.stream_tokens():
                if msg.get("is_final"):
                    text = msg.get("text", "").strip()
                    if text:
                        committed_sentences.append({
                            "text": text,
                            "timestamp": time.time(),
                            "reason": msg.get("commit_method", "unknown"),
                        })
                else:
                    text = msg.get("text", "").strip()
                    if text:
                        preview_events.append({
                            "text": text,
                            "timestamp": time.time(),
                        })
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(_consume_stream())

    # Feed audio frame-by-frame (32ms per frame)
    offset = 0
    t_start = time.perf_counter()

    try:
        chunk_samples = int(16000 * 0.032)  # 32ms frames
        chunk_bytes = chunk_samples * 2
        while offset < len(pcm_bytes_all):
            chunk = pcm_bytes_all[offset : offset + chunk_bytes]
            offset += len(chunk)
            vad.feed_chunk(chunk)
            await asyncio.sleep(0.010)

        # Silence flush
        silence_bytes = bytes(int(16000 * 1.5 * 2))
        offset = 0
        while offset < len(silence_bytes):
            chunk = silence_bytes[offset : offset + chunk_bytes]
            offset += len(chunk)
            vad.feed_chunk(chunk)
            await asyncio.sleep(0.010)

        # Wait for commits to settle
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

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Assemble final transcript
    full_transcript = " ".join(s["text"] for s in committed_sentences).strip()
    acc = evaluate_accuracy(ground_truth, full_transcript, language="zh")

    return {
        "condition_id": condition_id,
        "max_duration_sec": max_duration_sec if max_duration_sec != float("inf") else "inf",
        "split_count": len(committed_sentences),
        "committed_sentences": committed_sentences,
        "full_transcript": full_transcript,
        "char_count": len(full_transcript),
        "reference_char_count": len(ground_truth),
        "cer_pct": round(acc.cer * 100.0, 2),
        "wer_pct": round(acc.wer * 100.0, 2),
        "preview_count": len(preview_events),
        "elapsed_ms": round(elapsed_ms, 1),
    }


async def main() -> None:
    wav_path = Path("wav_test/Chinese_fast_speed_11s.wav")
    txt_path = Path("wav_test/Chinese_fast_speed_11s.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()

    logger.info(f"Loaded ground truth ({len(ground_truth)} chars): '{ground_truth[:60]}...'")

    matrix_conditions = [
        ("D0", 8.0, "Baseline (8.0s)"),
        ("D1", 10.0, "Continuous (10.0s)"),
        ("D2", 12.0, "Continuous (12.0s)"),
        ("D3", 15.0, "Saturation (15.0s)"),
        ("D4", float("inf"), "VAD-Only Natural (inf)"),
    ]

    results: Dict[str, Any] = {}

    for cid, max_dur, desc in matrix_conditions:
        logger.info(f"\n==========================================")
        logger.info(f"Running Condition {cid}: max_duration_sec={max_dur} ({desc})")
        logger.info(f"==========================================")
        res = await run_streaming_simulation_for_max_duration(
            wav_path=wav_path,
            ground_truth=ground_truth,
            max_duration_sec=max_dur,
            condition_id=cid,
        )
        results[cid] = res
        logger.info(f"Result {cid}: Splits={res['split_count']} | CER={res['cer_pct']}% | Transcript ({res['char_count']} chars):")
        for idx, s in enumerate(res["committed_sentences"]):
            logger.info(f"   Part {idx+1}: '{s['text']}'")
        logger.info(f"   Full: '{res['full_transcript']}'")

    output_path = Path("report/counterfactual_max_duration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
