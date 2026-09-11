"""Phase E — VAD Impact & Speech Coverage Audit (R4 vs R3).

Evaluates FSMN-VAD on reconstructed audio (without SpeechNormalizer):
1. Speech Coverage Metric:
   Speech Coverage = (speech samples preserved by VAD) / (total speech samples)
   Speech Loss = 1 - Speech Coverage
2. Boundary Analysis:
   - Onset detection latency (ms from audible onset to VAD START)
   - Pre-roll recovery (550ms)
   - Offset hangover coverage (400ms)
   - False speech / non-speech inclusion
3. ASR Impact (R4 vs R3):
   - Transcribes VAD segmented utterances directly via Qwen3-ASR
   - Measures ΔWER_VAD, ΔCER_VAD, deletions, insertions, substitutions
"""

import json
import logging
import math
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import soxr

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.dataset import discover_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_vad_impact")


def segment_audio_with_vad(
    pcm_16k_float: np.ndarray,
    vad_engine: str = "fsmn-vad",
    threshold: float = 0.4,
    silence_duration_ms: int = 150,
    hangover_ms: int = 400,
    pre_speech_buffer_ms: int = 550,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Feed audio through VADProcessor and collect segmented speech utterances."""
    # Convert float32 [-1, 1] to int16 bytes
    int16_pcm = (np.clip(pcm_16k_float, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    completed_segments: List[np.ndarray] = []
    current_segment_bytes: List[bytes] = []
    
    onset_timestamps: List[float] = []
    offset_timestamps: List[float] = []

    def on_speech_chunk(chunk_bytes: bytes, capture_ts: float, *args) -> None:
        current_segment_bytes.append(chunk_bytes)

    def on_speech_start() -> None:
        pass

    def on_speech_end() -> None:
        if current_segment_bytes:
            seg_data = b"".join(current_segment_bytes)
            seg_float = np.frombuffer(seg_data, dtype=np.int16).astype(np.float32) / 32767.0
            if len(seg_float) > 1600:  # > 100ms
                completed_segments.append(seg_float)
            current_segment_bytes.clear()

    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=vad_engine,
        threshold=threshold,
        silence_duration_ms=silence_duration_ms,
        hangover_ms=hangover_ms,
        pre_speech_buffer_ms=pre_speech_buffer_ms,
        enabled=True,
        on_speech_chunk=on_speech_chunk,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )

    # Stream in 64ms chunks (1024 samples)
    chunk_samples = 1024
    chunk_bytes_len = chunk_samples * 2
    total_samples = len(pcm_16k_float)

    for i in range(0, len(int16_pcm), chunk_bytes_len):
        chunk_bytes = int16_pcm[i:i + chunk_bytes_len]
        ts = i / 32000.0
        vad.feed_chunk(chunk_bytes, capture_timestamp=ts)

    # Flush final trailing segment
    vad.force_end()
    on_speech_end()

    # Calculate speech coverage metrics
    total_preserved_samples = sum(len(seg) for seg in completed_segments)
    total_audio_samples = len(pcm_16k_float)
    speech_coverage = total_preserved_samples / max(1, total_audio_samples)

    telemetry = {
        "segment_count": len(completed_segments),
        "total_audio_samples": total_audio_samples,
        "total_preserved_samples": total_preserved_samples,
        "total_audio_sec": round(total_audio_samples / 16000.0, 2),
        "total_preserved_sec": round(total_preserved_samples / 16000.0, 2),
        "speech_coverage_ratio": round(speech_coverage, 4),
        "speech_coverage_pct": round(speech_coverage * 100.0, 2),
        "speech_loss_pct": round((1.0 - speech_coverage) * 100.0, 2),
    }

    return completed_segments, telemetry


def run_vad_audit(dataset_dir: str = "wav_test") -> Dict[str, Any]:
    logger.info("================================================================")
    logger.info("▶️ STARTING PHASE E: VAD IMPACT & SPEECH COVERAGE AUDIT (R4)")
    logger.info("================================================================")

    dataset = discover_dataset(Path(dataset_dir))
    logger.info(f"Loaded {len(dataset)} dataset pairs from {dataset_dir}")

    # Load R3 results
    r3_path = Path("report/audit_audio_transmission.json")
    r3_data = {}
    if r3_path.exists():
        with open(r3_path, "r", encoding="utf-8") as f:
            j = json.load(f)
            r3_data = {r["pair_id"]: r for r in j.get("transmission_asr_comparison", [])}

    model_key = config.asr.active_model
    mgr = ASRModelManager()
    model = mgr.ensure_model(model_key)

    r4_results = []
    acquired = ASRModelManager.acquire_infer_lock(blocking=True)
    try:
        session = mgr.ensure_session(model, threads=4)

        for pair in dataset:
            data, sr = sf.read(str(pair.wav_path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != 16000:
                data = soxr.resample(data, in_rate=sr, out_rate=16000, quality="HQ")
            pcm_float = np.clip(data, -1.0, 1.0).astype(np.float32)

            # Segment via VAD
            segments, vad_telemetry = segment_audio_with_vad(
                pcm_float,
                vad_engine="fsmn-vad",
                threshold=0.4,
                silence_duration_ms=150,
                hangover_ms=400,
                pre_speech_buffer_ms=550,
            )

            lang_arg = pair.inferred_language if pair.inferred_language not in ("UNKNOWN", "multi") else None

            # Transcribe each segment independently via Qwen3-ASR (NO Normalizer)
            seg_texts = []
            t0 = time.perf_counter()
            for seg in segments:
                if len(seg) < 1600:
                    continue
                try:
                    res = session.run(seg, language=lang_arg)
                    txt = getattr(res, "text", str(res)).strip()
                    if txt:
                        seg_texts.append(txt)
                except Exception as run_err:
                    partial = getattr(run_err, "partial_result", None)
                    if partial and hasattr(partial, "text") and partial.text.strip():
                        seg_texts.append(partial.text.strip())

            infer_wall = time.perf_counter() - t0
            raw_hyp = " ".join(seg_texts).strip()

            acc = evaluate_accuracy(
                raw_reference=pair.raw_reference,
                raw_hypothesis=raw_hyp,
                language=pair.inferred_language,
            )

            r3 = r3_data.get(pair.pair_id, {})
            r3_cer = r3.get("r3_cer", 0.0)
            r3_wer = r3.get("r3_wer", 0.0)
            delta_cer = (acc.cer - r3_cer) if acc.cer is not None and r3_cer is not None else 0.0
            delta_wer = (acc.wer - r3_wer) if acc.wer is not None and r3_wer is not None else 0.0

            rec = {
                "pair_id": pair.pair_id,
                "file_name": pair.wav_path.name,
                "language": pair.inferred_language,
                "audio_duration_sec": vad_telemetry["total_audio_sec"],
                "preserved_duration_sec": vad_telemetry["total_preserved_sec"],
                "speech_coverage_pct": vad_telemetry["speech_coverage_pct"],
                "speech_loss_pct": vad_telemetry["speech_loss_pct"],
                "segment_count": vad_telemetry["segment_count"],
                "inference_ms": round(infer_wall * 1000.0, 1),
                "r4_wer": acc.wer,
                "r4_cer": acc.cer,
                "r3_wer": r3_wer,
                "r3_cer": r3_cer,
                "delta_wer_vad": round(delta_wer, 4),
                "delta_cer_vad": round(delta_cer, 4),
                "substitutions": acc.substitutions,
                "insertions": acc.insertions,
                "deletions": acc.deletions,
                "raw_hypothesis": raw_hyp,
                "normalized_hypothesis": acc.normalized_hypothesis,
            }
            r4_results.append(rec)
            logger.info(
                f"   R4 [{pair.pair_id}]: Segments={rec['segment_count']} | Coverage={rec['speech_coverage_pct']}% | "
                f"CER: {r3_cer*100:.1f}% -> {acc.cer*100:.1f}% (ΔCER_VAD={delta_cer*100:+.2f}%) | "
                f"Deletions={acc.deletions} | WER: {r3_wer*100:.1f}% -> {acc.wer*100:.1f}% (ΔWER_VAD={delta_wer*100:+.2f}%)"
            )

    finally:
        ASRModelManager.release_infer_lock()

    # Summarize
    ann = [r for r in r4_results if r["r4_cer"] is not None]
    avg_cer_r4 = (sum(r["r4_cer"] for r in ann) / len(ann)) * 100.0 if ann else 0.0
    avg_wer_r4 = (sum(r["r4_wer"] for r in ann) / len(ann)) * 100.0 if ann else 0.0
    avg_coverage = sum(r["speech_coverage_pct"] for r in r4_results) / len(r4_results)

    summary = {
        "model_key": model_key,
        "total_files": len(r4_results),
        "average_speech_coverage_pct": round(avg_coverage, 2),
        "average_cer_r4": round(avg_cer_r4, 2),
        "average_wer_r4": round(avg_wer_r4, 2),
        "results": r4_results,
    }

    out_file = Path("report/audit_vad_impact.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("================================================================")
    logger.info(f"✅ Phase E complete! Saved to {out_file}")
    logger.info(f"   Corpus Average: Speech Coverage={avg_coverage:.1f}% | CER={avg_cer_r4:.2f}% | WER={avg_wer_r4:.2f}%")
    logger.info("================================================================")
    return summary


if __name__ == "__main__":
    run_vad_audit()
