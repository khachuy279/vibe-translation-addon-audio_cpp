"""Phase A — Direct Native ASR Baseline (R1).

Executes raw audio directly through transcribe_cpp session.run().
Bypasses:
- VAD
- AudioBufferManager
- SpeechNormalizer
- WebSocket Transport
- Extension Simulator
- Sentence Segmentation / Subtitles

Distinguishes:
- Reference-RAW: native 16kHz mono samples (no resampling applied)
- Reference-ASR: exact production-equivalent downsampled 16kHz mono float32
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import soxr

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.config import config
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.dataset import DatasetPair, discover_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_direct_asr")


def load_audio_reference(wav_path: Path) -> np.ndarray:
    """Load audio with strict Reference-RAW vs Reference-ASR separation."""
    info = sf.info(str(wav_path))
    audio_data, sr = sf.read(str(wav_path), dtype="float32")

    # Downmix to mono if multi-channel
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # If already 16kHz mono PCM, do NOT resample (Reference-RAW)
    if sr == 16000:
        return audio_data

    # Production-equivalent polyphase sinc resampler to 16kHz (Reference-ASR)
    resampled = soxr.resample(audio_data, in_rate=sr, out_rate=16000, quality="HQ")
    return np.clip(resampled, -1.0, 1.0).astype(np.float32)


def run_direct_asr_baseline(dataset_dir: str = "wav_test") -> Dict[str, Any]:
    logger.info("================================================================")
    logger.info("▶️ STARTING PHASE A: DIRECT NATIVE ASR BASELINE (R1)")
    logger.info("================================================================")

    dataset = discover_dataset(Path(dataset_dir))
    logger.info(f"Loaded {len(dataset)} dataset pairs from {dataset_dir}")

    model_key = config.asr.active_model
    mgr = ASRModelManager()
    model = mgr.ensure_model(model_key)

    results: List[Dict[str, Any]] = []

    acquired = ASRModelManager.acquire_infer_lock(blocking=True)
    try:
        session = mgr.ensure_session(model, threads=4)

        for pair in dataset:
            logger.info(f"Testing {pair.pair_id} ({pair.actual_duration_sec:.2f}s, lang={pair.inferred_language})...")
            pcm = load_audio_reference(pair.wav_path)
            dur_sec = len(pcm) / 16000.0

            lang_arg = pair.inferred_language if pair.inferred_language not in ("UNKNOWN", "multi") else None

            t0 = time.perf_counter()
            max_window_samples = 30 * 16000
            
            if len(pcm) <= max_window_samples:
                try:
                    res = session.run(pcm, language=lang_arg)
                    raw_hyp = getattr(res, "text", str(res)).strip()
                except Exception as run_err:
                    partial = getattr(run_err, "partial_result", None)
                    if partial is not None and hasattr(partial, "text"):
                        raw_hyp = partial.text.strip()
                    else:
                        raise
            else:
                # Audio exceeds 30s context window: decode in 30s consecutive windows
                chunk_texts = []
                for start_idx in range(0, len(pcm), max_window_samples):
                    chunk = pcm[start_idx:start_idx + max_window_samples]
                    if len(chunk) < 1600: # < 100ms
                        continue
                    try:
                        res = session.run(chunk, language=lang_arg)
                        txt = getattr(res, "text", str(res)).strip()
                        if txt:
                            chunk_texts.append(txt)
                    except Exception as run_err:
                        partial = getattr(run_err, "partial_result", None)
                        if partial is not None and hasattr(partial, "text") and partial.text.strip():
                            chunk_texts.append(partial.text.strip())
                raw_hyp = " ".join(chunk_texts).strip()

            infer_wall = time.perf_counter() - t0
            rtf_asr = infer_wall / max(0.001, dur_sec)

            acc = evaluate_accuracy(
                raw_reference=pair.raw_reference,
                raw_hypothesis=raw_hyp,
                language=pair.inferred_language,
            )

            record = {
                "file_name": pair.wav_path.name,
                "pair_id": pair.pair_id,
                "language": pair.inferred_language,
                "audio_duration_sec": round(dur_sec, 2),
                "sample_count": len(pcm),
                "inference_ms": round(infer_wall * 1000.0, 1),
                "rtf_asr": round(rtf_asr, 3),
                "raw_reference": pair.raw_reference,
                "raw_hypothesis": raw_hyp,
                "normalized_reference": acc.normalized_reference,
                "normalized_hypothesis": acc.normalized_hypothesis,
                "is_annotated": acc.is_annotated,
                "wer": acc.wer,
                "cer": acc.cer,
                "substitutions": acc.substitutions,
                "insertions": acc.insertions,
                "deletions": acc.deletions,
                "ref_length": acc.ref_length,
                "hyp_length": acc.hyp_length,
            }
            results.append(record)
            logger.info(
                f"   Result: WER={record['wer']*100:.1f}% | CER={record['cer']*100:.1f}% | "
                f"Latency={record['inference_ms']}ms (RTF: {record['rtf_asr']}) | "
                f"Hyp: '{raw_hyp[:60]}...'" if len(raw_hyp) > 60 else f"Hyp: '{raw_hyp}'"
            )

    finally:
        ASRModelManager.release_infer_lock()

    # Aggregate metrics
    annotated = [r for r in results if r["is_annotated"]]
    avg_wer = (sum(r["wer"] for r in annotated if r["wer"] is not None) / len(annotated) * 100.0) if annotated else 0.0
    avg_cer = (sum(r["cer"] for r in annotated if r["cer"] is not None) / len(annotated) * 100.0) if annotated else 0.0
    avg_rtf = sum(r["rtf_asr"] for r in results) / len(results) if results else 0.0
    total_audio_sec = sum(r["audio_duration_sec"] for r in results)
    total_infer_sec = sum(r["inference_ms"] for r in results) / 1000.0

    summary = {
        "model_key": model_key,
        "total_files": len(results),
        "annotated_files": len(annotated),
        "total_audio_sec": round(total_audio_sec, 2),
        "total_infer_sec": round(total_infer_sec, 2),
        "corpus_rtf_asr": round(total_infer_sec / total_audio_sec, 3),
        "average_wer": round(avg_wer, 2),
        "average_cer": round(avg_cer, 2),
        "results": results,
    }

    out_file = Path("report/audit_direct_asr.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("================================================================")
    logger.info(f"✅ Phase A R1 Direct Baseline complete! Summary saved to {out_file}")
    logger.info(f"   Corpus Average: CER={avg_cer:.2f}% | WER={avg_wer:.2f}% | RTF_ASR={summary['corpus_rtf_asr']}")
    logger.info("================================================================")

    return summary


if __name__ == "__main__":
    run_direct_asr_baseline()
