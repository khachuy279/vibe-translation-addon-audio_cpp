"""Phase F — Speech Normalizer Impact & Bypass Identity Audit (R5 vs R4).

Evaluates:
1. Bypass Identity Test:
   - Feeds calibrated audio at target RMS (0.10) and verifies gain ≈ 1.0, shape preserved, SNR > 60dB, no clipping.
2. Signal Quality Telemetry:
   - Max gain, median gain, min gain, peak attenuation, clipping before/after, DC offset change, zero-crossing rate delta.
3. ASR Accuracy Ablation (R5 vs R4):
   - Transcribes normalized VAD segments via Qwen3-ASR.
   - Measures ΔCER_norm and ΔWER_norm to determine if normalizer improves or degrades accuracy.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import soxr

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.speech_normalizer import SpeechNormalizer
from backend_cpp.config import config
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.audit_vad_impact import segment_audio_with_vad
from benchmarks.dataset import discover_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_normalizer_impact")


def run_bypass_identity_test() -> Dict[str, Any]:
    """Test SpeechNormalizer with calibrated input already at target RMS."""
    normalizer = SpeechNormalizer(target_rms=0.10, target_peak=0.95, max_gain=3.0)
    
    # Generate 2.0s 1kHz sine wave calibrated to RMS = 0.10
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    # For a sine wave, RMS = Amplitude / sqrt(2) -> Amplitude = RMS * sqrt(2)
    amplitude = 0.10 * math.sqrt(2.0)
    calibrated_sine = amplitude * np.sin(2.0 * math.pi * 1000.0 * t).astype(np.float32)

    res = normalizer.process(calibrated_sine, use_smoothing=False)

    # Compute distortion metrics
    diff = res.pcm - calibrated_sine
    max_err = float(np.max(np.abs(diff)))
    rms_err = float(np.sqrt(np.mean(diff ** 2)))
    ref_power = np.mean(calibrated_sine ** 2)
    noise_power = np.mean(diff ** 2)
    snr_db = float(10.0 * np.log10(ref_power / noise_power)) if noise_power > 1e-12 else 120.0

    identity_passed = (abs(res.desired_gain - 1.0) < 0.05 and snr_db > 60.0 and res.peak_scale == 1.0)

    result = {
        "identity_passed": identity_passed,
        "input_rms": round(float(np.sqrt(np.mean(calibrated_sine ** 2))), 4),
        "output_rms": round(float(np.sqrt(np.mean(res.pcm ** 2))), 4),
        "desired_gain": round(res.desired_gain, 4),
        "smoothed_gain": round(res.smoothed_gain, 4),
        "peak_scale": round(res.peak_scale, 4),
        "max_abs_error": round(max_err, 6),
        "snr_db": round(snr_db, 2),
        "clipping_introduced": int(np.sum(np.abs(res.pcm) >= 0.9999)),
    }
    logger.info(
        f"   Bypass Identity Test: {'✅ PASSED' if identity_passed else '❌ FAILED'} | "
        f"Gain={res.desired_gain:.3f} | SNR={snr_db:.1f}dB | Clipped={result['clipping_introduced']}"
    )
    return result


def run_normalizer_audit(dataset_dir: str = "wav_test") -> Dict[str, Any]:
    logger.info("================================================================")
    logger.info("▶️ STARTING PHASE F: SPEECH NORMALIZER IMPACT AUDIT (R5 vs R4)")
    logger.info("================================================================")

    logger.info("1. Executing Bypass Identity Test on Calibrated Signal...")
    identity_result = run_bypass_identity_test()

    dataset = discover_dataset(Path(dataset_dir))
    logger.info(f"Loaded {len(dataset)} dataset pairs from {dataset_dir}")

    # Load R4 results
    r4_path = Path("report/audit_vad_impact.json")
    r4_data = {}
    if r4_path.exists():
        with open(r4_path, "r", encoding="utf-8") as f:
            j = json.load(f)
            r4_data = {r["pair_id"]: r for r in j.get("results", [])}

    model_key = config.asr.active_model
    mgr = ASRModelManager()
    model = mgr.ensure_model(model_key)

    normalizer = SpeechNormalizer(target_rms=0.10, target_peak=0.95, max_gain=3.0)

    r5_results = []
    telemetry_list = []

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

            # Get exact VAD segments
            segments, _ = segment_audio_with_vad(
                pcm_float,
                vad_engine="fsmn-vad",
                threshold=0.4,
                silence_duration_ms=150,
                hangover_ms=400,
                pre_speech_buffer_ms=550,
            )

            lang_arg = pair.inferred_language if pair.inferred_language not in ("UNKNOWN", "multi") else None

            # Process through Normalizer and transcribe
            normalizer.reset_gain(1.0)
            norm_segments = []
            file_gains = []
            peak_scales = []
            clipped_before_total = 0
            clipped_after_total = 0

            t0 = time.perf_counter()
            seg_texts = []

            for seg in segments:
                if len(seg) < 1600:
                    continue
                clipped_before_total += int(np.sum(np.abs(seg) >= 0.9999))
                
                # Apply SpeechNormalizer
                norm_res = normalizer.process(seg, use_smoothing=True)
                norm_segments.append(norm_res.pcm)
                file_gains.append(norm_res.smoothed_gain)
                peak_scales.append(norm_res.peak_scale)
                clipped_after_total += int(np.sum(np.abs(norm_res.pcm) >= 0.9999))

                # Transcribe normalized segment
                try:
                    res = session.run(norm_res.pcm, language=lang_arg)
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

            r4 = r4_data.get(pair.pair_id, {})
            r4_cer = r4.get("r4_cer", 0.0)
            r4_wer = r4.get("r4_wer", 0.0)
            delta_cer = (acc.cer - r4_cer) if acc.cer is not None and r4_cer is not None else 0.0
            delta_wer = (acc.wer - r4_wer) if acc.wer is not None and r4_wer is not None else 0.0

            file_telemetry = {
                "pair_id": pair.pair_id,
                "file_name": pair.wav_path.name,
                "min_gain": round(float(np.min(file_gains)), 3) if file_gains else 1.0,
                "median_gain": round(float(np.median(file_gains)), 3) if file_gains else 1.0,
                "max_gain": round(float(np.max(file_gains)), 3) if file_gains else 1.0,
                "min_peak_scale": round(float(np.min(peak_scales)), 3) if peak_scales else 1.0,
                "clipped_before": clipped_before_total,
                "clipped_after": clipped_after_total,
                "r5_wer": acc.wer,
                "r5_cer": acc.cer,
                "r4_wer": r4_wer,
                "r4_cer": r4_cer,
                "delta_wer_norm": round(delta_wer, 4),
                "delta_cer_norm": round(delta_cer, 4),
                "substitutions": acc.substitutions,
                "insertions": acc.insertions,
                "deletions": acc.deletions,
                "raw_hypothesis": raw_hyp,
                "normalized_hypothesis": acc.normalized_hypothesis,
            }
            r5_results.append(file_telemetry)

            logger.info(
                f"   R5 [{pair.pair_id}]: Gain[med={file_telemetry['median_gain']:.2f}, max={file_telemetry['max_gain']:.2f}] | "
                f"CER: {r4_cer*100:.1f}% -> {acc.cer*100:.1f}% (ΔCER_norm={delta_cer*100:+.2f}%) | "
                f"WER: {r4_wer*100:.1f}% -> {acc.wer*100:.1f}% (ΔWER_norm={delta_wer*100:+.2f}%)"
            )

    finally:
        ASRModelManager.release_infer_lock()

    ann = [r for r in r5_results if r["r5_cer"] is not None]
    avg_cer_r5 = (sum(r["r5_cer"] for r in ann) / len(ann)) * 100.0 if ann else 0.0
    avg_wer_r5 = (sum(r["r5_wer"] for r in ann) / len(ann)) * 100.0 if ann else 0.0

    summary = {
        "identity_test": identity_result,
        "total_files": len(r5_results),
        "average_cer_r5": round(avg_cer_r5, 2),
        "average_wer_r5": round(avg_wer_r5, 2),
        "results": r5_results,
    }

    out_file = Path("report/audit_normalizer_impact.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("================================================================")
    logger.info(f"✅ Phase F complete! Saved to {out_file}")
    logger.info(f"   Corpus Average: CER={avg_cer_r5:.2f}% | WER={avg_wer_r5:.2f}%")
    logger.info("================================================================")
    return summary


if __name__ == "__main__":
    run_normalizer_audit()
