"""Phase C & Phase D — Audio Transmission & Conversion Fidelity Audit.

Decouples:
1. Transport Loss: sent[i] vs received[i] over binary WebSocket Format A
   across chunk sizes: 20ms, 40ms, 64ms (production default), 100ms, 200ms.
2. Conversion Loss: Source WAV -> 16kHz mono Int16 PCM (resampling, quantization noise, clipping, SNR).
3. Transmission -> ASR (R3 vs R1): Feed reconstructed PCM directly into Qwen3-ASR without VAD or Normalizer.
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
from backend_cpp.ws.frame_protocol import parse_audio_frame
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.dataset import discover_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_audio_transmission")

CHUNK_SIZES_MS = [20, 40, 64, 100, 200]


def analyze_waveform_metrics(ref: np.ndarray, test: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive acoustic distortion metrics between reference and test signals."""
    min_len = min(len(ref), len(test))
    ref_trim = ref[:min_len]
    test_trim = test[:min_len]

    # Max absolute error & RMS error
    abs_err = np.abs(test_trim - ref_trim)
    max_abs_err = float(np.max(abs_err)) if len(abs_err) > 0 else 0.0
    rms_err = float(np.sqrt(np.mean((test_trim - ref_trim) ** 2))) if len(abs_err) > 0 else 0.0

    # Signal-to-Noise Ratio (SNR in dB)
    ref_power = np.mean(ref_trim ** 2)
    noise_power = np.mean((test_trim - ref_trim) ** 2)
    if noise_power > 1e-12 and ref_power > 1e-12:
        snr_db = float(10.0 * np.log10(ref_power / noise_power))
    else:
        snr_db = 120.0  # Essentially lossless

    # Peak, RMS, clipping
    peak_val = float(np.max(np.abs(test))) if len(test) > 0 else 0.0
    rms_val = float(np.sqrt(np.mean(test ** 2))) if len(test) > 0 else 0.0
    clipping_samples = int(np.sum(np.abs(test) >= 0.9999))
    has_nan = bool(np.isnan(test).any())
    has_inf = bool(np.isinf(test).any())

    return {
        "sample_count_ref": len(ref),
        "sample_count_test": len(test),
        "sample_count_delta": len(test) - len(ref),
        "max_abs_error": round(max_abs_err, 6),
        "rms_error": round(rms_err, 6),
        "snr_db": round(snr_db, 2),
        "peak": round(peak_val, 4),
        "rms": round(rms_val, 4),
        "clipping_samples": clipping_samples,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def run_transmission_audit(dataset_dir: str = "wav_test") -> Dict[str, Any]:
    logger.info("================================================================")
    logger.info("▶️ STARTING PHASE C & D: AUDIO TRANSMISSION & ASR FIDELITY AUDIT")
    logger.info("================================================================")

    dataset = discover_dataset(Path(dataset_dir))
    logger.info(f"Loaded {len(dataset)} dataset pairs from {dataset_dir}")

    report_data: Dict[str, Any] = {
        "transport_fidelity": {},
        "conversion_fidelity": {},
        "transmission_asr_comparison": {},
    }

    # Load Direct ASR baseline for comparison (R1)
    direct_baseline_path = Path("report/audit_direct_asr.json")
    r1_results = {}
    if direct_baseline_path.exists():
        with open(direct_baseline_path, "r", encoding="utf-8") as f:
            r1_json = json.load(f)
            r1_results = {r["pair_id"]: r for r in r1_json.get("results", [])}

    # -------------------------------------------------------------------------
    # PART 1: Transport Fidelity (sent[i] vs received[i] across chunk sizes)
    # -------------------------------------------------------------------------
    logger.info("----------------------------------------------------------------")
    logger.info("Part 1: Auditing Binary WebSocket Transport Fidelity (sent vs received)...")
    logger.info("----------------------------------------------------------------")

    transport_results_by_chunk = {}

    for chunk_ms in CHUNK_SIZES_MS:
        chunk_samples = int(16000 * (chunk_ms / 1000.0))
        chunk_bytes_len = chunk_samples * 2

        total_files = 0
        lossless_files = 0
        total_drift_ms = 0.0

        for pair in dataset:
            data, sr = sf.read(str(pair.wav_path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != 16000:
                data = soxr.resample(data, in_rate=sr, out_rate=16000, quality="HQ")
            data = np.clip(data, -1.0, 1.0)
            int16_pcm = (data * 32767.0).astype(np.int16).tobytes()

            # Pack into Format A frames
            sent_chunks = []
            capture_ts = 0.0
            ts_step = chunk_ms / 1000.0
            for i in range(0, len(int16_pcm), chunk_bytes_len):
                sub = int16_pcm[i:i + chunk_bytes_len]
                header = json.dumps({
                    "type": "audio_chunk",
                    "chunkIndex": i // chunk_bytes_len,
                    "captureTimestamp": capture_ts,
                }, separators=(",", ":")).encode("utf-8")
                pkt = struct.pack("<I", len(header)) + header + sub
                sent_chunks.append((pkt, capture_ts, sub))
                capture_ts += ts_step

            # Unpack via backend frame parser
            received_bytes_list = []
            for pkt, exp_ts, orig_sub in sent_chunks:
                pcm_out, got_ts, _ = parse_audio_frame(pkt)
                if pcm_out is not None:
                    received_bytes_list.append(pcm_out)

            reconstructed_bytes = b"".join(received_bytes_list)
            is_identical = (reconstructed_bytes == int16_pcm)
            total_files += 1
            if is_identical:
                lossless_files += 1

        transport_results_by_chunk[f"{chunk_ms}ms"] = {
            "chunk_ms": chunk_ms,
            "chunk_samples": chunk_samples,
            "total_files": total_files,
            "lossless_match_count": lossless_files,
            "lossless_ratio": round(lossless_files / max(1, total_files), 4),
            "is_strictly_lossless": (lossless_files == total_files),
        }
        logger.info(f"   Chunk {chunk_ms}ms: {lossless_files}/{total_files} files bit-identical reconstructed ({'✅ LOSSLESS' if lossless_files == total_files else '❌ LOSS DETECTED'})")

    report_data["transport_fidelity"] = transport_results_by_chunk

    # -------------------------------------------------------------------------
    # PART 2: Audio Conversion Fidelity (Source -> 16kHz Int16)
    # -------------------------------------------------------------------------
    logger.info("----------------------------------------------------------------")
    logger.info("Part 2: Auditing Audio Conversion Fidelity (Source -> 16kHz Int16)...")
    logger.info("----------------------------------------------------------------")

    conversion_metrics = []
    for pair in dataset:
        data_raw, sr = sf.read(str(pair.wav_path), dtype="float32")
        if data_raw.ndim > 1:
            data_raw = np.mean(data_raw, axis=1)

        # Reference-RAW or Reference-ASR (float32)
        if sr == 16000:
            ref_float = data_raw
            is_native_16k = True
        else:
            ref_float = soxr.resample(data_raw, in_rate=sr, out_rate=16000, quality="HQ")
            ref_float = np.clip(ref_float, -1.0, 1.0).astype(np.float32)
            is_native_16k = False

        # Extension conversion simulation: float32 -> int16 quantize -> float32
        int16_quant = (ref_float * 32767.0).astype(np.int16)
        reconstructed_float = int16_quant.astype(np.float32) / 32767.0

        metrics = analyze_waveform_metrics(ref_float, reconstructed_float)
        metrics["pair_id"] = pair.pair_id
        metrics["file_name"] = pair.wav_path.name
        metrics["is_native_16k"] = is_native_16k
        metrics["source_sample_rate"] = sr
        conversion_metrics.append(metrics)

        logger.info(
            f"   {pair.pair_id}: SNR={metrics['snr_db']}dB | MaxAbsErr={metrics['max_abs_error']} | "
            f"Peak={metrics['peak']} | RMS={metrics['rms']} | Clipping={metrics['clipping_samples']} samples"
        )

    report_data["conversion_fidelity"] = conversion_metrics

    # -------------------------------------------------------------------------
    # PART 3: Transmission -> ASR (R3 vs R1)
    # -------------------------------------------------------------------------
    logger.info("----------------------------------------------------------------")
    logger.info("Part 3: Testing Transmission -> ASR (R3 vs R1 Direct Baseline)...")
    logger.info("----------------------------------------------------------------")

    # Use 64ms production chunk reconstructed audio
    model_key = config.asr.active_model
    mgr = ASRModelManager()
    model = mgr.ensure_model(model_key)

    r3_results = []
    acquired = ASRModelManager.acquire_infer_lock(blocking=True)
    try:
        session = mgr.ensure_session(model, threads=4)

        for pair in dataset:
            data, sr = sf.read(str(pair.wav_path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != 16000:
                data = soxr.resample(data, in_rate=sr, out_rate=16000, quality="HQ")
            data = np.clip(data, -1.0, 1.0)
            int16_pcm = (data * 32767.0).astype(np.int16).tobytes()

            # Pack and unpack Format A 64ms chunks
            chunk_bytes_len = int(16000 * 0.064) * 2
            parsed_buffers = []
            for i in range(0, len(int16_pcm), chunk_bytes_len):
                sub = int16_pcm[i:i + chunk_bytes_len]
                hdr = json.dumps({"type": "audio_chunk", "captureTimestamp": i / 32000.0}).encode("utf-8")
                pkt = struct.pack("<I", len(hdr)) + hdr + sub
                pcm_chunk, _, _ = parse_audio_frame(pkt)
                if pcm_chunk:
                    parsed_buffers.append(pcm_chunk)

            reconstructed_pcm_bytes = b"".join(parsed_buffers)
            pcm_float = np.frombuffer(reconstructed_pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            dur_sec = len(pcm_float) / 16000.0

            lang_arg = pair.inferred_language if pair.inferred_language not in ("UNKNOWN", "multi") else None

            # 30s windowed inference matching Phase A
            max_window_samples = 30 * 16000
            t0 = time.perf_counter()
            if len(pcm_float) <= max_window_samples:
                try:
                    res = session.run(pcm_float, language=lang_arg)
                    raw_hyp = getattr(res, "text", str(res)).strip()
                except Exception as run_err:
                    partial = getattr(run_err, "partial_result", None)
                    raw_hyp = partial.text.strip() if partial and hasattr(partial, "text") else ""
            else:
                chunk_texts = []
                for start_idx in range(0, len(pcm_float), max_window_samples):
                    chunk = pcm_float[start_idx:start_idx + max_window_samples]
                    if len(chunk) < 1600:
                        continue
                    try:
                        res = session.run(chunk, language=lang_arg)
                        txt = getattr(res, "text", str(res)).strip()
                        if txt:
                            chunk_texts.append(txt)
                    except Exception as run_err:
                        partial = getattr(run_err, "partial_result", None)
                        if partial and hasattr(partial, "text") and partial.text.strip():
                            chunk_texts.append(partial.text.strip())
                raw_hyp = " ".join(chunk_texts).strip()

            infer_wall = time.perf_counter() - t0
            rtf_asr = infer_wall / max(0.001, dur_sec)

            acc = evaluate_accuracy(
                raw_reference=pair.raw_reference,
                raw_hypothesis=raw_hyp,
                language=pair.inferred_language,
            )

            # Compare with R1
            r1 = r1_results.get(pair.pair_id, {})
            r1_cer = r1.get("cer", 0.0)
            r1_wer = r1.get("wer", 0.0)
            r1_hyp = r1.get("normalized_hypothesis", "")
            r3_hyp = acc.normalized_hypothesis

            exact_match_r1 = (r1_hyp == r3_hyp)
            delta_cer = (acc.cer - r1_cer) if acc.cer is not None and r1_cer is not None else 0.0
            delta_wer = (acc.wer - r1_wer) if acc.wer is not None and r1_wer is not None else 0.0

            rec = {
                "pair_id": pair.pair_id,
                "file_name": pair.wav_path.name,
                "language": pair.inferred_language,
                "audio_duration_sec": round(dur_sec, 2),
                "inference_ms": round(infer_wall * 1000.0, 1),
                "rtf_asr": round(rtf_asr, 3),
                "r3_wer": acc.wer,
                "r3_cer": acc.cer,
                "r1_wer": r1_wer,
                "r1_cer": r1_cer,
                "delta_wer": round(delta_wer, 4),
                "delta_cer": round(delta_cer, 4),
                "exact_match_with_r1": exact_match_r1,
                "r1_hypothesis": r1_hyp,
                "r3_hypothesis": r3_hyp,
            }
            r3_results.append(rec)
            logger.info(
                f"   R3 [{pair.pair_id}]: Exact match R1={exact_match_r1} | "
                f"CER: {r1_cer*100:.1f}% -> {acc.cer*100:.1f}% (ΔCER={delta_cer*100:+.2f}%) | "
                f"WER: {r1_wer*100:.1f}% -> {acc.wer*100:.1f}% (ΔWER={delta_wer*100:+.2f}%)"
            )

    finally:
        ASRModelManager.release_infer_lock()

    report_data["transmission_asr_comparison"] = r3_results

    out_file = Path("report/audit_audio_transmission.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info("================================================================")
    logger.info(f"✅ Phase C & D complete! Results saved to {out_file}")
    logger.info("================================================================")
    return report_data


if __name__ == "__main__":
    run_transmission_audit()
