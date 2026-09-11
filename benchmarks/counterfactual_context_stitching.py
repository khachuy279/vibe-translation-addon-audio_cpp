"""Context Stitching Counterfactual Matrix on Chinese_fast_speed_11s.

Evaluates 5 cross-utterance acoustic context conditions:
C0: 0 ms (no context / cold start)
C1: 250 ms
C2: 500 ms
C3: 800 ms
C4: 1200 ms

Measures:
- Raw transcription of Utterance A (0..8s)
- Raw transcription of Utterance B with context (8s..end + context prefix)
- Overlap stripping performance (does it cleanly remove the context prefix?)
- Duplicate word/char count
- Full stitched transcript CER / WER vs ground truth
- Identification of the elbow point
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import rapidfuzz.distance.Levenshtein as lev

from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.simulator import StreamingAudioSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("context_stitching")


def run_counterfactual_context_matrix() -> Dict[str, Any]:
    wav_path = Path("wav_test/Chinese_fast_speed_11s.wav")
    txt_path = Path("wav_test/Chinese_fast_speed_11s.txt")
    
    with open(txt_path, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()

    # Load 16kHz audio
    sim = StreamingAudioSimulator(audio_source=wav_path)
    pcm_all = np.frombuffer(sim.pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    total_samples = len(pcm_all)
    total_dur = total_samples / 16000.0

    # Split boundary at 8.0s (the exact max_duration_sec boundary that triggers streaming cut)
    split_sample = int(8.0 * 16000)
    pcm_a = pcm_all[:split_sample]
    pcm_b_base = pcm_all[split_sample:]

    logger.info(f"Loaded {wav_path.name}: {total_dur:.2f}s ({total_samples} samples)")
    logger.info(f"Utterance A: 0.0s -> 8.0s ({len(pcm_a)} samples)")
    logger.info(f"Utterance B: 8.0s -> {total_dur:.2f}s ({len(pcm_b_base)} samples)")

    mgr = ASRModelManager()
    model = mgr.ensure_model("qwen3-asr-1.7b")

    ASRModelManager.acquire_infer_lock(blocking=True)
    try:
        session = mgr.ensure_session(model)

        # 1. Decode Utterance A
        t0 = time.perf_counter()
        res_a = session.run(pcm_a, language="zh")
        text_a = getattr(res_a, "text", str(res_a)).strip()
        infer_a_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"\n[UTTERANCE A (0..8s)]: '{text_a}' ({infer_a_ms:.1f}ms)")

        # 2. Also run Oracle (Uncut full 11.4s) for comparison
        res_oracle = session.run(pcm_all, language="zh")
        text_oracle = getattr(res_oracle, "text", str(res_oracle)).strip()
        acc_oracle = evaluate_accuracy(ground_truth, text_oracle, language="zh")
        logger.info(f"[ORACLE UNCUT (0..11.4s)]: '{text_oracle}' (CER: {acc_oracle.cer:.2%})")

        # 3. Test Context Conditions: 0ms, 250ms, 500ms, 800ms, 1200ms
        context_ms_list = [0, 250, 500, 800, 1200]
        matrix_results: List[Dict[str, Any]] = []

        for ctx_ms in context_ms_list:
            ctx_samples = int((ctx_ms / 1000.0) * 16000)
            if ctx_samples > 0:
                context_pcm = pcm_a[-ctx_samples:]
                pcm_b_with_ctx = np.concatenate([context_pcm, pcm_b_base])
            else:
                context_pcm = np.empty(0, dtype=np.float32)
                pcm_b_with_ctx = pcm_b_base

            t_start = time.perf_counter()
            res_b = session.run(pcm_b_with_ctx, language="zh")
            raw_text_b = getattr(res_b, "text", str(res_b)).strip()
            infer_ms = (time.perf_counter() - t_start) * 1000.0

            # Test Overlap Stripping Contract:
            # We want transcript(B), not transcript(A + B)
            # Use SentenceSegmenter.remove_prefix_overlap to strip overlapping context from raw_text_b
            stripped_text_b = SentenceSegmenter.remove_prefix_overlap(text_a, raw_text_b)

            # Combined transcript
            stitched_transcript = (text_a + " " + stripped_text_b).strip()

            # Accuracy evaluation
            acc = evaluate_accuracy(ground_truth, stitched_transcript, language="zh")

            # Check duplicate characters at boundary
            # Compare tail of A with head of raw B
            tail_a_chars = text_a[-10:] if len(text_a) >= 10 else text_a
            head_b_chars = raw_text_b[:10] if len(raw_text_b) >= 10 else raw_text_b
            
            # Simple boundary duplicate check: common characters between tail(A) and head(raw B)
            common_overlap = ""
            for k in range(min(len(text_a), len(raw_text_b)), 0, -1):
                if text_a.endswith(raw_text_b[:k]):
                    common_overlap = raw_text_b[:k]
                    break

            duplicate_chars = len(common_overlap) if stripped_text_b != raw_text_b else 0

            record = {
                "context_ms": ctx_ms,
                "context_samples": ctx_samples,
                "b_audio_duration_sec": round(len(pcm_b_with_ctx) / 16000.0, 2),
                "raw_text_b": raw_text_b,
                "stripped_text_b": stripped_text_b,
                "stitched_transcript": stitched_transcript,
                "common_overlap": common_overlap,
                "duplicate_chars_detected": duplicate_chars,
                "cer_pct": round(acc.cer * 100.0, 2),
                "wer_pct": round(acc.wer * 100.0, 2),
                "output_char_count": len(stitched_transcript),
                "reference_char_count": len(ground_truth),
                "infer_ms": round(infer_ms, 1),
            }
            matrix_results.append(record)
            logger.info(f"\n--- Condition C_{ctx_ms}ms ---")
            logger.info(f"   Raw B:      '{raw_text_b}'")
            logger.info(f"   Stripped B: '{stripped_text_b}'")
            logger.info(f"   Overlap:    '{common_overlap}' ({duplicate_chars} chars)")
            logger.info(f"   Stitched:   '{stitched_transcript}'")
            logger.info(f"   CER:        {acc.cer:.2%} (Ref: {len(ground_truth)} chars, Hyp: {len(stitched_transcript)} chars)")

    finally:
        ASRModelManager.release_infer_lock()

    summary = {
        "file": "Chinese_fast_speed_11s",
        "reference": ground_truth,
        "oracle_uncut": {
            "transcript": text_oracle,
            "cer_pct": round(acc_oracle.cer * 100.0, 2),
        },
        "utterance_a": text_a,
        "matrix": matrix_results,
    }

    report_path = Path("report/context_stitching_counterfactual.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✅ Results saved to {report_path}")

    return summary


if __name__ == "__main__":
    run_counterfactual_context_matrix()
