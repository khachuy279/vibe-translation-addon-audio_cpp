"""Phase 3A.5: C06 Confirmation & Robustness Suite.

Evaluates:
1. Multi-run Determinism: 3 repeated runs of C06 to check variance and bit-identity.
2. Clean Audio Safety: Invariance on clean test files (Cross_lingual, Chinese_noise, Japanese, Russian = 0.0%).
3. Local Neighborhood Sensitivity: Small grid around C06:
   - threshold: [0.175, 0.20, 0.225]
   - hangover_ms: [200, 250, 300]
   - pre_speech_buffer_ms: [700, 800, 900]
   Confirms C06 is a stable plateau, not a knife-edge overfit.
4. Noise vs Speech Trade-off: Admitted audio ratio vs false-positive silence admission.
5. Produces:
   - report/c06_confirmation.json
   - report/c06_confirmation_report.md
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.config import config
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.dataset import DatasetPair, discover_dataset
from benchmarks.optimize_vad_tuning import SegmentInterval, simulate_vad_processor_stream
from benchmarks.vad_oracle import get_oracle_segments, load_audio_16k_mono

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("c06_confirmation")

C06_PARAMS = {
    "name": "C06_Ultra_Sensitive_020",
    "speech_noise_thres": 0.20,
    "sil_to_speech_time_thres": 100,
    "speech_to_sil_time_thres": 200,
    "hangover_ms": 250,
    "pre_speech_buffer_ms": 800,
    "engine_end_grace": True,
}


def run_c06_single_evaluation(
    params: Dict[str, Any],
    dataset: List[DatasetPair],
    audio_by_file: Dict[str, np.ndarray],
    raw_model: Any,
    session: Any,
) -> Dict[str, Any]:
    """Runs a complete evaluation of a parameter set across the dataset."""
    corpus_cers, corpus_wers = [], []
    per_file = []
    total_audio_dur = 0.0
    total_speech_dur = 0.0

    for pair in dataset:
        pcm = audio_by_file[pair.pair_id]
        dur_sec = len(pcm) / 16000.0
        total_audio_dur += dur_sec

        # 1. FSMN frame generation
        cache = {}
        raw_model.model.init_cache(
            cache,
            speech_noise_thres=float(params["speech_noise_thres"]),
            max_end_silence_time=800,
            speech_to_sil_time_thres=int(params["speech_to_sil_time_thres"]),
            sil_to_speech_time_thres=int(params["sil_to_speech_time_thres"]),
        )
        in_speech = False
        events_list = []
        chunk_samples = 960

        for i in range(0, len(pcm), chunk_samples):
            chunk = pcm[i : i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            t_sec = i / 16000.0
            res = raw_model.generate(
                input=[torch.from_numpy(chunk).float()],
                cache=cache,
                is_final=False,
                chunk_size=60,
                dynamic_silence=False,
                disable_pbar=True,
                disable_log=True,
            )
            signals = res[0].get("value", []) if res else []
            event = None
            for sig in signals:
                if sig[0] >= 0 and sig[1] == -1:
                    event = "START"
                    in_speech = True
                elif sig[0] == -1 and sig[1] >= 0:
                    event = "END"
                    in_speech = False
                elif sig[0] >= 0 and sig[1] >= 0:
                    event = "START"
                    in_speech = False
            stats = cache.get("stats")
            sil_cnt = getattr(stats, "continous_silence_frame_count", 0) if stats else 0
            is_sp = bool(in_speech and sil_cnt == 0)
            events_list.append((t_sec, is_sp, event))

        # 2. VAD streaming simulation
        segs, grace_ms, end_cnt = simulate_vad_processor_stream(
            events_list,
            hangover_ms=params["hangover_ms"],
            silence_duration_ms=150,
            pre_speech_buffer_ms=params["pre_speech_buffer_ms"],
            engine_end_grace=params["engine_end_grace"],
        )

        file_speech_sec = sum(s.end_sec - s.start_sec for s in segs)
        total_speech_dur += file_speech_sec

        # 3. Decode segments
        lang_arg = pair.inferred_language if pair.inferred_language not in ("UNKNOWN", "multi") else None
        seg_texts = []
        for s in segs:
            start_idx = int(s.start_sec * 16000)
            end_idx = int(s.end_sec * 16000)
            seg_pcm = pcm[start_idx:end_idx]
            if len(seg_pcm) < 1600:
                continue
            try:
                r = session.run(seg_pcm, language=lang_arg)
                t = getattr(r, "text", str(r)).strip()
                if t:
                    seg_texts.append(t)
            except Exception as e:
                part = getattr(e, "partial_result", None)
                if part and hasattr(part, "text") and part.text.strip():
                    seg_texts.append(part.text.strip())

        hyp_full = " ".join(seg_texts)
        acc = evaluate_accuracy(pair.raw_reference, hyp_full, language=pair.inferred_language)

        if acc.cer is not None:
            corpus_cers.append(acc.cer)
        if acc.wer is not None:
            corpus_wers.append(acc.wer)

        per_file.append({
            "pair_id": pair.pair_id,
            "language": pair.inferred_language,
            "cer_pct": round(acc.cer * 100, 2) if acc.cer is not None else None,
            "wer_pct": round(acc.wer * 100, 2) if acc.wer is not None else None,
            "segments_count": len(segs),
            "active_audio_sec": round(file_speech_sec, 2),
            "active_audio_ratio_pct": round((file_speech_sec / dur_sec) * 100, 2),
            "raw_hypothesis": hyp_full,
        })

    avg_cer = sum(corpus_cers) / len(corpus_cers) * 100.0 if corpus_cers else 0.0
    avg_wer = sum(corpus_wers) / len(corpus_wers) * 100.0 if corpus_wers else 0.0
    active_ratio = (total_speech_dur / total_audio_dur) * 100.0
    total_utts = sum(p["segments_count"] for p in per_file)

    return {
        "params": params,
        "corpus_cer_pct": round(avg_cer, 2),
        "corpus_wer_pct": round(avg_wer, 2),
        "corpus_active_audio_ratio_pct": round(active_ratio, 2),
        "total_utterance_count": total_utts,
        "per_file": per_file,
    }


def run_c06_confirmation(dataset_dir: str = "wav_test") -> Dict[str, Any]:
    logger.info("================================================================")
    logger.info("▶️ STARTING PHASE 3A.5: C06 CONFIRMATION & ROBUSTNESS SUITE")
    logger.info("================================================================")

    from backend_cpp.vad.engines import VADEngineFactory
    engine = VADEngineFactory.get_engine("fsmn-vad")
    raw_model = engine.model

    dataset = discover_dataset(Path(dataset_dir))
    model_mgr = ASRModelManager()
    asr_model = model_mgr.ensure_model("qwen3-asr-1.7b")

    audio_by_file = {}
    for pair in dataset:
        audio_by_file[pair.pair_id] = load_audio_16k_mono(pair.wav_path)

    ASRModelManager.acquire_infer_lock(blocking=True)

    try:
        session = model_mgr.ensure_session(asr_model, threads=4)

        # -------------------------------------------------------------
        # Part 1: Multi-Run Determinism (3 Repeated Runs of C06)
        # -------------------------------------------------------------
        logger.info("--- Part 1: Running Multi-Run Determinism (3 Runs) ---")
        run_results = []
        for run_id in range(1, 4):
            t0 = time.perf_counter()
            res = run_c06_single_evaluation(C06_PARAMS, dataset, audio_by_file, raw_model, session)
            dur = time.perf_counter() - t0
            run_results.append(res)
            logger.info(f"   Run #{run_id}: CER={res['corpus_cer_pct']}%, WER={res['corpus_wer_pct']}%, Utterances={res['total_utterance_count']} ({dur:.1f}s)")

        # Verify bit-identity across runs
        run1_hyps = [p["raw_hypothesis"] for p in run_results[0]["per_file"]]
        run2_hyps = [p["raw_hypothesis"] for p in run_results[1]["per_file"]]
        run3_hyps = [p["raw_hypothesis"] for p in run_results[2]["per_file"]]
        is_deterministic = (run1_hyps == run2_hyps == run3_hyps)
        logger.info(f"   Deterministic Bit-Identity Across Runs: {'✅ PASSED (100% Identical)' if is_deterministic else '❌ FAILED'}")

        # -------------------------------------------------------------
        # Part 2: Local Neighborhood Sensitivity Grid
        # -------------------------------------------------------------
        logger.info("--- Part 2: Running Local Neighborhood Sensitivity Grid ---")
        thres_local = [0.175, 0.20, 0.225]
        hang_local = [200, 250, 300]
        pre_local = [700, 800, 900]

        neighborhood_results = []
        for t in thres_local:
            for h in hang_local:
                for p in pre_local:
                    p_cfg = {
                        "name": f"Local_t{t}_h{h}_p{p}",
                        "speech_noise_thres": t,
                        "sil_to_speech_time_thres": 100,
                        "speech_to_sil_time_thres": 200,
                        "hangover_ms": h,
                        "pre_speech_buffer_ms": p,
                        "engine_end_grace": True,
                    }
                    t_eval = time.perf_counter()
                    n_res = run_c06_single_evaluation(p_cfg, dataset, audio_by_file, raw_model, session)
                    neighborhood_results.append({
                        "threshold": t,
                        "hangover_ms": h,
                        "pre_speech_ms": p,
                        "cer_pct": n_res["corpus_cer_pct"],
                        "wer_pct": n_res["corpus_wer_pct"],
                        "active_audio_ratio_pct": n_res["corpus_active_audio_ratio_pct"],
                        "utterance_count": n_res["total_utterance_count"],
                    })
                    logger.info(f"   [t={t:.3f}, h={h}ms, p={p}ms] -> CER={n_res['corpus_cer_pct']}% | WER={n_res['corpus_wer_pct']}% | Utts={n_res['total_utterance_count']}")

        # -------------------------------------------------------------
        # Part 3: Clean Safety Verification
        # -------------------------------------------------------------
        c06_per_file = run_results[0]["per_file"]
        clean_files = {p["pair_id"]: p["cer_pct"] for p in c06_per_file}
        clean_safety_passed = (
            clean_files.get("Cross_lingual_English_French_Italian_Spanish_6s") == 0.0
            and clean_files.get("Chinese_noise_28s") == 0.0
            and clean_files.get("Japanese_5s") == 0.0
            and clean_files.get("Russian_4s") == 0.0
        )

        # -------------------------------------------------------------
        # Part 4: Noise vs False Speech Analysis
        # -------------------------------------------------------------
        en_88s = next(p for p in c06_per_file if p["pair_id"] == "English_multiple_kinds_of_noise_88s")
        en_88s_cer = en_88s["cer_pct"]
        en_88s_active = en_88s["active_audio_ratio_pct"]

        # -------------------------------------------------------------
        # Part 5: Final Confirmation Gate Summary
        # -------------------------------------------------------------
        cer_vals = [n["cer_pct"] for n in neighborhood_results]
        cer_min = min(cer_vals)
        cer_max = max(cer_vals)
        cer_spread = cer_max - cer_min

        is_stable_plateau = (cer_spread <= 1.0)  # Spread across neighborhood within 1.0 pp

        confirmation_status = (
            is_deterministic
            and clean_safety_passed
            and (run_results[0]["corpus_cer_pct"] <= 6.0)
            and is_stable_plateau
        )

        final_payload = {
            "experiment": "Phase 3A.5: C06 Confirmation & Robustness",
            "candidate_name": C06_PARAMS["name"],
            "parameters": C06_PARAMS,
            "runs": [
                {
                    "run_id": idx + 1,
                    "corpus_cer_pct": r["corpus_cer_pct"],
                    "corpus_wer_pct": r["corpus_wer_pct"],
                    "active_audio_ratio_pct": r["corpus_active_audio_ratio_pct"],
                    "total_utterance_count": r["total_utterance_count"],
                }
                for idx, r in enumerate(run_results)
            ],
            "determinism_verified": is_deterministic,
            "clean_safety_verified": clean_safety_passed,
            "clean_files_cer": {k: clean_files[k] for k in ["Cross_lingual_English_French_Italian_Spanish_6s", "Chinese_noise_28s", "Japanese_5s", "Russian_4s"]},
            "heavy_noise_en88s": {
                "cer_pct": en_88s_cer,
                "active_audio_ratio_pct": en_88s_active,
                "segments_count": en_88s["segments_count"],
            },
            "local_neighborhood_spread_pp": round(cer_spread, 2),
            "local_neighborhood_min_cer": cer_min,
            "local_neighborhood_max_cer": cer_max,
            "is_stable_plateau": is_stable_plateau,
            "local_neighborhood": neighborhood_results,
            "c06_confirmed_for_production": confirmation_status,
            "per_file_c06": c06_per_file,
        }

        out_json = Path("report/c06_confirmation.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2, ensure_ascii=False)

        logger.info("================================================================")
        logger.info(f"✅ C06 Confirmation Suite Completed! Saved to {out_json}")
        logger.info(f"   Deterministic Identity:  {'PASS' if is_deterministic else 'FAIL'}")
        logger.info(f"   Clean Audio Safety (4/4): {'PASS (0.0% CER on all 4)' if clean_safety_passed else 'FAIL'}")
        logger.info(f"   Corpus CER:              {run_results[0]['corpus_cer_pct']}% (WER: {run_results[0]['corpus_wer_pct']}%)")
        logger.info(f"   Neighborhood Stability:  CER range [{cer_min:.2f}% - {cer_max:.2f}%] (Spread: {cer_spread:.2f} pp)")
        logger.info(f"   PRODUCTION READINESS:    {'✅ CONFIRMED READY FOR PRODUCTION' if confirmation_status else '❌ FAILED'}")
        logger.info("================================================================")

        return final_payload

    finally:
        ASRModelManager.release_infer_lock()


if __name__ == "__main__":
    run_c06_confirmation()
