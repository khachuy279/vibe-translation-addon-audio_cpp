"""Benchmark Suite comparing ASR with and without normalize_speech().

Evaluates impact of dynamic RMS speech normalization on Qwen3-ASR 1.7B
over the entire /wav_test corpus.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import soxr

# Ensure workspace root in sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend_audio_cpp.asr.asr_engine import AudioCppASREngine
from backend_audio_cpp.asr.speech_normalizer import normalize_pcm_bytes, SpeechNormalizer
from backend_audio_cpp.commit.sentence_committer import SentenceCommitter, is_cjk
from backend_audio_cpp.commit.sentence_config import SentenceConfig
from backend_audio_cpp.vad.vad_engine import SileroVADEngine, VADConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bench_asr_norm")


def compute_levenshtein(ref: List[str], hyp: List[str]) -> int:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def calculate_error_rate(reference: str, hypothesis: str) -> Tuple[float, str]:
    ref_norm = re.sub(r'[.,!?;:，。！？；：…\'"“”‘’\-—()\[\]{}<>]', '', reference).strip().lower()
    hyp_norm = re.sub(r'[.,!?;:，。！？；：…\'"“”‘’\-—()\[\]{}<>]', '', hypothesis).strip().lower()

    if not ref_norm:
        return 0.0, "N/A"

    has_cjk = any(is_cjk(ch) for ch in ref_norm)
    if has_cjk:
        ref_tokens = [ch for ch in ref_norm if not ch.isspace()]
        hyp_tokens = [ch for ch in hyp_norm if not ch.isspace()]
        metric_name = "CER"
    else:
        ref_tokens = ref_norm.split()
        hyp_tokens = hyp_norm.split()
        metric_name = "WER"

    dist = compute_levenshtein(ref_tokens, hyp_tokens)
    rate = dist / max(len(ref_tokens), 1) * 100.0
    return rate, metric_name


def read_wav_16k_mono(wav_path: Path) -> Tuple[bytes, float, int]:
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        data = soxr.resample(data, sr, 16000)
    dur = len(data) / 16000.0
    int16_arr = np.clip(data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return int16_arr.tobytes(), dur, 16000


def run_benchmark_for_mode(
    wav_path: Path,
    txt_path: Optional[Path],
    use_normalization: bool,
    vad_engine: SileroVADEngine,
    asr_engine: AudioCppASREngine,
    poll_interval_sec: float = 0.45,
) -> Dict[str, Any]:
    pcm_bytes, duration_sec, sr = read_wav_16k_mono(wav_path)
    ground_truth = txt_path.read_text(encoding="utf-8").strip() if txt_path and txt_path.exists() else ""

    vad_state = vad_engine.create_state()
    sentence_config = SentenceConfig(
        max_chars=150,
        max_duration_sec=8.0,
        min_words_to_commit=2,
        split_on_stability=True,
        stability_duration_sec=1.0,
        stability_threshold_polls=3,
    )
    committer = SentenceCommitter(sentence_config)

    frame_bytes = 512 * 2  # 1024 bytes (32ms)
    total_frames = len(pcm_bytes) // frame_bytes

    current_speech_pcm = bytearray()
    committed_results: List[Tuple[str, str]] = []
    last_poll_time = 0.0

    start_bench = time.perf_counter()

    for i in range(total_frames):
        chunk = pcm_bytes[i * frame_bytes : (i + 1) * frame_bytes]
        ts = i * 0.032

        vad_res = vad_engine.process_frame(chunk, ts, vad_state)

        if vad_state.is_speech_active:
            current_speech_pcm.extend(chunk)

            if (ts - last_poll_time) >= poll_interval_sec and len(current_speech_pcm) >= 16000:
                last_poll_time = ts
                
                audio_to_send = bytes(current_speech_pcm)
                if use_normalization:
                    audio_to_send = normalize_pcm_bytes(audio_to_send)

                preview_text, _ = asr_engine.transcribe_chunk(
                    audio_to_send,
                    utt_id=vad_state.current_utterance_id,
                    is_partial=True,
                )

                commit_res = committer.check_stability_commit(
                    preview_text,
                    timestamp_sec=ts,
                    utt_id=vad_state.current_utterance_id,
                )
                if commit_res:
                    committed_results.append(commit_res)
                    current_speech_pcm.clear()

        if vad_res.event == "SPEECH_END":
            if current_speech_pcm:
                audio_to_send = bytes(current_speech_pcm)
                if use_normalization:
                    audio_to_send = normalize_pcm_bytes(audio_to_send)

                final_text, _ = asr_engine.transcribe_chunk(
                    audio_to_send,
                    utt_id=vad_res.utterance_id or vad_state.current_utterance_id,
                    is_partial=False,
                )

                if vad_res.reason == "SILENCE_TIMEOUT":
                    c_res = committer.commit_on_vad_silence(
                        final_text,
                        silence_dur_sec=0.55,
                        utt_id=vad_res.utterance_id or vad_state.current_utterance_id,
                    )
                elif vad_res.reason == "MAX_SPEECH_DURATION_REACHED":
                    c_res = committer.commit_on_max_duration(
                        final_text,
                        speech_dur_sec=vad_res.duration_sec,
                        utt_id=vad_res.utterance_id or vad_state.current_utterance_id,
                    )
                else:
                    c_res = committer.commit_on_stream_eof(
                        final_text,
                        utt_id=vad_res.utterance_id or vad_state.current_utterance_id,
                    )

                if c_res:
                    committed_results.append(c_res)
                current_speech_pcm.clear()

    flush_vad = vad_engine.flush(duration_sec, vad_state)
    if flush_vad and current_speech_pcm:
        audio_to_send = bytes(current_speech_pcm)
        if use_normalization:
            audio_to_send = normalize_pcm_bytes(audio_to_send)

        final_text, _ = asr_engine.transcribe_chunk(
            audio_to_send,
            utt_id=flush_vad.utterance_id or 999,
            is_partial=False,
        )
        c_res = committer.commit_on_stream_eof(final_text, utt_id=flush_vad.utterance_id or 999)
        if c_res:
            committed_results.append(c_res)
        current_speech_pcm.clear()

    total_proc_time = time.perf_counter() - start_bench
    rtf = total_proc_time / duration_sec if duration_sec > 0 else 0.0
    full_hyp = " ".join(text for text, _ in committed_results).strip()

    err_rate, metric_name = calculate_error_rate(ground_truth, full_hyp)
    acc = max(0.0, 100.0 - err_rate)

    return {
        "wav_file": wav_path.name,
        "use_normalization": use_normalization,
        "duration_sec": duration_sec,
        "total_proc_time": total_proc_time,
        "rtf": rtf,
        "metric_name": metric_name,
        "error_rate": err_rate,
        "accuracy": acc,
        "ground_truth": ground_truth,
        "hypothesis": full_hyp,
        "commits": committed_results,
    }


def main():
    wav_dir = _PROJECT_ROOT / "wav_test"
    target_files = [
        "Chinese_fast_speed_11s.wav",
        "Japanese_5s.wav",
        "Russian_4s.wav",
        "Cross_lingual_English_French_Italian_Spanish_6s.wav",
        "English_low_speech_quality_19s.wav",
        "Chinese_noise_28s.wav",
        "English_multiple_kinds_of_noise_88s.wav",
    ]

    vad_engine = SileroVADEngine()
    asr_engine = AudioCppASREngine()

    # Ensure model is qwen3-asr-1.7b
    asr_engine.switch_model("qwen3")

    results_baseline = []
    results_normalized = []

    print("\n" + "="*80)
    print("STARTING COMPARISON BENCHMARK: Baseline vs. With normalize_speech()")
    print("="*80)

    for fname in target_files:
        w_path = wav_dir / fname
        t_path = wav_dir / (w_path.stem + ".txt")

        print(f"\n>>> Running File: {fname}")

        # 1. Baseline (without normalization)
        print("  [1/2] Running Baseline (use_normalization=False)...")
        res_base = run_benchmark_for_mode(w_path, t_path, False, vad_engine, asr_engine)
        results_baseline.append(res_base)
        print(f"        Baseline Acc: {res_base['accuracy']:.1f}% ({res_base['metric_name']}: {res_base['error_rate']:.1f}%) | RTF: {res_base['rtf']:.3f}")

        # 2. With normalize_speech()
        print("  [2/2] Running Normalized (use_normalization=True)...")
        res_norm = run_benchmark_for_mode(w_path, t_path, True, vad_engine, asr_engine)
        results_normalized.append(res_norm)
        print(f"        Normaliz Acc: {res_norm['accuracy']:.1f}% ({res_norm['metric_name']}: {res_norm['error_rate']:.1f}%) | RTF: {res_norm['rtf']:.3f}")

    # Print Side-by-Side Summary Table
    print("\n" + "="*80)
    print("SIDE-BY-SIDE SUMMARY COMPARISON TABLE")
    print("="*80)
    print(f"{'Audio File':<38} | {'Metric':<4} | {'Base Acc':<8} | {'Norm Acc':<8} | {'Delta':<7} | {'Base RTF':<8} | {'Norm RTF':<8}")
    print("-" * 90)

    for b, n in zip(results_baseline, results_normalized):
        diff = n['accuracy'] - b['accuracy']
        sign = "+" if diff > 0 else ""
        delta_str = f"{sign}{diff:.1f}%"
        print(f"{b['wav_file']:<38} | {b['metric_name']:<4} | {b['accuracy']:>7.1f}% | {n['accuracy']:>7.1f}% | {delta_str:>7} | {b['rtf']:>8.3f} | {n['rtf']:>8.3f}")

    # Save detailed JSON for reporting
    out_json = _PROJECT_ROOT / "report" / "normalization_comparison_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"baseline": results_baseline, "normalized": results_normalized}, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed JSON saved to: {out_json}")


if __name__ == "__main__":
    main()
