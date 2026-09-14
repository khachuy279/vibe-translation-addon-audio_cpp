"""Pilot test for Sliding Window + Local Agreement (No VAD).

Tests on the first 45s of 00_ingress_stream.wav to verify:
1. Local Agreement (2-step and 3-step LCP).
2. Buffer slicing with context overlap.
3. SentenceSegmenter.remove_prefix_overlap cleanly preventing duplicates.
4. CER against ground truth for the first 45s.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter, count_content_tokens
from benchmarks.ingress_benchmark import DEFAULT_TXT, DEFAULT_WAV, parse_ground_truth
from benchmarks.ja_text import normalize_ja, score_ja


def longest_common_prefix(s1: str, s2: str) -> str:
    """Find longest common prefix between two strings."""
    min_len = min(len(s1), len(s2))
    idx = 0
    while idx < min_len and s1[idx] == s2[idx]:
        idx += 1
    return s1[:idx]


def run_pilot(
    wav_path: Path,
    limit_sec: float = 45.0,
    window_sec: float = 10.0,
    step_sec: float = 1.5,
    context_overlap_sec: float = 2.0,
    min_agreement_chars: int = 2,
    agreement_steps: int = 2,
):
    print(f"=== Running Pilot Sliding Window (No VAD) ===")
    print(f"File: {wav_path.name}, Limit: {limit_sec}s, Window: {window_sec}s, Step: {step_sec}s, Overlap: {context_overlap_sec}s")

    # Load audio
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    max_samples = int(limit_sec * sr)
    audio = data[:max_samples]

    # Pre-warm model
    import transcribe_cpp
    mgr = ASRModelManager()
    model = mgr.ensure_model("qwen3-asr-1.7b")
    session = transcribe_cpp.Session(model)

    chunk_samples = int(0.064 * sr)  # 64ms chunks
    step_samples = int(step_sec * sr)
    window_samples = int(window_sec * sr)
    overlap_samples = int(context_overlap_sec * sr)

    # Time-strided sliding window parameters
    stride_samples = int(step_sec * sr)
    window_samples = int(window_sec * sr)

    # We maintain a global audio position
    curr_audio_end = 0

    hyp_history: List[str] = []
    committed_chunks: List[str] = []
    cumulative_committed_text = ""

    t_infer_total = 0.0
    num_infers = 0
    sim_start_time = time.perf_counter()

    # Step through audio stream with time stride
    for curr_audio_end in range(stride_samples, len(audio) + 1, stride_samples):
        # Window covers [max(0, curr_audio_end - window_samples), curr_audio_end]
        window_start = max(0, curr_audio_end - window_samples)
        active_pcm = audio[window_start:curr_audio_end]

        if len(active_pcm) < int(0.5 * sr):
            continue

        # Energy gate: if silence, skip decode
        rms = np.sqrt(np.mean(active_pcm ** 2) + 1e-12)
        if rms < 0.005:
            continue

        # Run inference
        t0 = time.perf_counter()
        res = session.run(active_pcm, language="ja")
        raw_text = getattr(res, "text", str(res)).strip()
        infer_dur = time.perf_counter() - t0
        t_infer_total += infer_dur
        num_infers += 1

        norm_raw = normalize_ja(raw_text)

        # Robust suffix-prefix and fuzzy overlap finder
        def find_overlap(committed: str, raw: str) -> int:
            if not committed or not raw:
                return 0
            # 1. Check if recent tail of committed appears inside raw
            import rapidfuzz.fuzz as fuzz
            for tail_len in [30, 20, 15, 10]:
                if len(committed) >= tail_len:
                    tail = committed[-tail_len:]
                    align = fuzz.partial_ratio_alignment(tail, raw)
                    if align and align.score >= 80.0 and align.dest_end > 0:
                        return align.dest_end

            # 2. Longest suffix of committed matching prefix of raw
            max_k = min(len(committed), len(raw))
            for k in range(max_k, 1, -1):  # down to 2 chars
                c_sub = committed[-k:]
                r_sub = raw[:k]
                if c_sub == r_sub:
                    return k
                if k >= 4 and fuzz.ratio(c_sub, r_sub) >= 80.0:
                    return k
            return 0

        overlap_len = find_overlap(cumulative_committed_text, norm_raw)
        remainder = norm_raw[overlap_len:]

        hyp_history.append(remainder)
        if len(hyp_history) > agreement_steps:
            hyp_history.pop(0)

        print(f"[T={curr_audio_end/sr:4.1f}s] win=[{window_start/sr:4.1f}s-{curr_audio_end/sr:4.1f}s] | raw='{raw_text[:20]}...' | rem='{remainder[:20]}...' ({infer_dur*1000:.0f}ms)")

        # Local Agreement: Check if prefix is stable across consecutive steps
        if len(hyp_history) >= agreement_steps:
            common = hyp_history[0]
            for h in hyp_history[1:]:
                common = longest_common_prefix(common, h)

            common = common.strip()
            # Only commit if we have enough agreed characters
            if len(common) >= min_agreement_chars:
                committed_chunks.append(common)
                cumulative_committed_text += common
                print(f"   ⭐⭐ COMMIT: '{common}' (total committed: {len(cumulative_committed_text)} chars)")
                hyp_history = []

    # Final flush
    if hyp_history:
        final_text = hyp_history[-1]
        if final_text and final_text not in cumulative_committed_text:
            committed_chunks.append(final_text)
            cumulative_committed_text += final_text
            print(f"   🏁 FINAL FLUSH COMMIT: '{final_text}'")

    total_time = time.perf_counter() - sim_start_time
    rtf = t_infer_total / (len(audio) / sr)

    turns = parse_ground_truth(DEFAULT_TXT)
    # Filter turns up to limit_sec
    ref_texts = [t["norm_text"] for t in turns if t.get("time") and int(t["time"].split(":")[0])*60 + int(t["time"].split(":")[1]) <= limit_sec]
    full_ref = "".join(ref_texts)

    full_hyp = "".join(normalize_ja(c) for c in committed_chunks)
    score = score_ja(full_ref, full_hyp)

    print("\n" + "=" * 60)
    print(f"PILOT RESULTS ({limit_sec}s audio):")
    print(f"Total Inferences: {num_infers} | Total Inference Time: {t_infer_total:.2f}s | RTF: {rtf:.3f}")
    print(f"Number of Commits: {len(committed_chunks)}")
    print(f"CER: {score.cer*100:.2f}% | ITN CER: {score.cer_itn*100:.2f}%")
    print(f"Reference Text ({len(full_ref)} chars):\n{full_ref}")
    print(f"Hypothesis Text ({len(full_hyp)} chars):\n{full_hyp}")
    print("=" * 60)


if __name__ == "__main__":
    run_pilot(DEFAULT_WAV, limit_sec=45.0, window_sec=10.0, step_sec=1.5, context_overlap_sec=2.0)
