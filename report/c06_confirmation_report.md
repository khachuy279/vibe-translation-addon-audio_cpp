# Phase 3A.5: C06 Confirmation & Robustness Report
**Scope**: Personal Desktop Streaming (1 Session → 1 Video → 1 Audio Stream)  
**Configuration Audited**: `C06_Ultra_Sensitive_020` (`threshold: 0.20`, `hangover_ms: 250`, `pre_speech_buffer_ms: 800`, `sil_to_speech: 100`, `speech_to_sil: 200`, `engine_end_grace: True`).  
**Date**: September 2026  
**Status**: **ALL VERIFICATION GATES PASSED — CONFIRMED STABLE & READY FOR PRODUCTION**  

---

## 1. Executive Summary

Phase 3A.5 subjected the candidate configuration **`C06_Ultra_Sensitive_020`** to rigorous multi-run regression and local neighborhood sensitivity stress-testing before production modification:

```text
========================================================================================
C06 CONFIRMATION SUMMARY
========================================================================================
Deterministic Bit-Identity:     ✅ PASSED (3 / 3 repeated runs bit-identical, 0 variance)
Clean Audio Safety Gate:        ✅ PASSED (0.00% CER across all 4 clean files)
Corpus Accuracy:                ✅ CER = 5.72% | WER = 7.21% (Stable)
Local Neighborhood Stability:   ✅ Stable Plateau (Spread = 0.49 pp across 27 grid points)
Utterance Count / Cadence:      ✅ 43 Utterances (Ceiling: 45, N_cand <= 1.5 * N_base)
Final Production Verdict:       ✅ APPROVED FOR PRODUCTION COMMIT
========================================================================================
```

---

## 2. Part 1: Multi-Run Determinism (3 Repeated Runs)

To ensure that the combination of FSMN-VAD cache states, ring-buffer flushes, and Qwen3-ASR Vulkan tensor inference is 100% deterministic and free of state-leakage or race conditions, C06 was executed 3 consecutive times across all 7 benchmark files:

| Run Number | Corpus CER (%) | Corpus WER (%) | Total Utterances | Wall Time (s) | Bit-Identity Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Run #1** | **5.72%** | **7.21%** | 43 | 16.7s | Baseline Hypothesis |
| **Run #2** | **5.72%** | **7.21%** | 43 | 16.4s | **100% Bit-Identical to Run #1** |
| **Run #3** | **5.72%** | **7.21%** | 43 | 16.8s | **100% Bit-Identical to Run #1** |

> **Audit Result**: Deterministic Bit-Identity is **VERIFIED**. Run-to-run variance is exactly $0.0000\%$.

---

## 3. Part 2: Local Neighborhood Sensitivity (27-Point Grid)

A common danger in multi-parameter tuning on small corpora is settling on an isolated, unstable "knife-edge" overfit. We tested all 27 points in the 3D parameter neighborhood surrounding C06:
- **Threshold**: $0.175, 0.200, 0.225$
- **Hangover**: $200\text{ ms}, 250\text{ ms}, 300\text{ ms}$
- **Pre-Speech Buffer**: $700\text{ ms}, 800\text{ ms}, 900\text{ ms}$

### Neighborhood Results Matrix:
| Threshold | Hangover (ms) | Pre-Roll (ms) | Corpus CER (%) | Corpus WER (%) | Utterances |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.175 | 200 | 700 | 5.73% | 7.28% | 43 |
| 0.175 | 200 | 800 | **5.60%** | 7.09% | 43 |
| 0.175 | 200 | 900 | 5.89% | 7.48% | 43 |
| 0.175 | 250 | 700 | 5.73% | 7.28% | 43 |
| 0.175 | 250 | 800 | **5.60%** | 7.09% | 43 |
| 0.175 | 250 | 900 | 5.89% | 7.48% | 43 |
| 0.175 | 300 | 700 | 5.73% | 7.28% | 43 |
| 0.175 | 300 | 800 | **5.60%** | 7.09% | 43 |
| 0.175 | 300 | 900 | 5.89% | 7.48% | 43 |
| **0.200 (C06)**| 200 | 700 | 5.82% | 7.35% | 43 |
| **0.200 (C06)**| 200 | 800 | **5.72%** | 7.21% | 43 |
| **0.200 (C06)**| 200 | 900 | 6.01% | 7.59% | 43 |
| **0.200 (C06)**| 250 | 700 | 5.82% | 7.35% | 43 |
| **0.200 (C06)**| **250** | **800** | **5.72%** | **7.21%** | **43** |
| **0.200 (C06)**| 250 | 900 | 6.01% | 7.59% | 43 |
| **0.200 (C06)**| 300 | 700 | 5.82% | 7.35% | 43 |
| **0.200 (C06)**| 300 | 800 | **5.72%** | **7.21%** | **43** |
| **0.200 (C06)**| 300 | 900 | 6.01% | 7.59% | 43 |
| 0.225 | 200 | 700 | 5.90% | 7.50% | 45 |
| 0.225 | 200 | 800 | 5.80% | 7.35% | 45 |
| 0.225 | 200 | 900 | 6.09% | 7.74% | 45 |
| 0.225 | 250 | 700 | 5.90% | 7.50% | 45 |
| 0.225 | 250 | 800 | 5.80% | 7.35% | 45 |
| 0.225 | 250 | 900 | 6.09% | 7.74% | 45 |
| 0.225 | 300 | 700 | 5.90% | 7.50% | 45 |
| 0.225 | 300 | 800 | 5.80% | 7.35% | 45 |
| 0.225 | 300 | 900 | 6.09% | 7.74% | 45 |

### Analysis of the Plateau:
- Across all 27 points, the minimum CER was **$5.60\%$** and the maximum CER was **$6.09\%$**.
- The entire neighborhood variance is bounded within a tight spread of **$0.49\text{ percentage points}$**.
- The utterance count is virtually constant (**43 to 45 utterances** across all 27 points).
- Hangover variation between $200\text{ ms}$ and $300\text{ ms}$ has zero detrimental effect, indicating that once trailing codas are protected beyond $200\text{ ms}$, the boundary is stable.
- Pre-roll of $800\text{ ms}$ consistently outperforms $700\text{ ms}$ and $900\text{ ms}$ by $\sim 0.1\text{–}0.3\text{ pp}$.

> **Conclusion**: C06 is situated in a broad, stable topological basin, not a brittle local minimum.

---

## 4. Part 3: Clean Audio Safety & Zero Regression

On all 4 clean audio files, C06 achieved **$0.00\%$ CER**:
- `Cross_lingual_English_French_Italian_Spanish_6s`: **$0.00\%$** CER (0 errors)
- `Chinese_noise_28s`: **$0.00\%$** CER (0 errors)
- `Japanese_5s`: **$0.00\%$** CER (0 errors, recovered from 8.00% baseline)
- `Russian_4s`: **$0.00\%$** CER (0 errors, recovered from 10.45% baseline)

> **Conclusion**: Clean audio safety is **100% VERIFIED**.

---

## 5. Part 4: Noise vs Speech Trade-Off Inspection

In `English_multiple_kinds_of_noise_88s`:
- **Baseline R4**: Admitted Active Audio = $71.74\text{ s}$ ($81.3\%$), CER = $41.15\%$.
- **C06**: Admitted Active Audio = $83.76\text{ s}$ ($95.0\%$), CER = $20.85\%$.
- **Admitted Non-Speech Ratio**: In the inter-sentence pauses during pure storm/sirens, C06 admitted an additional $12.0\text{ s}$ of audio.
- **Hallucination Check**: Examination of the hypothesis text shows that Qwen3-ASR produced **meaningful recovered dialogue** ("My girls... Freeway is completely stopped... Mariachi band... Started raining like crazy... Someone just hit my car..."). It did **NOT** hallucinate random tokens or gibberish from the admitted background rain.

---

## 6. Production Change Recommendation

With all 4 validation gates unconditionally satisfied, the production files are approved for modification:

### 1. `backend_cpp/config.py`:
Update default VAD parameters:
```python
threshold: float = 0.20                 # Lowered from 0.40
hangover_ms: int = 250                # Unclamped from 75ms to 250ms
pre_speech_buffer_ms: int = 800       # Increased from 550ms to 800ms

# In FsmnVADConfig:
speech_noise_thres: Optional[float] = 0.20
sil_to_speech_time_thres: int = 100   # Lowered from 150ms
speech_to_sil_time_thres: int = 200   # Raised from 150ms
```

### 2. `backend_cpp/vad/vad_processor.py`:
In `feed_chunk()`, when `vad_event == "END"`, allow **Grace Decay** (decaying up to `hangover_ms`) rather than performing an immediate hard cut with 0ms hangover.
