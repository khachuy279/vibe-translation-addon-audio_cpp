# VAD Silence Duration Benchmark Matrix Report (100ms – 200ms)

## 1. Executive Summary
As directed by Phase 1 Review, silence duration was independently benchmarked across the 5 target thresholds (100ms, 120ms, 150ms, 180ms, 200ms) over all 7 files in /wav_test.

> [!NOTE]
> **Production Configuration Status**: In accordance with review instruction 3, **NO production config changes** have been made to silence_duration_ms. Production remains at 150ms.

## 2. Comparative Matrix

| Silence Duration | Avg TTFS (ms) | Final Latency (ms) | WER (%) | CER (%) | Missing Speech (Chars) | Total Commits | Total Partials | Subtitle Rollbacks | Utterance Fragmentation |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **100 ms** | 1280.1 ms | 813.6 ms | 18.9% | 10.9% | 72 chars | 47 | 203 | 3 | 6.7 utt/file |
| **120 ms** | 1271.6 ms | 865.2 ms | 17.2% | 9.9% | 69 chars | 48 | 203 | 7 | 6.9 utt/file |
| **150 ms** | 1271.5 ms | 957.0 ms | 22.8% | 13.7% | 80 chars | 47 | 198 | 9 | 6.7 utt/file |
| **180 ms** | 1274.5 ms | 946.7 ms | 23.2% | 14.1% | 93 chars | 48 | 206 | 7 | 6.9 utt/file |
| **200 ms** | 1271.3 ms | 948.4 ms | 24.2% | 16.3% | 84 chars | 40 | 193 | 10 | 5.7 utt/file |

## 3. Deep-Dive Metrics Analysis

### A. Time To First Subtitle (TTFS)
- Across all settings, TTFS remains flat between **1271.3 ms** and **1280.1 ms** (variance < 0.7%).
- **Architectural Reason**: TTFS is driven strictly by speech onset detection and token generation cadence at the start of a sentence; it is invariant to the trailing silence timeout.

### B. Final Subtitle Latency (Perceived Commit Lag)
- **100ms**: Achieves the fastest commit latency (**813.6 ms**), -143.4 ms faster than baseline 150ms.
- **120ms**: Achieves **865.2 ms** commit latency, -91.8 ms faster than baseline 150ms.
- **150ms**: Current production baseline sits at **957.0 ms**.
- **180ms & 200ms**: Commits are delayed to **946.7 ms** and **948.4 ms** respectively.

### C. Accuracy & Missing Speech (WER, CER, Deletions)
- **120ms delivers the best overall accuracy**: **WER 17.2%**, **CER 9.9%**, and **69 missing chars** (lowest deletion count).
- **100ms** is close with **WER 18.9%**, **CER 10.9%**, and **72 missing chars**.
- **Longer silence thresholds (180ms - 200ms) degrade accuracy**: CER climbs to **14.1%** at 180ms and **16.3%** at 200ms with up to **93 missing chars**. In continuous noisy environments, prolonged utterances force the ASR model to decode overly long audio contexts where hallucinations and token repetitions compound.

### D. Utterance Fragmentation & Number of ASR Commits
- **100ms – 180ms** show almost identical segmentation stability: **47 to 48 total commits** across the corpus (~6.7 - 6.9 utterances per file). There is virtually NO over-fragmentation at 120ms or 100ms.
- Only **200ms** forcibly merges utterances into fewer commits (40 commits, 5.7 utt/file), but this comes at the cost of higher CER (+6.4%).

### E. Subtitle Stability & Rollbacks
- **100ms** exhibited the fewest rollbacks (3), followed by **120ms** (7) and **180ms** (7).
- **200ms** exhibited the highest rollback count (10) due to longer partial re-transcription windows.

## 4. Architectural Trade-off Evaluation & Recommendation

| Option | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **100 ms** | Fastest final latency (813.6ms), few rollbacks (3) | Risk of premature cutoffs in speakers with very slow cadence | Viable candidate |
| **120 ms** | **Optimal sweet spot**: lowest WER (17.2%), lowest CER (9.9%), fewest missing chars (69), -92ms latency reduction without sentence fragmentation | Sligthly shorter pause tolerance than 150ms | **Strongly Recommended for Production Approval** |
| **150 ms (Current)** | Safe default conversational pause | Final latency is +92ms higher; CER is 13.7% vs 9.9% | Retain until user approves 120ms |
| **180 ms - 200 ms** | Merges sentences slightly more (5.7 utt/file) | Higher CER (14.1% - 16.3%), highest missing speech (93 chars), higher rollbacks | Not recommended |
