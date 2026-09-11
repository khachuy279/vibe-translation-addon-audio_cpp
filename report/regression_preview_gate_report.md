# Regression Verification: `preview_min_growth_ratio = 0.2` vs `0.0`

| File | Final Transcript Bit-Identical? | WER (0.0 → 0.2) | CER (0.0 → 0.2) | TTFS ms (0.0 → 0.2) | Final Latency ms (0.0 → 0.2) | Inferences (0.0 → 0.2) | Inference Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| `Chinese_fast_speed_11s.wav` | ✅ YES | 25.3% → 25.3% | 16.9% → 16.9% | 772 → 709 | 1913 → 959 | 18 → 13 | **-27.8%** |
| `Chinese_noise_28s.wav` | ✅ YES | 3.8% → 3.8% | 1.6% → 1.6% | 718 → 710 | 1526 → 1919 | 49 → 36 | **-26.5%** |
| `Cross_lingual_English_French_Italian_Spanish_6s.wav` | ✅ YES | 0.0% → 0.0% | 0.0% → 0.0% | 1159 → 1152 | 708 → 704 | 14 → 13 | **-7.1%** |
| `English_low_speech_quality_19s.wav` | ✅ YES | 25.8% → 25.8% | 21.7% → 21.7% | 2243 → 2176 | 0 → 0 | 30 → 29 | **-3.3%** |
| `English_multiple_kinds_of_noise_88s.wav` | ✅ YES | 60.9% → 60.9% | 41.0% → 41.0% | 1409 → 1409 | 896 → 896 | 151 → 141 | **-6.6%** |
| `Japanese_5s.wav` | ✅ YES | 4.0% → 4.0% | 4.0% → 4.0% | 1097 → 1088 | 896 → 767 | 10 → 9 | **-10.0%** |
| `Russian_4s.wav` | ✅ YES | 40.0% → 40.0% | 10.4% → 10.4% | 1609 → 1600 | 772 → 769 | 8 → 7 | **-12.5%** |

**Total Inferences**: 280 (Before) → 248 (After) = **-11.4% reduction in GPU compute**.
**Bit-identical Verification**: ✅ ALL PASSED