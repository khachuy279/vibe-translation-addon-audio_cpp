# Báo Cáo Benchmark Module FSMN-VAD (`fsmn_vad`)

- **Thời gian chạy**: 2026-09-15 00:50:46
- **Model**: `backend_audio_cpp/models/fsmn_vad.onnx`
- **Runtime Provider**: CPUExecutionProvider
- **Frame Size**: 512 samples (32.0 ms @ 16kHz)

## 1. Bảng Tổng Hợp Hiệu Năng

| Tệp Audio | Thời lượng (s) | Thời gian xử lý (s) | RTF | Tốc độ | Latency TB (ms) | P95 Latency (ms) | Số câu (Utterance) | Tỷ lệ tiếng nói (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `00_ingress_stream.wav` | 332.03 | 2.655 | **0.0080** | **125.1x** | 0.25 | 0.33 | 100 | 35.2% |
| `Chinese_fast_speed_11s.wav` | 11.38 | 0.088 | **0.0078** | **128.7x** | 0.25 | 0.29 | 2 | 99.3% |
| `Chinese_noise_28s.wav` | 28.26 | 0.220 | **0.0078** | **128.2x** | 0.25 | 0.30 | 4 | 100.0% |
| `Cross_lingual_English_French_Italian_Spanish_6s.wav` | 6.23 | 0.049 | **0.0078** | **128.4x** | 0.25 | 0.29 | 2 | 79.1% |
| `English_low_speech_quality_19s.wav` | 19.02 | 0.149 | **0.0078** | **127.9x** | 0.25 | 0.31 | 5 | 23.6% |
| `English_multiple_kinds_of_noise_88s.wav` | 88.19 | 0.699 | **0.0079** | **126.1x** | 0.25 | 0.31 | 36 | 36.4% |
| `Japanese_5s.wav` | 5.08 | 0.040 | **0.0079** | **126.5x** | 0.25 | 0.32 | 1 | 93.2% |
| `Russian_4s.wav` | 4.76 | 0.037 | **0.0078** | **127.6x** | 0.25 | 0.30 | 1 | 84.7% |

## 2. Chi Tiết Phát Hiện Đoạn Tiếng Nói (Speech Segments & Lý do ngắt)

### `00_ingress_stream.wav` (332.03s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.1s | 0.29s | 0.22s | `SILENCE_TIMEOUT` |
| 2 | 8.86s | 10.85s | 2.02s | `SILENCE_TIMEOUT` |
| 3 | 11.49s | 12.54s | 1.09s | `SILENCE_TIMEOUT` |
| 4 | 13.44s | 13.89s | 0.48s | `SILENCE_TIMEOUT` |
| 5 | 14.53s | 16.9s | 2.4s | `SILENCE_TIMEOUT` |
| 6 | 18.08s | 19.1s | 1.06s | `SILENCE_TIMEOUT` |
| 7 | 20.06s | 24.93s | 4.9s | `SILENCE_TIMEOUT` |
| 8 | 26.43s | 26.46s | 0.06s | `SILENCE_TIMEOUT` |
| 9 | 27.39s | 29.66s | 2.3s | `SILENCE_TIMEOUT` |
| 10 | 31.1s | 32.96s | 1.89s | `SILENCE_TIMEOUT` |
| 11 | 34.72s | 34.94s | 0.26s | `SILENCE_TIMEOUT` |
| 12 | 37.47s | 40.54s | 3.1s | `SILENCE_TIMEOUT` |
| 13 | 41.98s | 42.4s | 0.45s | `SILENCE_TIMEOUT` |
| 14 | 43.62s | 45.66s | 2.08s | `SILENCE_TIMEOUT` |
| 15 | 47.14s | 49.25s | 2.14s | `SILENCE_TIMEOUT` |
| 16 | 50.27s | 58.21s | 7.97s | `SILENCE_TIMEOUT` |
| 17 | 64.9s | 65.31s | 0.45s | `SILENCE_TIMEOUT` |
| 18 | 68.1s | 68.1s | 0.03s | `SILENCE_TIMEOUT` |
| 19 | 70.14s | 71.39s | 1.28s | `SILENCE_TIMEOUT` |
| 20 | 72.16s | 72.35s | 0.22s | `SILENCE_TIMEOUT` |
| 21 | 73.38s | 74.5s | 1.15s | `SILENCE_TIMEOUT` |
| 22 | 76.54s | 78.11s | 1.6s | `SILENCE_TIMEOUT` |
| 23 | 78.78s | 79.2s | 0.45s | `SILENCE_TIMEOUT` |
| 24 | 81.47s | 82.88s | 1.44s | `SILENCE_TIMEOUT` |
| 25 | 84.29s | 85.5s | 1.25s | `SILENCE_TIMEOUT` |
| 26 | 88.06s | 88.29s | 0.26s | `SILENCE_TIMEOUT` |
| 27 | 90.3s | 90.37s | 0.1s | `SILENCE_TIMEOUT` |
| 28 | 91.07s | 91.65s | 0.61s | `SILENCE_TIMEOUT` |
| 29 | 92.77s | 93.31s | 0.58s | `SILENCE_TIMEOUT` |
| 30 | 93.98s | 95.07s | 1.12s | `SILENCE_TIMEOUT` |
| 31 | 96.8s | 97.22s | 0.45s | `SILENCE_TIMEOUT` |
| 32 | 98.24s | 101.34s | 3.14s | `SILENCE_TIMEOUT` |
| 33 | 102.21s | 102.21s | 0.03s | `SILENCE_TIMEOUT` |
| 34 | 103.01s | 103.55s | 0.58s | `SILENCE_TIMEOUT` |
| 35 | 105.98s | 106.5s | 0.54s | `SILENCE_TIMEOUT` |
| 36 | 107.62s | 108.93s | 1.34s | `SILENCE_TIMEOUT` |
| 37 | 112.32s | 113.98s | 1.7s | `SILENCE_TIMEOUT` |
| 38 | 134.72s | 135.46s | 0.77s | `SILENCE_TIMEOUT` |
| 39 | 136.38s | 138.59s | 2.24s | `SILENCE_TIMEOUT` |
| 40 | 140.48s | 141.06s | 0.61s | `SILENCE_TIMEOUT` |
| 41 | 142.56s | 143.42s | 0.9s | `SILENCE_TIMEOUT` |
| 42 | 158.24s | 158.24s | 0.03s | `SILENCE_TIMEOUT` |
| 43 | 160.26s | 160.93s | 0.7s | `SILENCE_TIMEOUT` |
| 44 | 161.82s | 165.06s | 3.26s | `SILENCE_TIMEOUT` |
| 45 | 168.0s | 168.19s | 0.22s | `SILENCE_TIMEOUT` |
| 46 | 169.57s | 170.53s | 0.99s | `SILENCE_TIMEOUT` |
| 47 | 172.32s | 174.91s | 2.62s | `SILENCE_TIMEOUT` |
| 48 | 176.58s | 176.67s | 0.13s | `SILENCE_TIMEOUT` |
| 49 | 177.47s | 177.47s | 0.03s | `SILENCE_TIMEOUT` |
| 50 | 179.04s | 179.07s | 0.06s | `SILENCE_TIMEOUT` |
| 51 | 184.16s | 184.77s | 0.64s | `SILENCE_TIMEOUT` |
| 52 | 187.1s | 187.78s | 0.7s | `SILENCE_TIMEOUT` |
| 53 | 188.54s | 189.66s | 1.15s | `SILENCE_TIMEOUT` |
| 54 | 190.62s | 191.94s | 1.34s | `SILENCE_TIMEOUT` |
| 55 | 194.72s | 195.2s | 0.51s | `SILENCE_TIMEOUT` |
| 56 | 196.86s | 197.73s | 0.9s | `SILENCE_TIMEOUT` |
| 57 | 200.67s | 202.11s | 1.47s | `SILENCE_TIMEOUT` |
| 58 | 203.71s | 204.8s | 1.12s | `SILENCE_TIMEOUT` |
| 59 | 205.92s | 208.54s | 2.66s | `SILENCE_TIMEOUT` |
| 60 | 213.54s | 216.19s | 2.69s | `SILENCE_TIMEOUT` |
| 61 | 217.31s | 218.34s | 1.06s | `SILENCE_TIMEOUT` |
| 62 | 219.1s | 221.47s | 2.4s | `SILENCE_TIMEOUT` |
| 63 | 223.17s | 224.54s | 1.41s | `SILENCE_TIMEOUT` |
| 64 | 227.46s | 228.54s | 1.12s | `SILENCE_TIMEOUT` |
| 65 | 229.25s | 229.25s | 0.03s | `SILENCE_TIMEOUT` |
| 66 | 231.55s | 231.62s | 0.1s | `SILENCE_TIMEOUT` |
| 67 | 238.66s | 238.78s | 0.16s | `SILENCE_TIMEOUT` |
| 68 | 240.16s | 240.19s | 0.06s | `SILENCE_TIMEOUT` |
| 69 | 240.86s | 242.75s | 1.92s | `SILENCE_TIMEOUT` |
| 70 | 243.46s | 244.8s | 1.38s | `SILENCE_TIMEOUT` |
| 71 | 245.95s | 248.74s | 2.82s | `SILENCE_TIMEOUT` |
| 72 | 250.21s | 251.33s | 1.15s | `SILENCE_TIMEOUT` |
| 73 | 252.8s | 252.86s | 0.1s | `SILENCE_TIMEOUT` |
| 74 | 256.74s | 259.07s | 2.37s | `SILENCE_TIMEOUT` |
| 75 | 260.38s | 261.06s | 0.7s | `SILENCE_TIMEOUT` |
| 76 | 261.76s | 262.69s | 0.96s | `SILENCE_TIMEOUT` |
| 77 | 263.42s | 263.42s | 0.03s | `SILENCE_TIMEOUT` |
| 78 | 264.77s | 266.37s | 1.63s | `SILENCE_TIMEOUT` |
| 79 | 267.52s | 272.93s | 5.44s | `SILENCE_TIMEOUT` |
| 80 | 275.58s | 276.35s | 0.8s | `SILENCE_TIMEOUT` |
| 81 | 278.05s | 278.98s | 0.96s | `SILENCE_TIMEOUT` |
| 82 | 283.14s | 283.23s | 0.13s | `SILENCE_TIMEOUT` |
| 83 | 288.06s | 288.06s | 0.03s | `SILENCE_TIMEOUT` |
| 84 | 289.73s | 290.02s | 0.32s | `SILENCE_TIMEOUT` |
| 85 | 291.04s | 291.78s | 0.77s | `SILENCE_TIMEOUT` |
| 86 | 292.7s | 292.83s | 0.16s | `SILENCE_TIMEOUT` |
| 87 | 295.49s | 296.96s | 1.5s | `SILENCE_TIMEOUT` |
| 88 | 298.85s | 299.36s | 0.54s | `SILENCE_TIMEOUT` |
| 89 | 300.26s | 301.76s | 1.54s | `SILENCE_TIMEOUT` |
| 90 | 304.13s | 304.16s | 0.06s | `SILENCE_TIMEOUT` |
| 91 | 305.12s | 306.69s | 1.6s | `SILENCE_TIMEOUT` |
| 92 | 307.94s | 309.15s | 1.25s | `SILENCE_TIMEOUT` |
| 93 | 310.24s | 310.24s | 0.03s | `SILENCE_TIMEOUT` |
| 94 | 312.16s | 312.8s | 0.67s | `SILENCE_TIMEOUT` |
| 95 | 314.46s | 314.98s | 0.54s | `SILENCE_TIMEOUT` |
| 96 | 316.0s | 317.41s | 1.44s | `SILENCE_TIMEOUT` |
| 97 | 318.05s | 318.59s | 0.58s | `SILENCE_TIMEOUT` |
| 98 | 322.46s | 323.3s | 0.86s | `SILENCE_TIMEOUT` |
| 99 | 324.9s | 325.25s | 0.38s | `SILENCE_TIMEOUT` |
| 100 | 326.24s | 327.58s | 1.38s | `SILENCE_TIMEOUT` |

### `Chinese_fast_speed_11s.wav` (11.38s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0s | 8.0s | 8.03s | `MAX_SPEECH_DURATION_REACHED` |
| 2 | 8.1s | 11.33s | 3.26s | `STREAM_EOF` |

### `Chinese_noise_28s.wav` (28.26s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0s | 7.97s | 8.0s | `MAX_SPEECH_DURATION_REACHED` |
| 2 | 8.0s | 15.97s | 8.0s | `MAX_SPEECH_DURATION_REACHED` |
| 3 | 16.0s | 23.97s | 8.0s | `MAX_SPEECH_DURATION_REACHED` |
| 4 | 24.0s | 28.22s | 4.26s | `STREAM_EOF` |

### `Cross_lingual_English_French_Italian_Spanish_6s.wav` (6.23s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.48s | 2.24s | 1.79s | `SILENCE_TIMEOUT` |
| 2 | 2.88s | 5.98s | 3.14s | `STREAM_EOF` |

### `English_low_speech_quality_19s.wav` (19.02s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0s | 0.06s | 0.03s | `SILENCE_TIMEOUT` |
| 2 | 1.47s | 2.24s | 0.8s | `SILENCE_TIMEOUT` |
| 3 | 3.58s | 3.58s | 0.03s | `SILENCE_TIMEOUT` |
| 4 | 5.89s | 8.93s | 3.07s | `SILENCE_TIMEOUT` |
| 5 | 13.31s | 13.82s | 0.54s | `SILENCE_TIMEOUT` |

### `English_multiple_kinds_of_noise_88s.wav` (88.19s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 4.13s | 4.16s | 0.06s | `SILENCE_TIMEOUT` |
| 2 | 6.59s | 7.07s | 0.51s | `SILENCE_TIMEOUT` |
| 3 | 10.43s | 10.46s | 0.06s | `SILENCE_TIMEOUT` |
| 4 | 11.39s | 11.94s | 0.58s | `SILENCE_TIMEOUT` |
| 5 | 12.8s | 13.73s | 0.96s | `SILENCE_TIMEOUT` |
| 6 | 14.43s | 15.23s | 0.83s | `SILENCE_TIMEOUT` |
| 7 | 15.94s | 15.94s | 0.03s | `SILENCE_TIMEOUT` |
| 8 | 17.34s | 17.41s | 0.1s | `SILENCE_TIMEOUT` |
| 9 | 21.7s | 22.72s | 1.06s | `SILENCE_TIMEOUT` |
| 10 | 25.95s | 26.5s | 0.58s | `SILENCE_TIMEOUT` |
| 11 | 27.94s | 27.94s | 0.03s | `SILENCE_TIMEOUT` |
| 12 | 28.64s | 30.02s | 1.41s | `SILENCE_TIMEOUT` |
| 13 | 31.49s | 32.22s | 0.77s | `SILENCE_TIMEOUT` |
| 14 | 33.22s | 33.28s | 0.1s | `SILENCE_TIMEOUT` |
| 15 | 34.53s | 34.53s | 0.03s | `SILENCE_TIMEOUT` |
| 16 | 35.62s | 35.65s | 0.06s | `SILENCE_TIMEOUT` |
| 17 | 36.35s | 36.58s | 0.26s | `SILENCE_TIMEOUT` |
| 18 | 37.22s | 37.54s | 0.35s | `SILENCE_TIMEOUT` |
| 19 | 38.59s | 40.19s | 1.63s | `SILENCE_TIMEOUT` |
| 20 | 41.25s | 42.08s | 0.86s | `SILENCE_TIMEOUT` |
| 21 | 43.71s | 43.94s | 0.26s | `SILENCE_TIMEOUT` |
| 22 | 45.06s | 48.9s | 3.87s | `SILENCE_TIMEOUT` |
| 23 | 50.88s | 55.07s | 4.22s | `SILENCE_TIMEOUT` |
| 24 | 55.94s | 57.18s | 1.28s | `SILENCE_TIMEOUT` |
| 25 | 58.05s | 58.05s | 0.03s | `SILENCE_TIMEOUT` |
| 26 | 60.7s | 61.82s | 1.15s | `SILENCE_TIMEOUT` |
| 27 | 64.32s | 68.8s | 4.51s | `SILENCE_TIMEOUT` |
| 28 | 70.3s | 71.84s | 1.57s | `SILENCE_TIMEOUT` |
| 29 | 73.34s | 73.79s | 0.48s | `SILENCE_TIMEOUT` |
| 30 | 75.26s | 75.97s | 0.74s | `SILENCE_TIMEOUT` |
| 31 | 76.64s | 77.34s | 0.74s | `SILENCE_TIMEOUT` |
| 32 | 78.14s | 78.43s | 0.32s | `SILENCE_TIMEOUT` |
| 33 | 79.1s | 79.39s | 0.32s | `SILENCE_TIMEOUT` |
| 34 | 80.51s | 80.93s | 0.45s | `SILENCE_TIMEOUT` |
| 35 | 81.86s | 82.27s | 0.45s | `SILENCE_TIMEOUT` |
| 36 | 86.72s | 88.1s | 1.41s | `STREAM_EOF` |

### `Japanese_5s.wav` (5.08s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0s | 4.7s | 4.74s | `STREAM_EOF` |

### `Russian_4s.wav` (4.76s)
| Utterance ID | Bắt đầu (s) | Kết thúc (s) | Độ dài (s) | Lý do kết thúc (Reason) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.26s | 4.26s | 4.03s | `STREAM_EOF` |

## 3. Đánh Giá & Kết Luận

- **RTF trung bình**: **0.0079** (Nhanh gấp **127.3x** thời gian thực).
- **Độ trễ xử lý mỗi frame 32ms**: **0.25 ms** (Cực kỳ thấp, hoàn toàn không gây nghẽn CPU).
- **Khả năng bắt giọng**: FSMN-VAD phát hiện cực nhạy, phù hợp hoàn hảo với các video Whisper/ASMR và đa ngôn ngữ.
