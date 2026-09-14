# Vibe Translation Addon — Real-Time Video Subtitle, Neural Translation & Voice Cloning (`audio.cpp`)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![Firefox Extension](https://img.shields.io/badge/Firefox-Manifest%20V3-FF7139.svg)](https://addons.mozilla.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20CUDA-76B900.svg)](https://developer.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, completely offline, low-latency system for **real-time audio capture, streaming speech recognition (ASR via native `audio.cpp` CUDA runtime), CJK-aware sentence boundary detection, neural machine translation (Hunyuan-MT2 7B), and zero-shot voice cloning synthesis (OmniVoice-GGUF)** directly in your browser.

Tested and benchmarked on **NVIDIA GeForce RTX 5060 Ti**, delivering live bilingual subtitles and voice-over dubbing on any video streaming platform (YouTube, Bilibili, Coursera, Twitch, etc.) with **zero cloud dependencies, zero data leakage, and complete privacy**.

Repository: [https://github.com/khachuy279/vibe-translation-addon-audio_cpp](https://github.com/khachuy279/vibe-translation-addon-audio_cpp)

---

## 🚀 Key Highlights & Modular Architecture

- **⚡ Sub-Second E2E Pipeline Overhead**: Total latency from speech end to translated subtitle on screen & voice-cloned audio is **~400ms - 800ms**.
- **🎙️ Streaming Silero VAD v5 Engine**: Pure NumPy + ONNX Runtime execution on CPU (keeps GPU 100% free for LLMs).
  - **Chunk Slicing**: 512-sample (32ms @ 16kHz) frames with 64-sample context buffer.
  - **Latency:** `< 0.2 ms` per frame.
  - **RTF:** `0.0047` (~212x faster than real-time playback).
- **🤖 Multilingual ASR Streaming (`audio.cpp` CUDA C++ runtime)**:
  - Powered by native `audiocpp_server.exe` CUDA backend on port 8089.
  - Supported Models in `backend_audio_cpp/models.yaml`:
    - `qwen3-asr-1.7b`: Alibaba Qwen3 Audio-LLM 1.7B (`qwen3-asr-1.7b-q8_0.gguf`)
    - `nemotron-3.5-streaming`: NVIDIA Nemotron 3.5 Streaming 0.6B (`nemotron-3.5-asr-streaming-0.6b-q8_0.gguf`)
    - `voxtral-mini-4b-realtime`: Voxtral Mini 4B Realtime 2602 (`voxtral-mini-4b-realtime-2602-q4_k.gguf`)
  - **Live Model Switching:** Hot-swap ASR models live from the Firefox extension popup without restarting the backend.
- **🧠 Sentence Boundary & Commit Manager (`SentenceConfig`)**:
  - **CJK & Latin Token Counting (`count_content_tokens`)**: Accurately counts Japanese (Kanji/Hiragana/Katakana), Chinese (Hanzi), Korean (Hangul), and Latin words to prevent false utterance drops on spaceless scripts.
  - Multi-tier commit criteria: VAD Silence Timeout (`silence_duration_ms`), Stability Split, Max Duration (`8.0s`), and Max Characters (`150`).
- **🌐 Neural Machine Translation (`Hunyuan-MT2 7B`)**:
  - Model: **Hunyuan-MT2 7B** (`Hy-MT2-7B-UD-Q4_K_XL.gguf`, ~4.78GB) offloaded to GPU via `llama-cpp-python` CUDA backend.
  - **Generation Speed:** **69.5 tokens/second** on RTX 5060 Ti.
  - **Translation Latency:** `250ms - 800ms` per sentence with zero hallucination.
- **🗣️ Zero-Shot Voice Cloning TTS (`OmniVoice-GGUF`)**:
  - Model: `omnivoice-q8_0.gguf` (~1.35GB) running via native `audio.cpp` CUDA backend.
  - **Zero-Shot Speaker Reference:** Synthesizes natural Vietnamese voice cloning from reference audio sample `backend_audio_cpp/voices/speaker_01_0039.wav`.
  - **Real-Time Factor (RTF):** **~0.13 - 0.46** (2.2x to 7.6x faster than real-time playback).
  - **Prompt & Embedding Caching:** Caches speaker embeddings in VRAM for instant subsequent voice generation.
- **🛡️ Decoupled Async Pipeline & Concurrent VRAM Management**:
  - Runs all 4 models simultaneously on GPU/CPU with **~9.0 GB VRAM total** (out of 16 GB), leaving > 6.9 GB free VRAM without OOM.
  - Decoupled worker queues (`_translation_worker`, `_tts_worker`) for non-blocking real-time playback.

---

## 🏛️ End-to-End Pipeline Workflow

```
[Firefox Extension (Manifest V3)]
       │
       ▼ (Binary Audio Frame: Format A / 16kHz PCM16 Mono)
[WebSocket Gateway] (wss://localhost:8765/ws)
       │
       ├─► [SileroVADEngine (CPU ONNX)] ──────────► Voice Active Segments
       │                                                    │
       ▼                                                    ▼
[AudioCppASREngine] ◄───────────────────────────────────────┘
       │ (audio.cpp CUDA - Qwen3 / Nemotron / Voxtral)
       ├─► Live Partial Previews ──► WebSocket Out (`utterance_update`, is_final=False)
       ▼
[SentenceCommitter (CJK Token Count)]
       │ (VAD Silence / Stability / Max Duration / Min Words)
       ▼ (Committed Final Utterance)
[SessionState.translation_queue]
       │
       ▼
[HyMTTranslator] (Hunyuan-MT2 7B on CUDA GPU)
       │
       ├─► [WebSocket Out] ──► Subtitle Updates (`translation`, `utterance_update`)
       │
       ▼ (Optional TTS Dubbing)
[SessionState.tts_queue]
       │
       ▼
[OmniVoiceTTSEngine] (OmniVoice-GGUF Voice Cloning via audio.cpp CUDA)
       │
       ▼
[WebSocket Out] ──► Audio Playback (`tts_audio` base64 WAV 24kHz)
```

---

## 📁 Repository Structure

```
vibe-translation-addon-audio_cpp/
├── backend_audio_cpp/
│   ├── asr/              # AudioCppASREngine & ModelRegistry (models.yaml)
│   ├── bin/              # Native audiocpp_server.exe CUDA binary & DLLs
│   ├── commit/           # SentenceCommitter & SentenceConfig (CJK token support)
│   ├── models/           # Local models storage (.gguf, .onnx)
│   ├── translation/      # HyMTTranslator (Hunyuan-MT2 7B) & model_registry
│   ├── tts/              # OmniVoiceTTSEngine & VoiceManager (voices.json)
│   ├── vad/              # SileroVADEngine & VADStreamState
│   ├── utils/            # SSL certificate generator & CUDA helpers
│   ├── ws/               # WebSocket handler, frame protocol, serializers, session state
│   ├── config.py         # App configuration dataclasses
│   ├── models.yaml       # ASR models catalog
   ├── translation_models.yaml # Translation models catalog
│   └── main.py           # FastAPI + WSS server entry point (:8765)
├── extension_firefox/    # Firefox Manifest V3 Extension (Popup UI & Content Overlay)
├── benchmarks/           # Automated benchmark suite (VAD, ASR, Translation, TTS, E2E)
├── report/               # Markdown benchmark reports & tts_samples/
└── wav_test/             # Multi-language test audio samples
```

---

## 📊 Benchmark Summary & Performance Metrics

All modules have been benchmarked against multi-language audio samples in `wav_test/` on RTX 5060 Ti:

| Module | Benchmark Script | Key Metric | Measured Performance | Status |
| :--- | :--- | :--- | :--- | :---: |
| **VAD** | `benchmarks/bench_vad.py` | Latency / RTF | **0.15 ms / 0.0047 (212x real-time)** | ✅ PASS |
| **ASR** | `benchmarks/bench_asr.py` | Streaming RTF / Accuracy | **0.52 RTF (1.9x real-time) / 97.3% accuracy** | ✅ PASS |
| **Live Switching**| `benchmarks/bench_e2e_pipeline.py` | Hot-swap ASR Models | **Qwen3 1.7B ➔ Nemotron 3.5 ➔ Voxtral 4B (Instant)** | ✅ PASS |
| **Translation**| `benchmarks/bench_translation.py` | Generation Speed / Latency | **69.5 TPS / ~249ms - 835ms** | ✅ PASS |
| **TTS Voice Clone**| `benchmarks/bench_tts.py` | RTF / Output Format | **0.131 RTF (7.6x real-time) / 24kHz Mono WAV** | ✅ PASS |
| **E2E Pipeline** | `benchmarks/bench_e2e_pipeline.py` | Full E2E Streaming | **Smooth live subtitles + voice dubbing over WSS** | ✅ PASS |

Full detailed reports are available in [report/](file:///d:/vibe-translation-addon-transcribe_cpp/report/):
- `report/01_vad_benchmark_report.md`
- `report/02_asr_sentence_commit_report.md`
- `report/02b_asr_dynamic_model_switching_report.md`
- `report/02c_speech_normalization_benchmark_report.md`
- `report/03_translation_benchmark_report.md`
- `report/04_tts_benchmark_report.md`
- `report/05_final_e2e_report.md`

---

## 🛠️ Installation & Setup

### 1. Requirements
- **OS**: Windows 10/11 x64
- **GPU**: NVIDIA GPU with CUDA support (e.g. RTX 3060 / 4060 / 5060 Ti)
- **Python**: 3.10 / 3.11 / 3.12 / 3.13
- **Browser**: Firefox (Manifest V3)

### 2. Model Downloads & Placement
Place all model files inside `backend_audio_cpp/models/`:

1. **ASR Models**:
   - `qwen3-asr-1.7b-q8_0.gguf` (Download from [audio-cpp/audio.cpp-gguf Qwen3-ASR-1.7B-GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Qwen3-ASR-1.7B-GGUF))
   - `nemotron-3.5-asr-streaming-0.6b-q8_0.gguf` (Download from [audio-cpp/audio.cpp-gguf Nemotron-3.5-ASR-Streaming-0.6B-GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Nemotron-3.5-ASR-Streaming-0.6B-GGUF))
   - `voxtral-mini-4b-realtime-2602-q4_k.gguf` (Download from [audio-cpp/audio.cpp-gguf Voxtral-Mini-4B-Realtime-2602-GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Voxtral-Mini-4B-Realtime-2602-GGUF))
2. **Translation Model**:
   - `Hy-MT2-7B-UD-Q4_K_XL.gguf` (Download from `unsloth/Hy-MT2-7B-GGUF`)
3. **TTS Model**:
   - `omnivoice-q8_0.gguf` (Download from [audio-cpp/audio.cpp-gguf OmniVoice-GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/OmniVoice-GGUF))
4. **VAD Model**:
   - `silero_vad.onnx`

### 3. Voice Clone Reference Audio
Place reference speaker `.wav` files inside `backend_audio_cpp/voices/` (e.g. `speaker_01_0039.wav`) and update `backend_audio_cpp/voices/voices.json`.

---

## 🚀 Running the Backend Server

Launch the backend server (automatically generates SSL certificates for WSS):

```bash
python backend_audio_cpp/main.py
```

The server starts on:
- **HTTPS API**: `https://localhost:8765`
- **WSS Endpoint**: `wss://localhost:8765/ws`

---

## 🧩 Installing the Firefox Extension

1. Open Firefox and go to `about:debugging#/runtime/this-firefox`.
2. Click **Load Temporary Add-on...**
3. Select `extension_firefox/manifest.json`.
4. Open any video page (YouTube, Twitch, Coursera, etc.), open the extension popup:
   - **🤖 ASR Engine**: Select `Qwen3 ASR 1.7B`, `Nemotron 3.5 Streaming`, or `Voxtral Mini 4B Realtime`.
   - **🎙️ VAD Engine**: `Silero VAD`.
   - **🌐 Translation Model**: `Hunyuan-MT2 7B`.
   - **🎭 Voice Clone**: Select your reference voice (e.g. `speaker_01_0039.wav`).
   - Click **Start** to enjoy live bilingual subtitles and voice cloning dubbing!

---

## 📄 License
MIT License. Free for personal and commercial usage.
