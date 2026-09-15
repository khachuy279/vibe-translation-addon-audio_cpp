# Vibe Translation Addon — Phụ đề song ngữ thời gian thực + Dịch neural + Lồng tiếng (transcribe.cpp)

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Extension](https://img.shields.io/badge/Extension-Manifest%20V3-FF7139.svg)](https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions)
[![ASR](https://img.shields.io/badge/ASR-transcribe.cpp%20(Vulkan%2FCPU)-76B900.svg)](external/transcribe.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#-giấy-phép)

Hệ thống **chạy hoàn toàn ngoại tuyến** (100 % local inference) biến mọi video trên trình duyệt
thành video có **phụ đề song ngữ + giọng thuyết minh**, không gửi dữ liệu ra Internet:

```
Audio tab (extension) → VAD → ASR (transcribe.cpp) → cắt câu → Dịch GGUF → Phụ đề + TTS
```

Đo trên máy tham chiếu (**RTX 5060 Ti 16 GB**): commit câu p50 **~107 ms**, preview p95 **108 ms**,
token dịch đầu tiên **26 ms**, VRAM đỉnh **~9,5 GB** cho cả 4 model (ASR + VAD + dịch 7B + TTS).

Repository: [github.com/khachuy279/vibe-translation-addon-transcribe_cpp](https://github.com/khachuy279/vibe-translation-addon-transcribe_cpp)

> [!IMPORTANT]
> **Thiết kế cho sử dụng cá nhân: tối đa 1 phiên / 1 video.** ASR, dịch và TTS đều là singleton dùng
> chung; thư viện `transcribe.cpp` 0.x chỉ cho **một** stream in-flight trên mỗi model
> (`external/transcribe.cpp/include/transcribe.h`). Muốn N phiên song song phải nạp N bản model.
> Xem [Hạn chế đã biết](#-hạn-chế-đã-biết) và báo cáo audit đầy đủ trong `report/audit/`.

---

## 📑 Mục lục

- [Tính năng chính](#-tính-năng-chính)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Chuẩn bị mô hình](#-chuẩn-bị-mô-hình)
- [Khởi chạy backend](#-khởi-chạy-backend)
- [Cài extension & sử dụng](#-cài-extension--sử-dụng)
- [Kiến trúc & luồng dữ liệu](#-kiến-trúc--luồng-dữ-liệu)
- [Giao thức WebSocket (v3)](#-giao-thức-websocket-v3)
- [REST API](#-rest-api)
- [Cấu hình quan trọng](#-cấu-hình-quan-trọng)
- [Kiểm thử](#-kiểm-thử)
- [Kết quả đo](#-kết-quả-đo)
- [Khắc phục sự cố](#-khắc-phục-sự-cố)
- [Hạn chế đã biết](#-hạn-chế-đã-biết)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Giấy phép](#-giấy-phép)

---

## ✨ Tính năng chính

| Nhóm | Chi tiết |
| :--- | :--- |
| **Phụ đề** | Phụ đề gốc hiện **dần** (preview) rồi **chốt** khi hết câu; bản dịch hiện song song. Tuỳ chọn **“hiện bản dịch 1 lần”** (tắt chạy chữ) — bật mặc định. |
| **Cắt câu** | 4 bậc: `VAD_SILENCE` > `MAX_DURATION` > `STABLE_PREFIX` > `TIMEOUT_FORCE`; cắt ở ranh giới từ, có overlap 250 ms và lọc trùng ranh giới (không mất chữ). |
| **Seek/tua video** | Extension phát hiện `seeking`/`seeked` → gửi `reset_stream`; backend xoá audio + trạng thái cũ nên phụ đề không trộn nội dung trước/sau khi tua. |
| **Đổi model nóng** | Đổi ASR / VAD / model dịch / giọng TTS ngay trong popup, **không cần restart**. Model dịch chưa có file sẽ **tự tải** ở luồng nền (model đang chạy vẫn phục vụ). |
| **Chống treo** | Watchdog luồng thật dump stack mọi thread nếu event loop đứng (`[STALL WATCHDOG]`); watchdog suy luận + hàng rào RSS chống phình bộ nhớ native. |
| **Lồng tiếng** | OmniVoice voice-cloning, gửi **binary frame** (không base64), auto-ducking âm lượng video gốc. |
| **Riêng tư** | 100 % offline sau khi tải model; không telemetry, không API ngoài. |

---

## 💻 Yêu cầu hệ thống

| Thành phần | Yêu cầu |
| :--- | :--- |
| OS | Windows 10/11 64-bit (đã đo trên Windows; mã Python không phụ thuộc Windows trừ loader CUDA/DLL) |
| Python | 3.10 – 3.13 (máy tham chiếu dùng **3.13**) |
| GPU | NVIDIA RTX ≥ 8 GB VRAM; thoải mái nhất 12–16 GB (RTX 4060 Ti / 5060 Ti / 4070 / 4080 / 5080) |
| Driver | NVIDIA ≥ 535 (CUDA 12.x) cho **dịch (llama.cpp) + TTS (PyTorch)** |
| Trình duyệt | Firefox (khuyến nghị, `about:debugging`), Chrome/Edge (load unpacked) |
| RAM | ≥ 8 GB trống (native ASR/Vulkan có thể phình tạm thời; xem [Khắc phục sự cố](#-khắc-phục-sự-cố)) |

> **ASR dùng Vulkan, không phải CUDA.** Bản `transcribe-cpp-native` trên PyPI chỉ có backend
> **Vulkan + CPU**; `transcribe-cpp-native-cu12` hiện là *name reservation* (bản `0.0.0`, wheel ~1,4 KB)
> — cài vào **không có** CUDA. Muốn CUDA phải tự build từ
> `external/transcribe.cpp/bindings/python-native-cu12/`. Kiểm tra thực tế: `GET /health` →
> `asr_runtime.backend`.

---

## ⚙️ Cài đặt

```powershell
git clone https://github.com/khachuy279/vibe-translation-addon-transcribe_cpp.git
cd vibe-translation-addon-transcribe_cpp

python -m venv .venv
.venv\Scripts\activate

# 1) PyTorch CUDA 12.4 (cho TTS OmniVoice; VAD chạy torch trên CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2) Runtime backend
pip install fastapi "uvicorn[standard]" websockets pydantic pyyaml numpy soundfile scipy cryptography psutil orjson

# 3) ASR native (CPU + Vulkan — ĐANG dùng; KHÔNG có provider CUDA trên PyPI)
pip install transcribe-cpp-native

# 4) Dịch GGUF trên GPU
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# 5) VAD + TTS + tiện ích tải model/voice
pip install huggingface_hub fireredvad silero-vad funasr omnivoice
# (tuỳ chọn) chỉ cần khi tải file giọng mẫu từ HuggingFace Dataset:
pip install datasets
```

> `orjson` là tuỳ chọn — nếu thiếu, backend tự fallback sang `json` chuẩn (`backend/ws/connection.py`).

---

## 📦 Chuẩn bị mô hình

Tất cả model nằm trong `backend/models/` (thư mục này bị `.gitignore`).

**ASR** — catalog: `backend/models.yaml`

| Key (dùng trong popup/API) | File | Nguồn (HF repo) | Kiến trúc |
| :--- | :--- | :--- | :--- |
| `qwen3-asr-1.7b` *(mặc định)* | `Qwen3-ASR-1.7B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-1.7B-gguf` | offline LLM |
| `qwen3-asr-0.6b` | `Qwen3-ASR-0.6B-Q8_0.gguf` | `handy-computer/Qwen3-ASR-0.6B-gguf` | offline LLM |
| `sensevoice-small` | `SenseVoiceSmall-F32.gguf` | `handy-computer/SenseVoiceSmall-gguf` | non-autoregressive |
| `nemotron-3.5-streaming` | `nemotron-3.5-asr-streaming-0.6b-F16.gguf` | `handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf` | streaming |
| `cohere-transcribe` | `cohere-transcribe-03-2026-Q8_0.gguf` | `handy-computer/cohere-transcribe-03-2026-gguf` | offline LLM |
| `voxtral-mini-4b-realtime` | `Voxtral-Mini-4B-Realtime-2602-Q5_K_M.gguf` | `handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf` | streaming |

**Dịch** — catalog: `backend/translation_models.yaml`

| Key | File | Nguồn (HF repo) | Ghi chú |
| :--- | :--- | :--- | :--- |
| `tencent` *(mặc định)* | `Hy-MT2-7B-UD-Q4_K_XL.gguf` | `unsloth/Hy-MT2-7B-GGUF` | chất lượng cao (~4,6 GB) |
| `tencent-1.8b` | `Hy-MT2-1.8B-UD-Q8_K_XL.gguf` | `unsloth/Hy-MT2-1.8B-GGUF` | siêu nhanh (~2,0 GB) |
| `xiaomi` | `MiLMMT-46-4B-v1.0.Q4_K_M.gguf` | `mradermacher/MiLMMT-46-4B-v1.0-GGUF` | ~2,5 GB |
| `gemmax` | `GemmaX2-28-9B-v0.2.i1-Q4_K_M.gguf` | `mradermacher/GemmaX2-28-9B-v0.2-i1-GGUF` | 28 ngôn ngữ (~5,8 GB) |

**Model dịch tự tải.** Nếu file chưa có, chỉ cần chọn model trong popup: backend tải ở **luồng nền**
(log tiến độ ~10 s/lần), **giữ nguyên model đang chạy** cho tới khi nạp xong, rồi tự swap. Tắt bằng
`TranslationConfig.auto_download = False` (khi đó API trả lỗi 400 kèm đường dẫn cần copy).
Trạng thái tải xem ở `GET /api/config → translation.download` (popup hiển thị % và nhãn
`⤓ chưa tải` / `⚡` cho từng model).

**VAD** — tự chuẩn bị, không cần làm gì:

| Engine | Cách có model |
| :--- | :--- |
| `silero-vad` | copy từ gói `silero_vad` (bundled `silero_vad.jit`), nếu thiếu thì tải từ GitHub |
| `firered-vad` *(mặc định)* | `hf_hub_download("FireRedTeam/FireRedVAD", …)` vào `backend/models/firered_stream/` |
| `fsmn-vad` | `snapshot_download("funasr/fsmn-vad")` vào `backend/models/fsmn_vad/` |

> ⚠️ Lần đầu chọn một VAD engine mới cần mạng để tải model (chạy một lần, sau đó offline).
> **ASR thì KHÔNG tự tải** — phải copy file `.gguf` vào `backend/models/` (API chỉ báo `is_downloaded`).

**Giọng mẫu TTS:** đặt `.wav` (kèm `.txt` transcript nếu có) vào `backend/voices/` và khai báo trong
`backend/voices/voices.json`. Có sẵn script tải mẫu từ dataset tiếng Việt:

```powershell
python backend\voices\wav_downloader.py <tên_file_trong_dataset.wav>
```

---

## ▶️ Khởi chạy backend

```powershell
python backend\main.py
```

Lần đầu khởi động sẽ **pre-warm** ASR + dịch + VAD (dịch 7B mất ~47 s: ~11 s nạp + ~36 s warm-up
inference — giữ nguyên vì nếu không warm thì câu dịch **đầu tiên** bị đơ ~38 s):

```text
[STARTUP] Đang nạp và pre-warm ASR, VAD & Translation Models...
[ASR] Nạp thành công ASR Model 'qwen3-asr-1.7b' trên GPU (Backend: Vulkan0)
[TRANSLATE] Nạp thành công mô hình dịch 'tencent' trên GPU (n_ctx=512, n_batch=256, n_threads=4)
[VAD] Đã nạp model FireRed Stream-VAD từ: backend/models/firered_stream/Stream-VAD
[MAIN] Chế độ WSS (SSL) kích hoạt với cert: backend/cert.pem
INFO:  Uvicorn running on https://0.0.0.0:8765 (Press CTRL+C to quit)
```

- Chứng chỉ WSS **tự ký** được sinh tự động vào `backend/cert.pem` + `backend/key.pem`.
- Mở `https://localhost:8765` một lần và chọn **Nâng cao → Tiếp tục** để trình duyệt chấp nhận
  chứng chỉ (nếu không, extension sẽ báo *Server Offline*).

---

## 🧩 Cài extension & sử dụng

**Firefox**
1. `about:debugging#/runtime/this-firefox` → **Load Temporary Add-on…** → chọn `extension_firefox/manifest.json`.
2. Mở trang video (YouTube, Bilibili, Coursera, Twitch, Zoom web…) → bấm icon extension.
3. Chọn model/ngôn ngữ/giọng đọc → **Bắt đầu dịch**. Phụ đề hiện ở overlay; kéo/thu nhỏ tuỳ chỉnh trong popup.

**Chrome / Edge**: `chrome://extensions` → bật *Developer mode* → **Load unpacked** → chọn thư mục `extension_firefox`.

**Bảng điều khiển trong popup**

| Nhóm | Điều khiển |
| :--- | :--- |
| Nhận dạng | ASR Engine, VAD Engine, `VAD Silence` (ms), `Threshold`, `Min Words` |
| Dịch | Translation Model (kèm nhãn đã tải/chưa tải), Source Lang, Target Lang |
| Lồng tiếng | Bật/tắt TTS, Voice Clone, TTS Speed, Auto-Ducking, âm lượng audio gốc |
| Phụ đề | **Hiện bản dịch 1 lần** (tắt chạy chữ), Position Y, Width, cỡ chữ gốc/dịch, Font Family, Font weight, Max lines |

---

## 🏛️ Kiến trúc & luồng dữ liệu

```mermaid
flowchart TD
    EXT[Extension MV3<br/>AudioWorklet 48k→16k PCM16] -->|WSS binary frame| WS[FastAPI /ws<br/>protocol v3]
    WS --> BUF[CircularAudioBuffer 60s]
    WS --> VAD[VAD: FireRed / Silero / FSMN<br/>CPU, pre-warm]
    VAD -->|speech start/end| SEG[Segmenter + CommitManager 4 bậc]
    BUF --> NORM[Speech Normalizer<br/>RMS auto-gain + limiter]
    NORM --> ASR[transcribe.cpp<br/>Vulkan/CPU]
    ASR -->|preview tokens| WS
    ASR -->|committed text| DEDUP[3-layer Dedup]
    DEDUP --> TRANS[llama.cpp GGUF<br/>Hunyuan-MT2]
    TRANS -->|streaming translation| WS
    TRANS --> TTS[OmniVoice TTS<br/>CUDA]
    TTS -->|binary WAV frame| WS
    WS -->|utterance_update + translation + audio| EXT
```

**Thành phần & hiệu năng đo được**

| Thành phần | Công nghệ | Số đo |
| :--- | :--- | :--- |
| Audio ingress | Ring buffer 60 s, single-writer + mutex multi-reader | ghi `< 0,05 ms` |
| VAD | FireRed-VAD / Silero / FSMN, CPU, cả 3 pre-warm | `~3,2 ms` / chunk 25 ms (~13 % 1 nhân) |
| Chuẩn hoá | RMS auto-gain + soft-knee + peak limiter | `< 0,2 ms` / chunk |
| ASR | `transcribe.cpp` (Vulkan), preview cửa sổ ≤ 6 s | commit p50 **~107 ms** (câu 4,5 s, RTF ≈ 0,024) |
| Cắt câu | CommitManager 4 bậc + dedup 3 lớp | cả 3 loại lý do cắt đều xuất hiện trong log |
| Dịch | llama.cpp GPU, streaming token | **~67 token/s**, token đầu **26 ms** |
| TTS | OmniVoice PyTorch 24 kHz, cache prompt giọng | `~420 ms` / câu 3 s (RTF 0,077) |
| WebSocket | framing nhị phân, version hoá giao thức | cleanup phiên `~0,4 ms` |

---

## 🔌 Giao thức WebSocket (v3)

Backend và client **thương lượng phiên bản**: `protocolVersion: 3` ⇒ backend gửi payload **gọn**
(một tên cho mỗi giá trị) và cho phép **frame nhị phân** cho TTS.

**Client → server (text JSON)**

| Message | Ý nghĩa |
| :--- | :--- |
| `{"type":"set_config","action":"configure", …}` | Đồng bộ toàn bộ cấu hình popup (ngôn ngữ, VAD, TTS, model…) |
| `{"type":"reset_stream","reason":"seek"}` | Video vừa tua ⇒ xoá audio/trạng thái cũ (F-44) |
| `{"type":"ping","timestamp":…}` | Đo RTT, giữ kết nối |

**Client → server (binary)**: `[4 byte uint32 LE = độ dài header][header JSON][PCM16 LE]`, header gồm
`captureTimestamp`, `chunkIndex`… — xem `extension_firefox/lib/frame-builder.js`.

**Server → client**

| Message | Nội dung |
| :--- | :--- |
| `utterance_update` | Phụ đề gốc: `partial` (đang nói) hoặc bản chốt |
| `translation` | Bản dịch (có `partial` khi đang stream token) |
| `tts_audio` | Khung **nhị phân** (WAV/PCM 24 kHz) hoặc JSON base64 cho client cũ |
| `model_status` | `stage` (`asr`/`translation`), `state` (`downloading`/`loading`/`ready`/`error`) |
| `pong` | Trả lời `ping` |

---

## 🌐 REST API

| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/` | Trang HTML xác nhận chứng chỉ HTTPS/WSS đã được chấp nhận |
| `GET` | `/health` | Trạng thái: model ASR, `asr_runtime.backend`, VAD, model dịch, số phiên, `loop_stall_ms` |
| `GET` | `/api/config` | Cấu hình hiện tại + catalog model (kèm `is_downloaded`) + `translation.download` |
| `POST` | `/api/config` *(alias `/api/switch-engine`)* | Đổi ASR/VAD/model dịch/ngôn ngữ/TTS. Model dịch thiếu file ⇒ **202** (tải nền); tên sai ⇒ **400**; đang tải model khác ⇒ **409** |
| `GET` | `/api/voices` | Danh sách giọng clone khả dụng |
| `POST` | `/api/tts/prewarm` | Nạp trước model TTS |
| `GET` | `/api/metrics` | Báo cáo metric tổng hợp + cảnh báo nghẽn |
| `GET` | `/api/metrics/pipeline` | Ảnh chụp nhanh các stage hot path (asr/vad/queue) |
| `WS` | `/ws` *(alias `/`)* | Kênh streaming chính |

Ví dụ đổi model dịch (tự tải nếu thiếu):

```powershell
curl.exe -k -X POST https://localhost:8765/api/config `
  -H "Content-Type: application/json" `
  -d '{\"translation_model\":\"tencent-1.8b\"}'
```

---

## 🎛️ Cấu hình quan trọng

Toàn bộ cấu hình tập trung ở `backend/config.py` (Pydantic v2). Các cờ hay dùng:

| Cờ | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `ws.port` / `ws.protocol_version` | `8765` / `3` | Cổng WSS và phiên bản giao thức |
| `vad.vad_engine` / `vad.threshold` | `firered-vad` / `0,45` | Engine VAD và ngưỡng phát hiện nói |
| `vad.silence_duration_ms` / `hangover_ms` | `600` / `400` | Im lặng để ngắt câu / giữ trạng thái nói |
| `asr.min_transcribe_sec` | `0,35` | Audio tối thiểu để có preview đầu tiên |
| `asr.preview_window_sec` | `6,0` | Cửa sổ preview ⇒ chi phí preview bị chặn trên |
| `asr.max_inflight_infer` | `1` | Số suy luận song song (chống phình hàng đợi) |
| `asr.preview_reuse_for_commit` | `False` | **Giữ TẮT** (đo WER cho thấy bật thì xấu hơn) |
| `sentence.max_duration_sec` | `6,0` | Chốt an toàn cho câu nói liên tục |
| `sentence.min_words_to_commit` | `2` | Lọc tiếng ậm ừ / mảnh vụn |
| `translation.base` / `auto_download` | `tencent` / `True` | Model dịch đang dùng / tự tải khi thiếu file |
| `tts.enabled` / `default_voice` | `True` / `speaker_01_0039.wav` | Lồng tiếng và giọng mặc định |

---

## 🧪 Kiểm thử

```powershell
# Tầng A — mặc định, KHÔNG nạp model thật, ~10-15 s
python -m pytest

# Tầng B — cần model thật (chậm, tốn VRAM/RAM): benchmark + WER + độ trễ streaming
python -m pytest -m slow

# Tầng C — E2E đủ 3 model ASR + dịch + TTS
python -m pytest -m full
```

Cấu hình marker nằm ở `pytest.ini` (`addopts = -m "not slow and not full"`). Bộ test tầng A hiện có
**hơn 230 hàm test** trong `backend/tests/test_01…test_22`, phủ: ring buffer & chuẩn hoá, commit
manager, hiệu lực cấu hình popup, khoá/metric, giao thức compact, chống trùng dòng log, quy ước
logging, seek/reset, và tải model dịch + swap nguyên tử.

Có cả harness JS chạy bằng Node (không cần trình duyệt): `backend/tests/js/worklet_harness.js`
(AudioWorklet, 11 điểm), `backend/tests/js/subtitle_policy_test.js` (chính sách hiện bản dịch) và
`backend/tests/js/subtitle_renderer_harness.js` (**3 tầng phụ đề** với DOM giả lập — gồm ca bản dịch
đến muộn sau khi câu bị đẩy từ TẦNG 2 lên TẦNG 1).

> Chạy một harness tầng B/C có thể ngốn RAM lớn do lỗi phình bộ nhớ **bên trong native**; mọi harness
> đã gắn `backend/utils/mem_guard.py` (tự dừng tiến trình khi RSS vượt trần, mặc định 4000 MB).

---

## 📊 Kết quả đo

Nguồn: `report/audit/05_measurements_and_status.md` (§3, §13, §18). Số trên **file tĩnh** không so
trực tiếp được với số trên **pipeline streaming** — hãy dùng `/api/metrics/pipeline` để đo thật.

| # | Mục tiêu | Đo được | Trạng thái |
| :--- | :--- | :--- | :--- |
| K1 | p95 `preview_ms` < 250 ms | **108 ms** | ✅ |
| K2 | Preview đầu tiên < 500 ms | **570 ms** | ⚠️ vượt 14 % — đánh đổi có chủ ý để giữ độ chính xác preview (C4) |
| K3 | Token dịch đầu tiên < 120 ms | **26 ms** | ✅ |
| K4 | Ngừng nói → phụ đề gốc chốt < 1,2 s | **52 ms** (p95 110 ms) | ✅ |
| K5 | Không mất câu; mọi drop/merge có counter | `commit_carried_over`, `pending_commits`, `commit_slice_clamped`, `commit_dropped_stale`… | ✅ |
| K6 | WER không xấu đi | `reuse_preview_for_commit` bật ⇒ xấu hơn (+9,5 / +14,8 điểm ở 2 file đo ổn định) ⇒ giữ TẮT | ✅ |
| K7 | VRAM đỉnh < 14 GB | **9,5 GB** (ASR + dịch 7B + TTS) | ✅ |
| K8 | ASR dùng CUDA | Vulkan (`cuda_backend_available = False`) | ❌ bất khả thi với wheel hiện có |
| K9 | Mọi control trong popup có tác dụng | 11/11 nhóm control có test | ✅ |
| K10 | 0 crash khi đổi model lúc đang stream | soak **200 vòng** | ✅ |
| K11 | Độ trễ capture phía client < 70 ms | **64 ms** (worklet gom 1024 mẫu @16 kHz) | ✅ (chờ xác nhận trên Firefox thật) |
| K12 | Bộ test mặc định < 15 s | **10,9 s** (đo ở lần chạy đầy đủ gần nhất) | ✅ |

**Benchmark E2E trên `wav_test/`** (8 file audio: 7 file thoại EN/ZH/JA/RU + 1 file đa ngữ, kèm
transcript tham chiếu):

| Chỉ số | Giá trị |
| :--- | :--- |
| RTF suy luận ASR | **0,0232** (nhanh ~43× thời gian thực) |
| Độ trễ ASR trung bình | **278 ms** sau khi dứt tiếng |
| Tốc độ dịch (Hunyuan-MT2 7B) | **67,4 token/s** |
| Độ trễ TTS (OmniVoice) | **421 ms** (RTF 0,077) |
| Dọn phiên khi đổi tab/video | **0,4 – 0,6 ms** |
| VRAM cho cả 4 model | **~9,5 GB / 16 GB** |

---

## 🧯 Khắc phục sự cố

| Triệu chứng | Nguyên nhân & cách xử lý |
| :--- | :--- |
| Popup báo **Server Offline** dù backend đang chạy | Chưa chấp nhận chứng chỉ tự ký: mở `https://localhost:8765` → *Nâng cao → Tiếp tục*, rồi mở lại popup |
| `pip install transcribe-cpp-native-cu12` rồi vẫn không có CUDA | Gói này chỉ là *name reservation*. ASR chạy **Vulkan** — đây là đường được hỗ trợ. Kiểm tra `GET /health → asr_runtime.backend` |
| Đổi model dịch báo *Chưa có file GGUF cục bộ* | File chưa tải và `auto_download` đang tắt. Bật lại (mặc định bật) để backend tự tải, hoặc copy `.gguf` vào `backend/models/` |
| Đang tải model dịch, API trả **409** | Một lượt tải/nạp khác đang chạy. Xem `GET /api/config → translation.download`, đợi xong rồi thử lại |
| Backend đứng im, **Ctrl+C không tắt được** | Xem log có `[STALL WATCHDOG]` (dump stack mọi thread). Gửi kèm dump khi báo lỗi; đây là dạng treo event loop mà watchdog được thiết kế để bắt |
| RAM tăng liên tục khi chạy lâu | Lỗi phình bộ nhớ nằm trong native `transcribe.cpp`/Vulkan (đã giảm mạnh bằng `max_inflight_infer=1` + watchdog, chưa xử lý tận gốc). Theo dõi RSS và báo lại nếu vượt ~2 GB |
| Phụ đề trộn nội dung sau khi tua video | Cần extension v0.6+ (gửi `reset_stream`). Reload extension tại `about:debugging` |
| Chữ bị mất ở đầu câu | Kiểm tra log có `commit_slice_clamped`; đây là họ lỗi F-49 đã sửa — nếu tái xuất hiện, gửi log kèm `utteranceId` |
| Log quá nhiều dòng trùng | Đã có cơ chế chống trùng (`LOG_DEDUP_MS`). Nếu vẫn thấy, dùng `GET /api/metrics` thay vì đọc log |

---

## ⚠️ Hạn chế đã biết

1. **1 phiên / 1 video** — xem khối cảnh báo ở đầu tài liệu. Không có model pool.
2. **ASR không tự tải model** — phải copy `.gguf` vào `backend/models/` (khác với model dịch và VAD).
3. **ASR dùng Vulkan, không có CUDA** trên bản wheel hiện hành.
4. **Cửa sổ preview có đuôi độ trễ lẻ**: p50 ≈ 87 ms nhưng thỉnh thoảng spike ~2,5 s do tầng native
   dựng lại scheduler/compute context mỗi `run()`. Nhịp preview **bỏ nhịp** (không trôi) và commit
   được ưu tiên nên phụ đề chốt không bị chặn.
5. **`reuse_preview_for_commit` mặc định TẮT** — chỉ bật sau khi tự đo WER.
6. **`vad_enabled=False` không được hỗ trợ thực sự**: VAD là bắt buộc để phân câu; backend tự bật lại
   và ghi cảnh báo.
7. **Số liệu trên file tĩnh ≠ pipeline streaming** (có vòng lặp preview + VAD + queue).
8. **Pre-warm dịch tốn ~47 s mỗi lần khởi động** — đổi lấy việc câu dịch đầu tiên không bị đơ ~38 s.

---

## 📁 Cấu trúc thư mục

```
vibe-translation-addon-transcribe_cpp/
├── backend/
│   ├── main.py                   # FastAPI app: lifespan pre-warm, REST API, WSS endpoint
│   ├── config.py                 # Cấu hình tập trung (Pydantic v2)
│   ├── models.yaml               # Catalog model ASR (GGUF)
│   ├── translation_models.yaml   # Catalog model dịch (GGUF + repo HuggingFace)
│   ├── asr/                      # transcribe.cpp engine, registry, adapter, text cleaner
│   ├── core/                     # Ring buffer, SpeechNormalizer, CommitManager, dedup, metrics, heartbeat
│   ├── vad/                      # VADProcessor + engine FireRed / Silero / FSMN
│   ├── translation/              # GGUFTranslator, hotswap (đổi model nguyên tử), registry, prompts
│   ├── tts/                      # OmniVoice TTS, audio processor, VoiceManager
│   ├── ws/                       # handler, session, connection, serializers (payload v3)
│   ├── utils/                    # logger (quy ước tag), CUDA DLL, SSL tự ký, mem_guard, stall_watchdog, model_download
│   ├── voices/                   # Giọng mẫu (.wav/.txt), voices.json, wav_downloader.py
│   ├── models/                   # Model cục bộ (.gguf, .jit, firered_stream/, fsmn_vad/) — gitignore
│   └── tests/                    # test_01…test_22 + harness JS (Node)
├── extension_firefox/            # Extension MV3: content script, overlay, popup, worklet
├── wav_test/                     # 8 file audio (thoại EN/ZH/JA/RU + đa ngữ) kèm transcript tham chiếu
├── report/                       # Báo cáo theo phase (01…07) + report/audit/ (audit hiệu năng)
├── scratch/                      # Script đo đạc tạm thời (gitignore)
├── pytest.ini                    # Marker slow / gpu / full
└── README.md
```

---

## 📄 Giấy phép

Dự án phát hành theo **MIT License** — dùng, sửa, chia sẻ tự do cho mục đích cá nhân.

> Lưu ý: repo hiện **chưa kèm file `LICENSE`**. Nếu bạn publish/fork, hãy thêm file MIT chuẩn
> (năm + tên tác giả) để giấy phép có hiệu lực rõ ràng.

**Tài liệu liên quan**

- `report/audit/00_BAO_CAO_AUDIT_HIEU_NANG.md` — audit hiệu năng đầy đủ (F-01…F-51).
- `report/audit/03_KE_HOACH_TRIEN_KHAI.md` — kế hoạch triển khai & KPI.
- `report/audit/05_measurements_and_status.md` — trạng thái, số đo, nghiệm thu K1–K12.

Mọi đóng góp (Pull Request / Issue) đều được hoan nghênh!
