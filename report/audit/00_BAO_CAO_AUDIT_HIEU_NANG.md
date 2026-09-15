# BÁO CÁO AUDIT HIỆU NĂNG TOÀN DIỆN
## Vibe Translation Addon — Real-Time Subtitle / Neural Translation / Voice Cloning (`transcribe.cpp`)

| | |
| :--- | :--- |
| **Loại báo cáo** | Performance / Architecture / Concurrency Audit (read-only) |
| **Phạm vi** | Toàn bộ repository `D:\vibe-translation-addon-transcribe_cpp` |
| **Đối tượng chính** | `backend/**` (Python), `extension_firefox/**` (JS), `external/transcribe.cpp/**` (C++/ggml + Python bindings) |
| **Nguyên tắc** | **Không sửa bất kỳ dòng code nào.** Mọi phát hiện đều kèm `file:line` |
| **Mục tiêu hệ thống** | Live subtitle E2E < 1s, offline 100%, 1 GPU, browser extension |
| **Ngày thực hiện** | Audit tĩnh (static, source-level) |

---

## 0. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Hệ thống có kiến trúc module hoá tốt, logging/metrics đầy đủ và các tối ưu vi mô hợp lý (orjson, LRU cache cho voice prompt, `inference_mode`, tránh roundtrip F32→I16 cho FireRed-VAD, transferable buffer trong worklet). Tuy nhiên, **kiến trúc runtime hiện tại có 4 lỗi thiết kế gốc khiến hệ thống không thể đạt mục tiêu "< 1s E2E" một cách ổn định và không thể scale quá 1 session**:

### Bốn vấn đề gốc (root causes)

| # | Vấn đề gốc | Bản chất | Hệ quả |
| :--- | :--- | :--- | :--- |
| **RC-1** | **ASR preview là O(N²)** | Mỗi 350ms, engine transcribe lại **toàn bộ** đoạn speech đang lớn dần từ `_speech_start_sample` (`backend/asr/engine.py:330,358-363`) | Đến cuối câu dài, mỗi preview tốn ~0.7s GPU và **bản preview hiển thị trễ hơn cả bản commit**. GPU bị chiếm >50% chỉ để tạo preview vứt đi |
| **RC-2** | **Commit Manager 4 bậc là dead code** | `CommitManager.decide_commit_trigger()` **không được gọi ở đâu trong runtime** (chỉ có trong test) | Mất toàn bộ safeguard chống câu dài: `max_duration_sec`, stability split, inactivity timeout. Câu nói liên tục không bao giờ được chốt → khuếch đại RC-1 |
| **RC-3** | **Kiến trúc singleton + global lock** | `_shared_session`, `_infer_lock`, `_VAD_EXECUTOR(max_workers=1)`, `_TRANS_EXECUTOR(max_workers=1)`, TTS singleton (`backend/asr/engine.py:39,46-51`, `backend/ws/handler.py:36`, `backend/translation/engine.py:34,44`) | **Không thể chạy > 1 session**: mọi session dùng chung 1 transcribe.cpp session và xếp hàng trên cùng các lock. Chính thư viện C++ cũng ghi rõ giới hạn này (xem §C) |
| **RC-4** | **Mỗi `run()` là một pipeline native hoàn toàn mới** | Mel + encoder + prefill + decode đều tính lại từ `pcm[0]`; scheduler/compute context/threadpool đều bị dựng lại mỗi run (`external/transcribe.cpp/src/transcribe-mel.cpp:390-398`, `src/transcribe.cpp:2251-2253`, `ggml-cpu.c:3400-3416`) | Chi phí **cố định mỗi run** (F-31…F-35) bị nhân lên theo số poll; không có bất kỳ state reuse nào. Đây là **khuếch đại tầng native của RC-1** |

### Bảng xếp hạng phát hiện theo mức độ

| ID | Phát hiện | Mức độ | Tác động chính |
| :--- | :--- | :--- | :--- |
| **F-01** | ASR preview re-transcribe toàn bộ audio đang lớn dần (O(N²)) | 🔴 CRITICAL | Latency ↑↑, GPU ↑↑ |
| **F-02** | Commit Manager 4 bậc không được wire vào pipeline | 🔴 CRITICAL | Latency ↑↑, mất safeguard |
| **F-03** | Không hỗ trợ multi-session: shared session + global lock | 🔴 CRITICAL | Scale = 1 |
| **F-04** | Race condition giải phóng model/session khi đang inference (`unload_shared_model` không giữ `_infer_lock`) | 🔴 CRITICAL | Crash / UB (native) |
| **F-05** | `_pending_commits` (maxlen=8) **âm thầm drop câu** khi ASR nghẽn | 🔴 CRITICAL | Mất phụ đề |
| **F-06** | AudioWorklet là **dead code**; thực tế dùng `ScriptProcessorNode` trên main thread | 🔴 CRITICAL | Latency + jank + glitch |
| **F-07** | Không có backpressure ở bất kỳ đâu (0 lần `bufferedAmount`) | 🔴 CRITICAL | RAM ↑ vô hạn, lag tích luỹ |
| **F-08** | TTS audio gửi dạng **base64 trong JSON** (+33% + full copy) | 🟠 HIGH | Băng thông, CPU, I/O |
| **F-09** | Preview và commit **chạy trùng lặp 100%** cùng một đoạn audio | 🟠 HIGH | Lãng phí ~30–50% compute ASR |
| **F-10** | `session.run()` dùng `timestamps="auto"` → materialize toàn bộ words/tokens không dùng | 🟠 HIGH | CPU/alloc mỗi lần gọi |
| **F-11** | `VoiceManager` / `SpeechNormalizer` bỏ qua cấu hình; normalizer tạo ~7 mảng tạm per call | 🟠 HIGH | Alloc churn, cấu hình vô hiệu |
| **F-12** | Logging hot path: level DEBUG + `flush()` mỗi record + 1 lock toàn cục | 🟠 HIGH | CPU, latency jitter |
| **F-13** | VAD `is_speech()` chạy **bên trong** `RLock` của processor | 🟠 HIGH | Cleanup/`force_end` bị block (bằng chứng: cleanup 102.68ms) |
| **F-14** | Không dùng `Session.cancel()` / abort callback | 🟠 HIGH | Không thể huỷ inference cũ khi đổi tab/model |
| **F-15** | `metrics._checkpoints` tăng vô hạn (2 entry/session, không bao giờ xoá) | 🟠 HIGH | Memory growth |
| **F-16** | Hàng đợi TTS client **không giới hạn**, không có drop policy | 🟠 HIGH | RAM ↑, lệch tiếng vĩnh viễn |
| **F-17** | Pre-roll 300ms lấy nhầm audio của câu trước (off-by-4800 samples) | 🟠 HIGH | Chất lượng + dedup |
| **F-18** | `vad_enabled=False` ⇒ **không bao giờ có phụ đề** (đường chết) | 🟠 HIGH | Chức năng hỏng |
| **F-19** | ASR chạy trên **Vulkan**, translation/TTS chạy **CUDA** — 2 API GPU trên cùng 1 card | 🟠 HIGH | VRAM không thống nhất, perf thấp hơn |
| **F-20** | `_materialize()` copy toàn bộ segments/words/tokens mỗi lần `run()` | 🟡 MEDIUM | CPU/alloc |
| **F-21** | `n_ctx=512`, `n_batch=256`, `n_threads=4` hardcode trong translation, bỏ qua config | 🟡 MEDIUM | Khó tune |
| **F-22** | Không có metric nào cho ASR/VAD/queue depth → bottleneck vô hình | 🟡 MEDIUM | Observability |
| **F-23** | `asyncio.to_thread` cho TTS dùng default executor chung với prewarm | 🟡 MEDIUM | Tranh chấp thread |
| **F-24** | `CommitManager._last_activity_time` không bao giờ được update → nếu wire vào sẽ force-commit sai | 🟡 MEDIUM | Latent bug |
| **F-25** | Config chết: `max_payload_bytes`, `max_speech_segment_sec`, `dump_report_on_disconnect`, `hot_reload`, `normalize_*` | 🟡 MEDIUM | Cấu hình vô hiệu |
| **F-26** | `createScriptProcessor` + resample JS per-sample + `atob` per-byte | 🟡 MEDIUM | Main-thread jank |
| **F-27** | `findVideo()` quét toàn DOM mỗi sự kiện subtitle | 🟡 MEDIUM | Layout/CPU |
| **F-28** | TTS phát **2 lần** khi video nằm trong iframe | 🟡 MEDIUM | Chất lượng |
| **F-29** | Force 16kHz AudioContext làm giảm chất lượng audio gốc xuống 8kHz | 🟡 MEDIUM | Chất lượng |
| **F-30** | Payload JSON phình 2–3× do lặp field (`text`/`original`/`ui_text`, snake+camel) | 🟢 LOW | Băng thông, GC |
| **F-31** | Threadpool CPU của ggml là **dùng-một-lần mỗi graph compute** (chỉ parakeet có pool bền) ⇒ ~96 OS thread create/join mỗi poll | 🔴 CRITICAL | CPU ↑↑ |
| **F-32** | Scheduler + compute context của ggml bị destroy/dựng lại **mỗi** `run()` offline (thư viện tự đo 1–10 ms/run) | 🟠 HIGH | CPU, latency |
| **F-33** | Công việc host O(T²) + H2D mỗi run (causal mask `T_prompt²`, `build_cu_seqlens_mask`, `build_sinusoid_pe`) | 🟠 HIGH | CPU + PCIe |
| **F-34** | Output encoder bị **D2H về host rồi H2D lại** trong cùng một run | 🟠 HIGH | PCIe, latency |
| **F-35** | Front-end mel spawn `stft_threads-1` `std::thread` **mới mỗi `compute()`** | 🟠 HIGH | CPU |
| **F-36** | Không có pinned host memory; mọi `set/get` CUDA = `cudaMemcpyAsync` + `cudaStreamSynchronize` ngay lập tức | 🟡 MEDIUM | PCIe, latency |
| **F-37** | `run_batch`, `Session.cancel()`, `transcribe_session_get_limits` đều **không được dùng**; CUDA graphs TẮT trong build | 🟡 MEDIUM | Perf + an toàn |
| **F-38** | Harness `transcribe-bench` **về mặt cấu trúc không thể thấy** O(N²): chỉ chạy sample độ dài cố định, không bench streaming | 🟠 HIGH | Đo lường sai |

> **Ghi chú về F-31…F-37:** các finding này thuộc tầng native C++/ggml, được phát hiện trong phụ lục `02_phu_luc_transcribe_cpp.md`. Chúng **khuếch đại** F-01 (xem §14 của phụ lục 02 để biết bảng nối finding).

---

## 1. PHƯƠNG PHÁP & PHẠM VI

### 1.1 Phân loại mã nguồn

| Khu vực | Vai trò | Được audit |
| :--- | :--- | :--- |
| `backend/` | Toàn bộ pipeline runtime (FastAPI + WS + VAD + ASR + Translate + TTS) | ✅ Đọc toàn bộ |
| `extension_firefox/` | Capture audio, transport, render phụ đề, phát TTS | ✅ Đọc toàn bộ |
| `external/transcribe.cpp/` | Native ASR engine (ggml) + Python bindings (ctypes) | ✅ Đọc phần ảnh hưởng runtime |
| `report/`, `wav_test/` | Report benchmark & dữ liệu test | ✅ Đọc để đối chiếu claim |
| `benchmarks/`, `debug/`, `scratch/`, `vulkan-shaders/` | — | ⬜ **Rỗng, không tồn tại** |
| `backend/models/` | ~14 GB weights (GGUF/JIT/pt) | ⬜ Dữ liệu, không phải code |

> **Lưu ý về cấu trúc:** `benchmarks/`, `debug/`, `scratch/` là thư mục **rỗng**; `vulkan-shaders/` ở prompt không tồn tại (shader thực nằm trong `external/transcribe.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`).

### 1.2 Mô hình thực thi (execution model) — bản đồ thread/loop

```
┌─ uvicorn main thread ───────────────────────────────────────────────┐
│  asyncio event loop (1 thread)                                      │
│   ├─ handle_ws()          [1 coroutine / session]  ← receive loop    │
│   ├─ _stream_asr_tokens() [1 coroutine / session]                    │
│   ├─ _translation_worker()[1 coroutine / session]                    │
│   └─ _tts_worker()        [1 coroutine / session]                    │
└─────────────────────────────────────────────────────────────────────┘
        │ run_in_executor                      │ asyncio.to_thread
        ▼                                      ▼
┌───────────────────────┐  ┌──────────────────┐  ┌────────────────────────┐
│ _VAD_EXECUTOR         │  │ _EXECUTOR        │  │ default executor       │
│ max_workers=1 GLOBAL  │  │ max_workers=2    │  │ (min(32, cpu+4))       │
│ (handler.py:36)       │  │ GLOBAL           │  │ TTS synthesize+prewarm │
│ VAD + torch forward   │  │ ASR inference    │  │ (tts/engine.py:294)    │
│ + feed_audio          │  │ (engine.py:39)   │  │                        │
└───────────────────────┘  └──────────────────┘  └────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │ _shared_session (DUY NHẤT)          │
                    │ _shared_model   (DUY NHẤT)          │
                    │ _infer_lock (RLock, class-level)    │
                    │ ← mọi session xếp hàng tại đây      │
                    └─────────────────────────────────────┘
                                       ▲
                    ┌──────────────────┴──────────────────┐
                    │ _TRANS_EXECUTOR max_workers=1       │
                    │ + GGUFTranslator._infer_lock        │
                    │ + OmniVoiceTTS._infer_lock          │
                    └─────────────────────────────────────┘
```

**Quan sát then chốt:** chỉ có **một** GPU ASR session, **một** translation worker, **một** TTS engine cho **toàn bộ** server. Mọi tài nguyên GPU đều là singleton có lock. Đây là mô hình "tối ưu cho đúng 1 session" như README mô tả — nhưng nó là **giới hạn cứng**, không phải lựa chọn có thể mở rộng.

---

## 2. PHẦN A — ĐIỂM NGHẼN THEO TÀI NGUYÊN

### A1. CPU BOTTLENECK

#### F-12 — Logging hot path ở level DEBUG + flush mỗi record (🟠 HIGH)

`backend/utils/logger.py`
```python
174: logger = get_logger("backend", level=logging.DEBUG)     # ← root logger ở DEBUG
...
119: class SafeStreamHandler(logging.StreamHandler):
120:     _lock = threading.RLock()                                # ← LOCK TOÀN CỤC
121:     def emit(self, record):
125:         with self._lock:                                     # ← serialize mọi thread
128:                 stream.write(msg + self.terminator)
140:                 if written:
142:                     try: self.flush()                        # ← FLUSH MỖI RECORD
```

Vấn đề kép:
1. **`_lock` là class attribute** → dùng chung cho *mọi* logger instance trong tiến trình. Đây là điểm serialize giữa: event loop thread, VAD worker, 2 ASR worker, translation worker, TTS `to_thread` worker.
2. **`self.flush()` sau mỗi record** → mỗi dòng log là 1 syscall ghi stdout; trên Windows console đây là chi phí cỡ **0.1–1ms**, có khi hơn nếu console bị scroll.
3. Logger `"backend"` ở level **DEBUG** và được import trực tiếp bởi các module nóng:
   `backend/asr/engine.py:35`, `backend/asr/adapters.py:11`, `backend/vad/processor.py:22`, `backend/vad/engines/*.py`, `backend/translation/engine.py:19`.

Hệ quả: `backend/asr/engine.py:290` (`logger.debug("session.stream fallback to session.run")`) sẽ **bắn ra ở mỗi lần inference** nếu model khai báo `supports_streaming=True` nhưng `session.stream()` ném exception. Đây là trường hợp tệ nhất: exception + log + flush, 3 lần/giây, trong khi đang giữ `_infer_lock`.

Số record log trong hot path (đếm thực tế):

| Nguồn | Tần suất | File:line |
| :--- | :--- | :--- |
| VAD Speech START/END | ~1–2 / câu | `vad/processor.py:190,217,248` |
| ASR preview (mỗi preview) | **~3 / giây** | `asr/engine.py:338` (chỉ khi final), preview **không** log |
| ASR commit | 1 / câu | `asr/engine.py:338` |
| Translation queue + result | 2–3 / câu | `ws/handler.py:297,337` |
| TTS queued + sent + dedup | 2–3 / câu | `ws/handler.py:337,383,411` |

→ Trung bình **~8–15 record/giây** ở steady state, mỗi record là 1 lần acquire lock toàn cục + 1 lần flush. Không thảm hoạ, nhưng là **nguồn jitter latency** không cần thiết và tăng theo số session (tuyến tính × N session).

#### F-11 (phần CPU) — `SpeechNormalizer` tạo ~7 mảng tạm cho mỗi lần normalize

`backend/core/normalizer.py`:
```python
50:  return float(np.sqrt(np.mean(audio * audio)))   # alloc temp1 (audio*audio)
56:  return float(np.max(np.abs(audio)))            # alloc temp2 (np.abs)
78:  orig_rms = self.calculate_rms(audio)
79:  orig_peak = self.calculate_peak(audio)
110: normalized_audio = (audio * clamped_gain).astype(np.float32)  # alloc3 + alloc4
113: np.clip(normalized_audio, -1.0, 1.0, out=normalized_audio)   # in-place ✅
115: norm_rms  = self.calculate_rms(normalized_audio)   # alloc5
116: norm_peak = self.calculate_peak(normalized_audio)  # alloc6
```
Với đoạn speech 30s @16kHz = 480.000 mẫu float32 = **1.92 MB/mảng** → **~11.5 MB allocation + 6 lượt duyệt toàn mảng** cho *một* lần normalize. Và `normalize()` được gọi **mỗi lần inference** (preview + commit) ⇒ đến ~3 lần/giây ⇒ **~35 MB/s allocation churn**.

Có thể hợp nhất thành: `np.dot(a, a)` cho RMS (không tạo temp), `np.max(np.abs())` → `np.maximum.reduce` với `out=`, và `audio * gain` ghi thẳng vào buffer đích. Giá trị `norm_rms`/`norm_peak` **không được dùng ở bất kỳ đâu** (chỉ để log) — có thể bỏ.

#### F-11b — `SpeechNormalizer()` bỏ qua **toàn bộ** cấu hình normalize

`backend/asr/engine.py:103`
```python
103: self.normalizer = SpeechNormalizer()      # ← KHÔNG truyền tham số nào
```
Trong khi `ASRConfig` định nghĩa 11 tham số normalize:
`normalize_target_rms`, `normalize_target_peak`, `normalize_max_gain`, `normalize_min_gain`, `normalize_knee_start`, `normalize_knee_end`, `normalize_gain_smoothing`, `normalize_attack_alpha`, `normalize_release_alpha`, `normalize_speech_frame_ms`, `normalize_min_speech_frames` (`backend/config.py:112-124`).

**Tất cả đều bị bỏ qua** — `SpeechNormalizer.__init__` nhận `target_rms/target_peak/max_gain/min_gain/knee_start/knee_end` và dùng default hardcode. Thêm nữa, 5 tham số `*_smoothing/attack/release/frame/` **không tồn tại** trong class ⇒ 4 config nữa là dead. Hệ quả: người dùng chỉnh config không có tác dụng, và gain smoothing được quảng cáo trong README ("Gain smoothing") không hề được implement.

#### A1-tổng — Ngân sách thread CPU

| Thành phần | Threads | Nguồn |
| :--- | :--- | :--- |
| PyTorch (VAD + TTS) | 2 | `main.py:41` `torch.set_num_threads(2)` |
| llama.cpp (translation) | 4 | `translation/engine.py:109` `n_threads=4` |
| transcribe.cpp session | 4 | `config.asr.threads=4` → `model.session(n_threads=4)` |
| VAD executor | 1 | `handler.py:36` |
| ASR executor | 2 | `asr/engine.py:39` |
| Translation executor | 1 | `translation/engine.py:34` |
| `asyncio.to_thread` default | lên tới `min(32, cpu+4)` | TTS + prewarm |
| OMP/MKL | 2 (`PASSIVE`, `BLOCKTIME=0`) | `main.py:30-33` ✅ tốt |

Tổng **~10 thread compute thường trực**, cộng thêm browser (extension + page JS + audio thread + video decode/render) **trên cùng một máy**. Trên CPU 8 nhân đây là oversubscription thực sự. Việc project đã áp dụng patch `0001-fix-threadpool-oversubscription.patch` (đã **verify là ĐÃ được apply** vào `ggml/src/ggml-cpu/ggml-cpu.c:527-564,627-633`) là bằng chứng cho thấy vấn đề này đã từng gây **deadlock** thật.

---

### A2. GPU / VRAM BOTTLENECK

#### F-19 — ASR chạy Vulkan, Translation & TTS chạy CUDA (🟠 HIGH)

Bằng chứng 1 — chính log khởi động trong README (`README.md:143`):
```text
[ASR] Nạp thành công ASR Model 'qwen3-asr-1.7b' trên GPU (Arch: qwen3_asr, Backend: Vulkan0)
```
Bằng chứng 2 — lane wheel mặc định cho Windows (`external/transcribe.cpp/pyproject.toml`):
```toml
[tool.cibuildwheel.windows]
environment = { TRANSCRIBE_WHEEL_LANE = "cpu-vulkan", CMAKE_GENERATOR = "Ninja" }
```
Bằng chứng 3 — README hướng dẫn người dùng `pip install transcribe-cpp-native` (`README.md:123`), tức là **wheel CPU+Vulkan**, không phải CUDA.
Bằng chứng 4 — tồn tại provider CUDA riêng nhưng **không được dùng**:
`external/transcribe.cpp/bindings/python-native-cu12/pyproject.toml:33` `name = "transcribe-cpp-native-cu12"`.

**Bằng chứng 5 — kiểm chứng TRỰC TIẾP TẠI RUNTIME trên chính máy này** (`python -c "import transcribe_cpp; print(transcribe_cpp.native_provider())"`):
```text
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 5060 Ti (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | fp4: 0
                | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
load_backend: loaded Vulkan backend from ...\transcribe_cpp_native\_native\ggml-vulkan.dll
load_backend: loaded CPU backend   from ...\transcribe_cpp_native\_native\ggml-cpu-haswell.dll
```
⇒ Provider đang cài là `transcribe_cpp_native` (**CPU + Vulkan**), **KHÔNG** phải `transcribe_cpp_native_cu12`, và **không có backend CUDA nào được load**. Phát hiện này nâng F-19 từ suy luận (dựa trên README/pyproject) thành **sự kiện đã xác minh**.

Thêm một chi tiết: CPU backend là tier **`haswell`** (AVX2), **không** phải `icelake`/`alderlake`/`znver4` — nên mọi op rơi về CPU dùng SIMD thấp hơn khả năng của CPU hiện đại. Cả hai cây build checked-in (`external/transcribe.cpp/build`, `build_x64`) đều là **CPU-only** (`GGML_CUDA=OFF`, `GGML_VULKAN=OFF`) ⇒ nếu build từ source và dùng bản đó, ASR chạy **thuần CPU**.

Hệ quả:
1. Trên GPU NVIDIA, **Vulkan backend của ggml chậm hơn CUDA** đáng kể cho `mul_mat` (không dùng được tensor core / coopmat2 như CUDA path) — dù driver đã báo GPU này **hỗ trợ `NV_coopmat2`**.
2. **VRAM không được quản lý thống nhất**: Vulkan cấp phát VRAM qua driver Vulkan, CUDA cấp phát qua caching allocator của PyTorch/llama.cpp. Hai bên **không biết** về nhau ⇒ dễ OOM khi ASR mở rộng KV cache/scratch trong lúc TTS + llama.cpp đang giữ ~7–9 GB.
3. `config.asr.backend = "auto"` (`config.py:104`) không có cách nào ép CUDA ⇒ không thể kiểm soát. Muốn ép phải cài `transcribe-cpp-native-cu12` — **README không hề đề cập**.

README công bố "~9.5 GB / 16 GB" cho 4 model — nhưng đó là số đo tại 1 thời điểm tĩnh, không phải peak khi ASR + translate + TTS chạy đồng thời.

#### F-19b — Tranh chấp GPU không được điều phối (🟠 HIGH)

Ba model cùng chiếm 1 GPU, **không có cơ chế ưu tiên hay scheduling**:

| Stage | Engine | Deadline |
| :--- | :--- | :--- |
| ASR preview | transcribe.cpp (Vulkan) | cần xong trong 350ms |
| ASR commit | transcribe.cpp (Vulkan) | ảnh hưởng trực tiếp E2E latency |
| Translation | llama.cpp (CUDA), `_infer_lock` | 250–800ms/câu |
| TTS | PyTorch (CUDA), `_infer_lock` | ~420ms/câu, `torch.cuda.synchronize()` |

Vì translation và TTS chạy trong **worker coroutine riêng**, chúng có thể chiếm GPU **đúng lúc** ASR commit đang chạy. Kết quả là **priority inversion**: một câu TTS 420ms có thể đẩy ASR commit trễ thêm vài trăm ms, và mỗi lần như vậy là một **latency spike** nhìn thấy được trên phụ đề.

Thêm nữa `tts/engine.py:253-254`:
```python
252: audio_output = self._run_generate(gen_kwargs)
253: if torch.cuda.is_available() and "cuda" in str(self.device):
254:     torch.cuda.synchronize()          # ← drain toàn bộ GPU, chặn cả ASR
```
`torch.cuda.synchronize()` là **đồng bộ toàn device**, không phải chỉ stream của TTS. Nó buộc mọi công việc CUDA khác đang xếp hàng phải đợi ⇒ tăng đuôi latency.

#### F-VRAM-1 — TTS model không bao giờ được giải phóng khi tắt TTS (🟡 MEDIUM)

`OmniVoiceTTS` là singleton (`tts/engine.py:36-45`); `load_model()` được gọi lazy từ `synthesize_sync` (`:222-223`) và chỉ được giải phóng qua `reset_instance()` khi **shutdown server** (`main.py:156`).

Khi người dùng tắt `tts_enabled` trong popup (`main.py:340-343`), config chỉ đổi cờ — **mô hình vẫn nằm trong VRAM**. Vì `OmniVoice` load bằng `device_map=self.device` với `dtype=float16`, đây là vài GB VRAM bị giữ vô ích. Đây là **cơ hội giảm VRAM không giảm chất lượng**: gọi `unload_model()` (đã có sẵn, `:296-314`) khi `tts_enabled` chuyển sang `False`.

#### F-VRAM-2 — `_voice_prompt_cache` giữ object trên GPU

`tts/engine.py:64-65, 114-140`: LRU max 8 — **đã bounded tốt** ✅. Nhưng chú ý: mỗi `VoiceClonePrompt` có thể chứa embedding trên GPU; 8 entry × nhiều voice là mức trần cần đo. Với use case thực tế (1 voice) không đáng lo.

#### A2-tổng — Ước lượng mô hình chi phí GPU của ASR preview

Dùng chính số liệu project công bố (README: RTF `0.0232`; `config.asr.poll_interval_ms = 350`) và mô hình vòng lặp thực tế `t_{k+1} = t_k·(1+r) + P` (vì mỗi vòng = inference + `asyncio.sleep(P)`, xem `asr/engine.py:363,381`):

| Độ dài câu nói | Số preview | Tổng audio đã xử lý | Hệ số lặp | Thời gian GPU cho preview | Độ trễ nội tại của preview cuối |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 s | ~6 | ~9 s | **3.1×** | ~0.21 s | ~70 ms |
| 10 s | ~21 | ~102 s | **10.2×** | ~2.4 s | ~232 ms |
| 30 s | ~47 | ~690 s | **~23×** | ~16 s | **~700 ms** |

> **Đây là mô hình phân tích (analytical model), không phải số đo.** Nó dùng RTF do chính project công bố và công thức vòng lặp đọc trực tiếp từ code. Để xác nhận cần instrument (xem F-22).

**Hai kết luận quan trọng từ mô hình này:**
1. **Chi phí tăng siêu tuyến tính (bậc 2)** theo độ dài câu. Với câu 30s, ASR một mình chiếm >50% GPU chỉ để sản xuất preview sẽ bị vứt đi.
2. **Preview trở nên VÔ DỤNG với câu dài**: ở 30s, độ trễ nội tại của preview là ~700ms — tức bản "live preview" hiển thị **trễ hơn** bản commit sẽ đến ngay sau đó. Người dùng thấy phụ đề giật và lệch.

---

### A3. RAM / MEMORY GROWTH

#### F-15 — `metrics._checkpoints` tăng vô hạn (🟠 HIGH)

`backend/core/metrics.py`:
```python
36: self._checkpoints: Dict[str, float] = {}
...
115: def record_checkpoint(self, name: str) -> None:
117:     with self._lock:
118:         self._checkpoints[name] = time.time()
```
Và caller tạo key **duy nhất theo session** (`backend/ws/handler.py:54,127`):
```python
54:  metrics_collector.record_checkpoint(f"session_start_{session.session_id[:8]}")
127: metrics_collector.record_checkpoint(f"session_end_{session.session_id[:8]}")
```
Không có `maxlen`, không có prune, không có `pop`. Mỗi session để lại **2 entry vĩnh viễn**. Đổi tab/seek liên tục trong một buổi xem dài sẽ tích luỹ; mỗi entry ~100–120 byte (key string + float + dict overhead) ⇒ ~250 B/session. **Không phải thảm hoạ ngay, nhưng là memory leak thật** và mâu thuẫn trực tiếp với tuyên bố "Không rò rỉ bộ nhớ" (`README.md:173`).

Đối chiếu: `_latencies` dùng `deque(maxlen=10000)` ✅ **đã bounded đúng**.

#### F-16 — Hàng đợi TTS phía client không giới hạn (🟠 HIGH)

`extension_firefox/lib/tts-player.js`:
```js
5:   this.queue = [];
77:  this.queue.push(item);                       // ← KHÔNG có max length
...
94:  _playNext() {
103:   const item = this.queue.shift();
107:   const audio = new Audio(audioUrl);           // ← 1 element mới mỗi câu
```

Backend sinh TTS nhanh hơn realtime rất nhiều (RTF ~0.077 ⇒ **~13× realtime**). Nếu video có thoại liên tục và người dùng bật lồng tiếng, tốc độ **sản xuất** TTS vượt tốc độ **phát**. Không có drop policy, không có so sánh với `video.currentTime` ⇒ **queue phình vô hạn** và tiếng lồng **lệch khỏi video ngày càng nhiều** cho tới khi phiên kết thúc.

Hệ quả kép: **memory growth trên tab trình duyệt** (mỗi item giữ một chuỗi base64 WAV trong RAM: 3s @24kHz PCM16 ≈ 144 KB nhị phân → **~192 KB base64**; 100 câu tồn đọng ≈ **19 MB** chỉ riêng base64, chưa kể Blob) + **trải nghiệm hỏng dần**.

#### F-RAM-1 — Hàng đợi backend có bound nhưng **drop âm thầm** (xem F-05)

`backend/ws/session.py:146-147`:
```python
146: self.translation_queue = asyncio.Queue(maxsize=4)
147: self.tts_queue        = asyncio.Queue(maxsize=4)
```
Bound là **đúng** ✅ (back-pressure, chống OOM). Nhưng hành vi khi đầy là **vứt bỏ câu** (`backend/ws/handler.py:249-250`):
```python
249: except asyncio.QueueFull:
250:     logger.warning(f"Session ...: Hàng đợi Translation đầy, bỏ qua")
```
Xem F-05 để biết vì sao điều này đặc biệt nguy hiểm.

#### F-RAM-2 — Ảnh hưởng bộ nhớ của buffer

* `CircularAudioBuffer`: 60s × 16000 × 4 B = **3.84 MB** / session ✅ cố định, tốt.
* VAD `raw_buffer`: cap 3s (`vad/processor.py:150`) ✅ tốt.
* `pre_speech_ring`: `maxlen` tính đúng theo `pre_speech_buffer_ms` (`vad/processor.py:75-76`) ✅ tốt.
* `_pending_commits`: `deque(maxlen=8)` ✅ bounded — nhưng xem F-05.
* Slice tạm mỗi preview: tới **1.92 MB** (30s) × ~6 mảng tạm của normalizer ⇒ xem A1.

---

### A4. I/O BOTTLENECK

#### A4-1 — Model I/O lúc hot-switch (🟡 MEDIUM)

`backend/asr/engine.py:296` + `main.py:296-298`: đổi ASR model ⇒ `unload_shared_model()` (giải phóng VRAM) rồi `TranscribeEngine(target_model).prewarm()` **đọc lại GGUF từ đĩa** (1.3–3.3 GB) và chạy 1 inference dummy. Toàn bộ nằm trong request handler `/api/config` (`await asyncio.to_thread(engine.prewarm)`) ⇒ **request treo vài giây**, và trong khoảng đó **mọi session đang chạy đều không có ASR**.

Không có cơ chế "load model mới trước, swap sau" (double-buffer) và không có thông báo cho client.

#### A4-2 — 4.8 GB GGUF translation nạp trong request (🟡 MEDIUM)

`main.py:333-338` → `translator.load_model()` → `Llama(model_path=...)` đọc **4.78 GB** từ đĩa, cấp phát VRAM, **trong request HTTP**. Cùng vấn đề như A4-1.

#### A4-3 — TTS base64 hoá WAV mỗi câu (🟠 HIGH)

`backend/tts/audio_processor.py:131-140`:
```python
137: wav_buffer = io.BytesIO()
138: sf.write(wav_buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
139: wav_bytes = wav_buffer.getvalue()
140: return base64.b64encode(wav_bytes).decode("ascii")
```
Với 3s audio @24kHz: 144.000 mẫu × 2 B = **144 KB** → BytesIO → `getvalue()` copy → `b64encode` (alloc 192 KB) → `.decode("ascii")` (alloc **~192 KB str**) → nhét vào dict JSON → `_fast_dumps` encode UTF-8 lần nữa (~192 KB). Tổng **~5 bản copy + +33% kích thước** cho mỗi câu lồng tiếng.

Phía client đảo ngược đúng chuỗi đó bằng **vòng lặp per-byte JS** (`extension_firefox/lib/tts-player.js:83-92`):
```js
84:  const binary = atob(base64Str);
86:  const bytes = new Uint8Array(len);
87:  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);   // ← main thread!
```
144.000 vòng lặp + 192 KB string + 144 KB `Uint8Array` + Blob URL, **trên main thread** của trang web.

**Khuyến nghị:** gửi TTS dưới dạng **WebSocket binary frame** (opcode 2) với header nhỏ, client dùng `AudioContext.decodeAudioData(arrayBuffer)` (chạy off-main-thread trong browser). Giảm ~33% băng thông, bỏ 2 bản copy lớn, bỏ vòng lặp JS 144k iteration. **Không giảm chất lượng** (audio là PCM16 như cũ).

#### A4-4 — Serialization dư thừa trong preview (🟢 LOW — F-30)

`backend/ws/serializers.py:28-43` — mỗi message `utterance_update` chứa **cùng một giá trị** dưới 2–3 tên khác nhau:
```python
31: "utterance_id": utt_id,
32: "utteranceId":  utt_id,
33: "original":     text,
34: "ui_text":      text,
35: "text":         text,
36: "stable_text":  stable_text,
37: "stableText":   stable_text,
38: "unstable_text": unstable_text,
39: "unstableText":  unstable_text,
40: "translated":   translated,
41: "is_final":     is_final,
42: "isFinal":      is_final,
```
Với preview ~3 lần/giây, payload phình **~2.3×** so với cần thiết. Trên localhost băng thông không phải vấn đề, nhưng mỗi lần serialize/encode/parse là CPU + GC trên **cả hai** đầu, và nó làm chậm `send_json` trong lock của `SafeWebSocketConnection` (`ws/connection.py:61-64`) — mà `_stream_asr_tokens` **await** lệnh gửi này (`handler.py:234`), nên nó nằm trên critical path của ASR.

---

## 3. PHẦN B — VẤN ĐỀ CẤU TRÚC RUNTIME

### B1. 🔴 F-01 — ASR PREVIEW LÀ O(N²) — ĐIỂM NGHẼN LỚN NHẤT

**Code hiện tại** (`backend/asr/engine.py`):
```python
353: # 2. Neu dang trong trang thai noi -> Tham do preview text
354: if self._speech_active:
355:     current_total = self.audio_buffer.total_written
356:     start_s = self._speech_start_sample              # ← ĐIỂM BẮT ĐẦU CÂU (không đổi)
357:
358:     if current_total - start_s >= int(16000 * self.min_transcribe_sec):
359:         audio_slice = self.audio_buffer.get_slice(start_s, current_total)   # ← TOÀN BỘ câu
360:
361:         t0 = time.perf_counter()
362:         loop = asyncio.get_running_loop()
363:         preview_text = await loop.run_in_executor(_EXECUTOR, self._run_inference_sync, audio_slice)
...
381:     await asyncio.sleep(poll_interval)              # 350ms
```

**Phân tích:** `start_s` cố định ở đầu câu; `current_total` tăng liên tục. Mỗi vòng poll, engine **chạy lại inference trên toàn bộ đoạn từ đầu câu**. Không có incremental state, không có cache, không có giới hạn cửa sổ.

**Bốn hệ quả riêng biệt:**

**(a) Chi phí bậc hai.** Xem bảng ở §A2. Tổng công việc = Σ L_k ≈ O(T²/(2P)).

**(b) Sao chép bộ nhớ bậc hai.** `get_slice` (`core/audio_buffer.py:130`) `np.empty(requested_len)` rồi copy toàn bộ đoạn ⇒ mỗi preview copy lại toàn bộ câu. Ở 30s: **1.92 MB copy/preview × 47 preview ≈ 90 MB copy** cho một câu.

**(c) Preview và commit trùng lặp 100% (F-09).** Commit cũng chạy `_run_inference_sync` trên `get_slice(start_s, end_s)` (`engine.py:330-334`) — chính là đoạn audio mà preview cuối cùng vừa xử lý xong, chỉ dài thêm 300–600ms (hangover + silence). Với câu ngắn (2–4s, chiếm đa số trong thực tế), **một nửa tổng compute ASR bị lãng phí**. Đây là **tối ưu giảm latency mà không giảm chất lượng** rõ ràng nhất: tái sử dụng kết quả preview cuối nếu cửa sổ audio không đổi đáng kể.

**(d) Preview cũ dần (staleness).** `audio_slice` được chụp ở `:359` **trước** khi inference chạy. Trong lúc inference (r·L giây), audio mới vẫn tiếp tục vào buffer. Vậy preview hiển thị ở thời điểm `t + r·L` cho nội dung tại `t` ⇒ **trễ nội tại = r·L** (700ms ở L=30s). Với câu dài, "live preview" thực chất là "preview lịch sử".

**(e) XÁC NHẬN Ở TẦNG C++ — vấn đề còn nặng hơn mô hình Python dự đoán.** ✅ **[VERIFIED]**

Audit tầng native (phụ lục `02_phu_luc_transcribe_cpp.md`) xác nhận: **toàn bộ chuỗi front-end → encoder → prefill → decode được tính lại từ `pcm[0]` mỗi lần `run()`**:
* `MelFrontend::compute` luôn bắt đầu từ `pcm[0]`, không có frame cache, không có cursor "đã phát" — `external/transcribe.cpp/src/transcribe-mel.cpp:390-398, 341, 361`.
* qwen3_asr **không có streaming hook** (`src/arch/qwen3_asr/model.cpp:1716-1719`) ⇒ engine Python luôn đi đường offline.
* KV cache **bị wipe** và prefill làm lại mỗi run — `src/arch/qwen3_asr/model.cpp:768-775`:
  ```cpp
  } else {
      // Clear stale positions for a fresh prefill.
      if (cc->kv_cache.buffer != nullptr) {
          ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
      }
      cc->kv_cache.n    = 0;
      cc->kv_cache.head = 0;
  }
  ```
* **Không tồn tại API prefix/state reuse** ở bất kỳ đâu trong `src/causal_lm/` (grep `n_past|prefix|retain|state` chỉ thấy `n_past` là cursor **trong-run**).

⇒ Chi phí bậc hai không chỉ nằm ở model ASR mà ở **cả mel front-end**, cộng thêm chuỗi chi phí **cố định mỗi run** (F-31…F-35) bị nhân lên theo số poll. Đây là lý do F-01 được xếp CRITICAL và là ưu tiên số 1.

Cũng chính vì vậy, **phép đo hiện tại của project không thể phát hiện vấn đề này** (F-38): `backend/tests/test_03_asr_benchmark.py:88` gọi `engine._run_inference_sync(audio)` **một lần** trên một file tĩnh, và harness của thư viện (`tools/transcribe-bench/main.cpp:317-323, 408-414`) cũng chỉ replay **cùng một sample độ dài cố định**. Số "RTF 0.0232, nhanh gấp 43× realtime" hoàn toàn đúng — nhưng nó **không đo đường chạy thật**.

> 🎯 **Cách đo lại không cần viết code mới:** thư viện đã có sẵn instrumentation chi tiết dưới biến môi trường `TRANSCRIBE_PERF_DEBUG=1` (`src/arch/qwen3_asr/model.cpp:1034, 1040-1058`), in ra `mel` / `enc_build (graph + sched + uploads)` / `enc_compute` / `enc_d2h` / `prefill_build` / `prefill_compute` / `prefill_logits` / `step_loop`. Chạy kịch bản **nói liên tục 30 giây** (kịch bản `wav_test/` hiện **không có**) với biến này bật sẽ cho số đo thật cho F-01, F-31…F-35.

**Khuyến nghị (theo thứ tự ROI):**
1. **Giới hạn cửa sổ preview**: chỉ transcribe `[max(start_s, current_total - W), current_total)` với `W ≈ 4–6 s`, rồi ghép text với phần đã chốt. Giảm chi phí từ O(N²) về **O(N·W/P)** — tuyến tính. Không giảm chất lượng hiển thị nếu ghép đúng (đây chính là cách các hệ streaming caption thực tế làm).
2. **Tái sử dụng preview cuối cho commit** khi cửa sổ audio gần như không đổi (< ~300ms mới) → tiết kiệm ~1 inference/câu.
3. **Wire lại Commit Manager** (F-02) để câu không phình tới 30s.
4. **Dùng đúng streaming API của thư viện**: hiện `session.stream()` được **begin/finalize lại từ đầu mỗi lần gọi** (`asr/engine.py:284-287`), nên mọi trạng thái streaming bị vứt đi:
   ```python
   284: with session.stream(language=lang, family=stream_opts) as stream:
   285:     stream.feed(audio_to_infer)      # ← feed LẠI TOÀN BỘ từ đầu
   286:     stream.finalize()
   287:     raw_text = stream.text().full
   ```
   Thư viện có `transcribe_stream_feed` incremental + `committed_text`/`tentative_text` (xem `transcribe.h` doc & `bindings/.../__init__.py:1411-1438`). Nếu giữ **một** stream sống suốt câu và `feed()` chỉ phần audio mới, chi phí chuyển từ O(N²) sang **O(N)** và độ trễ preview xuống mức chunk-level. Đây là thay đổi lớn nhất về mặt kiến trúc nhưng cũng là mức tăng hiệu năng lớn nhất.

---

### B2. 🔴 F-02 — COMMIT MANAGER 4 BẬC LÀ DEAD CODE

`CommitManager` được khởi tạo (`asr/engine.py:104`) nhưng **`decide_commit_trigger()` không được gọi ở bất kỳ đâu trong runtime**. Kiểm chứng bằng grep toàn repo:

| Symbol | Nơi gọi trong runtime | Nơi gọi trong test |
| :--- | :--- | :--- |
| `decide_commit_trigger` | ❌ không có | `backend/tests/test_04_commit_logic.py:78,87,100,111` |
| `evaluate_preview_stability` | ❌ không có | `test_04_commit_logic.py:97,98` |
| `evaluate_inactivity_timeout` | ❌ chỉ từ `decide_commit_trigger` (cũng không được gọi) | gián tiếp |
| `check_max_duration` | ❌ chỉ từ `decide_commit_trigger` | gián tiếp |
| `record_activity` | ❌ chỉ từ `record_commit` (cũng không được gọi) | — |
| `CommitManager.record_commit` | ❌ không có | — |
| `CommitManager.is_duplicate` / `deduplicator` | ❌ không có | — |
| `CommitReason` (enum) | ❌ chỉ được định nghĩa, không bao giờ được tạo | `test_04:84,93,106,117` |

Trong runtime, đường commit **duy nhất** là VAD silence:
```
vad/processor.py: on_speech_end  →  asr/engine.py:237 _pending_commits.append({reason: "VAD_SILENCE"})
                                →  asr/engine.py:342 yield {"commit_reason": reason}
```

**Hệ quả nghiêm trọng:**
1. **Không có safeguard độ dài câu.** `SentenceConfig.max_duration_sec = 8.0` và `max_chars = 150` (`config.py:130-131`) **hoàn toàn không có tác dụng**. Câu nói liên tục bị chốt **chỉ khi** VAD phát hiện im lặng ≥ `silence_duration_ms=600ms` (`config.py:80`). FireRed postprocessor cho phép tới `max_speech_frame=2000` = **50 giây** (`vad/engines/firered.py:69`).
   → Người nói liên tục 30–50 giây (giảng bài, podcast, video review) sẽ tạo ra **một** utterance khổng lồ ⇒ kích hoạt chính xác thảm hoạ O(N²) ở §B1, và **không có phụ đề nào xuất hiện trong suốt 30–50 giây đó**.
2. **Không có stability split / inactivity timeout.** Preview ổn định (text không đổi qua 3 poll) lẽ ra phải chốt câu sớm để giảm latency — cơ chế đã viết, đã test, nhưng chưa nối.
3. **`CommitDeduplicator` (3 lớp dedup) ở tầng commit không hoạt động.** `core/dedup.py:94 record_commit` không được gọi ⇒ `_history` luôn rỗng ⇒ `is_duplicate()` luôn trả `False`. Dedup thực tế chỉ còn 1 lớp ở tầng translation (`translation/dedup.py`, so khớp exact + TTL 10s) và 1 lớp ở TTS.
   → README/report tuyên bố "3-Layer Dedup Engine — Loại bỏ 100% câu lặp rác" là **không đúng ở tầng commit**.
4. **F-24 latent bug:** `_last_activity_time` chỉ được set trong `__init__` (`commit_manager.py:74`) và `record_activity()`(không bao giờ gọi). Nếu ai đó wire `decide_commit_trigger` vào, `evaluate_inactivity_timeout` sẽ trả `True` ngay khi session sống > 1.2s ⇒ **force-commit mỗi vòng poll**. Cần sửa trước khi wire.

**File `report/04_commit_logic/report.md` (dòng 19-21) tuyên bố cả 4 bậc "✅ Kích hoạt".** Điều này đúng với *unit test của class* nhưng **sai với runtime E2E** — xem §D.

---

### B3. COPY & ALLOCATION KHÔNG CẦN THIẾT

#### B3-1 — Đường ống audio: ≥ 4 lần biến đổi/copy cho mỗi mẫu

```
[Browser] Int16Array (worklet) hoặc ScriptProcessor
   │  transferable / structured clone                      (copy #1)
[Content script] rawBuffer
   │  buildBinaryAudioPacket: new ArrayBuffer + .set()      (copy #2)
[WebSocket] ─────────────────────────────────────────────►
[backend] data: bytes
   │  parse_audio_frame: data[4+header_len:]  → bytes slice  (copy #3)
[VAD executor] raw_buf.extend(audio_data) → bytearray       (copy #4)
   │  per frame: bytes(raw_buf[offset:frame_end])            (copy #5, 40×/s)
[VAD engine]  np.frombuffer → .astype(np.float32)/32768     (alloc, 40×/s — chỉ silero/fsmn)
   │  on_speech_chunk(frame_bytes, ...)
[ASR engine]  np.frombuffer(int16) → .astype(float32)/32768 (alloc #6, 40×/s)
   │  audio_buffer.write() → copy vào ring                  (copy #7)
[ASR buffer]
   │  get_slice() → np.empty + copy                          (copy #8, mỗi preview)
[normalizer]  ~6 mảng tạm                                    (copy #9..#14, mỗi preview)
[ctypes]      (c_float * n).from_buffer_copy(floats)         (copy #15, mỗi inference)
```

Đáng chú ý:
* **FireRed-VAD đã được tối ưu đúng** — `vad/processor.py:167-171` truyền `chunk_raw` Int16 trực tiếp để tránh roundtrip F32→I16, và `vad/engines/firered.py:92-98` nhận thẳng Int16. ✅
* **Nhưng `asr_engine.feed_audio` ngay sau đó lại chuyển Int16 → Float32** (`asr/engine.py:205-210`) với một `astype` + một phép chia toàn mảng. Vậy tối ưu ở VAD bị "trả lại" ở ASR. Có thể giữ nguyên Int16 tới tận `CircularAudioBuffer` (buffer có thể là `int16`) và chỉ chuyển sang float32 **một lần**, ngay trước khi đưa vào model — giảm cả RAM (½) và CPU.
* `vad/processor.py:163` `bytes(raw_buf[offset:frame_end])` tạo **một object bytes mới cho mỗi frame 25ms** = 40 alloc/s. Có thể dùng `memoryview` để tránh copy.

#### B3-2 — `_materialize()` copy toàn bộ segments/words/tokens không dùng (F-10, 🟠 HIGH)

`backend/asr/engine.py:296`:
```python
296: res = session.run(audio_to_infer, language=lang, family=run_opts)
297: raw_text = getattr(res, "text", str(res))
```
Backend **chỉ dùng `res.text`**. Nhưng binding Python mặc định `timestamps="auto"` (`bindings/python/src/transcribe_cpp/__init__.py:1098`), và `_materialize()` (`:1266-1342`) copy **tất cả**:
```python
1300: for j in range(n_seg()):   ... segments.append(_segment_from(s))
1307: for j in range(n_word()):  ... words.append(_word_from(w))
1314: for j in range(n_tok()):   ... tokens.append(_token_from(tok))
1321: for j in range(n_speaker()): ...
```
Với Audio-LLM, số token sinh ra tỉ lệ với độ dài audio. Mỗi `_token_from()` tạo một `@dataclass(frozen=True)` object. Ở 30s audio có thể là hàng nghìn token × 47 preview ⇒ **hàng trăm nghìn object Python vứt đi**.

**Khuyến nghị:** truyền `timestamps="none"` (binding hỗ trợ) và/hoặc dùng `transcribe_full_text` trực tiếp. Đây là **tối ưu thuần tuý, không ảnh hưởng chất lượng**, có thể đo ngay.

Cùng vấn đề ở nhánh streaming: `_materialize` không được gọi (dùng `stream.text()` ✅ tốt).

#### B3-3 — `_pcm_to_carray` copy lần nữa (🟡 MEDIUM)

`bindings/python/src/transcribe_cpp/__init__.py:466`:
```python
466: return (ctypes.c_float * n).from_buffer_copy(floats), n
```
Mỗi `run()`/`feed()` copy **toàn bộ** mảng PCM vào một ctypes array mới. Ở 30s = 1.92 MB copy/lần. Với `feed()` incremental thì chi phí này biến mất.

---

### B4. LOCK / MUTEX CONTENTION & RACE CONDITION

#### B4-1 — 🔴 F-04 — Race giữa `unload_shared_model()` và inference (memory safety)

`backend/asr/engine.py`:
```python
50:  _shared_lock = threading.RLock()
51:  _infer_lock  = threading.RLock()
...
61:  @classmethod
62:  def unload_shared_model(cls) -> None:
64:      with cls._shared_lock:                 # ← CHỈ giữ _shared_lock
65:          if cls._shared_session is not None:
67:              cls._shared_session.close()    # ← ĐÓNG NATIVE SESSION
69:              cls._shared_session = None
72:          if cls._shared_model is not None:
74:              cls._shared_model.close()      # ← FREE NATIVE MODEL
...
271:  with self.__class__._infer_lock:           # ← _run_inference_sync giữ _infer_lock
272:      session = self.__class__._shared_session    # ← đọc KHÔNG có _shared_lock
...
284:      with session.stream(...) as stream:    # ← đang dùng native session
296:      res = session.run(...)
```

`unload_shared_model()` **không** acquire `_infer_lock`. Caller của nó là `main.py:296` (REST `/api/config` khi đổi model) và `main.py:144` (shutdown). Nếu một preview/commit đang chạy trong `_EXECUTOR` tại thời điểm đó:
* `_shared_session.close()` / `_shared_model.close()` giải phóng native handle
* trong khi thread kia vẫn đang `transcribe_run`/`transcribe_stream_feed` trên handle đó

→ **Use-after-free trên native memory ⇒ crash tiến trình (access violation), không phải exception Python.** Đây là rủi ro nghiêm trọng nhất về độ ổn định trong toàn bộ codebase. Tương tự với `tts/engine.py:296-314` (`unload_model` giữ `_init_lock` + `_infer_lock` ✅ **đúng**) và `translation/engine.py:130-145` (`unload_model` giữ `_shared_lock` + `_infer_lock` ✅ **đúng**) — chỉ ASR làm sai.

Ngoài ra `main.py:296-297` gọi `unload_shared_model()` rồi tạo engine mới **mà không dừng các session đang chạy** và **không kiểm tra** xem có session nào đang active.

#### B4-2 — 🟠 F-13 — VAD inference chạy bên trong RLock

`backend/vad/processor.py`:
```python
135: with self._lock:                       # ← RLock của processor
...
160:     while buf_len - offset >= frame_size:
...
173:         res = self._engine.is_speech(   # ← TORCH FORWARD BÊN TRONG LOCK
174:             samples_float32, state, self.threshold, chunk_raw=frame_bytes,
178:         )
...
262: if offset > 0:
263:     del raw_buf[:offset]
264: (hết `with`)                          # ← chỉ nhả lock ở đây
265: # Kích hoạt callbacks bên ngoài lock  ✅ (đúng, tránh deadlock)
266: for fn, args in callbacks_to_fire: fn(*args)
```
Một chunk 64ms = 2–3 frame ⇒ **2–3 lần PyTorch forward giữ RLock liên tục**. Bất kỳ ai gọi `update_config()` (`:87`) hay `force_end()` (`:274`) trong lúc đó sẽ **block**.

`force_end()` được gọi từ `SessionState.cleanup()` (`ws/session.py:278`) chạy trên **event loop thread** trong `handle_ws`'s `finally` (`handler.py:124`). Vậy **cleanup session bị chặn bởi VAD inference**.

**Bằng chứng thực nghiệm trong chính report của project** — `report/07_final_e2e_comparison/report.md:18`:

| File | Fast Cleanup |
| :--- | ---: |
| `00_ingress_stream.wav` | 0.56 ms |
| `Chinese_fast_speed_11s.wav` | 0.68 ms |
| `Chinese_noise_28s.wav` | 0.53 ms |
| `English_low_speech_quality_19s.wav` | **102.68 ms** ← |
| `English_multiple_kinds_of_noise_88s.wav` | 0.41 ms |
| `Japanese_5s.wav` | 0.63 ms |

Một mẫu **lệch 150–200× so với các mẫu còn lại** là dấu hiệu kinh điển của **lock contention**, không phải của chi phí cố định. Đây là bằng chứng độc lập (do chính project tạo ra) cho F-13.

**Khuyến nghị:** chuyển `self._engine.is_speech(...)` ra ngoài `with self._lock` — chỉ giữ lock cho phần đọc/ghi state (`state.total_samples_processed`, `state.is_speech`, `raw_buf`), nhả lock trong lúc chạy model. Với mô hình single-VAD-thread hiện tại điều này hoàn toàn an toàn.

#### B4-3 — `asyncio.Lock` của `SafeWebSocketConnection` serialize mọi writer (🟡 MEDIUM)

`ws/connection.py:55-72`. Có 3 coroutine/session cùng ghi: ASR preview, translation, TTS. Lock là **cần thiết** để tránh interleave frame ✅. Nhưng nó biến WebSocket thành **điểm serialize**: một message TTS ~192 KB (base64) đang được `send_text` sẽ **chặn** preview ASR tiếp theo. Kết hợp với F-08 (base64) ⇒ TTS gây **head-of-line blocking** cho phụ đề. Giải pháp: tách TTS audio sang binary frame (vẫn phải qua lock, nhưng nhỏ hơn 33% và không cần serialize JSON).

#### B4-4 — RLock không cần thiết cho read

`audio_buffer.py:113` — `get_slice()` giữ lock trong khi copy có thể tới 1.92 MB (`:130-141`). Điều này chặn `write()` từ VAD thread ⇒ **audio ingress bị chặn bởi ASR đang đọc**. Với single-writer/multi-reader có thể dùng double-buffer hoặc copy ngoài lock (chụp con trỏ trong lock, copy ngoài lock).

---

### B5. QUEUE / PRODUCER-CONSUMER IMBALANCE

Luồng sản xuất – tiêu thụ hiện tại:

```
Producer: extension ~15.6 audio chunks/s (1024 samples @16kHz = 64ms)
   │  KHÔNG backpressure (F-07)
   ▼
[VAD executor, 1 worker GLOBAL, await trong receive loop]  ← nút thắt #1
   │  on_speech_chunk → ASR buffer
   ▼
[ASR preview loop: 1 coroutine, poll 350ms + inference time]  ← nút thắt #2 (O(N²))
   │  _pending_commits deque(maxlen=8)  → DROP ÂM THẦM (F-05)
   ▼
[translation_queue maxsize=4] → [1 worker] → 250–800ms/câu  ← nút thắt #3
   │  QueueFull → DROP ÂM THẦM
   ▼
[tts_queue maxsize=4] → [1 worker] → ~420ms/câu  ← nút thắt #4
   │  QueueFull → DROP ÂM THẦM
   ▼
[Client tts-player queue: KHÔNG GIỚI HẠN] → phát tuần tự  ← nút thắt #5
```

#### F-05 — 🔴 `_pending_commits` drop câu âm thầm (mất phụ đề)

`backend/asr/engine.py:111`
```python
111: self._pending_commits: deque = deque(maxlen=8)  # Bounded: max 8 pending sentences, oldest dropped if ASR blocked
```
Comment thừa nhận rõ hành vi: **"oldest dropped if ASR blocked"**. Xảy ra chính xác khi nào? Khi `_stream_asr_tokens` bận inference preview/commit. Với O(N²) preview và câu dài (inference ~700ms), VAD có thể chốt câu nhanh hơn tốc độ ASR xử lý ⇒ deque tràn ⇒ **câu bị vứt, phụ đề không bao giờ xuất hiện, và KHÔNG có log/metric nào cho biết**.

Vấn đề tương tự ở hai queue khác nhưng **có** log warning (`handler.py:250,339`) — vẫn là mất dữ liệu người dùng nhưng ít nhất có dấu vết. Riêng `_pending_commits` **im lặng hoàn toàn**.

Thêm một hệ quả UI: khi translation bị drop, `_stream_asr_tokens` đã gửi `make_utterance_update_msg(..., translated="..." if is_final else "")` (`handler.py:227`). Client hiển thị **"..."** và không bao giờ nhận bản dịch ⇒ **phụ đề treo ở trạng thái "..."** cho tới khi có câu mới. Cần cơ chế timeout/cleanup ở phía client hoặc gửi message huỷ.

#### F-Queue-2 — `_VAD_EXECUTOR` global max_workers=1 (🔴 hệ quả cho scale)

`backend/ws/handler.py:36` — `ThreadPoolExecutor(max_workers=1)` ở **module level**, dùng chung cho **mọi session**. Với N session:
* N × 40 torch forward/s **xếp hàng tuần tự** trên 1 thread
* `_handle_binary_message` **await** future đó (`handler.py:186`) trong receive loop ⇒ mỗi session bị chặn chờ đến lượt

⇒ Session thứ 2 trở đi bị **chèn ép hoàn toàn** về VAD. Đây là điểm chặn scale cứng.

Thêm nữa: vì receive loop **await** VAD, nếu VAD chậm (ví dụ lần đầu load model FireRed/FSMN, hoặc `_ensure_engine()` đầu tiên), các frame audio tiếp theo **dồn vào buffer nội bộ của uvicorn/websockets** mà không có giới hạn ⇒ RAM tăng + audio bị xử lý trễ (lệch tiếng).

#### F-Queue-3 — Polling `asyncio.sleep(0.05)` khi lỗi

`handler.py:359,436` — khi worker gặp exception, sleep 50ms rồi lặp. Nếu lỗi có tính hệ thống (ví dụ model chưa nạp), vòng lặp sẽ **quay nóng 20 lần/giây vô hạn** kèm log ERROR (mỗi log = 1 flush dưới lock toàn cục). Nên có backoff + circuit breaker.

---

### B6. POLLING & EVENT

#### F-Poll-1 — Polling cố định 350ms thay vì event-driven (🟠 HIGH)

`backend/asr/engine.py:379-391`:
```python
379: if self._speech_active:
380:     # Dang speech: poll theo interval cho preview
381:     await asyncio.sleep(poll_interval)
382: else:
383:     # P2-1: Idle - khong waste CPU voi sleep, doi commit event
386:     await asyncio.wait_for(self._commit_event.wait(), timeout=poll_interval)
```
Nhánh idle dùng event ✅ **tốt** (cải tiến P2-1 có thật trong code). Nhánh speech vẫn poll.

Vấn đề không phải polling *bản thân* nó (preview cần nhịp đều), mà là **nhịp không đều**: nhịp thực = `inference_time + 350ms` chứ không phải 350ms. Ở câu 30s, nhịp thực ≈ **1.05s/preview**. Người dùng thấy phụ đề nhảy từng giây một. Nên đặt lịch **fixed-rate** (`next_deadline = t0 + k·P`, bỏ qua preview nếu inference vượt hạn) để UI cập nhật đều và để tự nhiên "rơi" preview khi không kịp.

#### F-Poll-2 — `wait_for(timeout=poll_interval)` như lưới an toàn

`engine.py:386-391` — comment giải thích đúng: timeout để không bị stuck nếu event bị miss. ✅ Hợp lý.

---

### B7. CACHE

| Cache | Đánh giá | Ghi chú |
| :--- | :--- | :--- |
| `_voice_prompt_cache` (LRU 8) `tts/engine.py:64-65,124-137` | ✅ **Tốt** | Có `move_to_end`, có eviction, tiết kiệm ~70ms/câu |
| `VoiceManager._cached_voices` `tts/voice_manager.py:23` | ✅ Tốt | Cache toàn cục, có `force_reload` |
| `TranslationContextTracker` `deque(maxlen=3)` | ✅ Tốt | Bounded |
| `TranslationDedupState` `deque(maxlen=20)` | ✅ Tốt | Bounded + TTL |
| `TTSDedupState` (max 15) | ✅ Tốt | — |
| `ModelRegistry._models` | ✅ Tốt | Nạp 1 lần |
| **KV cache của transcribe.cpp** | ❌ **Không dùng** | `session.stream()` bị begin/reset mỗi lần gọi ⇒ mọi state streaming/KV bị vứt. Đây là cache lớn nhất bị bỏ phí |
| **KV cache của llama.cpp (prompt cache)** | ❌ **Không dùng** | Mỗi `llm(prompt)` (`translation/engine.py:190`) prefill lại toàn bộ prompt từ đầu. Với prompt ChatML cố định ~30–60 token, chi phí nhỏ nhưng có thể tái dùng bằng prefix caching của llama.cpp |
| **Kết quả preview** | ❌ **Không cache** | F-09: commit luôn chạy lại inference mà preview vừa làm |
| **Metrics cache** | ❌ Không có | `/api/metrics` tính lại `np.percentile` cho mọi stage mỗi request |

---

### B8. OBJECT LIFETIME / GARBAGE COLLECTION

* **`normalize()` tạo object `NormalizationResult`** (dataclass 7 field) mỗi lần inference (`normalizer.py:16-25,118-126`). 4 trong 7 field (`original_rms/normalized_rms/original_peak/normalized_peak`) **không được đọc ở đâu** ⇒ vừa tính vừa cấp phát vô ích.
* **`callbacks_to_fire` list** (`vad/processor.py:133`): mỗi frame append một tuple `(callable, tuple)` ⇒ 40 tuple + 40 list append mỗi giây. Có thể tránh bằng cách gọi callback trực tiếp sau khi nhả lock theo từng nhóm.
* **`bytes(raw_buf[offset:frame_end])`** (`vad/processor.py:163`): 40 object bytes mới/giây.
* **`np.zeros(0, dtype=np.float32)`** trả về nhiều chỗ trong `get_slice` (`audio_buffer.py:117,127`) — tạo mảng mới liên tục; nên có sentinel dùng chung.
* **GC của PyTorch**: `tts/engine.py:198-200` gọi `gc.collect()` + `torch.cuda.empty_cache()` **chỉ khi load model** ✅ hợp lý. Nhưng `translation/engine.py:100-101` gọi `import gc; gc.collect()` **trong hàm load_model** (chỉ 1 lần) ✅ OK.
* **Không có `gc.freeze()` / tuning cho GC gen**: với allocation churn ~35 MB/s ở ASR path (§A1), có thể cân nhắc `gc.freeze()` sau khi nạp model để giảm thời gian quét của gen-2 GC.
* **`asyncio.Queue(maxsize=10)` `_preview_queue`** (`asr/engine.py:113`) — **được khởi tạo nhưng KHÔNG BAO GIỜ dùng**. Dead field.
* **`asr/engine.py:104` `self.commit_manager = CommitManager(...)`** — tạo object + `CommitDeduplicator` (deque) không bao giờ dùng. Dead field (xem F-02).

---

### B9. CHI PHÍ CỐ ĐỊNH MỖI LẦN `run()` Ở TẦNG NATIVE (F-31 … F-37)

Phần này tóm tắt các phát hiện ở tầng C++/ggml — chi tiết đầy đủ và bằng chứng `file:line` nằm ở **`report/audit/02_phu_luc_transcribe_cpp.md`**. Điểm mấu chốt: **mỗi poll 350 ms là một `run()` hoàn toàn mới**, nên mọi chi phí *cố định mỗi run* bị nhân lên theo số poll (tới ~47 lần cho một câu 30 giây).

#### F-31 — 🔴 CRITICAL: threadpool CPU của ggml là dùng-một-lần mỗi graph compute

`external/transcribe.cpp/src/transcribe-batch-util.cpp:69-88` chỉ đặt **số** thread (`ggml_backend_set_n_threads`), **không** gắn `ggml_threadpool`. Hệ quả: `ctx->threadpool` vẫn `NULL`, và `ggml_graph_compute` **dựng rồi phá huỷ cả một pool** mỗi lần gọi — `ggml/src/ggml-cpu/ggml-cpu.c:3400-3416, 3458-3464`:
```c
bool disposable_threadpool = false;
if (threadpool == NULL) {
    disposable_threadpool = true;
    struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
    threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
}
...
if (disposable_threadpool) { ggml_threadpool_free(threadpool); }
```
`ggml_threadpool_new_impl` tạo `n_threads - 1` OS thread (`ggml-cpu.c:3357-3370`). Vì OpenMP **OFF** trong build (`GGML_OPENMP:BOOL=OFF`), đường thread này là đường chạy thật.

Thư viện **đã biết** cách sửa và áp dụng ở **đúng một chỗ** — `src/arch/parakeet/decoder.cpp:505-521` có ghi chú *"Persistent threadpool so the per-step graph_compute reuses workers instead of spawning a transient pool each call"*. Mọi family khác (qwen3_asr, whisper, cohere, canary, sensevoice, voxtral, moonshine, funasr_nano) đều dùng-một-lần.

Với qwen3-asr-1.7b, một `run()` = encoder graph + prefill graph + **một step graph mỗi token** ⇒ ~**32 graph compute** ⇒ ~**96 OS thread create/join mỗi poll**, tức **~5–15 ms CPU thuần thread churn mỗi 350 ms** `[SUSPECTED về ms tuyệt đối]`.

#### F-32 — 🟠 HIGH: scheduler + compute context bị destroy/dựng lại mỗi `run()`

`src/transcribe.cpp:2251-2253` arm guard `release_scratch()` cho **mọi** `transcribe_run`; guard gọi `release_compute_scratch()` free cả `sched` lẫn `compute_ctx` (`src/transcribe-backend.cpp:152-159`).

Chính thư viện tự đo giá phải trả — `src/transcribe-session.h:305-307`:
> *"The measured recreation cost is about 1 ms per run on Metal and up to about 10 ms on CPU..."*

qwen3_asr **không** override `on_scratch_released()`, nên `cc->sched == nullptr` gần như luôn đúng và một `ggml_backend_sched` mới (`graph_size=16384`) được dựng mỗi run (`src/arch/qwen3_asr/model.cpp:634-641`). qwen3_asr còn `ggml_init` **hai lần** mỗi run (`:610-625` 16 MB, `:889-905` 8 MB).

#### F-33 — 🟠 HIGH: công việc host O(T²) + H2D mỗi run

`src/arch/qwen3_asr/model.cpp:812-824` dựng causal mask `T_prompt × T_prompt` bằng **vòng lặp lồng** rồi upload blocking:
```cpp
std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * T_prompt, mask_neg_inf);
for (int r = 0; r < T_prompt; ++r) {
    for (int c = 0; c <= r; ++c) {
        mask[static_cast<size_t>(r) * T_prompt + c] = mask_zero;
    }
}
ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
```
Với `T_prompt ≈ 800`: **1.28 MB cấp phát + 320k vòng lặp + 1.28 MB H2D blocking — mỗi poll**. Cộng thêm `build_cu_seqlens_mask` (`:662`, O(T²)) và `build_sinusoid_pe` (`:656`) đều dựng lại mỗi run. Tổng: **7 upload blocking + 2 download blocking mỗi run** (`:652, :657, :668, :801, :802, :809, :823` và `:709, :853`).

#### F-34 — 🟠 HIGH: output encoder round trip host vô nghĩa trong cùng một run

`src/arch/qwen3_asr/model.cpp:709` D2H encoder output về `cc->enc_host`, rồi `:802` H2D **ngược lại** vào prefill graph — dù cả hai graph nằm trên **cùng backend**. Đây là detour device→host→device thuần: 2 transfer blocking + 2 full pipeline sync + host allocation `d_enc × T_enc × 4` byte (≈12 MB ở 30 s với `d_enc = 1024`), **mỗi poll**.

#### F-35 — 🟠 HIGH: front-end mel spawn thread mới mỗi `compute()`

`src/transcribe-mel.cpp:493-507` — `run_threaded` tạo `stft_threads - 1` `std::thread` **mới, join, mỗi lần gọi**. Với `n_threads=4` và một poll 350 ms (≈35 mel frame), đó là **3 thread để làm 35 FFT 400 điểm** — thread creation chiếm phần không nhỏ của công việc. Đây là **fan-out thread độc lập thứ hai**, cộng lên trên pool ggml ở F-31.

#### F-36 — 🟡 MEDIUM: không pinned memory; mọi transfer CUDA blocking

`ggml/src/ggml-cuda/ggml-cuda.cu:786-787, 794-795` — mọi `set_tensor`/`get_tensor` là `cudaMemcpyAsync` + **`cudaStreamSynchronize` ngay lập tức**. Không overlap, không staging, không batch. Trên đường Vulkan (đang chạy thật, xem F-19) đây là transfer qua PCIe không có cơ hội che giấu độ trễ.

#### F-37 — 🟡 MEDIUM: các API đã có nhưng không dùng

| API | Có ở | Backend dùng? | Lợi ích nếu dùng |
| :--- | :--- | :--- | :--- |
| `session.run_batch()` | `bindings/python/src/transcribe_cpp/__init__.py:1136`; qwen3_asr có batched path (`src/arch/qwen3_asr/model.cpp:1714`) | ❌ | ~2× throughput cho multi-session |
| `Session.cancel()` + abort callback | `__init__.py:1359-1365`; trampoline đã cài ở `:1064-1067` | ❌ (F-14) | Huỷ inference cũ khi đổi tab/model |
| `session.limits.effective_max_audio_ms` | `__init__.py:1345-1355`; C API `transcribe_session_get_limits` | ❌ | Biết trần audio trước khi feed ⇒ tránh `OUTPUT_TRUNCATED` âm thầm |
| `session.stream()` **incremental** (`feed` từng phần, `committed_text`/`tentative_text`) | `__init__.py:1411-1438` | ❌ (bị begin/reset mỗi lần gọi — `asr/engine.py:284-287`) | Chuyển F-01 từ O(N²) sang O(N) |
| CUDA graphs (`GGML_CUDA_GRAPHS`) | build flag | ❌ OFF | Giảm sàn launch/sync mỗi token |

#### F-38 — 🟠 HIGH: harness benchmark **về mặt cấu trúc không thể phát hiện** O(N²)

`backend/tests/test_03_asr_benchmark.py:88` gọi `_run_inference_sync(audio)` **một lần** trên file tĩnh. Harness của thư viện (`tools/transcribe-bench/main.cpp:317-323, 408-414`) replay **cùng một sample độ dài cố định**. Cả hai **không bao giờ** chạy `run()` lặp lại trên buffer đang lớn dần, **không bao giờ** bench streaming API, và **không tính** `wall_ms − total_ms` (chính là phần overhead F-31/F-32/F-35).

⇒ Vì vậy con số "RTF 0.0232 / nhanh gấp 43× realtime" **đúng nhưng vô nghĩa** với đường chạy thật. Kèm theo đó, bộ `wav_test/` **không có file nào là nói liên tục dài** (dài nhất là `English_multiple_kinds_of_noise_88s` nhưng nhiều khoảng lặng) ⇒ kịch bản kích hoạt F-01 chưa bao giờ được test.

---

## 4. PHẦN C — KHẢ NĂNG SCALE NHIỀU STREAM / SESSION

### C1. 🔴 F-03 — Kiến trúc hiện tại bị chặn cứng ở 1 session

Bảng liệt kê mọi tài nguyên dùng chung:

| Tài nguyên | Khai báo | Ảnh hưởng khi N session |
| :--- | :--- | :--- |
| `TranscribeEngine._shared_session` | `asr/engine.py:48` (class attr) | **1 native session cho tất cả** — audio các session trộn vào cùng state |
| `TranscribeEngine._shared_model` | `asr/engine.py:46` | OK (model chia sẻ được) |
| `TranscribeEngine._infer_lock` | `asr/engine.py:51` (class-level RLock) | **Mọi inference ASR serialize toàn cục** |
| `_EXECUTOR` (2 worker) | `asr/engine.py:39` module-level | Xếp hàng |
| `_VAD_EXECUTOR` (1 worker) | `handler.py:36` module-level | **Serialize toàn cục** |
| `_TRANS_EXECUTOR` (1 worker) + `GGUFTranslator._infer_lock` | `translation/engine.py:34,44` | Serialize toàn cục |
| `OmniVoiceTTS._instance` (singleton) + `_infer_lock` | `tts/engine.py:36-37,67` | Serialize toàn cục |
| `metrics_collector` (singleton, 1 lock) | `core/metrics.py:22-23` | Contention tăng |
| `SafeStreamHandler._lock` | `logger.py:119` | Contention tăng |
| `VADEngineFactory._engines` | `vad/engines/__init__.py:18` | Model chia sẻ — **state thì per-session** ✅ |
| `CircularAudioBuffer`, `VADStreamProcessor`, queues | `ws/session.py` per-session | ✅ **Đúng** |

**Vấn đề nghiêm trọng nhất không phải hiệu năng mà là TÍNH ĐÚNG ĐẮN:** `_shared_session` là **một** `transcribe.cpp` session dùng cho mọi WebSocket session. Nhưng:

* `session.stream()` / `session.run()` thao tác trên **state của session** (`transcribe.h:24-26`: *"transcribe_session_* functions are not thread-safe. A session must be used by at most one thread at a time."*).
* `_infer_lock` bảo vệ điều đó ✅ **nhưng** ngữ nghĩa streaming bị chia sẻ: session A `stream()` → reset; session B `stream()` → reset state của A.

Với `qwen3-asr-1.7b` (`architecture_type: offline_llm`) mỗi `run()` là độc lập nên **tình cờ** đúng. Nhưng với `nemotron-3.5-streaming` / `voxtral-mini-4b-realtime` / `moonshine-streaming` (đều `architecture_type: streaming`), 2 session đồng thời sẽ **phá state của nhau**. Rủi ro thấp nếu thực tế chỉ 1 tab dùng — nhưng **không có gì trong code ngăn chặn** điều đó (không có semaphore/registry session).

### C2. Giới hạn được chính thư viện C++ ghi nhận

`external/transcribe.cpp/include/transcribe.h:11-20` (nguyên văn):
```
 * - KNOWN 0.x LIMITATION — concurrent COMPUTE is not yet supported: at most
 *   one transcribe_run / transcribe_run_batch / active stream may be in
 *   flight across ALL sessions of a given model at a time. Sessions share
 *   the model's backend instances and some per-family model state, so
 *   overlapping runs race (observed: corrupted decodes on CPU, command-
 *   buffer failures on Metal). Callers that want parallel transcription
 *   today should load one model per worker; a per-session backend
 *   architecture lifting this restriction is planned. Serialized use of
 *   many sessions on one model (e.g. a session pool behind a mutex) is
 *   fully supported.
```

⇒ Nếu muốn N session song song, phải **load 1 model instance cho mỗi worker** (tốn VRAM: 2.1 GB × N cho Qwen3-ASR-1.7B Q8_0). Với 16 GB VRAM đã dùng 9.5 GB cho 3 model khác, **thực tế không thể** scale quá 2–3 session ASR song song trên GPU này, kể cả khi refactor.

**Làm rõ quan trọng về bản chất ràng buộc (từ audit tầng native):** thư viện **KHÔNG có global lock nào** serialize session — mutex duy nhất trong `src/transcribe.cpp:895-900` là guard idempotency khi dlopen module, `ggml_cuda_lock` chỉ được giữ quanh CUDA-graph capture (`ggml-cuda.cu:4191,4281`) và **hoàn toàn inert** vì `GGML_CUDA_GRAPHS=OFF`. Nghĩa là:
* Ràng buộc multi-session là **về tài nguyên** (pool thread dùng-một-lần tranh core, không batching), **không phải** về lock.
* Và quan trọng hơn: thư viện **không tự bảo vệ**. Việc chạy 2 `run()` chồng lấn sẽ **race và corrupt decode** (đã quan sát trên CPU) — theo đúng cảnh báo ở header. Vì vậy `_infer_lock` (class-level RLock) ở `backend/asr/engine.py:51` là thứ backend **BẮT BUỘC** phải có, và **không được** gỡ nó ra khi tối ưu.
* Hệ quả thiết kế: muốn N session song song **thật** ⇒ N model instance. Không có đường tắt.

**Điều này nên được nêu rõ trong README** — hiện README không đề cập giới hạn single-session nào, cũng không nêu provider Vulkan (F-19) hay việc `transcribe-cpp-native-cu12` tồn tại.

### C3. Bảng đánh giá mức sẵn sàng scale

| Khả năng | Hiện trạng | Nút chặn chính |
| :--- | :--- | :--- |
| 1 session (use case thiết kế) | ✅ Chạy được | — (nhưng có F-01/F-02) |
| 2 tab cùng lúc | ⚠️ Chạy nhưng suy giảm nặng | F-03, `_VAD_EXECUTOR`=1, `_infer_lock` toàn cục |
| 2 tab dùng model streaming | 🔴 **Không đúng** | `_shared_session` chia sẻ state |
| > 2 tab | 🔴 Không khả thi | Serialize toàn cục + VRAM |
| Nhiều process (1 process/session) | ⚠️ Khả thi về code, không khả thi VRAM | Mỗi process nạp lại 3 model; cần API hướng `--worker` chưa có |
| Batch offline (`run_batch`) | ❌ Chưa dùng | Binding có `run_batch` (`__init__.py:1136`), backend không gọi |

---

## 5. PHẦN D — ĐỐI CHIẾU TUYÊN BỐ (README/REPORT) VỚI THỰC TẾ CODE

Bảng này quan trọng vì các tuyên bố hiệu năng đang được dùng làm cơ sở nghiệm thu.

| # | Tuyên bố | Vị trí | Thực tế trong code | Đánh giá |
| :--- | :--- | :--- | :--- | :--- |
| 1 | "Ring Buffer 60s Zero-Drop … **Lock-Free**" | `README.md:27`, `report/07:30` | `audio_buffer.py:28` `threading.Lock()`, acquire ở `:63` (write) và `:113` (read) | ❌ **Không lock-free** |
| 2 | "**Commit 4 Bậc Ưu Tiên**" | `README.md:31`, `report/04:5,19-21`, `report/07:31` | `decide_commit_trigger()` không được gọi trong runtime | ❌ **Chỉ 1/4 bậc hoạt động** |
| 3 | "3-Layer Dedup Engine — Loại bỏ 100% câu lặp rác" | `README.md:32` | `CommitDeduplicator.record_commit` không được gọi ⇒ `_history` rỗng ⇒ dedup tầng commit vô hiệu | ❌ **Sai ở tầng commit** |
| 4 | "ASR … RTF **0.0232**, nhanh gấp 43 lần realtime" | `README.md:30,167` | Số đo từ `engine._run_inference_sync(audio)` trên **file tĩnh** (`backend/tests/test_03_asr_benchmark.py:88`) — **không đo vòng lặp preview streaming**; harness của thư viện cũng chỉ replay sample cố định (F-38) | ❌ **Đo sai đường** |
| 5 | "Infer **~278ms**" | `README.md:30` | Trung bình trên file ngắn; câu dài tăng **bậc hai** (O(N²)) và độ trễ preview cuối ≈ 700 ms ở câu 30 s | ❌ **Không đại diện** |
| 6 | "VAD **< 0.2ms/frame**, RTF 0.004" | `README.md:28` | Mỗi frame chạy 1 PyTorch forward + feature extraction **bên trong RLock**; **không có metric runtime** nào xác thực | ⚠️ Không kiểm chứng được |
| 7 | "Zero-Copy Framing" | `README.md:35` | ≥ 4 bản copy/khung audio + base64 +33% cho TTS | ❌ **Không zero-copy** |
| 8 | "Session Cleanup **< 1ms**", "Fast Cleanup **0.52ms**" | `README.md:35,172` | Chính `report/07:18` ghi **102.68 ms** cho 1 mẫu (lock contention, F-13) | ⚠️ **Có mẫu vượt 200×** |
| 9 | "**Không rò rỉ bộ nhớ**" | `README.md:173` | `metrics._checkpoints` tăng vô hạn (F-15); TTS queue client không bound (F-16) | ❌ **Có memory growth** |
| 10 | "Tối ưu 1 Session" | `report/03:143` | Chính xác — nhưng là **giới hạn cứng**, không được nêu như một hạn chế | ⚠️ Cần nêu rõ |
| 11 | "AudioWorklet … low-latency, dedicated audio thread" | `extension_firefox/lib/audio-processor.js` | File **không bao giờ được load**; thực tế dùng `createScriptProcessor` trên **main thread** (`audio-capture.js:103`) | ❌ **Dead code** |
| 12 | "Speech Normalization: DC Offset removal, RMS auto-gain, soft limiter, Gain smoothing" | `README.md:29` | Không có DC offset removal; không có gain smoothing; 11 tham số config bị bỏ qua | ❌ **Một phần không tồn tại** |
| 13 | "Được đo lường và **tối ưu hóa hoàn hảo** trên RTX 5060 Ti 16GB" | `README.md:19` | ASR không hề dùng CUDA mà chạy **Vulkan** (`transcribe_cpp_native`, xác minh runtime — F-19) | ❌ **Chưa tối ưu cho GPU này** |
| 14 | "Bộ nhớ VRAM ~9.5 GB / 16 GB, **không rò rỉ**" | `README.md:173` | Số đo tĩnh 1 thời điểm; TTS giữ VRAM kể cả khi bị tắt (F-VRAM-1) | ⚠️ Không đại diện |
| 15 | "100% Offline Local Inference" | `README.md:9` | ✅ Đúng | ✅ |
| 16 | "Pre-warm song song ASR/Translation/VAD" | `main.py:125-130` | ✅ Đúng (`asyncio.gather` + `to_thread`) | ✅ |
| 17 | "orjson nhanh hơn 3-10x" | `ws/connection.py:16-26` | ✅ Đúng, có fallback | ✅ |
| 18 | "Fast Cleanup session **0.52ms**" | `README.md:172` | Phần lớn mẫu ~0.5ms ✅ | ✅ |
| 19 | Patch `0001-fix-threadpool-oversubscription.patch` | `external/transcribe.cpp/patches/ggml/` | ✅ **ĐÃ được apply** vào `ggml-cpu.c:527-534, 548-564, 624-636` | ✅ |

**Kết luận §D:** các tuyên bố về *kiến trúc* (commit 4 bậc, lock-free, zero-copy, dedup 3 lớp, worklet) **không khớp với code**. Các tuyên bố về *số đo* (RTF, ms) đúng về phép đo nhưng **đo sai đường** — chúng đo inference một lần trên file tĩnh, không đo pipeline runtime có vòng lặp preview lặp lại trên buffer đang lớn. Đây là điều cần chỉnh trước khi dùng các con số này cho nghiệm thu.

### Bổ sung §D — Ma trận nghiệm thu đề xuất

| Kịch bản | Bộ test hiện có | Có phát hiện F-01? | Cần bổ sung |
| :--- | :--- | :--- | :--- |
| File ngắn 4–11 s có khoảng lặng | ✅ `wav_test/` | ❌ | — |
| **Nói liên tục 30 s không ngừng nghỉ** | ❌ **KHÔNG CÓ** | ❌ | 🔴 **BẮT BUỘC** |
| Nói liên tục 60 s (chạm trần FireRed `max_speech_frame=2000` = 50 s) | ❌ | ❌ | 🟠 Nên có |
| **2 tab đồng thời** | ❌ | ❌ | 🔴 **BẮT BUỘC** (F-03) |
| **Tab bật model streaming (nemotron/voxtral)** | ❌ | ❌ | 🟠 (F-03 tính đúng đắn) |
| Phiên 1 giờ (leak) | ❌ | ❌ | 🟠 (F-15, F-16) |
| Seek/đổi video liên tục | ❌ | ❌ | 🟡 (F-14) |
| Backend nghẽn nhân tạo (inference chậm) | ❌ | ❌ | 🟠 (F-05, F-07) |


---

## 6. PHẦN E — KHUYẾN NGHỊ TỐI ƯU (XẾP THEO ROI)

### E1. P0 — Sửa ngay (đúng đắn / an toàn / tiết kiệm lớn)

| # | Hành động | File:line | Lợi ích kỳ vọng | Rủi ro |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | Thêm `_infer_lock` vào `unload_shared_model()` (hoặc dùng chung 1 lock), và **từ chối đổi model khi có session active** | `asr/engine.py:61-80`, `main.py:293-301` | Loại bỏ nguy cơ **crash use-after-free** | Thấp |
| **A2** | Wire `CommitManager.decide_commit_trigger()` vào `stream_tokens`; sửa `_last_activity_time` (gọi `record_activity()` khi có audio) | `asr/engine.py:308-391`, `core/commit_manager.py:74,76-78` | Chặn câu dài ⇒ triệt tiêu phần lớn O(N²); thêm stability split giảm latency | Trung bình (đổi hành vi phân câu) |
| **A3** | Giới hạn cửa sổ preview (`W = 4–6s`) hoặc chuyển sang **feed incremental** trên 1 stream sống suốt câu | `asr/engine.py:353-376` | **Giảm chi phí ASR từ O(N²) → O(N)**; độ trễ preview xuống mức chunk | Trung bình–cao (cần ghép text) |
| **A4** | Tái sử dụng kết quả preview cuối cho commit nếu cửa sổ audio không đổi | `asr/engine.py:322-351` | **−1 inference/câu** (~30–50% compute ASR thực tế) | Thấp |
| **A5** | `timestamps="none"` khi gọi `session.run()` | `asr/engine.py:296` | Bỏ materialize segments/words/tokens ⇒ giảm mạnh alloc/CPU | **Không** |
| **A6** | Chuyển TTS audio sang **binary WS frame**, bỏ base64; client dùng `decodeAudioData` | `tts/audio_processor.py:131-140`, `ws/serializers.py:70-91`, `tts-player.js:83-92` | −33% băng thông, bỏ ~5 bản copy, bỏ vòng lặp 144k iteration trên main thread | Thấp |
| **A7** | Thêm backpressure: kiểm tra `ws.bufferedAmount` trước khi gửi; drop frame cũ thay vì xếp hàng | `ws-client.js:236-250`, `service-worker.js:88-109` | Ngăn RAM tăng vô hạn + lag tích luỹ | Thấp |
| **A8** | Giới hạn `ttsPlayer.queue` (ví dụ 3) + drop theo `video.currentTime` | `tts-player.js:77,94-144` | Ngăn desync vĩnh viễn + memory growth | Thấp |
| **A9** | Load AudioWorklet thật (`audioWorklet.addModule` + `AudioWorkletNode`), bỏ `ScriptProcessorNode` | `audio-capture.js:101-176`, `audio-processor.js` | Giảm latency capture, bỏ jank main thread, khôi phục chất lượng audio gốc | Trung bình |

### E2. P1 — Tối ưu lớn (1–3 ngày)

| # | Hành động | File:line | Lợi ích |
| :--- | :--- | :--- | :--- |
| **B1** | Chuyển `is_speech()` ra ngoài `with self._lock` | `vad/processor.py:135-263` | Loại bỏ F-13; cleanup không còn spike 100ms |
| **B2** | Bounded `metrics._checkpoints` (dùng `deque(maxlen=… )` hoặc aggregate thay vì key theo session) | `core/metrics.py:36,115-118`, `ws/handler.py:54,127` | Sửa memory leak |
| **B3** | Thay `_pending_commits` (drop âm thầm) bằng cơ chế **back-pressure thật**: hoặc chặn VAD, hoặc gộp câu, và **luôn log + metric khi drop** | `asr/engine.py:111,237-242` | Không mất phụ đề âm thầm |
| **B4** | Giảm log level của logger `"backend"` xuống INFO (hoặc WARNING cho hot path); bỏ `flush()` mỗi record (dùng flush theo interval) | `utils/logger.py:119-146,174` | Giảm jitter + CPU |
| **B5** | Tối ưu `SpeechNormalizer`: `np.dot` cho RMS, `out=` cho các phép toán, bỏ 4 field không dùng; truyền config thật vào constructor | `core/normalizer.py:46-126`, `asr/engine.py:103` | Bỏ ~11 MB/alloc-per-call; config có tác dụng |
| **B6** | Giữ Int16 xuyên suốt tới trước model (buffer int16), chuyển float32 **một lần** | `core/audio_buffer.py`, `asr/engine.py:200-218` | −50% RAM buffer, bỏ 40 conversion/s |
| **B7** | Chuyển ASR sang provider **CUDA** (`transcribe-cpp-native-cu12`) để thống nhất CUDA với llama.cpp/PyTorch | `README.md:123`, `config.py:104` | Perf ASR tốt hơn + VRAM accounting thống nhất + giảm rủi ro OOM |
| **B8** | Dùng `Session.cancel()` / abort callback để huỷ inference cũ khi đổi tab/model | `asr/engine.py:334,363` (`bindings/.../__init__.py:1359`) | Loại bỏ latency spike khi seek/đổi kênh |
| **B9** | `unload_model()` TTS khi `tts_enabled` chuyển sang False | `main.py:340-343`, `tts/engine.py:296-314` | Giải phóng vài GB VRAM |
| **B10** | Thêm metrics cho ASR preview/commit/VAD + **độ sâu queue** | `asr/engine.py:335,364`, `ws/handler.py` | Biến bottleneck thành quan sát được (F-22) |
| **B11** | Fixed-rate preview scheduler thay vì `inference + sleep` | `asr/engine.py:379-391` | Nhịp cập nhật phụ đề đều, tự rơi preview khi quá hạn |
| **B12** | Hàng đợi chờ model load: "load mới trước, swap sau" cho hot-switch; trả 202 + thông báo WS | `main.py:293-338` | Không treo request vài giây, không mất ASR |
| **B13** | Bật `TRANSCRIBE_PERF_DEBUG=1` và chạy kịch bản **nói liên tục 30 s** để có **số đo thật** cho F-01/F-31…F-35 (không cần sửa code) | `src/arch/qwen3_asr/model.cpp:1034,1040-1058` | Biến mô hình phân tích thành số đo; cơ sở để verify mọi tối ưu sau |
| **B14** | Đọc `session.limits.effective_max_audio_ms` và kiểm tra `transcribe_was_truncated` trước/khi feed | `asyncio/engine.py:296`; `bindings/.../__init__.py:1345-1355` | Tránh `OUTPUT_TRUNCATED` âm thầm với câu dài |
| **B15** | Debounce poll lên ≥ 1 s khi VAD đang ở giữa câu dài (nếu A2/A3 chưa làm ngay) | `asr/engine.py:381`, `config.py:108` | Giảm trực tiếp số vòng O(N²) |

### E3. P2 — Tối ưu tầng native (cần sửa `external/transcribe.cpp`)

> ⚠️ Đây là thay đổi trong thư viện vendored. Repo có quy ước riêng (`external/transcribe.cpp/AGENTS.md`): dùng `uv run`, format bằng `scripts/ci/clang-format.sh`, và phải qua `scripts/validate.py`. Cân nhắc **đóng góp upstream** thay vì patch cục bộ.

| # | Hành động | File:line | Lợi ích |
| :--- | :--- | :--- | :--- |
| **N1** | Thêm **prefix/state reuse**: giữ KV cache + encoder output cho prefix đã transcribe và mở rộng, thay vì chạy lại trên toàn buffer | `src/arch/qwen3_asr/model.cpp:768-775`; không có API ở `src/causal_lm/` | Loại bỏ **gốc** của F-01 |
| **N2** | Gắn **persistent CPU threadpool** trong `configure_sched_n_threads` (dùng code parakeet làm template) | `src/transcribe-batch-util.cpp:69-88` vs `src/arch/parakeet/decoder.cpp:505-519` | Loại bỏ F-31 (~5–15 ms/run `[SUSPECTED]`) |
| **N3** | Ngừng gọi `release_scratch()` trên đường offline khi session sẽ chạy lại ngay (hoặc chỉ release trên ngưỡng kích thước) | `src/transcribe.cpp:2251-2253`; `src/transcribe-session.h:305-307` | Loại bỏ F-32 (1–10 ms/run, thư viện tự đo) |
| **N4** | Giữ encoder output **on-device**, feed thẳng prefill graph (bỏ round trip `enc_host`) | `src/arch/qwen3_asr/model.cpp:709` → `:802` | Loại bỏ F-34 (2 sync + ~12 MB transfer/poll) |
| **N5** | Cache causal mask `T_prompt²` + `build_sinusoid_pe` + `build_cu_seqlens_mask` khi chỉ đuôi thay đổi | `src/arch/qwen3_asr/model.cpp:812-824, 656, 662` | Loại bỏ F-33 (1.28 MB build + H2D/poll) |
| **N6** | Persistent mel thread pool (hoặc gộp STFT vào ngân sách pool ggml như gigaam) | `src/transcribe-mel.cpp:493-507`; đối chiếu `src/arch/gigaam/model.cpp:718-723` | Loại bỏ F-35 |
| **N7** | Thêm chế độ replay buffer-đang-lớn + metric overhead `wall_ms − total_ms` vào bench | `tools/transcribe-bench/main.cpp:391-415, 448-452` | Harness phát hiện được F-01/F-38 |

### E4. P2 — Kiến trúc & scale (1–2 tuần)

| # | Hành động | Lợi ích |
| :--- | :--- | :--- |
| **C1** | Tạo `ModelPool` cho transcribe.cpp: N model instance (theo `n_threads`/VRAM budget), cấp phát session theo kiểu pool-behind-mutex. Đúng như thư viện khuyến nghị (`transcribe.h:11-20`). **Lưu ý:** thư viện **không có** internal global lock — nó chỉ ghi rõ overlapping runs sẽ race (corrupted decodes). Muốn N session song song **thật** phải load 1 model instance/worker (2.1 GB VRAM mỗi instance cho Qwen3-ASR-1.7B Q8_0) | Cho phép N session song song thực sự |
| **C2** | Đưa `_VAD_EXECUTOR` / `_EXECUTOR` / `_TRANS_EXECUTOR` vào **per-session** hoặc một pool có kích thước theo cấu hình, thay vì global 1 worker | Loại bỏ serialize global |
| **C3** | Gộp batch: batch translation nhiều câu ngắn trong 1 lần gọi `llama`; dùng `run_batch` cho ASR offline (qwen3_asr có batched path thật) | Throughput ↑ (2× theo doc của binding) |
| **C4** | `n_ctx`, `n_batch`, `n_threads` của translation đưa vào config | Tune được |
| **C5** | Event-driven rendering ở client (`requestAnimationFrame` batching) + huỷ `findVideo()` per event (cache + `IntersectionObserver`) | Giảm main-thread cost |
| **C6** | Streaming translation token (llama.cpp `stream=True`) để hiển thị bản dịch dần dần | **Giảm latency cảm nhận ~250–800ms → ~100ms** mà không giảm chất lượng |
| **C7** | Ghi rõ giới hạn single-session + yêu cầu VRAM + provider Vulkan trong README | Tránh kỳ vọng sai |

### E5. Những tối ưu ĐÃ LÀM TỐT (không được regress)

| Hạng mục | Vị trí |
| :--- | :--- |
| `orjson` với fallback an toàn | `ws/connection.py:17-26` |
| `asyncio.Event` cho idle thay vì busy-poll (P2-1) | `asr/engine.py:114,244-247,383-391` |
| FireRed-VAD nhận Int16 trực tiếp, tránh F32↔I16 roundtrip | `vad/processor.py:167-171`, `vad/engines/firered.py:91-98` |
| `torch.inference_mode()` cho TTS | `tts/engine.py:96` |
| LRU cache cho `VoiceClonePrompt` | `tts/engine.py:114-140` |
| `KMP_BLOCKTIME=0` / `OMP_WAIT_POLICY=PASSIVE` / `torch.set_num_threads(2)` — chống busy-spin OpenMP | `main.py:29-45` |
| Patch ggml barrier spin→yield **đã được apply** | `ggml/src/ggml-cpu/ggml-cpu.c:527-564,627-633` |
| Bounded `deque` cho dedup/context/pending (đúng nguyên tắc, chỉ sai chính sách drop) | `core/dedup.py:42`, `translation/dedup.py:13`, `tts/dedup.py:16` |
| `deque(maxlen=10000)` cho metrics latencies | `core/metrics.py:34` |
| `SafeStreamHandler` chống `UnicodeEncodeError`/`WinError` | `logger.py:116-153` |
| Throttle log per-chunk (1 log / 200 chunk = 12.8s) ở extension | `audio-capture.js:159-160` |
| Fingerprint early-return + update text tại chỗ ở renderer | `subtitle-renderer.js:454-458,526-566` |
| Bounded `utteranceStore`/`completedSentences`/`playedIds` (50 / maxLines / 300) | `subtitle-renderer.js`, `tts-player.js:72-75` |
| `AbortController` cho listener lifecycle | `content-script.js:166-167` |
| Transferable `ArrayBuffer` trong worklet (dù worklet không được dùng) | `audio-processor.js:25-28` |
| Pre-warm song song 3 model | `main.py:125-130` |

---

## 7. PHẦN F — KẾ HOẠCH HÀNH ĐỘNG ĐỀ XUẤT

### Sprint 1 — An toàn & đúng đắn (ưu tiên tuyệt đối)
1. **A1** — sửa race `unload_shared_model` (crash risk)
2. **B2** — sửa memory leak `metrics._checkpoints`
3. **B3** — bỏ drop âm thầm ở `_pending_commits`, thêm log + metric
4. **B4** — hạ log level + bỏ flush-per-record
5. **B10** — thêm metric cho ASR preview/commit/VAD + queue depth ⇒ **có số liệu thật để đo các bước sau**

### Sprint 2 — Cắt chi phí ASR (ROI cao nhất)
6. **B13** — bật `TRANSCRIBE_PERF_DEBUG=1` + kịch bản nói liên tục 30 s ⇒ **đo trước khi tối ưu**
7. **A2** — wire lại Commit Manager (chặn câu dài) — đây cũng là bước "debounce" rẻ nhất cho F-01
8. **A4** — tái sử dụng preview cho commit
9. **A5** — `timestamps="none"`
10. **A3** — cửa sổ preview có giới hạn / feed incremental (hoặc **N1** nếu muốn sửa gốc)
11. **B11** — fixed-rate preview
12. **B15** — debounce poll ≥ 1 s (nếu A2/A3 lùi lịch)

### Sprint 3 — Transport & client
13. **A6** — TTS binary frame + `decodeAudioData`
14. **A7/A8** — backpressure + bounded TTS queue
15. **A9** — kích hoạt AudioWorklet thật
16. **C6** — streaming translation

### Sprint 4 — GPU, native & scale
17. **B7** — chuyển ASR sang CUDA provider (`transcribe-cpp-native-cu12`)
18. **B9** — giải phóng VRAM TTS khi tắt
19. **N2/N3** — persistent threadpool + bỏ release sched mỗi run (cân nhắc đóng góp upstream)
20. **N4/N5/N6** — bỏ round trip encoder, cache mask/pe, persistent mel pool
21. **B14** — kiểm tra `effective_max_audio_ms` / `was_truncated`
22. **C1/C2** — model pool + per-session executor (nếu thực sự cần multi-session)
23. **C7** — cập nhật README về giới hạn single-session + provider Vulkan

### Tiêu chí nghiệm thu lại (đề xuất)
* Đo **p50/p95/p99** cho: `vad_frame_ms`, `asr_preview_ms`, `asr_commit_ms`, `translate_ms`, `tts_ms`, `e2e_speech_to_subtitle_ms`, và **queue depth** của cả 3 queue.
* Bật `TRANSCRIBE_PERF_DEBUG=1` để có `mel` / `enc_build` / `enc_compute` / `enc_d2h` / `prefill_build` / `prefill_compute` / `step_loop` — đây là số đo **duy nhất** phơi ra F-01/F-31…F-35.
* Chạy kịch bản **nói liên tục 30s** (không im lặng) — kịch bản hiện tại trong `wav_test/` **không có** và do đó bỏ sót hoàn toàn F-01/F-02.
* Chạy kịch bản **2 tab đồng thời** để phát hiện F-03.
* Chạy **2 tab dùng model streaming** (`nemotron-3.5-streaming`) để xác nhận/loại trừ việc `_shared_session` phá state.
* Theo dõi RSS của backend và RSS của tab trong 1 giờ để xác nhận F-15/F-16.
* **Đo `wall_ms − total_ms`** từ `transcribe-bench` — khoảng cách này là F-31/F-32/F-35.

---

## 8. PHỤ LỤC

### 8.1 Chỉ mục file:line của các phát hiện chính

| ID | File:line |
| :--- | :--- |
| F-01 | `backend/asr/engine.py:330,353-376,381` |
| F-02 | `backend/asr/engine.py:104,308-391`; `backend/core/commit_manager.py:104-181` |
| F-03 | `backend/asr/engine.py:39,46-51,180,271-275`; `backend/ws/handler.py:36`; `backend/translation/engine.py:34,44`; `backend/tts/engine.py:36-37` |
| F-04 | `backend/asr/engine.py:61-80,271-296`; `backend/main.py:296` |
| F-05 | `backend/asr/engine.py:111,237-242` |
| F-06 | `extension_firefox/lib/audio-capture.js:101-176`; `extension_firefox/lib/audio-processor.js:1-38`; `extension_firefox/manifest.json:73` |
| F-07 | `extension_firefox/lib/ws-client.js:226-250`; `extension_firefox/background/service-worker.js:88-109` |
| F-08 | `backend/tts/audio_processor.py:131-140`; `backend/ws/serializers.py:70-91`; `extension_firefox/lib/tts-player.js:83-92` |
| F-09 | `backend/asr/engine.py:330-342` vs `:359-376` |
| F-10 | `backend/asr/engine.py:296`; `bindings/python/src/transcribe_cpp/__init__.py:1098,1266-1342` |
| F-11 | `backend/core/normalizer.py:46-126`; `backend/asr/engine.py:103`; `backend/config.py:112-124` |
| F-12 | `backend/utils/logger.py:119-146,174` |
| F-13 | `backend/vad/processor.py:135-263,274-282`; `backend/ws/session.py:278`; `report/07_final_e2e_comparison/report.md:18` |
| F-14 | `bindings/python/src/transcribe_cpp/__init__.py:1359-1365` (không dùng) |
| F-15 | `backend/core/metrics.py:36,115-118`; `backend/ws/handler.py:54,127` |
| F-16 | `extension_firefox/lib/tts-player.js:5,77,94-144` |
| F-17 | `backend/asr/engine.py:226` |
| F-18 | `backend/vad/processor.py:126-130` |
| F-19 | `README.md:143`; `external/transcribe.cpp/pyproject.toml` (cibuildwheel.windows); `bindings/python-native-cu12/pyproject.toml:33`; **+ xác minh runtime: `python -c "import transcribe_cpp; transcribe_cpp.native_provider()"` → `transcribe_cpp_native` + `ggml-vulkan.dll` + `ggml-cpu-haswell.dll`, KHÔNG có CUDA** |
| F-20 | `bindings/python/src/transcribe_cpp/__init__.py:1300-1326` |
| F-21 | `backend/translation/engine.py:104-111` |
| F-22 | `backend/core/metrics.py` (không có stage ASR/VAD); `backend/asr/engine.py:335,364` |
| F-23 | `backend/tts/engine.py:209,294` (`asyncio.to_thread` → default executor) |
| F-24 | `backend/core/commit_manager.py:74,76-78,135-147` |
| F-25 | `backend/config.py:39,177,185,199` |
| F-26 | `extension_firefox/lib/audio-capture.js:107-172` |
| F-27 | `extension_firefox/content/content-script.js:42-43,76,326` |
| F-28 | `extension_firefox/content/content-script.js:61`; `service-worker.js:117-125` |
| F-29 | `extension_firefox/lib/audio-capture.js:38-42` |
| F-30 | `backend/ws/serializers.py:28-43` |
| F-31 | `src/transcribe-batch-util.cpp:69-88`; `ggml/src/ggml-cpu/ggml-cpu.c:3400-3416,3357-3370,3458-3464`; đối chiếu `src/arch/parakeet/decoder.cpp:505-521` |
| F-32 | `src/transcribe.cpp:2251-2253`; `src/transcribe-backend.cpp:152-159`; `src/transcribe-session.h:305-307`; `src/arch/qwen3_asr/model.cpp:610-625,634-641,889-905` |
| F-33 | `src/arch/qwen3_asr/model.cpp:656,662,812-824` |
| F-34 | `src/arch/qwen3_asr/model.cpp:709` → `:802` |
| F-35 | `src/transcribe-mel.cpp:493-507` |
| F-36 | `ggml/src/ggml-cuda/ggml-cuda.cu:786-787,794-795` |
| F-37 | `bindings/python/src/transcribe_cpp/__init__.py:1136,1359-1365,1345-1355`; `GGML_CUDA_GRAPHS=OFF` |
| F-38 | `backend/tests/test_03_asr_benchmark.py:88`; `tools/transcribe-bench/main.cpp:317-323,408-414,448-452` |

### 8.2 Config & dead code

| Symbol | Vị trí | Trạng thái |
| :--- | :--- | :--- |
| `WSConfig.max_payload_bytes` | `config.py:39` | ❌ Không dùng ⇒ không có giới hạn payload |
| `AudioBufferConfig.max_speech_segment_sec` | `config.py:177` | ❌ Không dùng |
| `MetricsConfig.dump_report_on_disconnect`, `.report_file` | `config.py:184-185` | ❌ Không dùng |
| `AppConfig.hot_reload()` | `config.py:199-208` | ❌ Không dùng |
| `ASRConfig.normalize_*` (11 field) | `config.py:112-124` | ❌ Bị bỏ qua |
| `ArtifactRegistry.is_streaming_model()` | `asr/registry.py:72-77` | ❌ Không dùng (quyết định streaming dựa vào `capabilities`) |
| `TranscribeEngine._preview_queue` | `asr/engine.py:113` | ❌ Không dùng |
| `TranscribeEngine.commit_manager` | `asr/engine.py:104` | ❌ Không dùng |
| `BaseVADEngine.calculate_rms/peak` (result fields) | `core/normalizer.py:118-126` | ❌ 4/7 field không đọc |
| `AudioChunk`, `SpeechSegment`, `TranslationItem`, `TTSItem` | `core/pipeline_events.py:25-89` | ❌ Không dùng trong runtime (chỉ export) |
| `MoonshineStreamingOptions`, `ParakeetBufferedStreamOptions`, `VoxtralRealtimeStreamOptions` | binding | ❌ `build_family_options` không dựng các option này (`asr/adapters.py:88-105`) |
| `models.yaml: min_decode_interval_ms`, `context_window_sec` | `models.yaml:84,13` | ❌ Không dùng |
| `extension.../audio-processor.js` | — | ❌ Dead code (không có `audioWorklet.addModule` ở đâu) |
| `AudioProcessor.apply_time_stretch` | `tts/audio_processor.py:57-111` | ⚠️ Chỉ chạy khi `speed ≠ 1.0`; STFT/ISTFT numpy mỗi câu (đắt) — cần đo khi bật |
| `Session.run_batch()` | `bindings/python/src/transcribe_cpp/__init__.py:1136` | ❌ Không dùng (≈2× throughput cho multi-session) |
| `Session.cancel()` + abort trampoline | `__init__.py:1064-1067, 1359-1365` | ❌ Không dùng |
| `Session.limits` / `transcribe_session_get_limits` | `__init__.py:1345-1355` | ❌ Không dùng ⇒ không biết trần audio |
| `transcribe_was_truncated` | C API | ❌ Không kiểm tra ⇒ truncation âm thầm |
| `Stream.committed` / `Stream.tentative` | `__init__.py:1428-1438` | ❌ Không dùng (engine gán `stable_text=preview_text`, `unstable_text=""` — `asr/engine.py:372-373`) |
| `session.stream()` dạng incremental | `__init__.py:1411-1418` | ❌ Bị begin/reset mỗi lần gọi (`asr/engine.py:284-287`) |
| `GGML_CUDA_GRAPHS` | build flag | ❌ OFF trong cả hai build |
| Bench harness cho buffer-đang-lớn + `wall_ms − total_ms` | `tools/transcribe-bench/main.cpp` | ❌ Không tồn tại (nguyên nhân F-38) |

### 8.3 Ảnh hưởng của tối ưu đề xuất (tóm tắt định lượng)

| Tối ưu | Loại | Ước lượng |
| :--- | :--- | :--- |
| A3 giới hạn cửa sổ preview / feed incremental | Latency + GPU | O(N²) → O(N): ở câu 30s giảm ~**20×** công ASR preview |
| A4 tái sử dụng preview cho commit | GPU | **−30…50%** compute ASR tổng |
| A2 wire commit manager | Latency + GPU | Chặn câu > 8s ⇒ bỏ hẳn đuôi bậc hai |
| A5 `timestamps="none"` | CPU + alloc | Bỏ materialize words/tokens (hàng nghìn object/lần) |
| A6 TTS binary frame | Băng thông + CPU | **−33%** bytes, bỏ ~5 bản copy, bỏ 144k iteration JS/câu |
| B5 normalizer | CPU + alloc | Bỏ ~11 MB/call, 6 lượt duyệt mảng → 2 |
| B6 Int16 xuyên suốt | RAM + CPU | **−50%** RAM buffer audio, bỏ 40 conversion/s |
| B4 logging | CPU | Bỏ flush + lock contention trên hot path |
| B9 unload TTS khi tắt | VRAM | Giải phóng vài GB |
| C6 streaming translation | Latency cảm nhận | ~250–800ms → ~100ms |
| **B13 đo bằng `TRANSCRIBE_PERF_DEBUG`** | Đo lường | Biến mọi ước lượng `[SUSPECTED]` thành số đo; **làm trước khi tối ưu** |
| **N2 persistent threadpool** | CPU | **~5–15 ms/run** thread churn `[SUSPECTED]` × số poll |
| **N3 bỏ release sched/ctx mỗi run** | CPU + latency | **1–10 ms/run** (thư viện tự đo) × số poll |
| **N4 bỏ round trip encoder D2H/H2D** | PCIe + latency | 2 sync + ~12 MB transfer mỗi poll |
| **N5 cache causal mask + pe + cu_seqlens** | CPU + PCIe | 1.28 MB build + H2D mỗi poll (ở T_prompt≈800) |
| **N6 persistent mel thread pool** | CPU | Bỏ 3 thread create/join mỗi poll |
| **F-11 + F-20 + F native — chuỗi copy toàn buffer mỗi poll** | CPU + alloc | **3 tầng copy toàn bộ câu mỗi preview**: `get_slice` (`core/audio_buffer.py:130`) → `SpeechNormalizer` (~6 mảng) → `ctypes.from_buffer_copy` (`__init__.py:466`) ⇒ ~1.92 MB × 3 ở câu 30 s |

---

## 9. PHỤ LỤC CHI TIẾT (file riêng)

| File | Nội dung |
| :--- | :--- |
| `report/audit/01_phu_luc_extension_firefox.md` | Audit chi tiết `extension_firefox/**` (worklet, capture, transport, render, TTS, leak) |
| `report/audit/02_phu_luc_transcribe_cpp.md` | Audit chi tiết `external/transcribe.cpp/**` (threadpool, mel, KV cache, CPU↔GPU, locking, batching) |
| `report/audit/03_KE_HOACH_TRIEN_KHAI.md` | **Kế hoạch triển khai (Revision 2 — cá nhân hoá)**: tối ưu cho 1 GPU RTX 5060 Ti 16GB, **1 phiên duy nhất**, ưu tiên độ chính xác + tốc độ hiển thị + popup áp dụng tức thời. 4 phase, ~26–34 ngày, chiến lược test 3 tầng (< 15 s mặc định) |

> **Lưu ý phạm vi:** Revision 2 của kế hoạch **loại bỏ** toàn bộ phần multi-session/scale (RC-3, F-03, C1, C2 của báo cáo này) vì người dùng xác nhận chỉ chạy **1 phiên/1 video**. Các phát hiện đó vẫn đúng về mặt kỹ thuật nhưng **không cần khắc phục**.
>
> Revision 2 cũng bổ sung **§0.3–0.4 về popup** — 10 khoảng trống G1–G10 khiến thay đổi trong popup **không được áp dụng ngay**, phát hiện qua yêu cầu C6 của người dùng (xem chi tiết ở kế hoạch).

---

## 10. LỆNH KIỂM CHỨNG ĐÃ DÙNG (tái lập được)

| Mục đích | Lệnh | Kết quả |
| :--- | :--- | :--- |
| Xác định provider/backend ASR đang chạy (F-19) | `python -c "import transcribe_cpp; print(transcribe_cpp.__file__); print(transcribe_cpp.native_provider())"` | Load `ggml-vulkan.dll` + `ggml-cpu-haswell.dll`; **không có CUDA** |
| Xác nhận patch ggml barrier đã apply (F-31 nền tảng) | `grep -n "GGML_BARRIER_SPIN_BEFORE_YIELD\|YieldProcessor\|SwitchToThread" external/transcribe.cpp/ggml/src/ggml-cpu/ggml-cpu.c` | 10 match ở `:527-564,627-633` ⇒ **đã apply** |
| Xác nhận worklet không được dùng (F-06) | `grep -rn "audioWorklet\|AudioWorkletNode\|addModule" extension_firefox` | Chỉ `manifest.json:73` + `registerProcessor` ⇒ **dead code** |
| Xác nhận không có backpressure (F-07) | `grep -rn "bufferedAmount" extension_firefox` | **0 kết quả** |
| Xác nhận Commit Manager là dead code (F-02) | `grep -rn "decide_commit_trigger" backend` | Chỉ có trong `backend/tests/test_04_commit_logic.py` |
| Xác nhận lock trong audio buffer (claim "Lock-Free") | `grep -n "self._lock" backend/core/audio_buffer.py` | `:28` `threading.Lock()`, `:63` write, `:113` read |
| Xác nhận backend flags của build checked-in | `Select-String -Path external/transcribe.cpp/build*/CMakeCache.txt -Pattern "GGML_CUDA:BOOL\|GGML_VULKAN:BOOL\|GGML_OPENMP:BOOL"` | `GGML_CUDA=OFF`, `GGML_VULKAN=OFF`, `GGML_OPENMP=OFF` |
| Đo lại hot path **không cần sửa code** | `set TRANSCRIBE_PERF_DEBUG=1` rồi chạy kịch bản nói liên tục 30 s | In `mel` / `enc_build` / `enc_compute` / `enc_d2h` / `prefill_build` / `prefill_compute` / `step_loop` |

---

*Báo cáo được tạo ở chế độ **read-only**. Không có file mã nguồn nào bị sửa đổi.*
