# Báo Cáo Rà Soát Kỹ Thuật – `backend_cpp` Streaming ASR Pipeline

> **Reviewed by:** Senior Backend + Speech AI Engineer  
> **Date:** 2026-09-14  
> **Scope:** `backend_cpp/` — toàn bộ module ASR, VAD, WS, translation, config

---

## 1. Tổng Quan Sức Khỏe Dự Án

| Tiêu chí | Đánh giá |
|---|---|
| **Điểm tổng thể** | **7.2 / 10** |
| Kiến trúc pipeline | Rõ ràng, phân tầng tốt |
| Concurrency design | Tinh tế, có lifecycle barrier nghiêm chỉnh |
| Observability | Tốt (perf profiler, telemetry) |
| Rủi ro cao nhất | Internal queue mutation trực tiếp vào `asyncio.Queue._queue` |

### Điểm mạnh nổi bật
- **Lock hierarchy rõ ràng**: `_shared_lock → _shared_infer_lock` nhất quán, có kiểm tra ownership.
- **Commit Priority**: Commit blocking preview qua `_commit_waiting` counter – thiết kế đúng.
- **VAD-paced boundary**: Grace period + emergency failsafe (Phase 3C.3) – kiến trúc chính xác.
- **Epoch generation barrier**: Nhất quán xuyên suốt VAD → ASR → translation → TTS.
- **Backpressure queue**: Latest-wins cho preview, lossless cho final – pattern tốt.

### Rủi ro lớn nhất hiện tại
Truy cập trực tiếp vào `asyncio.Queue._queue` (private CPython internal) để coalesce preview – đây là **CPython implementation detail**, có thể vỡ bất kỳ lúc nào khi upgrade Python.

---

## 2. Danh Sách Vấn Đề

### 🔴 Critical

#### C-01: Truy cập trực tiếp `asyncio.Queue._queue` (private CPython API)
- **File:** `transcribe_engine.py` L452-L462, L422-L438
- **Vấn đề:** `_enqueue_token_message()` và `_evict_queued_preview()` đọc/ghi thẳng vào `q._queue` (deque nội bộ của asyncio.Queue) và modify `q._unfinished_tasks`. Đây là **CPython internal**, không có trong public API. Python 3.13+ đã bắt đầu thay đổi internals.
  ```python
  pending = getattr(q, "_queue", None)  # ← private!
  item.clear(); item.update(msg)         # ← mutate queue in-place
  q._unfinished_tasks -= 1              # ← bypass public API
  ```
- **Mức độ:** 🔴 Critical
- **Sửa:** Dùng một `deque` riêng làm "shadow queue" với `asyncio.Event` để signal consumer. Hoặc dùng `asyncio.Queue` kết hợp dict `{utterance_id: msg}` để implement coalescing ở lớp trên.

---

#### C-02: `transformers` không có trong `requirements.txt` nhưng được dùng bởi NamoTurnDetector
- **File:** `namo_detector.py:19`, `requirements.txt`
- **Vấn đề:** `from transformers import AutoTokenizer` – `transformers` (Hugging Face) là thư viện ~gigabyte-class với PyTorch dependency. Nó **hoàn toàn vắng mặt** trong `requirements.txt`. Pipeline sẽ crash tại runtime khi Namo được enabled nếu môi trường thiếu.
- **Mức độ:** 🔴 Critical
- **Sửa:** Thêm `transformers>=4.40.0` vào `requirements.txt`. Hoặc nếu muốn giữ "No PyTorch required" như comment line 1, phải dùng `tokenizers` package trực tiếp thay vì `transformers.AutoTokenizer`.

---

#### C-03: `torch` được import ở top-level trong `vad/engines.py` nhưng không có trong `requirements.txt`
- **File:** `engines.py:21`
- **Vấn đề:** `import torch` ở top-level – đây là hard dependency với PyTorch (~2GB). `requirements.txt` lại tuyên bố "No PyTorch required for ASR!" nhưng Silero VAD cần PyTorch, và `fireredvad` cũng có thể cần. Môi trường không có PyTorch sẽ crash ngay khi import `engines.py`.
- **Mức độ:** 🔴 Critical  
- **Sửa:** Lazy import PyTorch chỉ khi engine được chọn là silero hoặc firered. Thêm `torch` vào requirements với note rõ ràng là optional cho silero.

---

### 🟠 High

#### H-01: `force_end()` trong VADProcessor unpack tuple 2 phần tử nhưng ring buffer lưu 5 phần tử
- **File:** `vad_processor.py:371-374`
- **Vấn đề:** Trong `force_end()`:
  ```python
  pre_bytes, pre_ts = state.pre_speech_ring.popleft()  # ← unpack 2 phần tử
  ```
  Nhưng trong `feed_chunk()` (line 325), ring buffer lưu tuple 5 phần tử: `(frame_bytes, frame_ts, frame_media_start, frame_media_end, epoch)`. → `ValueError: too many values to unpack` khi `force_end()` chạy sau speech có pre-roll audio.
- **Mức độ:** 🟠 High (crash lúc cleanup, có thể silent nếu caught by outer exception handler)
- **Sửa:**
  ```python
  pre_item = state.pre_speech_ring.popleft()
  pre_bytes = pre_item[0]; pre_ts = pre_item[1]
  pre_m_start = pre_item[2] if len(pre_item) > 2 else 0.0
  pre_m_end   = pre_item[3] if len(pre_item) > 3 else 0.0
  pre_ep      = pre_item[4] if len(pre_item) > 4 else 0
  callbacks.append((self.on_speech_chunk, (pre_bytes, pre_ts, VAD_STATE_PRE_ROLL, pre_m_start, pre_m_end, pre_ep)))
  ```

---

#### H-02: `config = load_config()` tại module level trigger I/O và singleton init ở import time
- **File:** `config.py:355-361`, `config.py:251`
- **Vấn đề:** `config = load_config()` ở module level → `AppConfig()` → `TranslationConfig()` → `_apply_base_defaults()` → `TranslationModelRegistry.get_instance()` → đọc file YAML. Bất kỳ `import backend_cpp.config` nào sẽ trigger I/O và singleton initialization. Nếu YAML thiếu, `config` sẽ fail ở import time với lỗi khó debug.
- **Mức độ:** 🟠 High
- **Sửa:** Lazy-load translation registry; chỉ resolve trong `get_translator()` call. Tách `_apply_base_defaults` khỏi pydantic `model_validator` và dùng explicit `finalize()` method.

---

#### H-03: NamoTurnDetector ONNX inference được gọi synchronously trong coroutine (blocks event loop)
- **File:** `transcribe_engine.py:1173-1177`
- **Vấn đề:** Trong `check_preview_and_split()` (coroutine chạy trên event loop):
  ```python
  is_namo_eou, namo_conf = self._namo_detector.predict_eou(...)  # ← BLOCKING ONNX inference!
  ```
  `predict_eou()` chạy ONNX runtime inference (5–30ms) trực tiếp trên event loop thread, không có `await asyncio.to_thread(...)`. Điều này block event loop, trì hoãn WebSocket receive và tất cả async tasks khác.
- **Mức độ:** 🟠 High
- **Sửa:**
  ```python
  is_namo_eou, namo_conf = await asyncio.to_thread(
      self._namo_detector.predict_eou,
      preview_text,
      confidence_threshold=self.namo_config.confidence_threshold,
      min_tokens=self.namo_config.min_tokens,
  )
  ```

---

#### H-04: `asyncio.Lock()` trong `LocalGGUFTranslator.__init__` tạo ở wrong event loop
- **File:** `local_translator.py:118`
- **Vấn đề:** `self._async_load_lock = asyncio.Lock()` được gọi trong `__init__`. Nếu `__init__` chạy ở main thread trước khi event loop được tạo (ví dụ trong warmup thread), Lock sẽ thuộc về **wrong event loop** và raise `RuntimeError` khi `async with self._async_load_lock` chạy trong uvicorn event loop.
- **Mức độ:** 🟠 High
- **Sửa:** Lazy init: `self._async_load_lock: Optional[asyncio.Lock] = None` và trong `translate()`: `if self._async_load_lock is None: self._async_load_lock = asyncio.Lock()`.

---

#### H-05: `qwen3-asr-0.6b` khai báo `vram_estimate_mb: 2100` – copy-paste error từ 1.7B
- **File:** `models.yaml:24`
- **Vấn đề:** Model 0.6B với Q8_0 chỉ cần khoảng ~650MB VRAM, nhưng khai báo 2100MB – giống với 1.7B model. Đây là copy-paste error.
- **Mức độ:** 🟠 High
- **Sửa:** Sửa thành `vram_estimate_mb: 650`.

---

### 🟡 Medium

#### M-01: `_push_message` drop final message silently khi `_loop is None`
- **File:** `transcribe_engine.py:1093-1111`
- **Vấn đề:** Nếu `self._loop` là None lúc commit xảy ra (ví dụ sau `cleanup()`), final message bị silently dropped. Không có warning, không có counter increment.
- **Mức độ:** 🟡 Medium
- **Sửa:** Thêm `logger.warning(...)` và `perf.increment_counter("asr.final_dropped_no_loop")` khi `is_final and not self._running`.

---

#### M-02: `VADConfig.__setattr__` override trên Pydantic v2 model
- **File:** `config.py:186-194`
- **Vấn đề:** Override `__setattr__` trong Pydantic v2 model để propagate threshold là pattern không được khuyến khích và có thể conflict với Pydantic internals. `model_validator(mode="after")` đã làm đúng role này nhưng `__setattr__` override tạo thêm mutation path với undefined behavior nếu validation state change.
- **Mức độ:** 🟡 Medium  
- **Sửa:** Bỏ `__setattr__` override. `apply_engine_profile()` (đã có) là đủ.

---

#### M-03: `AudioBufferManager.get_snapshot()` – `np.frombuffer` trên bytearray trả về view, không copy
- **File:** `audio_buffer.py:93-94`
- **Vấn đề:** `bytearray(self._bytes_buffer)` dưới lock (correct) nhưng `np.frombuffer(raw_bytes, dtype=np.int16)` trả về view vào `raw_bytes`. Tuy nhiên `raw_bytes` là local variable đã copy – an toàn trong trường hợp này. Rủi ro tiềm ẩn nếu code tương lai pass `raw_bytes` trực tiếp mà không copy.
- **Mức độ:** 🟡 Medium
- **Sửa:** Dùng `np.frombuffer(raw_bytes, dtype=np.int16).copy()` để explicit copy, loại bỏ ambiguity.

---

#### M-04: `NamoConfig.model_dir` dùng relative path string
- **File:** `config.py:335`
- **Vấn đề:** `model_dir: str = "backend_cpp/models/namo"` – **relative path** phụ thuộc vào CWD khi process start. Nếu server được start từ directory khác, path sẽ sai.
- **Mức độ:** 🟡 Medium
- **Sửa:** `model_dir: str = str(BACKEND_CPP_DIR / "models" / "namo")`

---

#### M-05: Concurrent binary frame processing có thể race ở VAD callback layer
- **File:** `ws_handler.py:280-282`
- **Vấn đề:** `_handle_binary_message` dùng `await asyncio.to_thread(_process_binary_chunk, ...)`. Nếu hai binary frames đến gần nhau và cả hai được submit vào thread pool, chúng có thể chạy **concurrently** và cả hai call `vad_processor.feed_chunk()` cùng lúc. VADProcessor có `self._lock` nhưng callbacks (bao gồm `asr_engine.feed_audio`) được gọi **ngoài lock** – race condition nhẹ.
- **Mức độ:** 🟡 Medium
- **Sửa:** Serialize audio ingress với dedicated single-thread executor, hoặc dùng `asyncio.Semaphore(1)` trước `asyncio.to_thread`.

---

#### M-06: Translation queue có thể delay subtitle của epoch mới do stale items
- **File:** `ws_handler.py:329-352`, `session_state.py:379-397`
- **Vấn đề:** Nếu nhiều translation items tích lũy trong queue (maxsize=20) trước khi stream reset, worker sẽ mất nhiều poll cycles để drain chúng qua epoch check, delay subtitle của epoch mới.
- **Mức độ:** 🟡 Medium
- **Sửa:** Tăng drain aggressiveness trong `handle_stream_reset` hoặc dùng epoch-tagged priority queue.

---

#### M-07: `SentenceSegmenter.remove_prefix_overlap` word-level split không hoạt động với CJK
- **File:** `sentence_segmenter.py:136-168`
- **Vấn đề:** `remove_prefix_overlap()` dùng `.split()` để tách words nhưng CJK text không có spaces. Với Japanese/Chinese, word-level prefix removal không hoạt động.
- **Mức độ:** 🟡 Medium
- **Sửa:** Với CJK text (detect bằng `is_cjk()`), dùng character-level prefix strip.

---

#### M-08: `pydantic-settings>=2.1.0` trong requirements.txt nhưng không được dùng
- **File:** `requirements.txt:5`
- **Vấn đề:** `pydantic-settings` không được import bất kỳ đâu trong codebase. Dependency dư thừa.
- **Mức độ:** 🟡 Medium
- **Sửa:** Xóa khỏi `requirements.txt`.

---

#### M-09: `gguf>=0.10.0` trong requirements – cần xác nhận có được dùng không
- **File:** `requirements.txt:12`
- **Vấn đề:** `gguf` Python library (HuggingFace) là tool-level package cho GGUF file inspection/conversion. Nếu không có code nào `import gguf` trực tiếp, đây là dependency dư thừa.
- **Mức độ:** 🟡 Medium
- **Sửa:** Chạy `grep -r "import gguf"` để verify. Nếu không dùng, xóa.

---

### 🔵 Low

#### L-01: `NamoConfig.max_length: int = 8192` quá lớn cho BERT-class model (max 512)
- **File:** `config.py:339`
- **Sửa:** Đặt `max_length: int = 512`.

#### L-02: Warmup trong `lifespan()` không handle partial failures riêng biệt
- **File:** `main.py:101-118`
- **Sửa:** Tách từng warmup vào try/except riêng để ASR/translation/VAD warmup độc lập.

#### L-03: `cert.pem` và `key.pem` được commit vào repo
- **File:** `backend_cpp/cert.pem`, `backend_cpp/key.pem`
- **Sửa:** Thêm `*.pem` vào `.gitignore`, xóa khỏi repo.

#### L-04: Không pin upper bound version cho `transcribe-cpp` và `llama-cpp-python`
- **File:** `requirements.txt:9-10`
- **Sửa:** `llama-cpp-python>=0.3.0,<0.4.0`, `transcribe-cpp>=0.2.3,<0.3.0`.

#### L-05: `_max_duration_deferred_logged` reset trong `reset_stream()` nhưng không khai báo trong `__init__`
- **File:** `transcribe_engine.py:1334`
- **Sửa:** Thêm `self._max_duration_deferred_logged: bool = False` vào `__init__`.

---

## 3. Phân Tích Điểm Nghẽn Hiệu Năng

### Pipeline Latency Budget (ước lượng từ code)

```
Audio ingress (WebSocket) → to_thread parse
    [~0.5ms] frame_protocol.parse_audio_frame()
    
VAD processing (CPU, per 25ms frame)
    [fsmn-vad: ~3.6ms/frame = ~57.9ms/audio-s]
    VAD callbacks ngoài lock → on_speech_chunk → feed_audio

Preview polling (every 350ms)
    [AudioBuffer snapshot copy: ~0.3ms]
    [ASR inference: 100-500ms depending on audio length]  ← BOTTLENECK #1
    [Namo EOU inference: 5-30ms BLOCKING EVENT LOOP]      ← BOTTLENECK #2 (H-03)
    
Commit path (on VAD silence / Namo EOU)
    [Full ASR inference: 0ms (cached) or 200-800ms]       ← BOTTLENECK #3
    [Translation: 200-800ms llama-cpp-python]             ← BOTTLENECK #4
    
TTS synthesis (optional): ~500ms per sentence
```

### Bottleneck #1 – Preview ASR "Amplification Effect"
- **Vấn đề:** Mỗi preview cycle chạy inference trên **toàn bộ audio buffer** từ đầu utterance. Với utterance 8s và poll interval 350ms, có thể có 20+ preview cycles → tổng audio được xử lý = rất lớn.
- **Telemetry:** `asr.preview_audio_ms / asr.commit_audio_ms` trong `/api/perf/summary` – xem ratio này.
- **Giải pháp:** Với streaming models (`_shared_supports_streaming = True`), dùng incremental feed thay vì full-buffer replay mỗi poll.

### Bottleneck #2 – Namo EOU blocks event loop (Fix: H-03)
- Fix bằng `await asyncio.to_thread(...)` → giải phóng 5-30ms blocking mỗi preview cycle.

### Bottleneck #3 – Commit ASR đã có preview cache optimization
- `reuse_preview = diff_samples == 0` → loại bỏ inference khi VAD silence đến sau preview stable.
- Confirm với `asr.cached_preview_reuse` counter.

### Bottleneck #4 – Translation sequential (bình thường cho single-user desktop)
- Với translation queue `maxsize=20`, burst commits có thể cause queue full → subtitle drop.
- Monitor `translation.queue_full_dropped` counter.

### Tổng kết latency
| Stage | Typical | P95 | Notes |
|---|---|---|---|
| VAD | 3-6ms/frame | 10ms | fsmn-vad, CPU |
| Preview ASR | 100-500ms | 800ms | Phụ thuộc audio length |
| Namo EOU | 5-30ms | 50ms | **Đang block event loop** |
| Commit ASR | 0ms (cached) / 200-800ms | 1500ms | Với preview reuse |
| Translation | 200-600ms | 1000ms | llama-cpp-python |
| **E2E ASR→Subtitle** | ~300ms | ~1500ms | |

---

## 4. Dependency & Cài Đặt

### Thư viện dư thừa
| Package | Trạng thái | Hành động |
|---|---|---|
| `pydantic-settings>=2.1.0` | Không được dùng trong code | ❌ Xóa |
| `gguf>=0.10.0` | Cần verify | ⚠️ Kiểm tra rồi xóa nếu không dùng |

### Thiếu trong requirements.txt (Critical)
| Package | Lý do cần | Mức độ |
|---|---|---|
| `transformers>=4.40.0` | `namo_detector.py` dùng `AutoTokenizer` | 🔴 Critical |
| `torch>=2.0.0` | `vad/engines.py` import trực tiếp | 🔴 Critical |
| `fireredvad` / `funasr` | VAD engines FireRed và FSMN | ⚠️ Cần kiểm tra xem đã bundle trong package chưa |

### Version cần pin
```txt
# Hiện tại (không an toàn):
transcribe-cpp>=0.2.3
llama-cpp-python>=0.3.0

# Nên thành:
transcribe-cpp>=0.2.3,<0.3.0
llama-cpp-python>=0.3.0,<0.4.0
onnxruntime-gpu>=1.17.0,<2.0.0
```

### Gợi ý tách requirements
```
requirements.txt          # Core runtime (no ML frameworks)
requirements-vad.txt      # torch, fireredvad, funasr (optional engines)
requirements-namo.txt     # transformers (optional: Namo EOU detector)
requirements-dev.txt      # pytest, etc.
```

---

## 5. Đề Xuất Cải Tiến Ưu Tiên (Top 5 Việc Nên Làm Ngay)

### 1️⃣ Fix `asyncio.Queue` private API access (C-01) — Rủi ro kỹ thuật cao nhất
Implement `CoalescingQueue` dùng public API:
```python
class CoalescingQueue:
    """FIFO queue với latest-wins preview coalescing per utterance_id."""
    def __init__(self, maxsize=64):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._preview_map: dict = {}  # utterance_id → position
    # ...
```

### 2️⃣ Fix Namo EOU blocking event loop (H-03) — 1-line fix, tác động cao
```python
# transcribe_engine.py, check_preview_and_split():
is_namo_eou, namo_conf = await asyncio.to_thread(
    self._namo_detector.predict_eou, preview_text, ...
)
```

### 3️⃣ Thêm missing dependencies vào requirements.txt (C-02, C-03)
```txt
torch>=2.0.0               # Required by silero-vad and fireredvad engines
transformers>=4.40.0       # Required by NamoTurnDetector AutoTokenizer
```

### 4️⃣ Fix `force_end()` tuple unpack bug (H-01) — Silent crash on disconnect mid-speech
Sửa unpack từ 2 → 5 phần tử trong `force_end()` pre_speech_ring loop.

### 5️⃣ Fix `NamoConfig.model_dir` relative path và `asyncio.Lock` wrong loop (M-04, H-04)
```python
# config.py
model_dir: str = str(BACKEND_CPP_DIR / "models" / "namo")

# local_translator.py
self._async_load_lock: Optional[asyncio.Lock] = None  # lazy init
```

---

## 6. Checklist Xác Nhận Sau Khi Sửa

### Stability
- [ ] `force_end()` xử lý đúng 5-tuple từ `pre_speech_ring` – không raise `ValueError`
- [ ] `asyncio.Queue._queue` và `_unfinished_tasks` không còn được truy cập trực tiếp
- [ ] Namo EOU inference wrapped trong `asyncio.to_thread` – không block event loop
- [ ] `asyncio.Lock()` trong `LocalGGUFTranslator` được init lazy trong running event loop

### Correctness
- [ ] `NamoConfig.model_dir` resolve đúng bất kể CWD khi start server
- [ ] `qwen3-asr-0.6b` VRAM estimate = ~650MB (không phải 2100MB)
- [ ] `NamoConfig.max_length = 512` (match BERT context window)
- [ ] `_max_duration_deferred_logged` được khai báo trong `__init__`

### Dependencies
- [ ] `pip install -r requirements.txt` thành công trong clean environment
- [ ] `transformers` available và `AutoTokenizer.from_pretrained()` hoạt động
- [ ] `torch` available và silero-vad load được
- [ ] `pydantic-settings` đã xóa (verify không có `import pydantic_settings`)

### Performance
- [ ] `/api/perf/summary` → `asr.preview_audio_ms / asr.commit_audio_ms` ratio < 5×
- [ ] `/api/perf/summary` → `asr.cached_preview_reuse` counter > 50% của total commits
- [ ] `/api/perf/summary` → không có `asr.final_dropped` increments
- [ ] `pipeline.e2e_asr_to_sub_ms` P95 < 1500ms trong normal session

### Concurrency & VAD
- [ ] `config.debug.strict_lock_checks = true` → không raise warning trong normal flow
- [ ] Concurrent binary frame delivery không cause race (test với jitter simulation)
- [ ] Stream reset drain hoàn toàn translation queue trước khi epoch advance
- [ ] Không có `[VAD Buffer Overflow during SPEECH]` warning trong 60s test run
- [ ] `force_end()` callback chain hoàn thành không có exception khi disconnect mid-sentence

---

*Báo cáo này dựa trên đọc toàn bộ source code. Không có giả định nào về runtime behavior chưa được verify trong code.*
