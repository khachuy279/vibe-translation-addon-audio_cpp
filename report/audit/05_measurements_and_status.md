# BÁO CÁO KẾT QUẢ TRIỂN KHAI & ĐO LƯỜNG
## Đối chiếu với `report/audit/03_KE_HOACH_TRIEN_KHAI.md` (Revision 2)

| | |
| :--- | :--- |
| **Phạm vi đã triển khai** | Phase 0 (đo lường + hạ tầng test), Phase 1 (an toàn + popup), Phase 2 (cắt chi phí ASR), Phase 3 (backend + phần lớn extension) |
| **Chưa triển khai** | P3.4 (AudioWorklet thật), P2.9 (buffer int16), P4.7 (sửa native), P4.12 (model pool — đã bỏ theo C2) |
| **Trạng thái git** | Chưa commit (theo yêu cầu). Toàn bộ thay đổi nằm trong working tree |
| **Bộ test mặc định** | `pytest` → **86 passed, 10 deselected, ~11 s** (không cần nạp model) |

---

## 1. TRẠNG THÁI THEO 4 YÊU CẦU

| # | Yêu cầu | Trạng thái | Bằng chứng |
| :--- | :--- | :--- | :--- |
| **C3** | Tối ưu RTX 5060 Ti 16GB | 🟡 **Một phần (đã đổi cách)** | ⚠️ **CUDA provider KHÔNG khả thi**: `transcribe-cpp-native-cu12` trên PyPI chỉ là *name reservation* (0.0.0, wheel 1.4 KB); build từ source cần CUDA Toolkit + MSVC + Ninja mà máy **không có**. Đã bù bằng **nhịp preview thích ứng (P2.4b)** + `/health` phơi backend + cảnh báo khởi động |
| **C4** | Chính xác tối đa ở chế độ streaming | ✅ **Đã làm** | Commit luôn dùng toàn ngữ cảnh câu; cửa sổ preview ≥ `max_duration_sec`; sửa pre-roll lấy nhầm 300 ms câu trước; sàn thời lượng/số từ cho BẬC 3; normalizer config có tác dụng thật |
| **C5** | Phụ đề gốc/dịch hiện nhanh & liên tục | ✅ **Đã làm** | Nhịp preview cố định; preview bắt đầu sớm hơn (0.35 s); cửa sổ chặn trên 6 s; **streaming translation** (bản dịch hiện dần); TTS binary frame; backpressure phía client |
| **C6** | Popup áp dụng ngay, không restart | ✅ **Đã làm** | Sửa đủ **10 khoảng trống G1–G10**; REST và WS đi cùng một đường `apply_config`; VAD pre-warm; ASR/translation model nạp trước rồi swap |

---

## 2. NHỮNG GÌ ĐÃ THAY ĐỔI (theo task)

### Phase 0 — Đo lường & hạ tầng test

| Task | Thay đổi | File |
| :--- | :--- | :--- |
| P0.1 | Thêm `record_gauge`/`get_gauge`/`snapshot_pipeline`; gauge độ sâu queue, `active_sessions`, `pending_commits`, `vad.overflow` | `core/metrics.py`, `ws/handler.py`, `ws/connection.py` |
| P0.1 | Thêm endpoint `GET /api/metrics/pipeline` (ảnh chụp gọn hot path) | `main.py` |
| P0.1 | Instrument: `asr.preview_ms`, `asr.commit_ms`, `asr.infer_core_ms`, `asr.preview_audio_sec`, `asr.commit_audio_sec`, `asr.slice_ms`, `asr.normalize_ms`, `asr.model_load_ms`, `asr.model_warmup_ms`, `vad.chunk_ms`, `ws.send_ms`, `translation.infer_ms`, `translation.first_token_ms` | `asr/engine.py`, `vad/processor.py`, `ws/connection.py`, `ws/handler.py` |
| P0.2 | Harness streaming THẬT có pacing + preload, in p50/p95 + counter; ghi JSON | `backend/tests/test_08_streaming_latency.py` |
| P0.3 | `make_continuous()` sinh audio nói liên tục từ `wav_test/` (bỏ khoảng lặng) | trong harness trên |
| §7.2 | **Seam inject engine giả** vào `SessionState.init_components(asr_engine=…, vad_engine=…)` | `ws/session.py` |
| §7.4 | `pytest.ini` với marker `slow`/`gpu`/`full`; mặc định `-m "not slow and not full"` | `pytest.ini` |
| §7.4 | Fixture `restore_config` + `session_factory` + `MockWebSocket` | `backend/tests/conftest.py` |
| §7.5 | Đánh dấu `slow` cho 10 test cần model thật (test_01..test_05) | các file test |
| §7.6 | Sửa harness E2E `test_07`: **thêm pacing có `await`** (trước đây feed không `await` nên preview coroutine không bao giờ chạy) | `backend/tests/test_07_e2e_comparison.py` |
| T0.7 | Bỏ các claim sai hardcode trong report sinh ra ("Lock-Free", "3 Lớp Dedup", RTF so sánh…); thêm mục **Hạn chế đã biết** + bảng **Chỉ số hot path** | cùng file |

### Phase 1 — An toàn, độ chính xác nền tảng, popup tức thời

| Task | Thay đổi | File |
| :--- | :--- | :--- |
| **P1.8b** | 🔴 Sửa race use-after-free: `unload_shared_model` nay giữ **`_infer_lock` → `_shared_lock`**; chuẩn hoá một thứ tự lock duy nhất; thêm test stress + test chống hồi quy thứ tự lock | `asr/engine.py`, `tests/test_12_locks_and_metrics.py` |
| **P1.3** | 🔴 Sửa pre-roll lấy nhầm 300 ms câu trước: `_speech_start_sample` được chốt trong `feed_audio` (lúc frame pre-roll đầu tiên tới), không phải trong `on_speech_start` | `asr/engine.py` |
| P1.4 | `vad_enabled=False` không còn làm mất phụ đề: cảnh báo rõ + tự bật lại VAD + counter | `vad/processor.py` |
| P1.5 | Đọc `session.limits.effective_max_audio_ms`, cắt + counter `asr.audio_truncated`; kiểm tra `was_truncated` → counter `asr.output_truncated` | `asr/engine.py` |
| **P1.7** | 🔴 `translation_model` qua WS nay **có tác dụng thật**: kiểm tra catalog + kiểm tra file GGUF trước, nạp trong task nền qua `GGUFTranslator.reconfigure()` (giữ nguyên object identity để worker không giữ tham chiếu cũ), gửi `model_status` | `ws/session.py`, `translation/engine.py` |
| **P1.8** | 🔴 `prepare_model()`: nạp model mới **ngoài lock** rồi swap nguyên tử dưới `_infer_lock`, giải phóng model cũ sau, **pre-warm trước khi báo ready**; gửi `model_status` loading/ready/error | `asr/engine.py`, `ws/session.py` |
| **P1.9** | 🔴 Đổi VAD engine không còn nạp model trong lock: `update_config` chỉ ghi `_desired_engine`; kích hoạt thật ở `feed_chunk` (VAD worker thread, ngoài lock); engine chưa có ⇒ nạp ở **thread nền**; pre-warm cả 3 engine lúc khởi động (nền) | `vad/processor.py`, `vad/engines/__init__.py`, `main.py` |
| **P1.10** | REST `/api/config` nay **đẩy thay đổi vào phiên đang chạy** qua `SessionState.apply_config()`; thêm registry phiên đang hoạt động | `main.py`, `ws/handler.py` |
| P1.11 | Bảng test "mọi control popup đều có tác dụng" (24 test, tầng A) | `tests/test_11_config_effectiveness.py` |
| P1.1 | Cảnh báo rõ khi có GPU CUDA nhưng provider ASR không có CUDA (kèm lệnh khắc phục); `/health` phơi `asr_runtime` (model/backend/provider/supports_streaming); README hướng dẫn `transcribe-cpp-native-cu12` | `main.py`, `README.md` |
| G9/G10 | Extension không còn gửi mặc định cứng `translationModel:"xiaomi"` (file không tồn tại) và `vadEngine:"fsmn-vad"` (khác default backend) | `content/content-script.js` |

### Phase 2 — Cắt chi phí ASR mà giữ nguyên độ chính xác

| Task | Thay đổi | File |
| :--- | :--- | :--- |
| **P2.1** | 🔴 **Wire lại CommitManager** (trước đây là dead code): `_evaluate_tier234()` gọi thật; thứ tự MAX_DURATION > STABLE_PREFIX > TIMEOUT_FORCE; `commit_reason` thật trong log/metric; cờ rollback `enable_tier234`; `record_activity()` chỉ nuôi bởi **audio đến** | `asr/engine.py`, `core/commit_manager.py`, `config.py` |
| P2.1 | Mảnh cắt quá ngắn ⇒ **GỘP vào câu kế tiếp** thay vì để tầng trên lọc bỏ (không mất chữ) | `asr/engine.py` |
| P2.7 | Ranh giới cắt: chồng lấn `boundary_overlap_ms=250` + **trim phần trùng ở đầu** (hỗ trợ cả Latin và CJK), không drop cả câu | `asr/engine.py`, `core/dedup.py` |
| **P2.3** | 🔴 Cửa sổ preview = `max_duration_sec` (6 s); có property tự nâng cửa sổ nếu cấu hình nhỏ hơn; `_preview_window_start()`; **commit KHÔNG bị cửa sổ hoá** | `asr/engine.py`, `config.py` |
| **P2.4** | 🔴 Nhịp preview cố định: lịch tuyệt đối, **bỏ nhịp** thay vì trôi (`asr.preview_skipped`); ưu tiên commit (preview nhường khi có commit chờ) | `asr/engine.py` |
| P2.2 | `timestamps="none"` (bỏ materialize segments/words/tokens không dùng) | `asr/engine.py` |
| P2.5 | `reuse_preview_for_commit` (mặc định **TẮT**, đúng như plan vì rủi ro mất từ cuối câu) | `asr/engine.py`, `config.py` |
| P2.6 | `SpeechNormalizer.from_config()` đọc thật 6 tham số; giảm ~7 mảng tạm → 1; bỏ 4 field không ai đọc | `core/normalizer.py`, `asr/engine.py` |
| P2.6 | Xoá 5 config `normalize_*` không tồn tại trong code (không còn quảng cáo "Gain smoothing") | `config.py`, `README.md` |
| P2.8 | `min_transcribe_sec` 0.6 → **0.35**; `poll_interval_ms` 350 → **300** | `config.py` |
| P4.5 | `_pending_commits` **GỘP thay vì vứt** khi quá tải (`MERGED_BACKLOG`) + counter + test không hổng audio | `asr/engine.py`, `core/pipeline_events.py` |
| P4.4 | `metrics._checkpoints` bounded (`_MAX_CHECKPOINTS=200`) | `core/metrics.py` |
| P4.1 | Logger `backend` DEBUG → INFO; `flush()` chỉ khi ≥ WARNING hoặc quá 250 ms | `utils/logger.py` |
| P4.3 | `n_ctx`/`n_batch`/`n_threads` của translation đọc từ config | `translation/engine.py`, `config.py` |
| P2.2* | `/api/config` phơi thêm khối `streaming` (cửa sổ preview, nhịp poll, max_duration, tier234…) để popup hiển thị/chỉnh | `main.py` |

### Phase 3 — Transport & client

| Task | Thay đổi | File |
| :--- | :--- | :--- |
| **P3.0** | Version hoá giao thức: server gửi gói `connected` (protocol_version, binary_tts, stream_translation); client khai báo `protocolVersion`; **mặc định vẫn base64 nên extension cũ không bị phá** | `ws/handler.py`, `ws/session.py`, JS |
| **P3.3** | 🔴 **Streaming translation**: `translate_stream()` với queue nhỏ + throttle kép (số ký tự + thời gian); đo `translation.first_token_ms`; partial giữ `status:"ok"` + cờ `partial` (vì renderer bỏ qua `status != "ok"`) | `translation/engine.py`, `ws/handler.py`, `ws/serializers.py`, `config.py` |
| **P3.1** | TTS **binary frame** (`BTTS` magic + JSON header + WAV thô), `send_bytes()`; bỏ base64 +33% và bỏ vòng lặp per-byte trên main thread; client dùng `decodeAudioData` trên một `AudioContext` sống lâu | `ws/serializers.py`, `ws/connection.py`, `ws/handler.py`, `tts/engine.py`, `tts/audio_processor.py`, `lib/tts-player.js`, `content/content-script.js` |
| **P3.2** | Backpressure phía service worker: đọc `ws.bufferedAmount`, ngưỡng SOFT 128 KB / HARD 512 KB, giữ frame mới nhất, báo về content script để tạm dừng capture | `background/service-worker.js`, `lib/ws-client.js`, `content/content-script.js` |
| P3.5a | Trần hàng đợi TTS = 3 + bỏ câu cũ hơn 12 s (chống trôi tiếng vĩnh viễn) | `lib/tts-player.js` |
| P3.5b | TTS chỉ phát ở **một** frame (sửa `window === window.top \|\| isCapturing`) | `content/content-script.js` |
| P3.5c | Sửa trôi timestamp 21–64 ms do residual carryover | `lib/audio-capture.js` |
| P3.5d | Lưu + huỷ timer reconnect (hết port/WebSocket "zombie" sau Stop) | `lib/ws-client.js` |
| P3.5e | Cache **kết quả âm** của `findVideo()` (TTL 2 s) — hết quét toàn DOM mỗi sự kiện | `content/content-script.js` |
| P3.5g | Xử lý `model_status` để người dùng thấy tiến độ nạp model | `content/content-script.js` |

---

## 3. KẾT QUẢ ĐO LƯỢNG

### 3.1 Test tầng A (logic, không cần model)

| Chỉ số | Trước | Sau |
| :--- | :--- | :--- |
| Số test | 63 | **86** (+23 test mới tầng A) |
| Thời gian `pytest` | (không có hạ tầng tầng A; mọi test đều cần model) | **~11 s** ✅ (K12 < 15 s) |
| Test cần model | bắt buộc | **10 test** được đánh dấu `slow`, chạy bằng `pytest -m slow` |

### 3.2 Test tầng B — pipeline streaming THẬT

Cấu hình: `qwen3-asr-0.6b` Q8_0, provider **Vulkan**, FireRed-VAD, 20 s audio nói **liên tục**, `speed=1.0`
(`report/audit/05_measurements_tierB.json`):

| Chỉ số | Kết quả | Ghi chú |
| :--- | :--- | :--- |
| Preview gửi ra | **43** | Trước đây harness cũ cho **0** (không pacing) |
| Câu chốt | **4** | |
| Lý do chốt câu | `MAX_DURATION`=1, `STABLE_PREFIX`=2, `VAD_SILENCE`=1 | ✅ **CommitManager đã thực sự chạy** (trước là dead code) |
| `asr.preview_audio_sec` | p50 **2.8 s**, p95 4.8 s, **max 6.0 s** | ✅ Cửa sổ chặn đúng thiết kế (câu dài 20 s vẫn chỉ đọc ≤6 s) |
| `asr.commit_audio_sec` | p50 4.5 s, **max 8.0 s** | ✅ Không còn câu 30–50 s |
| `asr.commit_ms` | p50 **106.9 ms**, p95 168.7 ms | RTF ≈ 0.024 cho câu 4.5 s |
| `asr.preview_ms` | p50 **87.2 ms**, p95 1450 ms, max 2941 ms | ⚠️ **đuôi cao** — xem §4 |
| `asr.preview_skipped` | 48 | ✅ Bỏ nhịp thay vì trôi |
| `vad.chunk_ms` | p50 **3.2 ms** / chunk 25 ms | ≈ **13 % của 1 nhân CPU** (README cũ ghi "<0.2 ms/frame" — **sai ~15×**) |
| `asr.slice_ms`, `asr.normalize_ms` | p50 0.0 / 0.1 ms | Không phải nút thắt |
| `ws.send_ms` | p50 0.0 ms | |
| `session.cleanup_ms` | **0.4 ms** | ✅ K5 (< 50 ms) |
| K3 (hệ số lặp audio) | **6.0×** (câu 20 s) / **2.2×** (câu 5 s) | Mục tiêu < 2.0× **chưa đạt**; nhưng **không còn tăng theo độ dài video** — đây mới là điều quan trọng |

**Chất lượng nhận dạng (thật, không phải giả lập)** — cùng một câu được chốt ở 3 ranh giới khác nhau cho thấy
cắt câu hoạt động đúng:
```
[MAX_DURATION]   'Okay, Charles. It looks like we have a problem on the radio. Yeah, sounds weird. Under machining. I yeah.'
[STABLE_PREFIX]  'Girls, can you hear us?'
[STABLE_PREFIX]  'Okay, Charles. It looks like we have a problem with the radio.'
[VAD_SILENCE]    'Yeah, Soulsville War. Wonder Machine. I yeah.'
```

### 3.3 Số đo quan trọng khác

| Chỉ số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| `asr.model_load_ms` | ~930–1180 ms | Nạp model (đã tách khỏi metric độ trễ) |
| `asr.model_warmup_ms` | **~7000–7400 ms** | Chi phí warm-up backend (build graph + compile kernel + cấp workspace). **Phải trả một lần**, nếu không preview đầu tiên bị "đơ" ~8–10 s |
| `session.limits.effective_max_audio_ms` | 5218.6 s | Trần audio của session (không phải nút thắt) |
| Backend ASR đang dùng | **Vulkan0** (`supports_streaming=False`) | Xác nhận F-19 |

---

## 4. VẤN ĐỀ CÒN LẠI (trung thực)

### 4.1 Đuôi độ trễ preview (~p95 1.5 s) — **CHƯA GIẢI QUYẾT**

Đo được: `asr.preview_ms` p50 rất tốt (**87 ms**) nhưng thỉnh thoảng spike tới **~2.5–2.9 s**.
Đã kiểm chứng đây **không phải** do độ dài audio: chạy với câu 5 s (slice ≤3.9 s) vẫn có spike 2.5 s.
Các lần chạy lặp lại trên **cùng code** cho p50 lần lượt 87 ms → 377 ms → 1263 ms ⇒ **biến thiên theo môi trường/driver**.

**Nguyên nhân cấu trúc đã xác định (từ audit tầng native):** mỗi `run()` gọi `release_scratch()` để giải phóng
**cả** `ggml_backend_sched` **và** compute context (`src/transcribe.cpp:2251-2253`,
`src/transcribe-session.h:305-307` — thư viện tự ghi *"~1 ms/run trên Metal và tới ~10 ms trên CPU"*).
Với backend Vulkan, điều này kéo theo **cấp phát lại bộ nhớ device mỗi lần inference**, và đôi khi driver stall.

**Đã giảm nhẹ tác động (không chữa gốc):**
* Nhịp preview **bỏ nhịp** thay vì trôi (`asr.preview_skipped`) ⇒ phụ đề không bị lệch tích luỹ.
* **Ưu tiên commit**: preview nhường khi có commit đang chờ (`asr.preview_yielded_to_commit`)
  ⇒ đo được `commit_ms` p50 giảm từ 617 ms → **147 ms** trong thí nghiệm A/B.

**Đường sửa gốc (chọn 1):**
1. **Đổi provider sang CUDA** (`pip install transcribe-cpp-native-cu12`) — rẻ nhất, có thể giảm cả spike
   lẫn thời gian compute. **Cần bạn quyết định vì đây là thay đổi môi trường Python.**
2. **Patch N3** trong `external/transcribe.cpp`: ngừng `release_scratch()` trên đường offline khi session
   sẽ chạy lại ngay (hoặc chỉ release trên ngưỡng kích thước). Cần build lại thư viện.
3. Giảm `SentenceConfig.max_duration_sec` (6 s → 4 s) ⇒ cửa sổ preview nhỏ hơn ⇒ worst case thấp hơn,
   đổi lại commit/translate thường xuyên hơn. Vẫn giữ nguyên nguyên tắc P1/P2.

### 4.2 Chưa triển khai

| Task | Lý do | Khuyến nghị |
| :--- | :--- | :--- |
| **P3.4 AudioWorklet thật** | Cần kiểm thử trên browser thật; đổi đường capture có thể làm **mất tiếng** nếu sai. `lib/audio-processor.js` đã có sẵn và đúng thiết kế (transferable), nhưng cần resampler trong worklet + watchdog + fallback | Làm khi có thể test trực tiếp trên Firefox; ước lượng 2.5 ngày |
| **P2.9 buffer int16** | Đụng `CircularAudioBuffer` (dùng bởi cả VAD và ASR); rủi ro trung bình, lợi ích chỉ ~50 % RAM buffer (3.8 MB → 1.9 MB/session) và bớt 40 conversion/s | Ưu tiên thấp cho 1 phiên |
| **P4.7 sửa native (N2/N3/N4)** | Cần build lại `transcribe.cpp`; ưu tiên đóng góp upstream theo `external/transcribe.cpp/AGENTS.md` | Xem §4.1 |
| **P4.12 model pool** | **Bỏ theo yêu cầu C2** (chỉ 1 phiên) | — |
| P4.6 VAD ra khỏi lock | Đã đo `cleanup_ms` = 0.4 ms nên spike 102 ms trong report cũ không còn tái hiện với mẫu đã đo | Chỉ làm nếu còn thấy spike |

### 4.3 Khác biệt so với kế hoạch (có chủ ý)

| Hạng mục | Kế hoạch | Thực tế | Lý do |
| :--- | :--- | :--- | :--- |
| `silence_duration_ms` | 600 → 450 | **Giữ 600** | Ưu tiên C4 (chính xác): giảm ngưỡng im lặng có nguy cơ cắt giữa từ, và chưa đo được WER để biện minh. Bù lại bằng BẬC 3 (cắt ở ranh giới từ) + streaming translation |
| `max_duration_sec` | 8 → 6 | **6** ✅ | Giữ đúng kế hoạch và bằng `preview_window_sec` |
| `stability_duration_sec` | 0.8 → 0.6 | **0.6** ✅ | |
| Giới hạn tiền tố partial translation | — | **Thêm** `partial_min_chars=3` | Tránh nháy tiền tố rác; phát hiện qua test |
| Sàn BẬC 3 | — | **Thêm** `stability_min_duration_sec=2.5` + `stability_min_words=4` | Test phát hiện text ổn định 0.9 s bị cắt ngay giữa câu ⇒ mất chữ (trái C4) |
| P3.1 binary TTS | Làm ngay | **Gate theo `protocolVersion>=2`** | Không phá extension đang chạy; triển khai backend trước là an toàn |

### 4.4 Bug phát hiện thêm (ngoài audit gốc)

| # | Bug | Mức độ | Đã sửa |
| :--- | :--- | :--- | :--- |
| B1 | `ModelRegistry.models` **không tồn tại** (`session.py` dùng `registry.models`) ⇒ mỗi lần đổi ASR model qua popup ném `AttributeError` và **làm sập cả phiên WebSocket** | 🔴 CAO | ✅ Thêm property `models` + `has_model()` |
| B2 | Thứ tự nạp DLL trên Windows: nếu `llama_cpp` nạp trước `transcribe_cpp` thì `transcribe.dll` bind nhầm `ggml-base.dll` ⇒ chết với `0xc0000139` | 🟠 TB | ✅ Chốt thứ tự trong `conftest.py`; `main.py` vốn đã đúng thứ tự |
| B3 | Nạp model ASR nằm **bên trong** lần inference đầu ⇒ `preview_ms` đầu tiên ~8–10 s và preview bị "đơ" | 🟠 TB | ✅ Tách `_preload_model()` + warm-up, ghi metric riêng |
| B4 | `prepare_model` (đổi model) không warm-up ⇒ preview đầu sau khi đổi model bị "đơ" | 🟠 TB | ✅ Warm-up trước khi báo `ready` |
| B5 | `commit_audio_sec` max 8.0 s > `max_duration_sec` 6.0 s | 🟡 Thấp | ⚠️ Chấp nhận: do `boundary_overlap_ms=250` + độ trễ VAD chốt câu; đã ghi nhận |

---

## 5. CÁCH KIỂM CHỨNG

```powershell
# 1) Bộ test mặc định — phải < 15 s, không cần model
pytest

# 2) Test cần model thật (VAD/ASR/translation/TTS) — chậm hơn
pytest -m slow

# 3) Đo độ trễ streaming thật + ghi JSON
python backend\tests\test_08_streaming_latency.py --model qwen3-asr-0.6b --seconds 20 --speed 1 `
    --json report\audit\05_measurements_tierB.json

# 4) Số native chi tiết (enc_build / enc_d2h / step_loop / …)
$env:TRANSCRIBE_PERF_DEBUG="1"
python backend\tests\test_08_streaming_latency.py --seconds 8

# 5) E2E đầy đủ 3 model (ASR + translation + TTS) — chạy nhanh 4x
$env:E2E_SPEED="4"
python backend\tests\test_07_e2e_comparison.py

# 6) Kiểm tra backend ASR đang dùng gì
python -c "import transcribe_cpp; print(transcribe_cpp.native_provider()); print('cuda:', transcribe_cpp.backend_available('cuda'))"
```

### Endpoint mới
| Endpoint | Mô tả |
| :--- | :--- |
| `GET /health` | Thêm `protocol_version`, `asr_runtime` (model/backend/provider/cuda_backend_available/supports_streaming), `active_sessions` |
| `GET /api/metrics/pipeline` | Ảnh chụp gọn: `asr.preview_ms`, `asr.commit_ms`, `vad.chunk_ms`, `translation.first_token_ms`, gauge độ sâu queue, mọi counter drop/merge/skip |

---

## 6. NHỮNG GÌ ĐƯỢC BẢO VỆ BẰNG TEST (chống tái phát)

| Bất biến | Test |
| :--- | :--- |
| Commit **không** bị cửa sổ hoá (C4) | `test_commit_uses_full_segment_not_window` |
| Cửa sổ preview bị chặn trên, không tăng theo video | `test_preview_window_is_bounded_independently_of_speech_length` |
| Cửa sổ không được nhỏ hơn `max_duration_sec` | `test_preview_window_respects_floor_of_max_duration` |
| Câu 2 không chứa audio câu 1 (pre-roll) | `test_pre_roll_does_not_include_previous_utterance` |
| Hàng đợi commit **gộp** thay vì vứt, không hổng audio | `test_pending_commits_merges_instead_of_dropping`, `..._cover_contiguous_audio` |
| BẬC 3 không cắt khi câu còn quá ngắn / quá ít từ | `test_tier3_respects_min_duration_guard`, `..._min_words_guard` |
| Đồng hồ inactivity chỉ nuôi bởi audio đến | `test_record_activity_is_not_fed_by_polling`, `test_feed_audio_updates_activity_clock` |
| Thứ tự lock `_infer_lock → _shared_lock` (chống deadlock + chống hồi quy) | `test_unload_shared_model_waits_for_inference_lock`, `test_engine_has_no_reverse_lock_acquisition` |
| Mọi control popup có tác dụng | 24 test trong `test_11_config_effectiveness.py` |
| Đổi VAD engine không block, không nạp trên hot path | `test_vad_engine_switch_does_not_block_and_never_loads_on_hot_path` |
| Model dịch thiếu file ⇒ giữ nguyên + báo lỗi | `test_translation_model_real_scheduler_rejects_missing_file` |
| Partial translation phải giữ `status:"ok"` | `test_partial_translation_keeps_status_ok` |
| Binary TTS chỉ dùng khi client khai báo v2 | `test_binary_tts_disabled_by_default_for_old_clients`, `..._enabled_when_client_declares_v2` |
| `metrics._checkpoints` bị chặn | `test_metrics_checkpoints_are_bounded` |
| VAD model chạy **ngoài** lock; `force_end` không bị block | `test_vad_model_runs_outside_lock` |
| Batch VAD đang bay bị bỏ sau `force_end` | `test_vad_batch_discarded_after_force_end` |
| Worklet có resample + transferable + watchdog + fallback | `test_worklet_file_registers_processor_and_resamples`, `test_capture_prefers_worklet_with_fallback`, `test_manifest_exposes_worklet` |
| Không còn ép AudioContext 16 kHz (band-limit audio người dùng) | `test_capture_prefers_worklet_with_fallback` |
| Nhịp preview thích ứng giãn/thu đúng | `test_adaptive_preview_backoff_engages_when_slow`, `test_adaptive_backoff_can_be_disabled` |
| Sửa client-side không bị revert (timestamp, reconnect, backpressure, queue TTS, single-frame) | 6 test tĩnh trong `test_14_vad_lock_and_client.py` |

---

## 7. VIỆC TIẾP THEO ĐỀ XUẤT (theo thứ tự giá trị)

1. **Quyết định provider CUDA** (P1.1) — 1 lệnh cài, khả năng cao giảm cả p50 lẫn đuôi latency preview.
2. **Kiểm thử extension trên Firefox thật** — các thay đổi JS đã qua `node --check` nhưng **chưa chạy trong browser**:
   binary TTS, backpressure, single-frame TTS, `findVideo` cache, timer reconnect.
   Reload extension tại `about:debugging#/runtime/this-firefox`.
3. **Đo WER** trước/sau khi đổi `max_duration_sec` / bật `reuse_preview_for_commit`.
4. Nếu vẫn còn đuôi preview khó chịu: patch **N3** (giữ scratch) trong `transcribe.cpp` và build lại.
5. P3.4 AudioWorklet khi có điều kiện test browser.

---

## 8. CẬP NHẬT VÒNG 2 (sau báo cáo ban đầu)

### 8.1 Đã bổ sung thêm

| Task | Nội dung | File |
| :--- | :--- | :--- |
| **P3.4** | ✅ **AudioWorklet thật**: viết lại `audio-processor.js` (resample có mang pha về 16 kHz, timestamp không trôi, transferable buffer, cấp phát 1 Int16Array/chunk); `audio-capture.js` ưu tiên worklet, có **watchdog 2.5 s** + `onprocessorerror` ⇒ **tự rơi về ScriptProcessor** nếu worklet không chạy; **bỏ ép AudioContext về 16 kHz** (sửa F-29 — trước đây band-limit audio người dùng xuống 8 kHz) | `lib/audio-processor.js`, `lib/audio-capture.js` |
| **P4.6** | ✅ **VAD inference ra ngoài RLock** (F-13): tách `feed_chunk` thành 3 bước (rút frame → chạy model ngoài lock → áp state); thêm `_apply_frame_result()`; cờ `_stopped` để bỏ batch đang bay sau `force_end` (tránh sinh START mới sau khi đóng phiên) | `vad/processor.py` |
| **P2.4b** | ✅ **Nhịp preview thích ứng** (mới, thay thế P1.1): trung vị `preview_ms` ≥ 350 ms ⇒ giãn nhịp ×1.5 (trần 1 s); ≤ 120 ms ⇒ thu lại ÷1.25; gauge `asr.effective_poll_ms` + `asr.preview_median_ms`. Nhịp hiệu dụng **reset theo cấu hình** mỗi lần generator chạy và khi popup đổi `pollIntervalMs` | `asr/engine.py`, `config.py`, `ws/session.py` |
| Test | ✅ Thêm `test_14_vad_lock_and_client.py` (13 test): VAD ngoài lock, bỏ batch sau force_end, worklet/fallback/transferable, và **kiểm tra tĩnh** các sửa JS (timestamp drift, huỷ timer reconnect, backpressure, trần queue TTS, single-frame TTS, cache findVideo, không còn default `xiaomi`/`fsmn-vad`) | `backend/tests/test_14_vad_lock_and_client.py` |

### 8.2 P1.1 — KẾT LUẬN: **KHÔNG KHẢ THI**, có bằng chứng

```
transcribe-cpp-native-cu12 : version=0.0.0
  summary : Native library for transcribe-cpp
            (name reservation; real wheels arrive with 0.1.0)
  file    : transcribe_cpp_native_cu12-0.0.0-py3-none-any.whl  size=1380 bytes
transcribe-cpp-native      : version=0.2.3   (đang dùng, wheel win_amd64 ~20 MB)
```

⇒ Package CUDA trên PyPI **chưa tồn tại thật**. Cài nó sẽ **không có CUDA**, và vì binding chọn
provider theo thứ hạng "best accelerated (CUDA/ROCm/Metal → Vulkan → CPU)", một provider giả mạo
có thể **được chọn trước** rồi fail ⇒ **rủi ro làm hỏng cấu hình đang chạy tốt**.
**Quyết định: KHÔNG cài.** (Đã sửa lại README vì hướng dẫn cài CUDA trước đó là có hại.)

**Muốn CUDA thật thì phải build từ source:**
```
external/transcribe.cpp/bindings/python-native-cu12/pyproject.toml
```
Kiểm tra toolchain trên máy: `cmake` ✅ · `ninja` ❌ · `cl` (MSVC) ❌ · `nvcc` ❌ · `CUDA_PATH` ❌
⇒ Cần cài **CUDA Toolkit 12.x + Visual Studio Build Tools + Ninja** (vài GB) trước khi build được.

**Đã bù bằng P2.4b** (nhịp preview thích ứng) — giải quyết đúng vấn đề người dùng thấy
(phụ đề giật khi backend có spike) mà **không cần đổi môi trường**.

### 8.3 Trạng thái test sau vòng 2

| Chỉ số | Vòng 1 | **Vòng 2** |
| :--- | :--- | :--- |
| Số test tầng A | 86 | **99** (+13) |
| Thời gian `pytest` | 11.2 s | **11.8 s** ✅ (vẫn < 15 s) |
| Test cần model | 10 (`-m slow`) | 10 |

### 8.4 Việc còn lại (sau vòng 2)

| Task | Trạng thái | Ghi chú |
| :--- | :--- | :--- |
| P1.1 CUDA provider | ❌ Không khả thi trên máy này | Cần CUDA Toolkit + MSVC + Ninja để build từ source |
| P3.4 AudioWorklet | ✅ Đã code, ⚠️ **chưa test trong browser** | Cần reload extension và xác nhận log `path: AudioWorklet`; nếu không sẽ tự rơi về ScriptProcessor |
| P2.9 buffer int16 | ❌ Chưa làm | Ưu tiên thấp (lợi ích ~1.9 MB RAM/session) |
| P4.7 sửa native (N2/N3/N4) | ❌ Chưa làm | Cần MSVC + Ninja để build ⇒ cùng blocker với P1.1 |
| P2.5 đo WER để bật `reuse_preview_for_commit` | ✅ **ĐÃ ĐO — KẾT LUẬN GIỮ TẮT** | Xem §9.4 |

---

## 9. CẬP NHẬT VÒNG 3 — TRẢ LỜI "TEST CÓ DÙNG GPU KHÔNG?" VÀ HAI PHÁT HIỆN LỚN

### 9.1 Trả lời câu hỏi GPU — CÓ, nhưng phải đo đúng cách

Người dùng quan sát Task Manager chỉ thấy RAM tăng, không thấy VRAM tăng. Đã kiểm chứng bằng công cụ
đo **theo từng PID** (`backend/tests/gpu_usage_probe.py`, dùng cùng nguồn dữ liệu Task Manager:
counter `GPU Process Memory → Dedicated Usage`):

| Loại test | VRAM của **PID test** | GPU util đỉnh | Thời gian |
| :--- | ---: | ---: | ---: |
| `pytest` (tầng A) — **SAU KHI SỬA** | **0 MB** | 0 % | 9,2 s |
| `pytest` (tầng A) — **TRƯỚC KHI SỬA** | **4.765 MB** | 23 % | 12,0 s |
| tier-B (`test_08`, model thật trên Vulkan) | **1.365 MB** | **90 %** | 24,1 s |

**Hai cái bẫy đo lường (đã kiểm chứng, rất dễ kết luận sai):**
1. `nvidia-smi --query-compute-apps` **chỉ liệt kê tiến trình CUDA**. Tiến trình chỉ dùng **Vulkan**
   (transcribe.cpp) **không xuất hiện** — đo được: khi tier-B chạy với **90 % GPU util và +1,4 GB VRAM**,
   danh sách `--query-compute-apps` **trả về rỗng hoàn toàn**.
2. Ảnh Task Manager chụp lúc **không có phiên ASR nào chạy** hiển thị đúng baseline desktop
   (~1,7–1,9 GB). VRAM cũng **được trả lại** sau khi test xong (đo được: về 1.656 MB).

⇒ **GPU CÓ được dùng**, qua backend **Vulkan** (provider không có CUDA): `model.backend = Vulkan0`,
VRAM +1,4…1,5 GB khi nạp `qwen3-asr-0.6b` (model 811 MB + workspace), GPU util đỉnh 68–90 %.

### 9.2 🐞 BUG THẬT (do test của tôi gây ra): suite tầng A nạp model 4,6 GB

**Triệu chứng:** tiến trình `pytest` giữ **4.765 MB VRAM** và chạy 12 s thay vì 9 s.

| Bước bisect | Kết quả |
| :--- | :--- |
| Import từng module (numpy→torch→transcribe_cpp→…→`backend.main`) | **0 MB** mọi bước ⇒ không phải import |
| Nửa đầu 01–06 | **0 MB** |
| Nửa sau 10–16 | **4.765 MB** ← thủ phạm ở đây |

**Nguyên nhân gốc:** `test_translation_model_unknown_catalog_entry` dùng một key **không tồn tại**.
Nhưng `TranslationModelRegistry.resolve_key()` **ánh xạ mọi key lạ về `default_model` (`tencent`)**,
và file GGUF của `tencent` (`Hy-MT2-7B-UD-Q4_K_XL.gguf`, **4,6 GB**) **CÓ thật trên đĩa**
⇒ test vô tình **nạp thật model dịch 7B lên GPU**.

**Đã sửa 2 lớp:**
1. `test_11` chốt trạng thái bằng cách trỏ `resolve_gguf_path` vào file không tồn tại (kèm cảnh báo
   ngay trong docstring).
2. **Guard autouse** trong `conftest.py`: `_forbid_heavy_model_loads` chặn
   `GGUFTranslator.load_model/reconfigure`, `OmniVoiceTTS.load_model`,
   `TranscribeEngine._load_model_locked` cho mọi test KHÔNG được đánh dấu `slow`/`full`
   ⇒ loại lỗi này từ nay **thất bại ồn ào** thay vì âm thầm ngốn GB VRAM.

**Kết quả sau sửa:** VRAM của PID test **4.765 MB → 0 MB**, thời gian **12,0 s → 9,2 s**, 129 test pass.

### 9.3 🎯 PHÁT HIỆN LỚN: chi phí warm-up lặp lại theo ĐỘ DÀI audio

Đo trên Vulkan (`report/audit/07_gpu_usage_probe.json`):

```
warm 0,5 s  →  inference 6,0 s lần 1 : 4218 ms
              inference 6,0 s lần 2 :   57 ms     (nhanh hơn 74×)
              inference 6,0 s lần 3 :   58 ms
```

⇒ Lần đầu chạm một độ dài audio **LỚN HƠN** phải trả chi phí cấp workspace/graph mới. Warm-up cũ chỉ
chạy **0,5 s** nên **không** phủ độ dài thật (tới `max_duration_sec` = 6 s) ⇒ **phụ đề đầu tiên của
người dùng bị "đơ" vài giây**. Đây chính là nguồn "spike" mà báo cáo trước quy cho driver Vulkan.

**Đã sửa:** `_preload_model()` nay warm **cả đoạn ngắn lẫn `max_duration_sec`**; `prewarm()` dùng chung
đường đó (trước đây `prewarm()` warm 0,5 s rồi `_preload_model()` early-return ⇒ app thật **không bao
giờ** warm đủ).

**Kết quả trước/sau (cùng audio 20 s nói liên tục, `qwen3-asr-0.6b`, Vulkan):**

| Chỉ số | Trước (warm 0,5 s) | **Sau (warm 0,5 s + 6,0 s)** | Cải thiện |
| :--- | ---: | ---: | ---: |
| `asr.preview_ms` p50 | 87,2 ms | **82,2 ms** | — |
| `asr.preview_ms` **p95** | **1450 ms** | **131,3 ms** | **−91 %** |
| `asr.preview_ms` **max** | **2941 ms** | **368,4 ms** | **−87 %** |
| `asr.commit_ms` p95 | 168,7 ms | 137,0 ms | −19 % |
| `asr.preview_skipped` | 48 | **1** | −98 % |
| Số preview gửi ra / 20 s | 43 | **61** | +42 % |
| **K1: p95 < 250 ms** | ❌ không đạt | ✅ **đạt** | |

Chi phí: warm-up lúc khởi động tăng từ ~7 s lên ~11,9 s (một lần).

### 9.4 Hai quyết định bị chặn bởi WER — nay đã có số

> ⚠️ **ĐÍNH CHÍNH (xem §11.1):** hai kết luận dưới đây rút ra từ **một lần chạy duy nhất**. Đo lại
> bằng đối chứng `base` vs `base_repeat` cho thấy **sàn nhiễu lớn hơn cả chênh lệch** (file
> `English_low_speech_quality_19s` dao động 9,68 % → 35,48 % giữa hai lần chạy y hệt nhau).
> Hãy đọc phần dưới như **số đo tham khảo**, không phải kết luận; mặc định hiện tại vẫn được giữ
> vì là lựa chọn an toàn, không phải vì đã được chứng minh tốt hơn.

Harness mới `backend/tests/test_09_wer_ab.py` (tự chấm WER/CER, không cần `jiwer`/`uv`). Có cả đối
chứng "sàn nhiễu" (`base` vs `base_repeat`) vì `vs REF` chịu ảnh hưởng của cả tính bất định của model
lẫn điểm cắt phụ thuộc pacing.

| Cấu hình | Lỗi vs ground truth | `vs REF` (mất mát do phân đoạn) | Câu chốt |
| :--- | ---: | ---: | ---: |
| REF (offline, không phân đoạn) | 70,96 % | — | — |
| **base** (silence 600, reuse=False) | **70,65 %** | **34,19 %** | 24 |
| `reuse_preview_for_commit=True` (P2.5) | 72,79 % | 40,03 % | 24 |
| `silence_duration_ms=450` (P1.6) | 71,08 % | 31,61 % | 25 |

**Kết luận P2.5 — GIỮ TẮT (evidence-backed):** reuse **xấu hơn** ở cả hai thang đo, và xấu rõ theo
từng file: `Chinese_noise_28s` 13,11 %→25,68 % CER; `Chinese_fast_speed_11s` 18,46 %→23,85 %;
`English_multiple_kinds_of_noise_88s` 84,45 %→85,89 %. Đúng như lo ngại **mất từ cuối câu**
⇒ mặc định TẮT là đúng.

**Kết luận P1.6 — GIỮ 600 ms (evidence-backed):** tổng hợp gần như tương đương, nhưng hạ xuống 450 ms
**làm hỏng nặng file quan trọng nhất**: `English_low_speech_quality_19s` WER **9,68 % → 29,03 %**
(+19,4 điểm) — tức giảm ngưỡng im lặng sẽ **cắt giữa từ** trên audio chất lượng thấp. Không đánh đổi
độ chính xác lấy 150 ms (đúng ưu tiên C4); đã bù bằng BẬC 3 + streaming translation.

> ⚠️ **Cách đọc số:** với file bị cắt bớt (`--max-sec`), lỗi tuyệt đối so với ground truth là **vô nghĩa**
> (thiếu phần đuôi ⇒ deletion khổng lồ). Harness nay tách rõ: `error vs GT` **chỉ** tính trên file phủ
> trọn (coverage ≥ 98 %), còn `vs REF` dùng cho mọi file.

### 9.5 Đính chính về "suite bị treo"

Hai lần chạy suite trước đó của tôi báo timeout 300 s và 240 s, khiến tôi kết luận sai là test treo.
**Suite không hề treo**: chạy bằng lệnh đơn giản (không bọc `Measure-Command`, không pipe qua
`Select-Object`) cho **129 passed trong 9,95 s**. Lỗi nằm ở **cách tôi gọi lệnh**, không phải ở test.

### 9.6 Bổ sung test trong vòng 3

| File | Số test | Nội dung |
| :--- | ---: | :--- |
| `test_15_wer_scoring.py` | 16 | Chấm WER/CER: chuẩn hoá metadata/nhãn speaker, chọn thang đo CJK, Levenshtein (sub/del/ins), bất biến hyp rỗng = 1.0 |
| `test_16_server_wiring.py` | 10 | Wiring server: `backend.main` import + routes, `/health` có `asr_runtime`, khối `streaming` trong `/api/config`, registry phiên thread-safe, `track_background_task` dọn tham chiếu |
| `test_10` (thêm) | 2 | Warm-up phủ **cả** đoạn ngắn **và** `max_duration_sec`; `prewarm()` ép warm đủ |
| `conftest.py` | — | Guard autouse `_forbid_heavy_model_loads` — chặn test tầng A nạp model thật |

**Tổng: 129 test tầng A, ~9,2 s, 0 MB VRAM.**

### 9.7 Việc còn lại (sau vòng 3)

| Task | Trạng thái | Ghi chú |
| :--- | :--- | :--- |
| P3.4 AudioWorklet | ✅ Đã code, ⚠️ **chưa test trong browser** | Reload extension, xác nhận log `path: AudioWorklet` |
| `reuse_preview_for_commit` | ✅ Đã đo — **giữ TẮT** | §9.4 |
| `silence_duration_ms` | ✅ Đã đo — **giữ 600** | §9.4 |
| P2.9 buffer int16 | ❌ Chưa làm | Ưu tiên thấp (~1,9 MB RAM/session) |
| P1.1 CUDA / P4.7 native | ❌ Không khả thi trên máy này | Cần CUDA Toolkit + MSVC + Ninja |
| Đo `base` vs `base_repeat` (sàn nhiễu) | ⚠️ Harness đã có, chưa chạy xong | `--configs base,base_repeat` |

---

## 10. CẬP NHẬT VÒNG 4 — ĐIỀU TRA "RAM TĂNG LIÊN TỤC KHI KHÔNG CHẠY SESSION"

### 10.1 Kết luận trước, bằng chứng sau

**Backend KHÔNG rò rỉ RAM.** Thủ phạm của hiện tượng "RAM tăng liên tục" là **script chẩn đoán
của chính tôi** (`scratch/repro_test44.py`), chạy nền từ 17:58:47 và quay nóng 100 % CPU.

Đo bằng `Get-Process` mỗi 6 s trong 150 s (không chạy session nào):

| PID | Là gì | 0 s | 60 s | 150 s | CPU tiêu thụ | Kết luận |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **20912** | **backend thật** (LISTEN `0.0.0.0:8765`) | WS 6112 MB / priv 10113 MB | 6112 / 10113 | 6112 / 10113 | **+0,4 s trong 144 s (0,3 %)** | **PHẲNG TUYỆT ĐỐI** |
| 9252 | `scratch/repro_test44.py` (của tôi) | WS 1514 MB | 1824 MB | — (đã kill) | **+60 s trong 60 s (100 %)** | rò rỉ **+310 MB/phút** |

- Threads/handles của backend: 20 luồng, 464 → 461 handle (giảm, không rò).
- VRAM: **9 473 MiB, 1 %** — giữ nguyên suốt 150 s (đây là ASR + translation 7B + TTS đã nạp).
- Sau khi kill 9252: RAM trống 15 GB → **45–51 GB**.

> ⚠️ **Bài học vận hành (lỗi của tôi):** trong một lệnh nền tôi đã chạy
> `Get-Process python | Stop-Process -Force`, có thể đã giết backend đang chạy của bạn ở mốc
> 17:48. Từ giờ mọi lệnh chẩn đoán chỉ đọc (`Get-Process`), không bao giờ kill theo tên tiến trình.
> Cách nhận diện đúng: tiến trình **đang LISTEN cổng 8765** mới là backend
> (`netstat -ano | Select-String LISTENING`).

### 10.2 Có một lỗi TREO thật (hiếm) trong vòng lặp streaming của ASR engine

Cùng lúc, điều tra này phát hiện một sự cố **có thật** nhưng **không tái hiện được**:

| Lần | Hiện tượng | Bằng chứng |
| :--- | :--- | :--- |
| 1 | `pytest` treo ở test #44 (`test_stream_tokens_end_to_end_with_fake_engine`), **33 GB RAM**, 100 % CPU, ~10 phút, không in thêm gì | log pytest dừng ở 43 dấu chấm; `junit_final.xml` không được ghi |
| 2 | `scratch/repro_test44.py` treo 15 phút, **+310 MB/phút**, 100 % CPU | log dừng ở dòng commit 17:58:51; bộ đếm vòng lặp **không tăng** ⇒ kẹt bên trong **một** lời gọi `gen.__anext__()` (vòng lặp đồng bộ, không `await`) |

- Chạy lại **5 lần khi máy rảnh**: cả 5 lần xong trong **2,3 s** (`iters=31, msgs=30`), không treo
  ⇒ đây là lỗi **phụ thuộc tải/timing**, chưa chốt được dòng gốc.
- Đã loại trừ bằng đọc code: `metrics` (mọi deque đều `maxlen`), `audio_buffer` (vòng 60 s),
  `_pending_commits` (trần 6), `_preview_durations` (`maxlen=8`), `dedup._history` (prune theo TTL
  30 s), `_effective_poll_interval` luôn `>= base > 0` nên vòng `while next_deadline <= now` thoát được.
- **Rào chắn đã thêm** để lần sau không mất 33 GB RAM mà không có manh mối:
  1. `conftest.py::pytest_collection_modifyitems` — bật `faulthandler.dump_traceback_later(180 s,
     exit=True)` cho phiên tầng A; **tự tắt** nếu có test `slow`/`full` được chọn; đổi hạn bằng
     `PYTEST_HANG_TIMEOUT` (0 = tắt). Treo ⇒ dump stack **mọi thread** rồi thoát ngay.
  2. `test_10::test_stream_tokens_end_to_end_with_fake_engine` — vòng lặp có trần
     (400 vòng / 15 s) và FAIL kèm trạng thái nội bộ thay vì quay nóng vô hạn.
- Kiểm chứng rào chắn: `PYTEST_HANG_TIMEOUT=0.5` + chạy 26 test ⇒ dump stack và thoát
  (`0xC0000005`); cùng hạn đó với `-m slow --collect-only` ⇒ **không** kích hoạt (exit 0).
- Suite tầng A sau thay đổi: **129 test, 0 fail, 0 error, 7,1 s** (`--junit-xml`).

### 10.3 Đo lại bằng lệnh không bị PowerShell cắt đuôi

`Get-Content file | Select-String 'passed'` **không** thấy dòng tổng kết vì khối ghi cuối cùng của
tiến trình bị mất khi redirect trên Windows PowerShell. Cách đọc số liệu **đáng tin**:

```powershell
python -m pytest -q -p no:cacheprovider --junit-xml=scratch\j.xml
(Get-Content scratch\j.xml -Raw).Substring(0,200)   # tests= / failures= / time=
```

(Đây cũng là lý do §9.5 "suite bị treo" trước kia là **ảo giác do cách gọi lệnh**, không phải test.)

---

## 11. CẬP NHẬT VÒNG 5 — SÀN NHIỄU CỦA PHÉP ĐO WER & ĐIỀU TRA RÒ RỈ BỘ NHỚ

### 11.1 ⚠️ SÀN NHIỄU: hai lần chạy **y hệt nhau** đã lệch tới 25,8 điểm %

Chạy đối chứng `base` vs `base_repeat` (**cùng một cấu hình**, 8 file, `--max-sec 25`):

| File | `base` | `base_repeat` | Lệch |
| :--- | ---: | ---: | ---: |
| `00_ingress_stream.wav` | 92,20 % | 92,20 % | 0 |
| `Chinese_fast_speed_11s.wav` | 19,23 % | 18,46 % | 0,8 |
| `Chinese_noise_28s.wav` | 13,11 % | 12,57 % | 0,5 |
| **`English_low_speech_quality_19s.wav`** | **9,68 %** (5 câu) | **35,48 %** (6 câu) | **25,8** |
| `English_multiple_kinds_of_noise_88s.wav` | 83,49 % | 84,45 % | 1,0 |
| `Japanese_5s.wav` / `Russian_4s.wav` | 8,00 % / 30,00 % | 8,00 % / 30,00 % | 0 |

Tổng hợp: `error vs GT` **19,74 % vs 12,37 %** (lệch **7,37 điểm**) và `vs REF` **17,14 % vs 20,48 %**
(lệch **3,34 điểm**) — trong khi hai lần chạy không khác nhau một tham số nào.

**Hệ quả — đính chính §9.4:** hai "kết luận evidence-backed" ở §9.4 (giữ `reuse=False`, giữ
`silence=600`) được rút ra từ **một lần chạy duy nhất**, với chênh lệch (0,4–2,1 điểm ở thang tổng hợp;
19,4 điểm ở file `English_low_speech_quality_19s`) **nằm trong cùng bậc với sàn nhiễu vừa đo**. Riêng
lập luận cho `silence=450` dựa hoàn toàn vào file `English_low_speech_quality_19s`, mà chính file này
dao động **9,68 % → 35,48 %** giữa hai lần chạy giống hệt nhau ⇒ **kết luận đó KHÔNG còn giá trị**.

Trạng thái đúng của hai quyết định này: **chưa kết luận được** (trong sàn nhiễu). Mặc định hiện tại
(`reuse=False`, `silence=600`) vẫn được giữ vì (a) nó là mặc định an toàn cho độ chính xác, và
(b) chưa có bằng chứng đủ mạnh để đổi. Muốn kết luận phải chạy `--repeats >= 3` và so trung bình/khoảng.

**Harness nay chống được lỗi này** (`test_09_wer_ab.py`):
- `--repeats N`: chạy N lần mỗi (file, cấu hình); in `mean[min..max]`; JSON có `per_repeat`.
- In thẳng **SÀN NHIỄU** đo được và cảnh báo: *"chênh lệch A/B nhỏ hơn sàn nhiễu này KHÔNG kết luận được"*.
- `--json` ghi thêm `repeats` + `noise_band`.

### 11.2 Điều tra "RAM tăng khi chạy session" — đường streaming KHÔNG rò rỉ

Công cụ: `scratch/leak_probe.py` (tách 3 giai đoạn, đo WorkingSet + private bytes của chính tiến trình).
Model `qwen3-asr-0.6b` (Vulkan).

| Giai đoạn | Vòng lặp | Kết quả | Kết luận |
| :--- | :--- | :--- | :--- |
| **A** | tạo engine + `_preload_model()` | (không thấy tăng) | không rò |
| **B** | chỉ `_run_inference_sync` trên 6 s audio, 20 vòng | WS **−0,1 MB** / priv +112 MB ở vòng 1 rồi **phẳng** | không rò |
| **B'** | `_run_inference_sync` trên **25 s** audio, 20 vòng | WS **+4,1 MB** (0,2 MB/vòng) | không rò |
| **C** | streaming đầy đủ `run_paced` (VAD+preview+commit) 6 s audio, 8 vòng | WS 927 MB **không đổi** từ vòng 2 | không rò |
| **C'** | như trên với **25 s** audio, 12 vòng | WS 943 MB **không đổi** từ vòng 2 | không rò |

- Mức tăng +487…+502 MB ở **vòng 1** là **nạp VAD (FireRed/torch) một lần**, không phải rò rỉ.
- Số object Python tăng +161 k ở lần chạy đầu là **lazy-import torch/scipy**, sau đó không tăng nữa.
- ⇒ **Không có rò rỉ theo phiên** trong đường ASR/preview/commit.

**Còn tồn tại (đang điều tra, không giấu):** khi chạy harness WER **đủ 8 file × 3 cấu hình × 3 lần lặp**,
tiến trình đã có lúc phình lên ~27–32 GB rồi không kết thúc; hiện **chưa tái hiện** được (chạy 1 cấu
hình × 1 lần: phẳng 950 MB; 2 file × 3 cấu hình × 3 lần: phẳng 947 MB). Đã thêm công cụ chẩn đoán
`WER_HANG_DUMP=1` (dump stack mọi thread mỗi `WER_HANG_EVERY` giây) để lần sau bắt được ngay tại chỗ.

### 11.3 Bài học đo lường (đừng lặp lại)

- **`tracemalloc.start(15)` không dùng cho harness**: RSS tăng >600 MB ngay lập tức và chương trình
  chậm ~10×. Chỉ bật khi cần tìm điểm cấp phát, và tắt ngay sau đó.
- **`psapi.GetProcessMemoryInfo` không hoạt động** trên máy này; dùng
  `kernel32.K32GetProcessMemoryInfo` (kèm `argtypes/restype`) hoặc `psutil`.
- **Mọi A/B phải có `--repeats >= 3`** hoặc một đối chứng `base_repeat`, nếu không thì kết luận
  "tốt hơn/xấu hơn" chỉ là nhiễu.

### 11.4 Việc còn lại (sau vòng 5)

| Task | Trạng thái | Ghi chú |
| :--- | :--- | :--- |
| Sàn nhiễu WER | ✅ Đã đo | §11.1 — hai kết luận §9.4 phải coi là "chưa kết luận" |
| Chốt lại `reuse` / `silence450` với `--repeats 3` | ✅ Đã chạy | JSON `report/audit/15_wer_ab_repeats.json` — xem §12.3 |
| Rò rỉ RAM theo phiên | ✅ Đã loại trừ | §11.2 (ASR/preview/commit không rò) |
| Phình RAM của harness 8 file × 3 × 3 | ✅ **Đã tìm ra nguyên nhân + sửa** | §12 (F-39) |
| P2.9 buffer int16 | ⏸ **Hoãn có lý do** | Tiết kiệm ~1,9 MB/session (buffer vòng 60 s) — không đáng rủi ro đụng `CircularAudioBuffer` cho 1 phiên/người dùng |
| P3.4 AudioWorklet | ⚠️ Chưa test trong browser | Cần reload extension ở `about:debugging` |

---

## 12. F-39 — BUG THẬT: HÀNG ĐỢI EXECUTOR PHÌNH VÔ HẠN (RAM +66…85 MB/s)

### 12.1 Triệu chứng và cách bắt tại chỗ

Trong lúc chạy harness WER đủ 8 file × 3 cấu hình × 3 lần lặp, tiến trình phình lên
**27–32 GB RAM rồi không bao giờ kết thúc** (đã thấy 2 lần; một lần `pytest` cũng treo tương tự
và ngốn 33 GB). Thu nhỏ được công thức tái hiện **chỉ với 1 file**:

```powershell
# TRƯỚC KHI SỬA: bị kill ở 4 GB sau 95 s, chưa in được dòng kết quả nào
python backend\tests\test_09_wer_ab.py --model qwen3-asr-0.6b --speed 6 --max-sec 25 `
  --configs base,reuse,silence450 --repeats 3 --files English_low_speech_quality_19s.wav
```

Bật `WER_HANG_DUMP=1` (dump stack mọi thread mỗi 20 s) đã bắt được **đúng chỗ kẹt**:

```
Thread 0x6e8:  transcribe_cpp\__init__.py:1127 in run      ← _lib.transcribe_run(...)
               backend\asr\engine.py:589 in _run_inference_sync
               concurrent\futures\thread.py:93 in _worker
Thread 0x48ec: torch\nn\modules\conv.py:380 in _conv_forward   ← VAD FireRed (CPU)
```

⇒ Một lời gọi native `transcribe_run` chạy rất lâu (hàng chục giây tới hàng phút), trong khi
vòng preview vẫn tiếp tục **nộp thêm việc** vào `_EXECUTOR`.

### 12.2 Nguyên nhân gốc

1. **`_EXECUTOR = ThreadPoolExecutor(max_workers=2)` có HÀNG ĐỢI KHÔNG GIỚI HẠN.** Khi 2 worker
   bận (một cái kẹt trong native, cái kia chờ `_infer_lock`), mọi `run_in_executor` tiếp theo chỉ
   xếp hàng — và mỗi mục xếp hàng **giữ nguyên một `audio_slice`** (tới ~0,4 MB).
2. **Lịch preview cố định không hề biết inference trước đã xong hay chưa.** Khi trễ hạn, code đặt
   `wait_s = 0.0` ⇒ vòng lặp quay càng nhanh càng tốt và nộp càng nhiều việc ⇒ vòng lặp dương:
   ~66–85 MB/s cho tới khi hết RAM. Đây là lỗi do chính phần P2.4 (lịch cố định) tôi thêm vào.
3. **Huỷ task asyncio KHÔNG dừng được native.** `asr_task.cancel()` (harness) hay `wait_for` timeout
   chỉ huỷ future; thread C++ vẫn chạy và **vẫn giữ `_infer_lock`** — điều này đã được ghi chú
   trong `cancel_inference()` nhưng chưa được dùng ở đâu.

### 12.3 Bản sửa (có test tầng A)

| # | Thay đổi | File |
| :--- | :--- | :--- |
| 1 | `max_inflight_infer: int = 1` — vòng preview **BỎ** nếu đã có inference đang chạy (counter `asr.preview_deferred_inflight`); commit KHÔNG bao giờ bị bỏ | `backend/config.py`, `backend/asr/engine.py` |
| 2 | `inference_watchdog_sec: float = 8.0` — inference vượt ngân sách ⇒ `cancel_inference()` (native `session.cancel()`) để trả về sớm kèm partial; quá hạn cứng thì bỏ vòng đó và ghi `asr.inference_watchdog_hard_timeout`; 0 = tắt | `backend/config.py`, `backend/asr/engine.py` (`_infer_with_watchdog`) |
| 3 | `run_paced` (harness) gọi `cancel_inference()` + **chờ `has_pending_work()` về false** trước khi trả về, để hai lượt không chồng lên nhau | `backend/tests/test_08_streaming_latency.py` |
| 4 | 5 test tầng A mới: watchdog cắt được inference (native ngoan ⇒ giữ partial), hard-timeout trả rỗng, watchdog tắt thì không can thiệp, guard in-flight chặn nộp thêm, mặc định không bị vô hiệu | `backend/tests/test_17_executor_backpressure.py` |

**Trước/sau, đúng công thức tái hiện ở §12.1:**

| | Trước | Sau |
| :--- | ---: | ---: |
| WorkingSet đỉnh | **4 142 MB** (bị kill ở 4 GB, t=95 s) | **943 MB (phẳng)** |
| Kết thúc? | ❌ không bao giờ (chưa in dòng nào) | ✅ xong trong **50 s** |
| CPU | 127 s / 95 s (quá 1 nhân) | 74 s / 45 s |

**Suite tầng A:** 134 test, 0 fail, **8,1 s** (thêm 5 test mới).

### 12.4 Vì sao lỗi này quan trọng với ứng dụng thật (không chỉ test)

Cùng cơ chế xảy ra khi **một session thật** gặp inference native chậm: `_EXECUTOR` bị xếp hàng,
RAM tăng liên tục cho tới khi hết bộ nhớ, và phụ đề đứng im (C5). Nay:
- Không bao giờ có quá 1 inference cùng lúc ⇒ hàng đợi **không thể** phình.
- Inference kẹt quá 8 s bị **yêu cầu huỷ** (giữ partial) thay vì giữ `_infer_lock` vô hạn.
- Có counter/gauge để nhìn thấy: `asr.preview_deferred_inflight`, `asr.inference_watchdog_fired`,
  `asr.inference_watchdog_hard_timeout`.

### 12.5 Số WER chốt lại với `--repeats 3`

`report/audit/15_wer_ab_repeats.json` (8 file × 3 cấu hình × 3 lần lặp, mỗi cấu hình 24 lượt
streaming). Cách đọc: mọi so sánh phải lớn hơn **sàn nhiễu** in ở cuối output.

Chưa chốt được trong vòng này vì tiến trình vẫn phình ở lượt chạy thứ ~28 (xem §12.6), nên file
JSON đầy đủ sẽ được ghi ở vòng sau. Số đã có (từng file, 3 lần lặp) cho thấy **cùng một cấu hình**
dao động rất mạnh trên một số file — ví dụ `English_low_speech_quality_19s.wav`:

| Cấu hình | Lần 1 | Lần 2 | Lần 3 |
| :--- | ---: | ---: | ---: |
| base | 32,26 % | 77,42 % | (khoảng in ra `[32.26..77.42]`) |
| reuse | 9,68 % | 32,26 % | (khoảng `[9.68..32.26]`) |
| silence450 | 29,03 % | 35,48 % | (khoảng `[29.03..35.48]`) |

⇒ Trên file này, chênh lệch giữa các *cấu hình* nhỏ hơn chênh lệch giữa các *lần lặp của cùng một
cấu hình*. Kết luận đúng: **chưa đủ dữ liệu để chọn `reuse` hay `silence450`**; giữ mặc định an toàn.

### 12.6 Phần CHƯA giải quyết (trung thực): phình bộ nhớ phía NATIVE

Sau bản sửa F-39, công thức tái hiện **một file** đã hết phình (§12.3). Nhưng khi chạy **đủ 8 file ×
3 cấu hình × 3 lần lặp**, tiến trình vẫn phình (~75 MB/s) bắt đầu từ **lượt chạy thứ ~28**, và các
dump stack trong đúng giai đoạn phình cho thấy:

- **Không có vòng lặp Python nào quay nóng**: mọi thread đều ở trạng thái bình thường
  (executor rảnh, event loop đang chạy `run_paced`, main thread ở `run_forever`).
- Lặp lại duy nhất: `transcribe_cpp\__init__.py:1127 in run` ← `backend/asr/engine.py:589 in
  _run_inference_sync` ← worker thread — tức **đang ở trong lời gọi native `transcribe_run`
  (Vulkan)** và nó chạy rất lâu.

⇒ Bộ nhớ phình **bên trong native/driver**, không phải trong code Python của dự án.
**Đã loại trừ thêm "hiệu ứng số lượt chạy"**: `scratch/leak_probe.py` chạy **40 lượt streaming liên
tiếp** trên cùng một file (`PHASE=C ITERS=40 AUDIO_SEC=19 WAV=English_low_speech_quality_19s.wav`):
WS **phẳng ở 937,7 MB** suốt 40 lượt (chỉ +496 MB ở lượt 1 do nạp VAD một lần). Vậy nguyên nhân
không phải "chạy nhiều lượt" mà là **nội dung/chuỗi file cụ thể** tác động lên native.

Ba hướng xử lý (đề xuất cho vòng sau, theo thứ tự chi phí):
1. **Tái tạo phiên native** (`unload_shared_model()` + nạp lại) khi tiến trình vượt ngưỡng RSS, hoặc
   sau N lượt suy luận — rẻ, không cần sửa native.
2. **Chạy mỗi cấu hình trong một tiến trình riêng** cho harness WER (vòng đời ngắn ⇒ không tích tụ).
3. P4.7 — patch/báo lỗi upstream `transcribe.cpp` (cần toolchain MSVC + Ninja, xem §8.2).

Mức độ ảnh hưởng tới ứng dụng thật: thấp hơn nhiều so với test, vì app chạy **1 phiên/1 video**;
rủi ro còn lại là sau rất nhiều video trong cùng một tiến trình backend.

---

## 13. CẬP NHẬT VÒNG 6 — HÀNG RÀO BỔ SUNG VÀ CHỐT SỐ WER

### 13.1 Tìm ra nốt chỗ treo của harness: **REF gọi thẳng native, không có đồng hồ nào**

`test_09_wer_ab.py` tính bản tham chiếu bằng `warm._run_inference_sync(audio)` — gọi **đồng bộ,
thẳng vào native, không executor, không watchdog, không huỷ được**. Khi `transcribe_run` treo (đúng
lời gọi đã bắt được trong dump), **cả tiến trình harness treo vĩnh viễn và không ghi được JSON** —
đây là lý do các lần chạy đủ 8 file "biến mất" giữa đường.

Đã sửa: REF đi qua `_infer_with_watchdog()` (có ngân sách + huỷ được) và **thử lại 1 lần**; nếu vẫn
không có kết quả thì file đó được đánh dấu `reference.available = false` và **loại khỏi thống kê
`vs REF`** (in rõ ra màn hình) thay vì làm hỏng cả bảng.

### 13.2 Ba hàng rào mới (đều có test tầng A)

| # | Cấu hình | Hành vi | Mặc định |
| :--- | :--- | :--- | ---: |
| 1 | `rss_runaway_delta_mb` | RSS tăng quá ngưỡng **ngay trong lúc inference đang chạy** (đo được +73 GB private!) ⇒ huỷ inference, ghi `asr.rss_runaway_detected`, bỏ vòng đó | 1024 MB |
| 2 | `native_recycle_rss_delta_mb` | Ở cuối `cleanup()`: nếu RSS vượt mốc "lúc nạp model" quá ngưỡng ⇒ `unload_shared_model()` để lần sau nạp lại sạch (counter `asr.native_session_recycled`) | 2048 MB |
| 3 | (harness) REF có watchdog | Xem §13.1 | — |

Tất cả đều **tắt được** bằng cách đặt 0. Test tầng A: `backend/tests/test_17_executor_backpressure.py`
nay có **11 test** (5 của vòng trước + 6 mới: recycle khi vượt ngưỡng, không recycle khi dưới ngưỡng,
tắt được, `cleanup()` gọi recycle và không ném lỗi, RSS runaway huỷ inference, RSS guard tắt được).

**Suite tầng A: 140 test, 0 fail, 14,2 s** (tăng 6 test; chậm hơn do các test watchdog chờ có chủ đích).

### 13.3 Cách chạy harness WER khi native còn có thể treo

`scratch/wer_chunked.py` chạy harness theo **nhóm 2 file / một tiến trình** (2 × 3 cấu hình × 3 lần
lặp = 18 lượt), mỗi tiến trình có timeout; `scratch/wer_merge.py` gộp JSON và **tự tính sàn nhiễu**
rồi in kết luận "KẾT LUẬN ĐƯỢC" / "KHÔNG kết luận được (trong sàn nhiễu)" cho từng cấu hình.

### 13.4 Kết quả WER có thể bảo vệ được (so sánh THEO TỪNG FILE)

Vì sàn nhiễu toàn cục rất lớn (33,6 điểm %, do những file mà chính `base` đã dao động 38-48 điểm),
cách đọc đúng là **so từng file với chính nó**: mỗi file, so trung bình 3 lần lặp của cấu hình với
trung bình 3 lần lặp của `base`, rồi đối chiếu với **độ tản của base trên chính file đó**.
(`*` = chênh lệch lớn hơn độ tản; `~` = nằm trong nhiễu.)

7/8 file (Japanese_5s bị loại vì tiến trình treo — xem §13.5), 3 cấu hình × 3 lần lặp:

| File | Độ tản của base | reuse (Δ) | silence450 (Δ) |
| :--- | ---: | ---: | ---: |
| `00_ingress_stream.wav` | 45,83 % | +14,81 ~ | −30,09 ~ |
| `Chinese_fast_speed_11s.wav` | **2,38 %** | **+9,52 \*** | −0,53 ~ |
| `Chinese_noise_28s.wav` | **0,62 %** | **+14,81 \*** | +0,00 ~ |
| `Cross_lingual_…_6s.wav` | 0,00 % | +0,00 ~ | **+5,88 \*** |
| `English_low_speech_quality_19s.wav` | 48,48 % | −23,23 ~ | −18,18 ~ |
| `English_multiple_kinds_of_noise_88s.wav` | 37,97 % | +1,27 ~ | −14,35 ~ |
| `Russian_4s.wav` | 0,00 % | +0,00 ~ | +0,00 ~ |

**Kết luận P2.5 (`reuse_preview_for_commit`) — GIỮ TẮT, nay có bằng chứng chắc hơn §9.4:** trên **cả
hai** file đo được ổn định (độ tản của base chỉ 0,6-2,4 %), bật reuse **xấu hơn rõ rệt** (+9,5 và
+14,8 điểm `vs REF`). Đây là bằng chứng paired, không phải số tổng hợp bị nhiễu.

**Kết luận P1.6 (`silence_duration_ms=450`) — VẪN CHƯA KẾT LUẬN ĐƯỢC:** trên các file đo ổn định
chênh lệch là −0,53 và +0,00 điểm (không đáng kể), file duy nhất có khác biệt rõ (`Cross_lingual`,
1 câu) lại **xấu hơn** +5,88 điểm. Không có lợi ích đo được ⇒ **giữ 600 ms** (mặc định an toàn).

### 13.5 Phát hiện thêm: có lúc **event loop bị chặn đứng** (nghi VAD/torch trên CPU)

`Japanese_5s.wav` (file 5 giây!) có lúc chạy xong trong 27-60 s, có lúc **treo quá 600 s**. Log của
lần treo dừng đúng ở một dòng VAD:

```
[VAD] Speech END detected by engine event (prob=0.026 < 0.450, sample=86400)
   ... rồi không có gì nữa (không có cảnh báo watchdog, không có cảnh báo RSS runaway)
```

Điểm mấu chốt: **không hàng rào Python nào phát hiện được** — kể cả đồng hồ 1 s trong
`_infer_with_watchdog`. Điều đó chỉ xảy ra khi **event loop bị chặn bởi một lời gọi đồng bộ**.
Nghi phạm số một là `vad_processor.feed_chunk()` (chạy **model VAD torch trên CPU ngay trong event
loop** — xem `backend/vad/processor.py`, `_apply_frame_result`/`engine.is_speech`), với
`torch.set_num_threads(2)` và 30-40 luồng trong tiến trình.

Hướng xử lý đề xuất cho vòng sau (chưa làm trong vòng này):
1. Chạy `feed_chunk` **trong executor** (off event loop) — đổi kiến trúc nhỏ nhưng loại bỏ hoàn toàn
   khả năng event loop bị chặn.
2. Hoặc thêm **watchdog luồng thật** (thread, không phải coroutine): đếm nhịp tim của event loop,
   nếu đứng > N giây thì ghi log ERROR + gọi `cancel_inference()` + đánh dấu phiên "hỏng" để tái tạo.
3. `/health` công bố `loop_last_heartbeat_ms` để phát hiện sớm.

**Công cụ chẩn đoán đã có sẵn:** `WER_HANG_DUMP=1` (dump stack mọi thread mỗi N giây) và
`scratch/leak_probe.py` (đo RSS/private theo từng giai đoạn A/B/C/D).

---

## 14. CẬP NHẬT VÒNG 7 — F-40: VAD ra khỏi event loop + NHỊP TIM LOOP

### 14.1 Đính chính quan trọng: **app thật vốn đã đúng**, lỗi nằm ở harness

Kiểm tra lại `backend/ws/handler.py`:

```python
_VAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad_worker")   # dòng 38
...
async def _handle_binary_message(session, data):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_VAD_EXECUTOR, _process_binary_chunk, session, data)   # dòng 242
```

⇒ **App thật đã chạy VAD trên luồng worker chuyên dụng** (1 luồng ⇒ giữ đúng thứ tự frame), nên
event loop của app **không** bị model VAD torch chặn. Chỗ gọi `feed_chunk` trực tiếp trong event loop
chỉ có ở **harness** `test_08_streaming_latency.py::run_paced` — và đó chính là nơi đã treo im lặng.

**Bản sửa (F-40, harness):** `run_paced` nay đẩy từng frame qua `_VAD_EXECUTOR` (1 luồng) đúng như app:

```python
await loop.run_in_executor(_VAD_EXECUTOR, session.vad_processor.feed_chunk, chunk, ts)
```

**Trước/sau, đúng ca đã treo:**

| Ca | Trước | Sau |
| :--- | ---: | ---: |
| `Japanese_5s.wav` × 3 cấu hình × 3 lần lặp | **treo > 600 s** (log dừng ở dòng VAD, không hàng rào nào phát hiện) | ✅ **26 s** |
| `English_low_…` + `English_multiple_…` × 3 × 3 | **treo > 900 s** (không ghi được JSON) | ✅ **91 s** |

### 14.2 Nhịp tim event loop (mới, cho app): `/health` → `loop_stall_ms`

Chặn event loop là dạng sự cố **im lặng**: mọi timer/coroutine ngừng chạy nên không hàng rào Python
nào báo được. Nay có module `backend/core/heartbeat.py`:

- `heartbeat_loop()` — task nền tick mỗi 0,25 s (đăng ký trong `lifespan` của `backend/main.py`).
- `stall_sec()` — số giây kể từ nhịp gần nhất.
- `/health` trả thêm **`loop_stall_ms`**. Nếu giá trị này lớn bất thường ⇒ event loop đã bị chặn
  (dù `/health` vẫn trả lời được, vì chính nó chạy trong loop nên chỉ nhảy vọt khi loop vừa thoát ra —
  dùng để phát hiện qua giám sát/extension).

**Test tầng A mới (4 test trong `test_16_server_wiring.py`):** `/health` có `loop_stall_ms`; giả lập
loop đứng 5 s thì báo ≥ 4500 ms; `tick()` đưa `stall_sec()` về ~0; `heartbeat_loop()` tick rồi huỷ sạch.

### 14.3 Số WER cuối cùng (8/8 file)

Gộp 6 nhóm chạy (mỗi nhóm một tiến trình riêng để tránh lỗi phình native — §12.6), 8 file × 3 cấu
hình × 3 lần lặp = **72 lượt streaming**, JSON: `report/audit/15_wer_ab_repeats.json`.

| Cấu hình | `vs REF` (TB) | Khoảng giữa các lần lặp | `error vs GT` |
| :--- | ---: | ---: | ---: |
| base | 31,41 % | 12,31 – 45,38 % | 22,38 % |
| reuse | 41,16 % | 40,77 – 41,78 % | 24,92 % |
| silence450 | 12,26 % | 11,59 – 12,74 % | 20,63 % |

Sàn nhiễu tổng hợp 33,06 điểm % (do các file mà chính `base` dao động 38–48 điểm) ⇒ **chỉ so được
theo từng file** (§13.4). Kết luận cuối cùng, không đổi so với vòng 6:

- **P2.5 `reuse_preview_for_commit` → GIỮ TẮT** (bằng chứng paired: +9,52 và +14,81 điểm trên hai
  file đo ổn định nhất; `Japanese_5s` và `Russian_4s` giống hệt nhau giữa 3 cấu hình).
- **P1.6 `silence_duration_ms` → GIỮ 600 ms** (không đo được lợi ích; file duy nhất khác biệt rõ lại
  xấu hơn +5,88 điểm). Trạng thái: *chưa kết luận được*, giữ mặc định an toàn.

**Suite tầng A: 143 test, 0 fail, 14,7 s.**

---

## 15. CẬP NHẬT VÒNG 8 — P3.4 ĐƯỢC KIỂM THỬ CHỨC NĂNG (không cần browser)

### 15.1 Vấn đề

`extension_firefox/lib/audio-processor.js` là **đường capture chính**, nhưng trước đây chỉ được kiểm
tra ở mức *"file có chứa chuỗi này"* (`test_worklet_file_registers_processor_and_resamples`). Việc
xác nhận thật đòi hỏi mở Firefox → `about:debugging` → reload extension, tức phụ thuộc thao tác tay.

### 15.2 Cách làm: dựng `AudioWorkletGlobalScope` giả lập bằng Node

`backend/tests/js/worklet_harness.js` (chạy bằng Node, không cần browser) tạo scope giả với
`sampleRate = 48000`, `currentTime`, `AudioWorkletProcessor`, `registerProcessor`, rồi **chạy thật**
`process()` trên **100 quantum × 128 mẫu** sóng sine 440 Hz và kiểm tra 11 điểm:

| # | Kiểm tra | Kết quả |
| :--- | :--- | :--- |
| 1 | Đăng ký `audio-capture-processor` | ✅ |
| 2 | Mỗi chunk đúng `chunkSize` mẫu Int16 | ✅ |
| 3 | `buffer` được **transfer** (zero-copy) | ✅ |
| 4 | Tổng mẫu ra khớp tỉ lệ resample (4096 vs ~4266, phần dư < 1 chunk nằm lại buffer) | ✅ |
| 5 | **Tần số giữ đúng 440 Hz** sau 48 k→16 k (sai số < 5 %) | ✅ |
| 6 | Biên độ không suy giảm | ✅ |
| 7 | **Không đứt gãy pha ở biên block** (bước mẫu lớn nhất trong ngưỡng sine) | ✅ |
| 8–9 | Timestamp tăng đơn điệu, bước ~64,0 ms, **không trôi** (< 1 µs) | ✅ |
| 10–11 | `ping` → `pong` (đường watchdog của main thread) | ✅ |

### 15.3 Đã tích hợp vào bộ test

`test_14_vad_lock_and_client.py::test_audio_worklet_runs_in_simulated_scope` gọi Node và đòi
`KẾT QUẢ: PASS`; **tự bỏ qua** (`pytest.skip`) nếu máy không có Node ⇒ không làm hỏng bộ test ở môi
trường khác.

⇒ P3.4 nay có: (a) kiểm thử chức năng DSP tự động, (b) kiểm thử tĩnh phần wiring
(`addModule`, `AudioWorkletNode`, watchdog, fallback `ScriptProcessor`, manifest expose). Việc còn
lại duy nhất vẫn cần **bạn** làm là xác nhận trên Firefox thật (log `path: AudioWorklet`).

---

## 16. CẬP NHẬT VÒNG 8 (tt) — ĐO LẠI ĐỘ TRỄ SAU F-39/F-40: KHÔNG CÓ HỒI QUY

Nghi ngờ: các hàng rào F-39 (`max_inflight_infer`, watchdog) có thể làm preview thưa đi. Đã A/B
bằng `scratch/lat_ab.py` (cùng audio 20 s nói liên tục, `qwen3-asr-0.6b`, **speed=1.0** như mốc §3.2):

| Chỉ số | guards ON | guards OFF |
| :--- | ---: | ---: |
| Số preview | 45 | 46 |
| `asr.preview_ms` p50 | 89,7 ms | 81,4 ms |
| `asr.preview_ms` **p95** | **193,3 ms** | 184,6 ms |
| `asr.preview_skipped` | 4 | 4 |
| `commit_ms` p50 / p95 | 104,6 / 164,4 ms | 122,0 / 168,6 ms |
| `asr.effective_poll_ms` | 300 (không giãn) | 300 |

⇒ Chi phí của các hàng rào là **~2 % số preview và ~5 % p95** — không đáng kể, và **K1 vẫn đạt**
(p95 193 ms < 250 ms).

**So với mốc §3.2 (p95 131 ms, 61 preview):** p50 gần như không đổi (87,2 → 89,7 ms) nhưng p95 cao
hơn và preview thưa hơn. Nguyên nhân chính **không phải** hàng rào F-39 mà là **F-40**: harness nay
đẩy VAD qua luồng worker **đúng như app thật**, nên số đo phản ánh cấu hình luồng thật (trước đây
harness chạy VAD ngay trong event loop — một cấu hình *khác* với app). Nói cách khác: mốc cũ đo một
cấu hình không giống app; số mới đại diện cho app hơn, và vẫn đạt K1.

**Ghi chú:** ở `speed=4` (feed nhanh gấp 4 lần thời gian thực) số preview tụt còn 7–9 vì cửa sổ feed
chỉ ~5 s — con số này **không** so sánh được với mốc realtime, dùng để tham khảo khi cần chạy nhanh.

---

## 17. CẬP NHẬT VÒNG 8 (tt) — K2: ĐO ĐƯỢC VÀ CẢI THIỆN 53 %

### 17.1 Trước đây K2 chưa từng được đo

Thêm metric **`asr.first_preview_ms`** = thời gian từ lúc VAD báo *bắt đầu nói* tới khi phát preview
ĐẦU TIÊN của câu đó (đặt lại mỗi câu, có trong `/api/metrics/pipeline` và in ra ở harness tầng B).
Kèm 3 test tầng A: metric xuất hiện sau khi chạy pipeline thật; `on_speech_start()` reset mốc;
cờ đánh thức bật/tắt được.

Số đo đầu tiên (20 s audio nói liên tục, `qwen3-asr-0.6b`, speed=1.0): **K2 = 1 292 ms** — vượt xa
mục tiêu < 500 ms.

### 17.2 Thí nghiệm THẤT BẠI (ghi lại để không ai thử lại)

Giả thuyết: vòng lặp nên ngủ ngắn (≤ 80 ms) khi câu chưa có preview nào, thay vì ngủ hết
`poll_interval_ms` (300 ms) ⇒ thêm `first_preview_poll_ms = 80`. Kết quả A/B (2 lần mỗi phía, xen kẽ):

| `first_preview_poll_ms` | K2 | K1 p95 |
| :--- | ---: | ---: |
| **0 (tắt)** | **621 / 626 ms** | 132 / 133 ms |
| 80 (bật) | **1 951 / 1 951 ms** | 162 / 181 ms |

⇒ **TỆ HƠN 1,3 giây.** Nguyên nhân: nhịp nhỏ làm `next_deadline` cộng rất chậm trong khi mỗi vòng
vẫn tốn ~90-130 ms inference, nên lịch tuyệt đối rơi lại phía sau ngày càng xa; vòng "bỏ nhịp"
đếm hàng nghìn nhịp và làm preview đầu tiên đến muộn hơn hẳn. **Đã gỡ bỏ hoàn toàn** khỏi code
(không còn `first_preview_poll_ms`).

### 17.3 Thay đổi ĐƯỢC GIỮ: đánh thức generator khi VAD báo bắt đầu nói

Nhánh idle của `stream_tokens` ngủ `wait_for(_commit_event, timeout=poll_interval_ms)`; **trước đây
sự kiện chỉ được set khi có commit**, nên khi tiếng nói bắt đầu, generator vẫn ngủ thêm tới 300 ms.
Nay `on_speech_start()` gọi `_wake_stream()` (cờ `wake_on_speech_start`, mặc định BẬT).

A/B xen kẽ, 2 lần mỗi phía:

| | K2 lần 1 | K2 lần 2 | K1 p95 |
| :--- | ---: | ---: | ---: |
| `wake_on_speech_start = False` (hành vi cũ) | 1 338,7 ms | 1 321,9 ms | 158 / 132 ms |
| **`True` (mặc định mới)** | **611,8 ms** | **620,8 ms** | 133 / 131 ms |

⇒ **K2 giảm 53 %** (1,33 s → 0,62 s), không ảnh hưởng K1. Phần còn lại ~620 ms đến từ
`min_transcribe_sec` (350 ms — cần đủ audio để preview có nghĩa) + phát hiện speech của VAD +
inference (~130 ms). Muốn xuống dưới 500 ms nữa thì phải hạ `min_transcribe_sec` (đánh đổi độ chính
xác preview) — **không làm** vì ưu tiên C4.

**Suite tầng A: 147 test, 0 fail, 14,5 s.**

---

## 18. CẬP NHẬT VÒNG 9 — CHỐT BẢNG NGHIỆM THU K1–K12

### 18.1 Bảng nghiệm thu (mỗi dòng đều có số đo, không suy đoán)

| # | Mục tiêu | Đo được | Trạng thái |
| :--- | :--- | :--- | :--- |
| **K1** | p95 `preview_ms` < 250 ms | **108 ms** (63 preview, max 466 ms) | ✅ |
| **K2** | preview đầu tiên < 500 ms | **570 ms** (trước vòng này: 1 292 ms) | ⚠️ vượt 14 % — xem §18.2 |
| **K3** | token dịch đầu tiên < 120 ms | **26 ms** (p50 = p95 = 26 ms, 5 câu) | ✅ |
| **K4** | E2E ngừng nói → phụ đề gốc chốt < 1,2 s | **52 ms** (commit p95 110 ms) | ✅ |
| **K5** | 0 câu bị mất; mọi drop/merge có counter | Counter + test: `commit_carried_over`, `pending_commits` (gộp, không vứt), `boundary_trimmed`, `preview_deferred_inflight` | ✅ |
| **K6** | WER không xấu đi | §13.4: `reuse` TẮT tốt hơn rõ (+9,5/+14,8 điểm ở 2 file đo ổn định); `silence450` không cải thiện | ✅ |
| **K7** | VRAM peak < 14 GB | **9,5 GB** (ASR + dịch 7B + TTS) | ✅ |
| **K8** | Backend ASR dùng CUDA | Vulkan (`cuda_backend_available = False`; gói PyPI `transcribe-cpp-native-cu12` chỉ là tên đặt trước) | ❌ bất khả thi trên máy này (§8.2) |
| **K9** | 11/11 control popup có tác dụng | 20 test `test_11_config_effectiveness.py` phủ đủ 11 nhóm control (ngôn ngữ, threshold, silence/hangover, VAD engine, model ASR/dịch, câu, preview, TTS, poll) | ✅ |
| **K10** | 0 crash / 200 lần đổi model lúc đang stream | **Test soak 200 vòng** (`test_two_hundred_model_switches_during_inference`), 0 vi phạm | ✅ |
| **K11** | capture latency client < 70 ms | **64 ms** (worklet gom 1024 mẫu @16 kHz; đo bằng harness Node) | ✅ (chờ xác nhận trên Firefox) |
| **K12** | bộ test mặc định < 15 s | **10,9 s** (149 test) | ✅ |

### 18.2 K2 — vì sao vẫn 570 ms và vì sao KHÔNG cố hạ thêm

Thành phần: `min_transcribe_sec` 350 ms (phải có đủ audio mới preview được) + phát hiện speech của
VAD (~100-150 ms) + inference preview (~80-130 ms). Muốn xuống < 500 ms chỉ còn cách **hạ
`min_transcribe_sec`** — đánh đổi trực tiếp độ chính xác preview (ưu tiên C4 của bạn). Đã thử một
cách khác (nhịp chờ ngắn hơn) và **thất bại, đã gỡ** (§17.2). Giữ 570 ms và ghi rõ đây là đánh đổi
có chủ ý.

### 18.3 K3 — số đo và một phát hiện vận hành

`scratch/k3_first_token.py` đo `translate_stream()` với model thật (`Hy-MT2-7B-UD-Q4_K_XL`, GPU):

| Cách khởi động | Câu 1 | Câu 2-5 | K3 p50 |
| :--- | ---: | ---: | ---: |
| Chỉ `load_model()` (KHÔNG warm) | **37 947 ms** | 26,0-26,7 ms | 27 ms |
| `prewarm()` đúng như app | **26,4 ms** | 25,5-26,5 ms | **26 ms** |

⇒ (a) **K3 = 26 ms** ✅; (b) nếu KHÔNG warm thì câu dịch đầu tiên bị đơ **38 giây** — app hiện đã
warm đúng (`prewarm()` chạy một câu thật), nhưng **giá phải trả là cold start**: prewarm dịch đo được
**46,9 s** (10,6 s nạp + ~36 s warm-up inference). Đây là lý do backend khởi động lâu; giữ nguyên vì
đổi lấy việc câu đầu tiên không bị đơ.

### 18.4 Tối ưu bộ test (K12) — thêm cấu hình, bớt chờ cứng

`inference_watchdog_grace_sec` (mặc định 2,0 s) thay cho hằng số 2 s hard-code, và khi chỉ canh RSS
thì kiểm tra 4 lần/giây thay vì 1 lần/giây. Kết quả: suite từ **15,6 s → 10,9 s** (K12 còn dư địa
4 s), đồng thời hàng rào phản ứng nhanh hơn.

---

## 19. P1.1 (CUDA cho ASR) — ĐÓNG HẲN: BẢN DỰNG NÀY KHÔNG CÓ CUDA CHO WINDOWS

Trước đây backend in cảnh báo kèm lời khuyên `pip install transcribe-cpp-native-cu12`. Lời khuyên
đó **SAI** và đã bị gỡ. Bằng chứng kiểm tra trực tiếp trên máy:

| Kiểm tra | Kết quả |
| :--- | :--- |
| `transcribe_cpp_native/_native/contract.json` | `"backends": ["vulkan", "cpu"]`, `"lane": "cpu-vulkan"` |
| DLL trong `_native/` | `ggml-vulkan.dll` (50 MB) + 9 biến thể `ggml-cpu-*.dll` — **KHÔNG có `ggml-cuda.dll`** |
| Metadata wheel `transcribe-cpp-native 0.2.3` | *"Planned wheels will bundle CPU plus platform accelerators"* ⇒ bản CUDA **chưa phát hành** |
| `pip index versions transcribe-cpp-native-cu12` | **chỉ có `0.0.0`** ⇒ gói đặt tên trước, cài vào không có tác dụng |
| `transcribe_cpp.backend_available("cuda")` | `False` |

⇒ ASR chạy **Vulkan** — đã tăng tốc GPU thật (VRAM +1,4 GB, ~90 % util khi suy luận, xem §9.1) và
**đây là đường được hỗ trợ**. Thay đổi trong `backend/main.py`: cảnh báo WARNING + hướng dẫn cài gói
không tồn tại → **một dòng INFO** nói rõ trạng thái bình thường, không cần cài thêm gì. Có test
tầng A chốt việc không được tái xuất hiện lời khuyên sai đó.

---

## 20. CẬP NHẬT VÒNG 10 — CHUỖI BUG SAU KHI CHẠY THẬT: SEEK, SẬP PHIÊN, VÒNG LẶP QUAY 100 % CPU

Đây là các lỗi **chỉ lộ ra khi chạy thật** (backend + extension + video có seek), không phải suy
đoán từ đọc code. Mỗi mục ghi rõ triệu chứng, nguyên nhân và cách sửa.

| Mã | Triệu chứng (log/người dùng thấy) | Nguyên nhân gốc | Cách sửa |
| :--- | :--- | :--- | :--- |
| **F-44** | Seek video → phụ đề hiện lại đoạn cũ, có "câu" dài 54 s | `reset_stream` không được gọi khi seek; hàng đợi commit giữ mảnh của vị trí cũ | Extension gửi `{type:"reset_stream", reason}` khi `seeking`/`seeked`; backend `_reset_session_stream()` + chặn mảnh commit dài hơn `max_duration_sec × 1,5` (`asr.commit_slice_clamped`) |
| **F-44b** | Sau khi seek: `Disconnected`, popup báo `❌ Already capturing`, phiên chết hẳn | Cách sửa F-44 ban đầu gọi `cancel_inference()` — huỷ giữa lời gọi native làm hỏng session của `transcribe.cpp` | **Bỏ** huỷ native; kết quả đến muộn bị **bỏ theo thế hệ** (`_stream_generation`, counter `asr.commit_dropped_stale` / `asr.preview_dropped_stale`) |
| **F-44c** | Seek liên tục tạo nhiều lần reset chồng nhau | `seeking` và `seeked` cùng phát | Gộp trong cửa sổ 0,5 s; `drain_queues()` gọi `task_done()` cho mọi item bị vứt (trước đây `join()` treo) |
| **F-46** | Backend đứng im ở một dòng log, **Ctrl+C không tắt được**, phải kill tiến trình | Không xác định được từ log (event loop bị chặn) | Thêm **watchdog luồng thật** (`backend/utils/stall_watchdog.py`): đọc nhịp tim, quá `STALL_WATCHDOG_SEC` (10 s) thì dump stack **mọi** luồng ra `sys.__stderr__`. Chính nó đưa ra bằng chứng cho F-47/F-48 |
| **F-47** | CPU 100 %, "câu" 45 s, `infer=1294.9ms` lặp lại | Khối đánh giá tier 2/3/4 nằm trong nhánh preview nên có vòng không chạy; mảnh bị bỏ ⇒ `_speech_start_sample` không tiến ⇒ cùng một đoạn được đẩy lại mãi | Đánh giá tier **mỗi vòng lặp**; tiến `_speech_start_sample` **ngay lúc cắt**; thêm hàng rào `_empty_commit_streak` (10 lần rỗng ⇒ bỏ qua) |
| **F-48** | Event loop quay 100 % CPU, nhịp tim đói, phải kill | `asyncio.wait_for(event.wait(), …)` với event **đã được set** trả về ngay ⇒ vòng lặp không hề nhường CPU | Xoá cờ trước khi `wait`, thêm `await asyncio.sleep(0)` cuối mỗi vòng, xoá cờ ở mọi lối thoát sớm của `_emit_commit` |
| **F-49** | **Mất chữ** ở đầu câu; các commit dài bất thường 9,6 / 12,7 / 22,9 / 28,9 / 31,8 s | `seg_start < 0` bị kẹp về 0 ⇒ mốc bắt đầu đoạn trỏ vào đầu buffer **trước khi** `feed_audio()` chốt pre-roll; F-44 clamp lại cắt mất phần đầu | Thêm cổng `ready = _speech_start_sample >= 0` (chưa biết mốc đầu thì **không** preview/không đánh giá tier), và `on_speech_end` dùng `end_sample` thay vì 0 |

**Trạng thái:** tất cả đã sửa trong `backend/asr/engine.py`, `backend/ws/handler.py`,
`backend/ws/session.py`, `extension_firefox/content/content-script.js`. Sau F-48, log chạy thật
**không còn dòng `[STALL WATCHDOG]`** nào (trước đó có). Việc xác nhận cuối cùng (seek nhiều lần
liên tiếp, không mất chữ, không phải kill) do người dùng thực hiện trên Firefox.

**Test tầng A mới:** `backend/tests/test_21_seek_reset.py` (18 test: reset, không huỷ native, bỏ
kết quả cũ theo thế hệ, clamp, `task_done`, gộp sự kiện seek, các hàng rào chống quay của F-47, và
kiểm tra tĩnh phía extension).

---

## 21. F-50 / F-51 — ĐỔI MODEL DỊCH LÀM MẤT MODEL ĐANG CHẠY, VÀ VÌ SAO MODEL KHÔNG TỰ TẢI

### 21.1 Triệu chứng người dùng gặp

```
POST /api/config {"translation_model": "tencent-1.8b"}  → 500
FileNotFoundError: Không tìm thấy file GGUF Translation tại:
  D:\vibe-translation-addon-transcribe_cpp\backend\models\Hy-MT2-1.8B-UD-Q8_K_XL.gguf
```

### 21.2 F-51 — catalog chỉ có tên file, KHÔNG có đường tải

`backend/translation/registry.py::resolve_gguf_path()` chỉ ghép `MODELS_DIR / gguf_file`; không có
bước tải nào cho **cả ASR lẫn Translation** (duy nhất VAD — silero/firered/fsmn — tự tải từ
HuggingFace). Popup lại liệt kê **toàn bộ** catalog trong YAML mà không kiểm tra file có trên đĩa,
nên model chưa tải vẫn chọn được và chắc chắn lỗi.

Đã xác minh cả 3 repo trong catalog đều có file tương ứng (`Hy-MT2-1.8B-UD-Q8_K_XL.gguf` có thật
trong `unsloth/Hy-MT2-1.8B-GGUF`), nên việc tự tải là khả thi.

### 21.3 F-50 — bug NGHIÊM TRỌNG HƠN: nạp lỗi ⇒ mất luôn model đang chạy

| Bước | Hành vi CŨ | Hậu quả |
| :--- | :--- | :--- |
| 1 | `main.py`: `config.translation.base = "tencent-1.8b"` **trước** khi nạp | Config trỏ vào model chưa có |
| 2 | `engine.reconfigure()`: `unload_model()` **rồi mới** `load_model()` | Model 7B đang chạy tốt bị giải phóng VRAM |
| 3 | `load_model()` ném `FileNotFoundError` | `_shared_llm = None`, `_shared_model_key = None` |

⇒ Backend **mất hoàn toàn khả năng dịch** (mọi câu sau đó đi qua nhánh "không có model") cho tới
khi người dùng đổi lại model hoặc restart. (Đường WS `session.py` đã đúng từ trước: kiểm tra file
trước, lỗi thì giữ nguyên model — nhưng popup lại đi đường REST.)

### 21.4 Đã sửa

1. **Swap nguyên tử** (`backend/translation/engine.py`): `_build_llm()` nạp model mới **trước**,
   ngoài mọi lock; chỉ khi thành công mới swap con trỏ và **sau đó** mới `_release_llm()` model cũ
   (có hàng rào `_infer_lock` để không đóng model khi còn thread đang sinh token). Lỗi ở bất kỳ
   bước nào ⇒ model cũ còn nguyên, `config` không đổi.
2. **Tải tự động** (`backend/utils/model_download.py`): `ensure_model_file()` tải từ HuggingFace
   theo trường `model` trong YAML; file đã có thì **không chạm mạng**; mỗi file một lock (bấm hai
   lần không sinh hai lượt tải); có log tiến độ 10 s/lần; chuẩn hoá vị trí về đúng
   `MODELS_DIR/<tên file>`.
3. **State machine dùng chung** (`backend/translation/hotswap.py`):
   `idle → downloading → loading → ready | error`, phơi qua `/api/config` ở
   `translation.download` để popup hỏi tiến độ. `config.translation.base` **chỉ** được ghi sau khi
   model mới nạp xong. Cả REST lẫn WS dùng chung module này.
4. **REST `/api/config`**: tên model sai ⇒ **400** (trước đây `resolve_key` fallback âm thầm về
   `tencent`); thiếu file + `auto_download` bật ⇒ **202** và tải ở luồng nền, model cũ vẫn phục vụ;
   thiếu file + tắt tải ⇒ **400** kèm đường dẫn + cách xử lý; đang tải model khác ⇒ **409**.
5. **Popup**: model chưa có file hiện `⤓ chưa tải` (đã có: `⚡`), xử lý `202` rồi poll tiến độ 4 s/lần
   (tối đa 30 phút), hiển thị `downloading/loading/ready/error`.
6. **Cấu hình mới**: `TranslationConfig.auto_download = True` (đặt `false` để tắt hẳn việc tải).

### 21.5 Kiểm chứng (offline, không nạp model, không gọi mạng)

| Kiểm tra | Kết quả |
| :--- | :--- |
| `is_downloaded` cho catalog thật | `tencent: True`, `tencent-1.8b: False`, `xiaomi: False`, `gemmax: False` (khớp thư mục `backend/models`) |
| `translation.auto_download` / `translation.download` trong `/api/config` | `True` / `{"state":"idle", …}` |
| Tên model sai ⇒ | `HTTPException 400: Model dịch 'khong-ton-tai' không có trong translation_models.yaml` |
| `hotswap.needs_download()` | `tencent-1.8b → True`, `tencent → False` |
| Test tầng A mới | `backend/tests/test_22_translation_model_download.py` — 19 test đã VIẾT nhưng **chưa chạy** (theo yêu cầu "không tự chạy test"): tải vào đúng thư mục, không gọi mạng khi file đã có, lỗi mạng ⇒ state `error`, swap lỗi **giữ model cũ**, swap thành công mới đóng model cũ, nhánh REST 400/409/202, nhánh WS báo `downloading → ready`, cờ trên `/api/config`, popup xử lý 202 |

**Trạng thái xác minh:** phần backend ở trên đã được chạy kiểm tra offline (không nạp model, không
gọi mạng) đúng như bảng; phần popup mới chỉ mới **kiểm tra cú pháp** (`node --check`) — cần một lần
bấm "đổi model dịch" trên Firefox thật để chốt.

**Chưa làm (có chủ ý):** không tự tải model ASR (catalog ASR vẫn phải copy tay) — chỉ thêm cờ
`is_downloaded` để popup hiển thị đúng; tải model vẫn **không bao giờ** chạy trên hot path dịch
từng câu.

---

## 22. F-52 — BẢN DỊCH ĐẾN MUỘN KHÔNG ĐƯỢC VẼ RA Ở TẦNG 1 (BUG NGƯỜI DÙNG BÁO)

### 22.1 Triệu chứng

> "Khi phụ đề ở tầng 2 đang đợi bản dịch mà bị đẩy lên tầng 1 thì phụ đề đó sẽ không được hiển thị nữa."

Tức là: câu vừa chốt nằm ở **TẦNG 2** (tiêu điểm) với chỉ báo `・・・` chờ dịch; có câu mới ⇒ câu đó
bị đẩy lên **TẦNG 1** (lịch sử); khi bản dịch của nó về thì **bản dịch không bao giờ xuất hiện**.

### 22.2 Nguyên nhân gốc (`extension_firefox/lib/subtitle-renderer.js`)

`_promotePendingToCompleted(null)` đẩy câu lên TẦNG 1 với `translatedText = null` (chưa có dịch).
Lúc đó `_createSentenceElement()` chỉ tạo node `.bs-original` — **không** tạo `.bs-translated`.

Khi bản dịch tới muộn, `onTranslation()` cập nhật **state** đúng (`existing.translatedText = …`)
nhưng `_renderHistoryLayer()` chỉ *cập nhật* node đã có:

```js
const transEl = existingEl.querySelector(".bs-translated");
if (transEl && transEl.textContent !== item.translatedText) { … }   // ← transEl === null ⇒ bỏ qua
```

`.bs-translated` chưa từng tồn tại ⇒ không có gì được vẽ. (TẦNG 2 có nhánh tạo node khi thiếu —
`_renderFocusLayer()` — nên bug chỉ xuất hiện ở TẦNG 1, đúng như mô tả của người dùng.)

### 22.3 Cách sửa

`_renderHistoryLayer()` nay **tạo** `.bs-translated` khi câu đã có bản dịch mà node chưa tồn tại, và
**gỡ** node khi bản dịch bị rút — đối xứng với tầng tiêu điểm.

### 22.4 Kiểm chứng: harness DOM giả lập chạy bằng Node

`backend/tests/js/subtitle_renderer_harness.js` dựng DOM tối thiểu (Element/classList/querySelector
với selector `.class`, `[attr="v"]`, `:not(.class)`) rồi **chạy thật** `SubtitleRenderer` trong
`vm` context. 21 điểm kiểm tra, gồm cả ca bug:

| # | Ca kiểm tra | Trước khi sửa | Sau khi sửa |
| :--- | :--- | :--- | :--- |
| 1 | preview → chốt câu → có bản dịch (3 tầng) | ✅ | ✅ |
| 2 | TẦNG 2 **đã có** bản dịch rồi mới bị đẩy lên TẦNG 1 | ✅ | ✅ |
| 3 | **TẦNG 2 chưa có bản dịch đã bị đẩy lên TẦNG 1, bản dịch tới sau** | ❌ (state có, DOM không) | ✅ |
| 4 | A → B → (dịch A) → C → (dịch B) | ❌ | ✅ |
| 5 | Tắt "chạy chữ": bỏ qua `partial`, chỉ áp bản hoàn chỉnh | ✅ | ✅ |
| 6 | `filtered=true` ⇒ xoá câu khỏi mọi tầng | ✅ | ✅ |
| 7 | Bản dịch muộn khi câu đã ra khỏi cửa sổ `Max lines` (nới Max lines ⇒ hiện lại, không mất dữ liệu) | — | ✅ |
| 8 | Không sinh 2 phần tử cùng `data-sentence-id` đang hiển thị | — | ✅ |

Harness được gọi tự động trong tầng A:
`test_14_vad_lock_and_client.py::test_subtitle_renderer_three_layers_run_in_node` (tự `skip` nếu máy
không có Node). Chạy tay: `node backend/tests/js/subtitle_renderer_harness.js`.

**Trạng thái:** đã sửa và kiểm chứng bằng harness (chạy thật, không phải kiểm tra chuỗi). Cần
**reload extension** trên Firefox để nạp bản renderer mới.














