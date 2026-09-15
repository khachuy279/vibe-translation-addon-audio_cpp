# KẾ HOẠCH TRIỂN KHAI (IMPLEMENTATION PLAN) — BẢN CÁ NHÂN HOÁ
## Vibe Translation Addon — Tối ưu cho 1 GPU RTX 5060 Ti 16GB, 1 phiên, ưu tiên độ chính xác & tốc độ hiển thị

| | |
| :--- | :--- |
| **Revision** | **2** — viết lại theo nhu cầu sử dụng cá nhân (không thương mại hoá) |
| **Trạng thái triển khai** | ⚠️ **ĐÃ TRIỂN KHAI PHẦN LỚN** — xem `report/audit/05_measurements_and_status.md` để biết task nào xong, số đo thật, và việc còn lại. |
| **Tài liệu nguồn** | `00_BAO_CAO_AUDIT_HIEU_NANG.md`, `01_phu_luc_extension_firefox.md`, `02_phu_luc_transcribe_cpp.md` |
| **Thay thế** | Revision 1 (đã bỏ — bản đó tối ưu cho multi-session/scale, không còn phù hợp) |

---

## 0. BỐI CẢNH SỬ DỤNG & HỆ QUẢ THIẾT KẾ

### 0.1 Ràng buộc thực tế (do người dùng nêu)

| # | Ràng buộc | Hệ quả then chốt |
| :--- | :--- | :--- |
| **C1** | Máy cá nhân, **không thương mại hoá** | Không cần hardening multi-tenant, không cần auth/rate-limit, không cần HA |
| **C2** | **Tối đa 1 session/stream cho 1 video** | **Toàn bộ RC-3 / F-03 / C1 / C2 của audit trở thành KHÔNG LIÊN QUAN.** Bỏ hẳn model pool, per-session executor, VRAM budget đa phiên |
| **C3** | Tối ưu **RTX 5060 Ti 16GB** | Provider **CUDA** thay vì Vulkan là ưu tiên cao; có dư VRAM để đổi lấy độ chính xác (quant cao hơn, context lớn hơn) |
| **C4** | **Chính xác tối đa** trong khả năng model ASR ở chế độ streaming | Mọi tối ưu **không được** đánh đổi độ chính xác của bản **commit**; ưu tiên cắt câu ở ranh giới từ; bật lại normalizer config |
| **C5** | Phụ đề gốc/dịch hiển thị **nhanh nhất có thể, liên tục** | Preview phải có nhịp đều và bắt đầu sớm; bản dịch phải xuất hiện sớm (commit sớm + streaming translation) |
| **C6** | Thay đổi trong popup áp dụng **ngay, không restart backend** | Trở thành **yêu cầu hạng nhất**, có task riêng — hiện có nhiều khoảng trống thật (xem §0.3) |

### 0.2 Điều gì thay đổi so với Revision 1

| Hạng mục | Revision 1 | **Revision 2** | Lý do |
| :--- | :--- | :--- | :--- |
| Model pool + per-session executor (T4.12) | Có, 5 ngày | ❌ **XOÁ** | C2: 1 session |
| Multi-session correctness (2 tab, streaming state) | KPI + test | ❌ **XOÁ** | C2 |
| Provider CUDA (T4.1) | Phase 4, ưu tiên thấp | ⬆️ **Phase 1, P0** | C3 |
| Popup immediacy | Rải rác (T4.4) | ⬆️ **Phase 1, P0, 5 task riêng** | C6 |
| Reuse preview cho commit (T2.3) | Bật mặc định | ⬇️ **TẮT mặc định**, chỉ bật sau khi đo WER | C4 (có thể mất từ cuối câu) |
| Cửa sổ preview (T2.4) | "Giới hạn cửa sổ để giảm chi phí" | ⬆️ **Thiết kế lại: cửa sổ = câu hiện tại ⇒ KHÔNG mất độ chính xác** (xem §1) | C4 + C5 |
| Tối ưu native N1–N7 | 10 ngày | ⬇️ **GIẢM còn N2/N3/N4, gộp 1 sprint phụ** | ROI thấp hơn khi đã có CUDA + cửa sổ câu |
| Chiến lược test | Đo nặng, 3 model | ⬆️ **3 tầng, mặc định < 15 s** | Yêu cầu test nhanh |
| Tổng effort | ~48–66 ngày | **~26–34 ngày** | Bỏ scale, gọn test |

### 0.3 Phát hiện mới về popup (yêu cầu C6) — qua kiểm chứng code

Có **HAI đường cấu hình song song** và chúng hành xử **khác nhau**:

| Đường | Ai gọi | File |
| :--- | :--- | :--- |
| **REST `/api/config`** | `popup.js:391, 471, 695, 764` (nút Apply, đổi model) | `backend/main.py:286-380` |
| **WS `set_config`** | `content-script.js:11-16, 142, 181, 304` (mỗi lần đổi setting) | `backend/ws/session.py:173-259` |

**6 khoảng trống thực tế khiến "áp dụng ngay" không đúng:**

| # | Vấn đề | Bằng chứng | Tác động với bạn |
| :--- | :--- | :--- | :--- |
| **G1** | Đổi **translation model** qua WS **không làm gì cả** — chỉ ghi vào dict | `ws/session.py:186-187` chỉ `updates["translation_model"] = ...`; **không** switch engine | Bạn chọn model dịch mới, backend vẫn chạy model cũ |
| **G2** | Đổi **ASR model** qua WS chỉ gán `engine.model_key`; việc nạp lại xảy ra **bên trong lần inference kế tiếp** | `ws/session.py:222-227` + `asr/engine.py:128-188` được gọi từ `_run_inference_sync:258` | Im lặng vài giây, phụ đề đứng, không có thông báo |
| **G3** | Đổi **VAD engine** gọi factory **trong khi đang giữ `RLock`**, và factory có thể **tải model từ HuggingFace** | `vad/processor.py:88-100` (trong `with self._lock`) + `vad/engines/fired.py:50-55`, `fsmn.py:46-53` (`snapshot_download`) | **Backend ngừng nhận audio** trong suốt thời gian tải; WS receive loop `await` VAD executor (`ws/handler.py:186`) |
| **G4** | Cấu hình phân câu qua WS ghi vào `commit_manager.cfg` — object **không bao giờ được dùng** | `ws/session.py:249-250` → `commit_manager.py:157-181` (dead) | `max_duration_sec`, `max_chars`, `split_on_stability`, `min_words_to_commit` **không có tác dụng** |
| **G5** | Cấu hình REST ghi vào `config` **toàn cục**, nhưng session đang chạy giữ **snapshot** riêng lấy lúc khởi tạo | `ws/session.py:62-79` (`SessionConfig.__init__`) vs `main.py:303-347` | Đổi qua REST có thể **không** ảnh hưởng session đang chạy |
| **G6** | `vad_enabled=False` ⇒ **không bao giờ có phụ đề** | `vad/processor.py:126-130` không gọi `on_speech_start/end` ⇒ engine không preview, không commit (F-18) | Bạn tắt VAD trong popup ⇒ phụ đề biến mất hoàn toàn |

Plus (từ audit): **G7** — 11 tham số `normalize_*` bị bỏ qua vì `SpeechNormalizer()` gọi không tham số (`asr/engine.py:103`, F-11). **G8** — `n_ctx`/`n_batch`/`n_threads` của translation hardcode, không tune được từ popup (F-21).

### 0.4 Hai giá trị mặc định không khớp với model có sẵn cục bộ (làm nặng G1/G3)

| # | Vấn đề | Bằng chứng | Hệ quả |
| :--- | :--- | :--- | :--- |
| **G9** | Extension mặc định gửi **`translationModel: "xiaomi"`** khi người dùng chưa chọn | `content-script.js:15`, `popup.js:102` | `translation_models.yaml` có `xiaomi` (MiLMMT-4.6B) nhưng **file GGUF không có cục bộ** (chỉ có `Hy-MT2-7B-UD-Q4_K_XL.gguf`). Hiện **không gây lỗi** vì G1 khiến WS bỏ qua field này. ⚠️ **Nhưng ngay khi P1.7 được implement**, giá trị mặc định này sẽ cố nạp một model không tồn tại ⇒ **P1.7 bắt buộc** phải (a) xử lý file thiếu một cách êm (giữ model cũ + thông báo), và (b) sửa default về `tencent` trong extension |
| **G10** | Extension mặc định gửi **`vadEngine: "fsmn-vad"`**, khác default backend (`fired-vad`) | `content-script.js:16` vs `config.py:78` | Mỗi lần content script khởi động, nó **ghi đè** VAD engine về `fsmn-vad` (`vad/processor.py:88-100`) ⇒ kích hoạt đúng đường nguy hiểm của G3. Cục bộ model FSMN đã có (`backend/models/fsmn_vad/`) nên không tải mạng, nhưng **vẫn nạp model trong lúc giữ `RLock`**. Sửa default cho khớp backend và/hoặc để backend tự quyết khi client không chỉ định |

**Hệ quả cho kế hoạch:** P1.9 (pre-warm + load ngoài lock) trở thành **bắt buộc**, không chỉ là tối ưu — vì đường này được kích hoạt **mỗi lần** content script khởi động, không phải chỉ khi người dùng đổi VAD thủ công.

---

## 1. NGUYÊN TẮC THIẾT KHAI LAI (accuracy-first, 1 phiên)

Đây là phần **quan trọng nhất** của kế hoạch. Nó giải quyết cùng lúc C4 (chính xác) và C5 (nhanh, liên tục) mà không phải đánh đổi.

> **P1. Commit LUÔN dùng toàn bộ ngữ cảnh câu. Không bao giờ cắt cửa sổ ở commit.**
> Chỉ **preview** được cắt cửa sổ. Bản dịch và bản gốc "chốt" luôn là kết quả của một lần suy luận trên **trọn câu**.

> **P2. Giới hạn độ dài câu, và đặt cửa sổ preview = độ dài câu tối đa.**
> Nếu `preview_window_sec ≥ max_duration_sec`, cửa sổ preview **luôn bao trùm đúng câu hiện tại**.
> ⇒ Preview thấy **cùng lượng ngữ cảnh** như commit ⇒ **không mất độ chính xác**, mà chi phí preview bị **chặn trên bởi một hằng số** thay vì tăng theo độ dài video.
> ⇒ Đây chính là cách biến O(N²) thành O(N) **mà không hy sinh chất lượng**.

> **P3. Ưu tiên cắt câu ở ranh giới từ.**
> BẬC 3 (`STABLE_PREFIX` — text preview bất biến ≥ 0.8 s) là cơ chế cắt **chính** vì nó rơi vào khoảng ngắt tự nhiên. BẬC 2 (`MAX_DURATION`) chỉ là **chốt an toàn** chống câu phình.

> **P4. Chồng lấn ở ranh giới cắt + trim trùng, không drop cả câu.**
> Câu kế tiếp bắt đầu lùi lại 200–400 ms; phần trùng ở đầu được **cắt bỏ**, không phải vứt cả câu.
> ⚠️ Điều này đòi hỏi **đổi ngữ nghĩa `CommitDeduplicator`** (hiện `is_duplicate` drop cả câu — `core/dedup.py:49-92`).

> **P5. Hiển thị liên tục.**
> Nhịp preview **cố định** (không phải `inference + sleep`); bắt đầu preview **sớm hơn** (giảm `min_transcribe_sec`); bản dịch xuất hiện sớm nhờ commit sớm + **streaming translation**.

> **P6. Dùng dư VRAM để mua độ chính xác.**
> 1 phiên trên 16GB ⇒ có thể dùng **CUDA provider**, quant cao hơn (F16), `n_ctx` lớn hơn. Không cần tiết kiệm VRAM.

### 1.1 Bài toán chi phí sau khi áp dụng P1–P2

Với `max_duration_sec = 8`, `preview_window_sec = 8`, `poll_interval_ms = 300`, RTF `r`:

| Đại lượng | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| Chi phí 1 preview cho câu dài `L` | `r · L` | Chặn trên = `r · 8 s` |
| Chi phí 1 preview ở `L = 8 s` (r = 0.0232) | **~186 ms** | Chặn trên tuyệt đối, **bất kể video dài bao nhiêu** |
| Chi phí 1 preview trung bình trong câu | ~93 ms | `L` trung bình 4 s |
| Nhịp preview | `P = 300 ms` (fixed-rate, bỏ nhịp nếu trễ) | Cảm giác "cập nhật liên tục" |
| GPU cho preview / 8 s câu | ~1.5–1.9 s ⇒ **~19–23 %** | Có thể giảm còn ~12–15 % khi chuyển CUDA (giảm `r`) |
| GPU cho commit | `r · 8` ≈ 186 ms/câu ⇒ ~2 % | |
| **Tổng GPU ASR** | **~21–25 %** | Còn dư cho translation (~5 %) + TTS (~5 %) |
| **Hệ số lặp audio** | ~10× **trong phạm vi 1 câu** | Không tăng theo độ dài video — đây là điểm mấu chốt |

> **Khác biệt cốt lõi so với hiện tại:** hiện tại ở câu 30 s, chi phí preview là ~700 ms/lần và **tăng mãi**; sau khi áp dụng, chi phí preview **bị chặn ở ~186 ms và không đổi** dù video dài 5 phút hay 5 giờ. Độ trễ hiển thị vì thế cũng bị chặn.

### 1.2 Cấu hình mục tiêu đề xuất

| Tham số | Hiện tại | **Đề xuất** | Lý do |
| :--- | :--- | :--- | :--- |
| `SentenceConfig.max_duration_sec` | 8.0 | **6.0** | Cân bằng: preview rẻ hơn, commit sớm hơn ⇒ dịch xuất hiện sớm hơn (C5) |
| `SentenceConfig.split_on_stability` | True (nhưng **dead**) | **True + wire thật** | Cắt ở ranh giới từ (P3) |
| `SentenceConfig.stability_duration_sec` | 0.8 | **0.6** | Bản dịch xuất hiện sớm hơn ở khoảng ngắt tự nhiên |
| `SentenceConfig.min_words_to_commit` | 2 | **2** (giữ) | Lọc tiếng ậm ừ |
| `ASRConfig.preview_window_sec` | *(chưa có)* | **6.0** (= `max_duration_sec`, P2) | Không mất ngữ cảnh |
| `ASRConfig.poll_interval_ms` | 350 | **300** | Cập nhật mượt hơn (C5) |
| `ASRConfig.min_transcribe_sec` | 0.6 | **0.35** | Preview đầu tiên xuất hiện sớm hơn ~250 ms (C5) |
| `ASRConfig.threads` | 4 | **4** (giữ, đo lại) | Với CUDA, CPU thread ít quan trọng hơn |
| `VADConfig.silence_duration_ms` | 600 | **450** | Commit sớm hơn 150 ms. ⚠️ Đo WER: giảm quá tay sẽ cắt giữa từ |
| `VADConfig.hangover_ms` | 400 | **300** | (`grace = min(hangover, silence*0.5) = 225 ms`) |
| `ASRConfig.reuse_preview_for_commit` | *(chưa có)* | **False** | C4 — chỉ bật sau khi đo WER (§Phase 2) |
| Provider ASR | Vulkan (mặc định wheel) | **CUDA** (`transcribe-cpp-native-cu12`) | C3 |

---

## 2. YÊU CẦU → TASK (TRACEABILITY)

| Yêu cầu | Task chính | Task hỗ trợ |
| :--- | :--- | :--- |
| **C3** — Tối ưu 5060 Ti 16GB | **P1.1** (CUDA provider), **P1.2** (VRAM/quant/context cho độ chính xác) | P4.2 (giải phóng TTS), P4.3 (đưa `n_ctx` vào config) |
| **C4** — Chính xác tối đa | **P2.1** (wire CommitManager — không cắt cửa sổ ở commit), **P2.3** (cửa sổ preview = câu), **P1.3** (sửa pre-roll 300 ms sai), **P2.6** (normalizer config) | P2.7 (ranh giới cắt: overlap + trim), P1.4 (`vad_enabled`), P1.5 (session limits/truncation) |
| **C5** — Hiển thị nhanh & liên tục | **P2.4** (fixed-rate preview), **P2.2** (`timestamps=none`), P2.3 (cửa sổ nhỏ), **P3.3** (streaming translation), P3.4 (worklet capture) | P1.6 (giảm silence_duration), P4.4 (giảm nghẽn transport) |
| **C6** — Popup áp dụng ngay | **P1.7** (translation model qua WS — G1), **P1.8** (ASR reload tường minh — G2), **P1.9** (VAD pre-warm + load ngoài lock — G3), **P1.10** (REST↔session coherence — G5), P2.1 (sentence config — G4), P2.6 (G7), P4.3 (G8) | P1.11 (bảng test "mọi control có tác dụng") |

---

## 3. PHASE 1 — P0: GPU, ĐỘ CHÍNH XÁC NỀN TẢNG, POPUP TỨC THỜI (~8–10 ngày)

> Đây là phase mang lại giá trị lớn nhất cho 4 yêu cầu của bạn và **rủi ro thấp**.

### P1.1 — ⭐ Chuyển ASR sang provider CUDA (C3)
**Effort:** 0.5 ngày · **Rủi ro:** TRUNG BÌNH (đổi wheel)

> 🔴 **ĐÍNH CHÍNH SAU KHI KIỂM CHỨNG (quan trọng):** task này **KHÔNG khả thi như đã viết**.
> - `transcribe-cpp-native-cu12` trên PyPI chỉ là **name reservation**: bản `0.0.0`, wheel **1.380 byte**,
>   mô tả *"name reservation; real wheels arrive with 0.1.0"*. **Cài vào sẽ không có CUDA** và có thể
>   gây nhầm lẫn khi binding chọn provider (nó đăng ký cùng entry-point group và được xếp hạng
>   "best accelerated").
> - Muốn CUDA phải **build từ source** (`external/transcribe.cpp/bindings/python-native-cu12/`), cần
>   **CMake + Ninja + MSVC Build Tools + CUDA Toolkit 12.x**.
> - Trên máy này: có `cmake`, **KHÔNG có** `ninja`, `cl`, `nvcc`, và **không có** `CUDA_PATH`
>   ⇒ **không thể build** mà không cài thêm vài GB công cụ.
>
> **Đường thay thế đã triển khai:** giữ Vulkan và bù bằng **P2.4b — nhịp preview thích ứng**
> (tự giãn nhịp khi backend chậm, tự thu lại khi nhanh) ⇒ phụ đề cập nhật đều hơn thay vì giật,
> không cần đổi môi trường.
> **Bằng chứng PyPI:** `scratch/check_pypi_providers.py`.

**Bằng chứng:** kiểm chứng runtime cho thấy hiện tại chỉ nạp `ggml-vulkan.dll` + `ggml-cpu-haswell.dll`, **không có CUDA**, dù translation (llama.cpp) và TTS (PyTorch) đều chạy CUDA trên cùng card ⇒ 2 API GPU, VRAM không thống nhất.

**Thay đổi:**
1. `pip install transcribe-cpp-native-cu12` (thay `transcribe-cpp-native`); gỡ provider cũ để tránh nhầm.
2. `README.md:123` — cập nhật hướng dẫn.
3. `backend/main.py` `/health` — thêm `asr_backend` (lấy từ `getattr(model, "backend", "unknown")`) để xác nhận bằng mắt.
4. `backend/config.py:104` `ASRConfig.backend` — giữ `"auto"` nhưng log rõ backend đã chọn lúc prewarm (đã có ở `asr/engine.py:182-187`).

**Acceptance:**
* Log prewarm hiện `Backend: cuda` (không còn `Vulkan0`).
* `p95 asr.preview_ms` (đo ở P0.1) **không tệ hơn**, kỳ vọng tốt hơn.
* Không OOM; ghi lại VRAM trước/sau bằng `nvidia-smi`.

### P1.2 — ⭐ Dùng dư VRAM để mua độ chính xác (C3 + C4)
**Effort:** 1.5 ngày (gồm thời gian tải model nếu cần)

**Ngân sách VRAM 16GB cho 1 phiên:**

| Thành phần | Hiện tại | Phương án chính xác hơn |
| :--- | :--- | :--- |
| Qwen3-ASR-1.7B | Q8_0 ≈ 2.1 GB | **F16/BF16 ≈ 3.4 GB** (nếu có GGUF) |
| Hy-MT2-7B | Q4_K_XL ≈ 4.6 GB | Q5_K_M/Q6_K ≈ 5.5–6.5 GB (nếu thấy chất lượng dịch chưa đủ) |
| OmniVoice | ~1–2 GB | giữ |
| CUDA/Vulkan workspace | ~1 GB | — |
| **Tổng** | **~9–10 GB** | **~11–13 GB** — vẫn còn dư trong 16 GB |

**Thay đổi:**
1. Kiểm tra `hf_repo: handy-computer/Qwen3-ASR-1.7B-gguf` (`backend/models.yaml:7`) có bản F16/BF16 không. Nếu có ⇒ tải, thêm entry vào `models.yaml`, đặt làm model mặc định.
2. Đo A/B độ chính xác: chạy cùng bộ `wav_test/*.txt` với Q8_0 vs F16, so text bằng `scripts/wer/score.py`. **Chỉ đổi nếu WER tốt hơn thật.**
3. Truyền `n_ctx` cho session ASR: `model.session(n_threads=..., n_ctx=...)` (`backend/asr/engine.py:180`) — hiện **không truyền**. Đặt `n_ctx` theo `session.limits.effective_max_audio_ms` để câu 8 s không bao giờ bị cắt cụt.
4. Ghi kết quả vào `report/audit/04_accuracy_ab.md`.

**Acceptance:** có bảng A/B WER; nếu F16 tốt hơn ⇒ dùng F16; VRAM peak < 14 GB.

### P1.3 — 🔴 Sửa pre-roll lấy nhầm 300 ms audio câu trước (C4)
**Effort:** 0.75 ngày

**Vấn đề:** `backend/asr/engine.py:226`:
```python
self._speech_start_sample = max(0, self.audio_buffer.total_written - 4800)  # pre-roll 300ms
```
`on_speech_start` được gọi **TRƯỚC** khi VAD flush pre-roll (`vad/processor.py:197-211` xếp `on_speech_start` rồi mới tới pre-roll chunk). Nên `total_written` lúc đó = **cuối câu trước** ⇒ `−4800` lùi vào **300 ms audio đã transcribe của câu trước**.

**Tác động tới độ chính xác:** mỗi utterance bị **thừa 300 ms audio cũ ở đầu** ⇒ model có thể sinh lại từ/cụm từ cuối của câu trước, hoặc decode lệch ngữ cảnh ngay từ token đầu.

**Thay đổi đề xuất:**
```python
# on_speech_start():
self._awaiting_pre_roll = True
self._speech_start_sample = -1   # chưa biết

# feed_audio(): chunk ĐẦU TIÊN sau on_speech_start chính là frame pre-roll đầu
if self._awaiting_pre_roll:
    self._speech_start_sample = self.audio_buffer.total_written
    self._awaiting_pre_roll = False
```
Cách này cho `_speech_start_sample` = **đúng mép đầu của pre-roll**, không lùi vào câu trước.

**Acceptance:** unit test (tầng A, không cần model): feed 2 câu có khoảng lặng; assert `get_slice` của câu 2 **không** chứa mẫu nào trong `[end_câu_1 − 4800, end_câu_1)`. Đo lại số từ lặp đầu câu trên `wav_test/`.

### P1.4 — 🟠 `vad_enabled=False` không bao giờ ra phụ đề (C6 / G6)
**Effort:** 0.5 ngày

**Thay đổi (chọn A):** Vì README và thiết kế đều coi VAD là **bắt buộc** (không có VAD thì không có ranh giới câu), hãy:
1. `backend/vad/processor.py:126-130` — nếu `enabled=False`, log **WARNING một lần** nói rõ "VAD bị tắt ⇒ không thể phân câu ⇒ phụ đề sẽ không xuất hiện; đã tự bật lại VAD", rồi **vẫn chạy VAD**.
2. `extension_firefox/popup/popup.html` + `popup.js` — ẩn hoặc khoá toggle VAD, kèm tooltip giải thích.
3. **Hoặc (B)** — nếu bạn thực sự muốn chế độ không-VAD: gọi `on_speech_start` ngay chunk đầu và chốt câu bằng BẬC 2 `max_duration_sec` của CommitManager (phụ thuộc P2.1).

**Acceptance:** bật/tắt `vadEnabled` trong popup ⇒ phụ đề **vẫn chạy** trong cả hai trạng thái (kèm cảnh báo ở log).

### P1.5 — 🟡 Kiểm tra trần audio & truncation (C4)
**Effort:** 0.5 ngày

**Vấn đề:** engine **không đọc** `session.limits.effective_max_audio_ms` (binding có: `transcribe_cpp/__init__.py:1345-1355`) và **không kiểm tra** `transcribe_was_truncated` ⇒ câu quá dài có thể bị cắt cụt **âm thầm** ⇒ mất chữ mà không biết.

**Thay đổi:** đọc `limits` khi nạp model, ghi log; trong `_run_inference_sync` nếu `len(audio) > limit` ⇒ cắt + `increment_counter("asr.audio_truncated")` + WARNING; sau `run()` kiểm tra `transcribe_was_truncated` ⇒ `increment_counter("asr.output_truncated")`.
**Acceptance:** câu vượt trần ⇒ có counter + log rõ ràng; không bao giờ trả text rỗng im lặng.

### P1.6 — 🟡 Giảm ngưỡng im lặng để commit sớm hơn (C5)
**Effort:** 0.25 ngày + đo WER

**Thay đổi:** `backend/config.py:80-81` — `silence_duration_ms: 600 → 450`, `hangover_ms: 400 → 300`.
**⚠️ Bắt buộc đo WER trước/sau** — giảm quá tay sẽ cắt giữa từ (vi phạm C4).
**Acceptance:** E2E commit latency giảm ~150 ms; WER chênh **≤ 0.3 %** tuyệt đối. Nếu WER xấu hơn ⇒ giữ 600 và bù bằng P3.3 (streaming translation).

### P1.7 — 🔴 Đổi translation model qua WS phải có tác dụng (C6 / G1)
**Effort:** 1 ngày

**Vấn đề:** `backend/ws/session.py:186-187` chỉ ghi `updates["translation_model"]` — **không switch engine nào cả**. Popup đổi model dịch qua WS (`content-script.js:15`) ⇒ không có tác dụng.

**Thay đổi:** trong `SessionState.apply_config`, xử lý `translation_model` giống `main.py:320-338`:
1. `translation_registry.resolve_key(...)` → `get_model(...)` để kiểm tra hợp lệ.
2. Khởi động **task nền** (`asyncio.create_task` + `track_background_task`) gọi `get_translation_engine(new_cfg).load_model()` — **không** block receive loop.
3. Cập nhật `config.translation.base`.
4. Gửi thông báo WS mới `{"type":"model_status","stage":"translation","state":"loading"|"ready"|"error","model":key}`.
5. Nếu file GGUF không tồn tại ⇒ log ERROR rõ ràng + gửi `state:"error"` (hiện chỉ raise 500 ở REST).

**Lưu ý:** chỉ có `Hy-MT2-7B-UD-Q4_K_XL.gguf` tồn tại cục bộ; các model khác trong `translation_models.yaml` sẽ `FileNotFoundError` ⇒ thông báo rõ thay vì im lặng. **Đồng thời bắt buộc sửa default `"xiaomi"` trong extension** (`content-script.js:15`, `popup.js:102`) về `tencent` — nếu không, P1.7 sẽ cố nạp MiLMMT không tồn tại ngay khi content script khởi động (G9).
**Acceptance:** đổi model dịch trong popup ⇒ log backend hiện đang nạp model nào, và sau khi xong thì bản dịch tiếp theo dùng model mới — **không cần restart**. Chọn model không có file ⇒ **giữ model cũ**, báo lỗi rõ, **không treo** backend.

### P1.8 — 🔴 Đổi ASR model phải tường minh, không "nạp trộm" trong inference (C6 / G2)
**Effort:** 1.5 ngày · **Phụ thuộc:** P1.3 không bắt buộc

**Vấn đề:** `ws/session.py:222-227` chỉ gán `engine.model_key`; `_ensure_model_loaded` (`asr/engine.py:128-188`) chạy **bên trong lần inference kế tiếp** (`:258`) ⇒ phụ đề đứng im vài giây, không thông báo, và audio vẫn dồn vào buffer.

**Thay đổi:**
1. Thêm vào `TranscribeEngine`: `def reload_model_async(self, new_key)` — nạp model mới **trước**, giữ model cũ phục vụ tới khi sẵn sàng, rồi swap và `close()` model cũ.
2. Giữ `_infer_lock` trong lúc swap để không đụng native handle (đúng ràng buộc thư viện: `transcribe.h:11-20`).
3. Trong `stream_tokens`, **không** gọi `_ensure_model_loaded` nếu đã có model; tách "nạp" khỏi "suy luận".
4. Gửi WS `model_status` (`loading` → `ready`) và **giữ preview bằng model cũ** trong lúc nạp.
5. `unload_shared_model()` **phải** giữ `_infer_lock` (sửa luôn race F-04 — xem P1.8b).

**P1.8b — Sửa race use-after-free (F-04, CRITICAL):** `asr/engine.py:61-80` chỉ giữ `_shared_lock`, **không** giữ `_infer_lock`, trong khi `_run_inference_sync:271-296` đang dùng native session ⇒ nguy cơ **use-after-free / crash** khi đổi model lúc đang stream. Chuẩn hoá **một thứ tự lock duy nhất: `_infer_lock` → `_shared_lock`** cho cả `unload_shared_model` và `_run_inference_sync`. Không có chu trình ⇒ không deadlock.

**Acceptance:** đổi ASR model trong popup giữa lúc đang phát video ⇒ có thông báo `loading`/`ready`, phụ đề không đứng quá ~100 ms, và **stress 200 lần đổi model lúc đang inference ⇒ 0 crash**.

### P1.9 — 🔴 Đổi VAD engine không được làm đứng backend (C6 / G3 / G10)
**Effort:** 1 ngày

**Vấn đề:** `vad/processor.py:88-100` gọi `VADEngineFactory.get_engine(eng)` **trong khi giữ `RLock`**; factory có thể **tải từ HuggingFace** (`vad/engines/fired.py:50-55`, `fsmn.py:46-53`). Vì `ws/handler.py:186` **await** VAD executor, backend **ngừng nhận audio** suốt thời gian đó. ⚠️ **Đường này được kích hoạt mỗi lần content script khởi động**, vì extension mặc định gửi `vadEngine: "fsmn-vad"` (`content-script.js:16`) — không phải chỉ khi bạn đổi VAD thủ công (G10).

**Thay đổi:**
1. **Pre-warm tất cả VAD engine lúc khởi động** — `backend/main.py:117-123` hiện chỉ warm 1 engine. Warm cả `firered-vad`, `silero-vad`, `fsmn-vad` (model files đã có cục bộ: `firered_stream/`, `silero_vad.jit`, `fsmn_vad/`).
2. Trong `update_config` (`vad/processor.py:88-100`): **chỉ đổi sang engine đã có trong cache**; nếu chưa có ⇒ log WARNING, **giữ engine hiện tại**, và khởi động nạp ở background (không giữ lock).
3. `create_initial_state` gọi **ngoài** `self._lock`.
4. Tuyệt đối **không** để `snapshot_download` chạy trên đường hot path — kiểm tra file trước, nếu thiếu thì báo lỗi rõ.

**Acceptance:** đổi VAD engine giữa lúc đang stream ⇒ backend **không** ngừng nhận audio (đo `ws.audio_chunks_received` không có khoảng trống); thời gian áp dụng < 100 ms (vì đã pre-warm).

### P1.10 — 🟠 Thống nhất REST và WS: một nguồn sự thật (C6 / G5)
**Effort:** 1 ngày

**Vấn đề:** REST `/api/config` ghi vào `config` **toàn cục** (`main.py:303-347`), nhưng `SessionState.config` là **snapshot** lấy lúc khởi tạo (`ws/session.py:62-79`) ⇒ đổi qua REST có thể không ảnh hưởng session đang chạy.

**Thay đổi (chọn cách đơn giản nhất cho 1 phiên):**
* REST cập nhật `config` toàn cục **và** (nếu có session active) **đẩy cùng thay đổi vào session qua `session.apply_config(...)`** — dùng lại đúng code đường WS ⇒ một đường xử lý duy nhất.
* Giữ một **registry session đang hoạt động** (thêm vào `ws/handler.py`) để REST tìm được session; với C2 (1 phiên) chỉ cần 1 phần tử.
* Ghi log rõ "đã áp dụng cho session đang chạy" vs "chỉ áp dụng cho phiên mới".

**Acceptance:** đổi `target_lang` / `vad_threshold` / `min_words_to_commit` qua REST **trong lúc đang phát video** ⇒ có hiệu lực ngay trên phiên đang chạy (kiểm chứng bằng log + hành vi).

### P1.11 — 🟠 Bảng test "mọi control trong popup đều có tác dụng" (C6)
**Effort:** 1 ngày · **Đây là task bảo hiểm cho C6**

**Thay đổi:** tạo `backend/tests/test_20_config_effectiveness.py` — **tầng A, không cần model**, dùng `FakeASREngine` / `FakeVADEngine` / `FakeTranslator`:

| Control | Kỳ vọng quan sát được |
| :--- | :--- |
| `asrEngine` | `session.asr_engine.model_key` đổi + task nạp được spawn |
| `vadEngine` | `vad_processor.vad_engine` đổi, **không** giữ lock khi nạp |
| `vadThreshold` | `vad_processor.threshold` đổi |
| `silenceDurationMs` / `hangoverMs` | thuộc tính tương ứng đổi |
| `vadEnabled` | **không** được dẫn tới "không có phụ đề" |
| `ttsEnabled` / `ttsVoice` / `ttsSpeed` | `session.config` + engine phản ánh |
| `translationModel` | task nạp model dịch được spawn (P1.7) |
| `targetLang` / `sourceLang` | `config` + `engine.language` đổi |
| `maxDurationSec` / `minWordsToCommit` / `splitOnStability` | **`commit_manager.cfg` đổi** (P2.1) |
| `stabilityDurationSec` | `commit_manager.cfg` đổi |

**Acceptance:** mọi dòng pass; **bất kỳ control nào không có tác dụng ⇒ test fail** (chống tái phát G1–G4).

---

## 4. PHASE 2 — P0: CẮT CHI PHÍ ASR MÀ GIỮ NGUYÊN ĐỘ CHÍNH XÁC (~7–9 ngày)

> Đây là phase hiện thực hoá nguyên tắc P1–P4 ở §1.

### P2.1 — 🔴 Wire lại CommitManager 4 bậc (C4 + C5 + C6/G4)
**Effort:** 2.5 ngày · **Rủi ro:** TRUNG BÌNH · **Làm TRƯỚC P2.3**

**Vấn đề:** `core/commit_manager.py:157-181` `decide_commit_trigger()` **không được gọi ở đâu**; `CommitReason` không bao giờ được tạo; `max_duration_sec`, `max_chars`, `split_on_stability`, `min_words_to_commit` **đều vô hiệu** ⇒ câu có thể dài tới **50 s** (FireRed `max_speech_frame=2000`, `vad/engines/fired.py:69`).

**Thay đổi:**
| File | Nội dung |
| :--- | :--- |
| `backend/asr/engine.py:308-391` | Gọi `commit_manager.decide_commit_trigger(vad_silence, duration, preview_text, is_speech_active)` mỗi vòng poll; nhánh `MAX_DURATION`/`STABLE_PREFIX`/`TIMEOUT_FORCE` ⇒ tạo commit + **reset `_speech_start_sample` sang `end_sample` của commit** |
| `backend/asr/engine.py:322-351` | Dùng `commit_req["reason"]` cho trường `commit_reason` (đang hardcode `"VAD_SILENCE"`) |
| `backend/core/commit_manager.py:74` | ⚠️ **Bắt buộc:** khởi tạo `_last_activity_time` đúng và **gọi `record_activity()`** mỗi khi có audio. Nếu không, `evaluate_inactivity_timeout` trả `True` sau 1.2 s kể từ lúc tạo object ⇒ force-commit mỗi vòng poll |
| `backend/config.py:127-135` | Thêm `SentenceConfig.enable_tier234: bool = True` (feature flag rollback) |

**Ba điểm khó phải xử lý đúng:**
1. **Không mất audio ở ranh giới.** Câu kế tiếp phải bắt đầu tại `end_sample` của commit (hoặc lùi 200–400 ms theo P4).
2. **Không tạo mảnh câu rác.** Nếu cắt `MAX_DURATION` mà mảnh < `min_words_to_commit` ⇒ **gộp vào câu kế tiếp**, **không** để `_stream_asr_tokens` lọc bỏ (hiện `handler.py:206-221` lọc bỏ ⇒ mất chữ).
3. **Tương tác với P2.7** (overlap + trim trùng, không drop cả câu).

**Acceptance (tầng A + B):**
* `commit_reason` có `MAX_DURATION`/`STABLE_PREFIX` xuất hiện thật trong log/metric.
* Tổng độ dài các segment ≈ độ dài speech (**test assert**, không mất audio).
* Trên `wav_test/` (tầng B): **WER không xấu đi**.

### P2.2 — ⭐ `timestamps="none"` (C5, gần như miễn phí)
**Effort:** 0.25 ngày

**Thay đổi:** `backend/asr/engine.py:296` — thêm `timestamps="none"` vào `session.run(...)`.
**Cơ sở:** binding mặc định `"auto"` (`transcribe_cpp/__init__.py:1098`) ⇒ `_materialize()` (`:1266-1342`) copy **toàn bộ** segments/words/tokens thành dataclass Python, trong khi engine **chỉ dùng `.text`** (`:297`). Với Audio-LLM, số token tỉ lệ với độ dài audio ⇒ hàng nghìn object vứt đi mỗi preview.
**Acceptance:** `p95 asr.preview_ms` giảm; text **không đổi** (so sánh trên `wav_test/`). Kiểm tra `transcribe_full_text` vẫn trả đúng khi `timestamps="none"`.

### P2.3 — 🔴 Cửa sổ preview = câu hiện tại (C4 + C5) — *task quan trọng nhất*
**Effort:** 3 ngày · **Rủi ro:** TRUNG BÌNH (đã giảm nhờ thiết kế §1)

**Vấn đề:** `backend/asr/engine.py:358-363` transcribe **toàn bộ** `[speech_start, current_total)` mỗi poll ⇒ O(N²).

**Thay đổi:**
```python
# CHỈ preview bị cửa sổ hoá. Commit (P1: toàn ngữ cảnh) KHÔNG bị.
W = config.asr.preview_window_sec          # = max_duration_sec = 6.0
win_start = max(self._speech_start_sample, current_total - int(16000 * W))
preview_text = infer(get_slice(win_start, current_total))
```
Vì `W = max_duration_sec` (P2), cửa sổ **luôn bao trùm trọn câu hiện tại** ⇒ preview thấy **cùng ngữ cảnh** như commit ⇒ **không mất độ chính xác**.
Thêm `increment_counter("asr.preview_audio_sec")` để theo dõi.

**Feature flag:** `ASRConfig.preview_window_sec` — đặt `0` để tắt (quay lại hành vi cũ).

**Acceptance:**
* `p95 asr.preview_ms` **< 250 ms và không tăng** khi video chạy 10 phút (đo ở phút 1 vs phút 10).
* Độ trễ hiển thị preview ≈ `asr.preview_ms` (không tăng theo độ dài video).
* **WER của preview vs commit không lệch** (vì cùng ngữ cảnh) — đây là bằng chứng cho C4.

### P2.4 — ⭐ Fixed-rate preview scheduler (C5)
**Effort:** 0.75 ngày

**Vấn đề:** `asr/engine.py:363,381` — `await inference` rồi `await sleep(0.35)` ⇒ nhịp thực = `inference + 0.35`, **không đều** ⇒ cảm giác phụ đề "nhảy".
**Thay đổi:** lịch tuyệt đối `deadline_k = t0 + k·P`. Nếu inference vượt hạn ⇒ **bỏ** preview đó (`increment_counter("asr.preview_skipped")`) thay vì trôi nhịp.
**Acceptance:** độ lệch chuẩn khoảng cách giữa các preview thấp; preview cập nhật đều ~3 lần/s trong suốt video dài.

### P2.5 — 🟡 Bật reuse preview cho commit — **CHỈ SAU KHI ĐO WER** (C4)
**Effort:** 0.75 ngày + đo WER · **Mặc định: TẮT**

**Vấn đề:** commit (`asr/engine.py:330-334`) chạy lại inference trên đúng đoạn audio preview cuối vừa xử lý.
**Rủi ro với C4:** dùng text preview cho commit có thể **mất từ cuối câu**.
**Thay đổi:** `ASRConfig.reuse_preview_for_commit: bool = False`. **Chỉ bật nếu** đo WER cho thấy chênh ≤ 0.3 % tuyệt đối **và** delta audio < 0.4 s.
**Acceptance:** nếu bật ⇒ tiết kiệm ~30 % compute ASR commit, WER chênh ≤ 0.3 %. Nếu không đạt ⇒ **giữ TẮT** và ghi lý do vào report.

### P2.6 — 🟠 Normalizer: sửa bug + truyền config thật (C4 + C6/G7)
**Effort:** 1 ngày

**Thay đổi:**
1. `backend/asr/engine.py:103` — `SpeechNormalizer()` gọi **không tham số** ⇒ **bỏ qua toàn bộ** 11 field `config.asr.normalize_*` (`config.py:112-124`). Truyền config thật vào.
2. `backend/core/normalizer.py:50,56,78-79,110,115-116` — dùng `np.dot(a,a)` cho RMS (không tạo temp), gộp thành ít phép toán hơn, **bỏ 4 field không ai đọc** (`original_rms/normalized_rms/original_peak/normalized_peak`).
3. **Quyết định dứt điểm** về 5 config không tồn tại trong code (`normalize_gain_smoothing`, `*_attack_alpha`, `*_release_alpha`, `*_speech_frame_ms`, `*_min_speech_frames`): implement **hoặc** xoá khỏi config + README (`README.md:29` đang quảng cáo "Gain smoothing").

**Acceptance:** đổi `normalize_target_rms` trong config ⇒ gain áp dụng thay đổi **quan sát được**; số mảng tạm/call giảm từ ~7 → ≤ 3.

### P2.7 — 🟠 Ranh giới cắt câu: overlap + trim trùng (C4)
**Effort:** 1.5 ngày · **Phụ thuộc:** P2.1

**Vấn đề:** khi cắt giữa câu (BẬC 2/3), ranh giới có thể rơi **giữa từ** ⇒ mất chữ hoặc lặp chữ.
**Thay đổi:**
1. Câu kế tiếp bắt đầu lùi lại `boundary_overlap_ms = 250`.
2. Đổi ngữ nghĩa dedup ở tầng commit: hiện `CommitDeduplicator.is_duplicate` (`core/dedup.py:49-92`) **drop cả câu** — cần thêm chế độ **trim phần trùng ở đầu** rồi giữ phần mới, thay vì vứt cả câu.
3. Ưu tiên `STABLE_PREFIX` (rơi vào khoảng ngắt tự nhiên — P3) trước `MAX_DURATION`.

**Acceptance (tầng A):** test cắt cưỡng bức giữa câu ⇒ **không mất từ**, không lặp từ ở đầu câu kế; tổng text ghép lại khớp text đầy đủ (sai khác ≤ 1 từ ở ranh giới).

### P2.8 — 🟡 Bật `min_transcribe_sec` thấp hơn + `poll_interval_ms` 300 (C5)
**Effort:** 0.25 ngày
**Thay đổi:** `backend/config.py:107-108` — `min_transcribe_sec: 0.6 → 0.35`, `poll_interval_ms: 350 → 300`.
**Lưu ý:** preview trên audio < `min_transcribe_sec` có thể cho kết quả kém; vì preview chỉ để hiển thị và commit (toàn ngữ cảnh) mới là bản chốt, đây là đánh đổi **chấp nhận được** cho C5.
**Acceptance:** preview đầu tiên xuất hiện sớm hơn ~250 ms; text preview 0.35–0.6 s không rác vô lý.

### P2.9 — 🟡 Int16 xuyên suốt tới trước model (C5, giảm tải CPU)
**Effort:** 1.5 ngày · **Rủi ro:** TRUNG BÌNH (đụng `CircularAudioBuffer`)
**Thay đổi:** buffer lưu `int16` (−50 % RAM); chuyển `float32` **một lần** ngay trước inference; `feed_audio` (`asr/engine.py:205-218`) không còn `astype/32768` mỗi frame; `vad/processor.py:163` dùng `memoryview` để bỏ 40 alloc/s.
**Acceptance:** test bit-exact (`test_01_core_audio.py:77`) pass; không sai lệch text.

---

## 5. PHASE 3 — P1: HIỂN THỊ NHANH & LIÊN TỤC (~5–7 ngày)

### P3.1 — 🟠 TTS qua binary frame + `decodeAudioData` (C5)
**Effort:** 2 ngày (backend 1 + extension 1)
**Vấn đề:** TTS gửi **base64 trong JSON** (`tts/audio_processor.py:131-140`, `ws/serializers.py:70-91`) ⇒ +33 % bytes + ~5 bản copy; client decode bằng `atob` + vòng lặp **per-byte** trên **main thread** (`tts-player.js:83-92`) ⇒ 144k vòng lặp/câu **tranh chấp** với thread giao audio.
**Thay đổi:** backend trả `wav_bytes` thô, gửi **binary frame** (thêm `send_bytes` cho `SafeWebSocketConnection`); extension xử lý `ArrayBuffer` + `decodeAudioData` trên **một** `AudioContext` sống lâu.
**Điều kiện tiên quyết:** **P3.0 — version hoá giao thức** (0.5 ngày): backend gửi `protocol_version`; extension tự giảm cấp hành vi nếu lệch ⇒ triển khai được backend trước.
**Acceptance:** −33 % bytes; không còn vòng lặp per-byte trên main thread; TTS phát đúng.

### P3.2 — 🟠 Backpressure (C5)
**Effort:** 1.5 ngày
**Vấn đề:** **0 lần** `bufferedAmount` trong toàn extension ⇒ nếu backend nghẽn, client vẫn gửi 32 KB/s vào buffer không giới hạn ⇒ **latency không bao giờ hồi phục**.
**Thay đổi:** `service-worker.js:88-109` + `ws-client.js:226-250` — kiểm tra `ws.bufferedAmount`; vượt ngưỡng ⇒ bỏ frame cũ + counter; vượt ngưỡng cao ⇒ tạm dừng capture + báo content script. Truyền `[rawBuffer]` làm **transfer list** cho `port.postMessage`.
**Acceptance:** backend nghẽn nhân tạo 10 s ⇒ `bufferedAmount` không tăng vô hạn; latency **phục hồi** sau khi hết nghẽn.

### P3.3 — ⭐ Streaming translation token (C5) — *đòn bẩy lớn nhất cho "dịch hiện nhanh"*
**Effort:** 2 ngày
**Vấn đề:** bản dịch chỉ xuất hiện **sau khi** sinh xong toàn bộ (~250–800 ms cảm nhận).
**Thay đổi:** `backend/translation/engine.py:190-198` dùng `stream=True`; đẩy token dần qua WS (`make_translation_msg` thêm `partial=True`); extension hiển thị dần (đã có `bs-translating` + `_renderFocusLayer` cập nhật tại chỗ — `subtitle-renderer.js:526-566` ✅ hạ tầng đã sẵn).
**Acceptance:** **thời gian tới token đầu** < 120 ms (đo mới); chất lượng bản dịch cuối **không đổi**.

### P3.4 — 🟠 Kích hoạt AudioWorklet thật (C5)
**Effort:** 2.5 ngày · **Rủi ro:** TRUNG BÌNH
**Vấn đề:** `lib/audio-processor.js` (worklet, thiết kế đúng, có transferable) là **dead code** — không có `audioWorklet.addModule` nào trong repo. Capture thật là `createScriptProcessor(4096)` trên **main thread** (`audio-capture.js:103`) ⇒ **+170 ms (48 kHz) đến +512 ms (16 kHz)** latency trước khi audio rời browser.
**Thay đổi:** dùng `audioWorklet.addModule(runtime.getURL("lib/audio-processor.js"))` + `AudioWorkletNode`; chuyển resampler 16 kHz **vào worklet**; **bỏ ép `sampleRate: 16000`** lên graph playback (`audio-capture.js:38,65,76`) để không band-limit audio gốc xuống 8 kHz. Giữ đường ScriptProcessor làm fallback có log.
**Acceptance:** capture latency < 70 ms; audio gốc không bị band-limit; không jank khi TTS hoạt động.

### P3.5 — 🟠 Các sửa nhanh ở extension
**Effort:** 1.5 ngày
| Task | File | Nội dung |
| :--- | :--- | :--- |
| P3.5a | `lib/tts-player.js:77` | Cap `queue` (3) + drop theo `audioTimeSec` ⇒ hết desync TTS tích luỹ (F-16) |
| P3.5b | `content/content-script.js:61` | Sửa `\|\|` ⇒ chỉ **một** frame phát TTS (hết phát 2 lần, F-28) |
| P3.5c | `lib/audio-capture.js:157` | Trừ `residualSamples.length / 16000` khỏi `chunkTime` (hết trôi timestamp 21–64 ms) |
| P3.5d | `lib/ws-client.js:160-174` | Lưu handle `setTimeout` reconnect + `clearTimeout` trong `disconnect()` (hết zombie port/WS) |
| P3.5e | `content/content-script.js:42-43` | Cache **kết quả âm** của `findVideo()` ⇒ hết quét toàn DOM mỗi sự kiện |
| P3.5f | `lib/subtitle-renderer.js` + `overlay-manager.js:221` | Thêm `destroy()` clear timer |
| P3.5g | `content/overlay-manager.js:445,463` | Bỏ animate `max-height`/`margin`/`padding` ⇒ chỉ `transform`/`opacity` |
| P3.5h | `lib/subtitle-renderer.js:440-444` | Coalesce `_renderAll` bằng `requestAnimationFrame` |

---

## 6. PHASE 4 — P2: DỌN DẸP & TỐI ƯU SÂU (~6–8 ngày, tuỳ chọn)

> Làm nếu sau Phase 1–3 vẫn còn dư địa. **Không bắt buộc** để đạt 4 yêu cầu.

### P4.1 — 🟡 Logging hot path (giảm jitter)
**Effort:** 0.5 ngày — `utils/logger.py:174` DEBUG → INFO; chỉ `flush()` khi `levelno >= WARNING` (`:140-144`); hạ log per-utterance (`asr/engine.py:338`, `ws/handler.py:297,337,411`).

### P4.2 — 🟡 Giải phóng VRAM TTS khi tắt (C3)
**Effort:** 0.5 ngày — `main.py:340-343`: khi `tts_enabled → False`, gọi `OmniVoiceTTS.get_instance().unload_model()` (`tts/engine.py:296-314`, đã có sẵn). Bật lại ⇒ prewarm (đã có).

### P4.3 — 🟡 Đưa tham số translation vào config (C6/G8)
**Effort:** 0.5 ngày — `translation/engine.py:104-111`: `n_ctx=512`, `n_batch=256`, `n_threads=4` hardcode → đọc từ `TranslationConfig` (`config.py:138-155`) + yaml. `n_ctx` lớn hơn giúp câu dài + `use_context`.

### P4.4 — 🟡 Bounded `metrics._checkpoints` (sửa memory leak)
**Effort:** 0.25 ngày — `core/metrics.py:36,115-118` dùng `deque(maxlen=200)` hoặc bỏ checkpoint theo session (đã có counter `sessions_connected` ở `ws/handler.py:53`).

### P4.5 — 🟡 `_pending_commits` không được vứt câu âm thầm
**Effort:** 0.75 ngày — `asr/engine.py:111` `deque(maxlen=8)` comment tự thừa nhận *"oldest dropped if ASR blocked"*. Đổi thành **gộp** `end_sample` của commit cũ nhất với commit kế (không mất audio) + `increment_counter("asr.commits_merged")` + WARNING. **Chỉ có ý nghĩa sau P2.1** (nếu không, gộp sẽ tạo câu dài).

### P4.6 — 🟡 VAD inference ra ngoài RLock
**Effort:** 1 ngày — `vad/processor.py:135-263` đang giữ RLock xuyên suốt `is_speech()` (PyTorch forward) ⇒ spike cleanup **102.68 ms** (`report/07:18`). Nhả lock trong lúc gọi model (an toàn vì hiện chỉ có 1 VAD thread).

### P4.7 — 🟡 Tối ưu native (`external/transcribe.cpp`) — **tuỳ chọn, có thể bỏ**
**Effort:** 3–5 ngày · **Chỉ làm nếu P2.3/P1.1 chưa đủ**
Chỉ 3 mục ROI cao nhất, **ưu tiên đóng góp upstream** (cây vendored có `patches/` + quy ước `AGENTS.md`):

| ID | Nội dung | File | Lợi ích |
| :--- | :--- | :--- | :--- |
| N2 | Persistent CPU threadpool (template: `parakeet/decoder.cpp:505-519`) | `src/transcribe-batch-util.cpp:69-88` | Bỏ ~96 thread create/join mỗi poll ⇒ ~5–15 ms/run |
| N3 | Ngừng `release_scratch()` mỗi `run()` offline | `src/transcribe.cpp:2251-2253` | Bỏ 1–10 ms/run (thư viện tự đo) |
| N4 | Bỏ round trip encoder D2H→H2D | `src/arch/qwen3_asr/model.cpp:709` → `:802` | Bỏ 2 sync + ~12 MB transfer/poll |

**N1 (prefix/state reuse)** — loại bỏ O(N²) tận gốc — **KHÔNG khuyến nghị** trong bản này: cần API mới trong `src/causal_lm/`, effort lớn, và **P2.1+P2.3 đã chặn được chi phí** ở mức đủ tốt cho 1 phiên.

---

## 7. CHIẾN LƯỢC TEST NHANH (yêu cầu: test phải tiết kiệm thời gian)

### 7.1 Ba tầng test

| Tầng | Chạy khi nào | Thời gian mục tiêu | Cần model? | Nội dung |
| :--- | :--- | :--- | :--- | :--- |
| **A — Logic thuần** | **Mặc định, mọi lần sửa** | **< 15 s** | ❌ Không | Commit tiers, drop/merge, cửa sổ preview, ranh giới pre-roll, config effectiveness, metrics bounded, normalizer, dedup, serializer, VAD frame slicing |
| **B — Model nhỏ** | Trước khi merge task Phase 2/3 | **30–60 s** | ✅ 1 model | Latency preview/commit thật, streaming liên tục, WER smoke |
| **C — Full profile** | Chỉ khi **chốt phase** (thủ công) | 2–4 phút | ✅ 3 model | E2E đầy đủ ASR + dịch + TTS, bảng KPI |

### 7.2 Chìa khoá để tầng A nhanh: **seam để inject engine giả**

Hiện `SessionState.init_components()` (`ws/session.py:142-160`) hardcode `TranscribeEngine(...)`. Thêm seam:
```python
# ĐỀ XUẤT — cho phép test không cần model
def init_components(self, asr_engine=None, vad_engine=None):
    self.asr_engine = asr_engine or TranscribeEngine(session_id=self.session_id)
    self.vad_processor = VADProcessor(..., engine_override=vad_engine, ...)
```
`FakeASREngine` trả text cấu hình trước với delay cấu hình trước ⇒ test được **logic phân câu, merge, cửa sổ, drop** **mà không nạp model nào**. Đây là thay đổi nhỏ nhưng biến phần lớn test từ "phút" thành "giây".

### 7.3 Chọn model cho tầng B — dùng model **nhỏ & nhanh** có sẵn cục bộ

| Model có sẵn | Kích thước | Vì sao phù hợp tầng B |
| :--- | :--- | :--- |
| **`SenseVoiceSmall-F32.gguf`** | 893 MB | **Non-autoregressive**, `models.yaml:42` ghi latency ~20 ms ⇒ test latency gần như tức thì ⭐ |
| **`Qwen3-ASR-0.6B-Q8_0.gguf`** | 811 MB | **Cùng family** với model mặc định 1.7B ⇒ **cùng code path**, nạp nhanh, compute ~1/3 ⭐ |
| `nemotron-3.5-asr-streaming-0.6b-F16.gguf` | 1.2 GB | Model **streaming thật** ⇒ dùng để kiểm chứng đường incremental (qwen3_asr `offline_llm` **không** có streaming hook) |

**Quy tắc:** tầng B dùng `qwen3-asr-0.6b` (đại diện code path) + `sensevoice-small` (latency tối thiểu); test đường streaming dùng `nemotron-3.5-streaming`.

### 7.4 Cấu hình pytest (hiện **chưa có** file config nào)

Tạo `pytest.ini` ở gốc repo:
```ini
[pytest]
testpaths = backend/tests
markers =
    slow: test cần model thật (tầng B/C)
    gpu:  test cần GPU
    full: E2E 3 model (tầng C, mặc định bỏ qua)
addopts = -m "not full" -q
```
* Fixture `session` scope="session" giữ model đã nạp giữa các test ⇒ **không nạp lại** model cho mỗi test (đây là nguyên nhân chính khiến test hiện tại chậm).
* Tầng B/C **tái dùng** `TRANSCRIBE_PERF_DEBUG` + `metrics_collector.generate_report()` thay vì tự đo lại.

### 7.5 Cách chạy

| Lệnh | Thời gian | Nội dung |
| :--- | :--- | :--- |
| `pytest` | **< 15 s** | Tầng A (mặc định, dùng hằng ngày) |
| `pytest -m "slow"` | ~60 s | Tầng A + B |
| `pytest -m "full" --speed 4` | 2–4 phút | Tầng C (chốt phase) |
| `python backend/tests/test_08_streaming_latency.py --model qwen3-asr-0.6b --speed 4 --seconds 10` | ~40 s | Đo KPI nhanh |

### 7.6 Sửa harness E2E hiện tại (bắt buộc)

`backend/tests/test_07_e2e_comparison.py:146-154` đẩy toàn bộ audio trong vòng `for` **không có `await`** ⇒ coroutine `_stream_asr_tokens` **không có cơ hội chạy** ⇒ **đường preview chưa bao giờ được test** (đây là lý do gốc khiến vấn đề O(N²) vô hình).
**Sửa:** thêm `await asyncio.sleep(...)` để **pacing** theo `chunk_ms / speed`. Kèm sửa report generator đang hardcode claim sai (`:271-272,288`: "Lock-Free", "4 Bậc", "3 Lớp Dedup") và thêm mục **"Hạn chế đã biết"** vào report sinh ra.

---

## 8. KPI — BASELINE → TARGET

| # | Chỉ số | Baseline | **Target** | Đo ở |
| :--- | :--- | :--- | :--- | :--- |
| K1 | **p95 preview latency** (câu 6 s, video dài 10 phút) | Tăng vô hạn theo độ dài câu | **< 250 ms**, **không tăng** theo thời gian video | P2.3 |
| K2 | **Thời gian tới preview đầu tiên** | ~600 ms speech + inference | **< 500 ms** | P2.8 |
| K3 | **Thời gian tới token dịch đầu tiên** | ~250–800 ms (sau commit) | **< 120 ms** | P3.3 |
| K4 | **E2E speech → phụ đề gốc chốt** (p95) | Không chặn trên (tới 50 s) | **< 1.2 s** ở câu ≤ 6 s | P2.1 + P1.6 |
| K5 | **Số câu bị mất** | Không đếm được (drop âm thầm) | **= 0**, mọi drop/merge có counter + log | P4.5 |
| K6 | **WER trên `wav_test/`** | Mốc so sánh | **Không xấu đi** (chênh ≤ 0.3 % tuyệt đối) | mọi task Phase 2 |
| K7 | **VRAM peak** | ~9–10 GB | **< 14 GB** (dùng dư cho độ chính xác) | P1.2 |
| K8 | **Backend ASR dùng CUDA** | ❌ (Vulkan) | ✅ `Backend: cuda` trong log | P1.1 |
| K9 | **Mọi control popup có tác dụng ngay** | 6–8 khoảng trống | **11/11 control pass** test P1.11 | P1.7–P1.11 |
| K10 | **0 crash khi đổi model lúc đang stream** | Nguy cơ use-after-free | 0 crash / 200 lần | P1.8b |
| K11 | **Capture latency client** | ~170–512 ms | **< 70 ms** | P3.4 |
| K12 | **Thời gian bộ test mặc định** | (không có tầng A) | **< 15 s** | §7 |

---

## 9. THỨ TỰ THỰC THI

| Sprint | Nội dung | Task | Cổng |
| :--- | :--- | :--- | :--- |
| **S0** (2–3 ngày) | Đo lường tối thiểu + sửa harness | P0.1 (instrument), §7.4 (`pytest.ini`), §7.6 (sửa pacing + report gen), seam tầng A | Bộ test tầng A chạy < 15 s; có số baseline K1/K2 |
| **S1** (3–4 ngày) | GPU + popup tức thời | **P1.1**, P1.7, P1.8, P1.9, P1.10, P1.11 | K8 ✅, K9 ✅, K10 ✅ |
| **S2** (2–3 ngày) | Độ chính xác nền tảng | P1.3, P1.4, P1.5, P1.2 | WER không xấu đi; K7 ✅ |
| **S3** (5–6 ngày) | Cắt O(N²) giữ nguyên độ chính xác | **P2.1**, **P2.2**, **P2.3**, P2.4, P2.8 | **K1, K2, K4 ✅**; K6 ✅ |
| **S4** (3–4 ngày) | Hiển thị nhanh | **P3.3**, P3.0, P3.1 | **K3 ✅**; TTS −33 % bytes |
| **S5** (3–4 ngày) | Client & transport | P3.2, P3.4, P3.5 | K11 ✅; latency phục hồi sau nghẽn |
| **S6** (3–4 ngày) | Dọn dẹp tuỳ chọn | P2.5 (đo WER), P2.6, P2.7, P2.9, P4.1–P4.6 | K5 ✅, K6 ✅, K12 ✅ |
| **S7** (tuỳ chọn) | Native | P4.7 | Chỉ nếu cần |

**Tổng:** ~26–34 engineer-day. **Đường tối thiểu để đạt 4 yêu cầu: S0 → S1 → S2 → S3 → S4** (~16–20 ngày), phần còn lại là tối ưu thêm.

**Nhịp:** mỗi sprint kết thúc bằng `pytest -m "slow"` (~60 s) + cập nhật `report/audit/04_measurements.md`. Chỉ chạy tầng C khi chốt phase.

---

## 10. RISK REGISTER

| # | Rủi ro | Khả năng | Tác động | Giảm thiểu |
| :--- | :--- | :--- | :--- | :--- |
| R1 | **P2.1** (cắt câu ở `MAX_DURATION`) cắt giữa từ ⇒ mất chữ | TRUNG BÌNH | **C4 — độ chính xác** | Ưu tiên `STABLE_PREFIX`; overlap 250 ms + trim trùng (P2.7); mảnh < `min_words` thì **gộp**, không drop; **đo WER bắt buộc** |
| R2 | **P2.3** (cửa sổ preview) — nếu `W < max_duration` thì preview mất ngữ cảnh | THẤP (thiết kế đã chặn) | C4 | Assert `preview_window_sec >= max_duration_sec` khi khởi động; feature flag `preview_window_sec=0` để tắt |
| R3 | **P1.6** (giảm silence) cắt giữa từ | TRUNG BÌNH | C4 | Đo WER; nếu xấu ⇒ giữ 600 và bù bằng P3.3 |
| R4 | **P1.8b** (đổi thứ tự lock) gây deadlock | THẤP | Treo backend | Một thứ tự duy nhất `_infer_lock → _shared_lock`; test stress có timeout |
| R5 | **P1.1** (đổi wheel CUDA) lỗi driver/OOM | THẤP–TB | Không chạy được | Test trên máy đích; đo VRAM; giữ khả năng quay lại wheel Vulkan |
| R6 | **P1.2** (F16) không có GGUF sẵn ⇒ phải tải và có thể không tốt hơn | TRUNG BÌNH | Mất thời gian | Đo A/B; **chỉ đổi nếu WER tốt hơn thật**; không có thì giữ Q8_0 |
| R7 | **P1.9** (pre-warm cả 3 VAD) tốn RAM/thời gian khởi động | THẤP | Khởi động chậm hơn | Pre-warm song song (`asyncio.gather` đã có ở `main.py:125`); hoặc chỉ pre-warm engine mặc định + engine đã từng dùng |
| R8 | **P3.1/P3.4** đổi giao thức/worklet ⇒ extension hỏng | TRUNG BÌNH | Mất tính năng | P3.0 version hoá; backend trước extension; fallback ScriptProcessor |
| R9 | **P2.5** (reuse preview) mất từ cuối câu | CAO nếu bật | C4 | **Mặc định TẮT**; chỉ bật sau khi đo WER |
| R10 | Tối ưu không đo được ⇒ không biết có hiệu quả | TRUNG BÌNH | Lãng phí effort | P0.1 instrument **trước**; mỗi sprint cập nhật `04_measurements.md` |
| R11 | Tầng A (FakeASREngine) không phản ánh hành vi thật | TRUNG BÌNH | Test pass nhưng thực tế lỗi | Mỗi phase **bắt buộc** chạy tầng B; tầng C khi chốt |

---

## 11. ROLLBACK

| Thay đổi | Cách rollback |
| :--- | :--- |
| P1.1 CUDA provider | Cài lại `transcribe-cpp-native` (Vulkan) |
| P1.2 quant/context | Đổi `active_model` / `n_ctx` trong config |
| P1.6 silence/hangover | Sửa `config.py` hoặc qua popup |
| P2.1 CommitManager | `SentenceConfig.enable_tier234 = False` |
| P2.3 cửa sổ preview | `ASRConfig.preview_window_sec = 0` |
| P2.4 fixed-rate | `ASRConfig.preview_fixed_rate = False` |
| P2.5 reuse preview | `ASRConfig.reuse_preview_for_commit = False` (mặc định đã tắt) |
| P2.6 normalizer | `ASRConfig.normalize_speech = False` |
| P2.9 int16 buffer | Đổi dtype buffer qua config |
| P3.0/P3.1 giao thức | Extension tự giảm cấp theo `protocol_version` |
| P3.4 worklet | Tự động fallback `ScriptProcessorNode` có log |
| P4.7 native | Không patch; dùng bản wheel nguyên gốc |

**Nguyên tắc:** mọi thay đổi **hành vi** đều có feature flag. Các fix **an toàn** (P1.3, P1.8b, P3.2, P4.4) **không** có flag — phải luôn bật.

---

## 12. DEFINITION OF DONE

Một task xong khi **tất cả** đúng:

1. ✅ Có **số đo trước/sau** (không chấp nhận "cảm thấy nhanh hơn").
2. ✅ Test tầng A tương ứng đã thêm và pass; **bộ test mặc định vẫn < 15 s**.
3. ✅ Với task Phase 2: **WER không xấu đi** (K6).
4. ✅ Counter/metric liên quan đã có để phát hiện tái phát.
5. ✅ Feature flag (nếu là thay đổi hành vi) đã có, mặc định hợp lý.
6. ✅ `report/audit/04_measurements.md` được cập nhật.
7. ✅ **Fail loudly:** mọi drop/merge/truncate/skip/fallback phải có counter **và** log — không còn đường thất bại âm thầm (bài học từ F-05, G1, G4).

---

## 13. NHỮNG VIỆC **KHÔNG** LÀM (bản cá nhân hoá)

| ❌ Không làm | Lý do |
| :--- | :--- |
| Model pool, per-session executor, multi-session correctness | **C2: chỉ 1 phiên.** Đã bỏ khỏi plan |
| Scale test 2 tab / nhiều tab | C2 |
| Hardening multi-tenant (auth, rate-limit, quota) | C1 |
| Fork sâu `external/transcribe.cpp` cho prefix reuse (N1) | Effort lớn; P2.1+P2.3 đã chặn chi phí ở mức đủ tốt cho 1 phiên |
| Gỡ `_infer_lock` | Thư viện C++ ghi rõ overlapping `run()` sẽ corrupt decode và **không tự lock** (`transcribe.h:11-20`) |
| Bật `reuse_preview_for_commit` mặc định | Rủi ro mất từ cuối câu — trái **C4** |
| Đặt `preview_window_sec < max_duration_sec` | Preview sẽ mất ngữ cảnh — trái **C4** và vi phạm nguyên tắc P2 |
| Tắt VAD để "giảm latency" | Sẽ không còn phụ đề nào (G6 / F-18) |
| Chạy tầng C (3 model) trong vòng lặp phát triển | Trái yêu cầu test nhanh |
| Tối ưu trước khi có số đo | Không chứng minh được cải thiện |
| Sửa vendored lib không theo `external/transcribe.cpp/AGENTS.md` | Quy ước repo (`uv run`, `clang-format.sh`, `validate.py`) |
| Đổi payload TTS sang binary mà không version hoá | Sẽ phá extension đang chạy |

---

## 14. TÓM TẮT MỘT TRANG

| | |
| :--- | :--- |
| **Ý tưởng cốt lõi** | **Giới hạn độ dài câu (≤ 6 s) và đặt cửa sổ preview = độ dài câu.** Khi đó preview thấy **đúng cùng ngữ cảnh** như commit ⇒ **chi phí bị chặn ở ~190 ms bất kể video dài bao nhiêu** mà **không mất độ chính xác**. Đây là cách biến O(N²) → O(N) mà vẫn phục vụ C4. |
| **Việc đầu tiên** | Instrument metrics + sửa harness E2E (đang feed **không có `await`** ⇒ chưa bao giờ test đường preview) + `pytest.ini` + seam engine giả ⇒ **test mặc định < 15 s** |
| **Việc giá trị nhất cho C3/C6** | **P1.1** (CUDA provider) + nhóm **P1.7–P1.11** (popup áp dụng ngay — hiện có **6 khoảng trống thật**, nặng nhất là đổi VAD engine có thể **tải model từ HuggingFace trong lúc giữ lock** ⇒ backend ngừng nhận audio) |
| **Việc giá trị nhất cho C5** | **P2.4** (nhịp preview cố định) + **P3.3** (streaming translation ⇒ dịch hiện sau ~120 ms thay vì ~800 ms) |
| **Việc giá trị nhất cho C4** | **P1.3** (sửa pre-roll lấy nhầm 300 ms câu trước) + **P2.1** (wire CommitManager) + **P2.6** (bật lại normalizer config) |
| **Rủi ro số 1** | P2.1/P2.7 cắt câu giữa từ ⇒ **bắt buộc đo WER**; ưu tiên `STABLE_PREFIX`, overlap + trim trùng, mảnh ngắn thì gộp chứ không drop |
| **Đường tối thiểu** | S0 → S1 → S2 → S3 → S4 (**~16–20 ngày**) đạt cả 4 yêu cầu |
| **Có thể bỏ** | Toàn bộ Phase 4 và P4.7 (native) — không cần cho 4 yêu cầu |

---

*Kế hoạch là tài liệu **read-only** — không có file mã nguồn nào bị sửa đổi. `file:line` phản ánh trạng thái code tại thời điểm audit.*
