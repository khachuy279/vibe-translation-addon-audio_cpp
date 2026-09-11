# TODO — backend_cpp + extension_firefox

> File trạng thái công việc tách riêng khỏi kế hoạch chi tiết.
> Chi tiết/bằng chứng: `report/backend_cpp_extension_performance_audit_implementation_plan.md` (§0.1–§0.8)
> Cập nhật: 2026-09-11 · Rev 16

`[x]` xong · `[~]` xong nhưng chờ quyết định · `[ ]` chưa làm

---

## ✅ PHASE 0 + 1 — Correctness (P0)

- [x] **W0.1** `IndentationError` trong `model_manager.py` — không còn cần (bạn đã phục hồi bản cũ)
- [x] **W0.2** Backup `scratch/backup_before_perf_plan/` + `backend_cpp/scripts/verify_all.py`
- [x] **W1.1** Bound + coalesce ASR token queue (P0-02) — 9 test
- [x] **W1.2** Bỏ dump conversion vô điều kiện (N-01) — 3 test
- [x] **W1.3** Lifecycle barrier ASR (P0-01) — 12 test
- [x] **W1.4** Một capture owner duy nhất (P1-04) — + hotfix Rev 4–10 cho iframe cross-origin, 10 test bất biến

## ✅ PHASE 2 — Performance

- [x] **W2.1** Baseline TRƯỚC khi sửa → `report/baseline_perf.json`
- [~] **W2.2** Preview growth gate (P1-01/P1-02)
  - Đo được **−52 % audio ASR, −49 % thời gian infer, transcript GIỐNG HỆT**
  - ⚠️ **Mặc định TẮT** theo quyết định của bạn (đổi nhịp cập nhật preview là hành vi người dùng thấy)
  - Bật khi cần: `config.asr.preview_min_growth_ratio` hoặc `--preview-growth-ratio`
  - ⚠️ Spike đã phủ định thiết kế gốc: `qwen3-asr-1.7b` có `supports_streaming = False`
- [ ] **W2.3** Preview normalization idempotent (N-03) — chưa làm
- [ ] **W2.4** Giảm copy chain (P1-03/N-02) — **đo rồi: chỉ ~0.7 ms/audio-giây ⇒ ROI rất thấp, đề xuất bỏ**
- [ ] **W2.5** Dedicated executors (P1-08) — ~~chưa làm; đáng làm hơn sau khi biết VAD nặng CPU~~ **đã đo: không cần, xem dưới**
- [ ] **W2.6** Audio dumper bounded (P1-09) — chưa làm; chỉ ảnh hưởng khi `dump_audio=True` (mặc định TẮT)
- [x] ~~**W2.5** Dedicated executors (P1-08)~~ → **ĐO RỒI: KHÔNG CẦN** (`scratch/measure_executor_pressure.py`)
  - dispatch delay (submit → worker chạy) **p50 0.06 ms**, p99 0.11 ms
  - max concurrent `to_thread` = **2** trên pool **16 worker** (mặc định `min(32, cpu+4)`)
  - VAD chỉ chiếm ~7 % một worker (7 ms mỗi chunk 100 ms)
  - ⇒ tối đa W2.5 loại bỏ được **< 1 ms**; lại thêm phức tạp shutdown + phải sửa `test_ws_handler`
  - (Ghi chú: VAD nặng GIL do `deepcopy` cấp Python — nhưng GIL là toàn process, executor riêng **không sửa được**)

## ⏸️ PHASE 3 — Transport & scale (chỉ khi benchmark chứng minh cần)

- [ ] **W3.1** Binary TTS transport (P2-09/P1-10) — chưa có bằng chứng cần
- [ ] **W3.2** Event-driven preview scheduling (P2-01) — **chồng lấn W2.2**, cân nhắc gộp
- [ ] **W3.3** Multi-session — chưa có bằng chứng cần

## ✅ PHASE 4 — Housekeeping

- [x] **W4.1** Logging `DEBUG` → `INFO`, DEBUG qua `BS_LOG_LEVEL`
- [x] **W4.2** `[ASR UTT LEVEL]` 1 dòng/commit ở INFO — **đã đúng sẵn, không cần sửa**
- [ ] **W4.3** Cache catalog/config cho `/api/*`
- [ ] **W4.4** `AudioCapture` log mỗi 200 chunk → DEBUG
- [ ] **W4.5** Dọn alias/wrapper trùng (chỉ sau khi behavior freeze)

---

## ✅ VIỆC PHÁT SINH (ngoài kế hoạch ban đầu)

- [x] **Điều tra VAD** (stage CPU lớn nhất mà audit chưa từng đo)
  - firered 108.7 / fsmn 57.9 / silero 13.3 ms mỗi audio-giây
  - firered là engine **đắt nhất** mà lại là default cũ
- [x] **Đổi VAD default → `fsmn-vad`** (−47 % CPU, transcript GIỐNG HỆT, cùng 8 segment)
- [x] Sửa **fallback âm thầm** của VAD engine (gõ sai tên → engine đắt nhất, im lặng)
- [x] **Điều tra "transcript trùng lặp"** → là **artifact của cách đếm**, không phải bug
      (`_process_translation_item` gửi 2 message `is_final` là **cố ý**, để tương thích ngược)
- [x] **`RESOURCE_LEAK_WARNING` +655 MB** → **dương tính giả** (load model một lần); đo 3 lần liên
      tiếp cho +644 / +6.4 / +4.8 MB ⇒ đã sửa detector để không báo động giả ở lần load đầu
- [x] **Audit load/chạy model dịch** → GPU đúng (33/33 layer); **sửa bug `setup_cuda_dll_paths()`
      đăng ký RỖNG** (lỗi indentation) — translation trước đó chỉ load được **do may mắn** thứ tự import
- [x] **Ngân sách độ trễ thật** → phát hiện **480 ms im lặng VAD (44 %)** bị bỏ sót; đính chính
      `hangover_ms` **không** cộng vào độ trễ
- [x] **Đổi `silence_duration_ms` 450 → 150** → độ trễ phụ đề **~1 073 → ~781 ms (−27 %)**,
      transcript GIỐNG HỆT

---

## 📊 CỔNG CHẤT LƯỢNG (hiện tại)

| Cổng | Kết quả |
|---|---|
| `python backend_cpp/scripts/verify_all.py` | **88/88 py, 10/10 js** |
| `python -m pytest backend_cpp/tests -q` | **197 passed, 2 skipped** |
| `python scratch/verify_guards_catch_regressions.py` | **16/16 mutations caught** |

## ✅ ĐÃ CHỐT (quyết định cuối)

- [x] **Model dịch: GIỮ `tencent` 7B** — ưu tiên chất lượng dịch (quyết định 2026-09-11).
  Đã đo 4 model (§0.7D): `xiaomi` 173 ms (2.61×) có lỗi từ vựng ("Rice" → "Cám"),
  `tencent-1.8b` 193 ms (2.34×) bỏ sót từ, `gemmax` 241 ms nhưng load 12.7 s.
  ⇒ **Không đổi model.** 435 ms còn lại là chi phí của sự lựa chọn chất lượng, chấp nhận.
- [x] **Trễ lồng tiếng TTS ~1.29 s: CHẤP NHẬN** — TTS là **tính năng phụ, không phải tính năng
  chính** (quyết định 2026-09-11). Đo được 499 ms/câu + 507 ms e2e, và **không ảnh hưởng đường
  phụ đề** (`e2e_asr_to_sub` 438.7 ms có TTS vs 435 ms không TTS) ⇒ không cần tối ưu.
- [x] **W2.2 preview gate**: giữ **mặc định TẮT** (đổi nhịp preview là hành vi người dùng thấy)

## 🎯 BƯỚC TIẾP THEO (ứng viên, theo ROI)

| # | Việc | Lợi ích đo được | Trạng thái |
|---|---|---|---|
| 1 | **W2.3** normalization idempotent | nhỏ, ≤ ~1 ms/audio-giây | chưa làm |
| 2 | **W4.3** cache catalog/config cho `/api/*` | nhỏ (chỉ lúc mở popup) | chưa làm |
| 3 | **W4.4** `AudioCapture` log mỗi 200 chunk → DEBUG | vệ sinh log | chưa làm |
| 4 | **W2.6** audio dumper bounded | chỉ khi `dump_audio=True` (mặc định TẮT) | chưa làm |
| 5 | **W4.5** dọn alias/wrapper trùng | vệ sinh code | chờ behavior freeze |
| — | **W2.4** / **W2.5** / **W3.1** | đã đo: ~0.7 ms, 0.06 ms, 0.285 ms | **đã đo: bỏ** |
| — | **W3.2** / **W3.3** | trùng W2.2 / chưa có bằng chứng | **đề xuất hoãn** |
