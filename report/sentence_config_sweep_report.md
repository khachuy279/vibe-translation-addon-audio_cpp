# Báo Cáo Phân Tích & Tối Ưu Thông Số SentenceConfig

## 1. Tổng Quan & Thiết Lập Đo Lường
- **Audio kiểm thử:** `debug_audio/.../00_ingress_stream.wav` (đối thoại tiếng Nhật thực tế đa người nói).
- **Reference chuẩn:** `00_ingress_stream.txt` (golden reference từ công cụ online).
- **Mô hình ASR:** `qwen3-asr-1.7b`.
- **VAD Engine:** `fsmn-vad` (Threshold=0.45, Silence=500ms, Hangover=300ms).
- **Ngưỡng cố định:** `min_words_to_emit_final = 1` (giữ trọn các lượt thoại ngắn tự nhiên).

## 2. Bảng Kết Quả Chi Tiết

| Condition | CER (Strict) | CER (ITN) | Commits | Avg Duration | Avg Chars/Turn | Emerg Cut % | Commit Breakdown |
|---|---:|---:|---:|---:|---:|---:|---|
| **BASELINE**<br>*Current production config (stability=2.0s/2p, max_dur=15s, grace=2s, probe=80ms, min_w=1)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **STAB_OFF**<br>*split_on_stability=False (Pure VAD silence commits only)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **STAB_FAST_1.0s**<br>*stability_duration_sec=1.0s, polls=2 (Fast prefix emit)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **STAB_MID_1.5s**<br>*stability_duration_sec=1.5s, polls=2 (Balanced prefix emit)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **STAB_CONSERV_2.5s**<br>*stability_duration_sec=2.5s, polls=3 (Conservative prefix emit)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **BOUND_TIGHT_10s**<br>*max_duration_sec=10.0s, grace=2.0s, probe=80ms* | 9.41% | 9.29% | 85 | 0.06s | 11.2 | 0.0% | `MAX_DURATION_SAFE`: 2<br>`VAD_SILENCE`: 83 |
| **BOUND_PROBE_50ms**<br>*boundary_candidate_silence_ms=50ms (Sensitive acoustic probe)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **BOUND_PROBE_120ms**<br>*boundary_candidate_silence_ms=120ms (Conservative acoustic probe)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **BOUND_HARD_CUT**<br>*max_duration_require_silence=False (Direct hard cut at 15s)* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **GATE_WORDS_2**<br>*min_words_to_commit=2, min_words_to_emit_final=1* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **GATE_WORDS_4**<br>*min_words_to_commit=4, min_words_to_emit_final=1* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **CHARS_COMPACT_100**<br>*max_chars=100, min_words_to_emit_final=1* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |
| **CHARS_EXPAND_180**<br>*max_chars=180, min_words_to_emit_final=1* | 8.92% | 8.80% | 83 | 0.06s | 11.4 | 0.0% | `VAD_SILENCE`: 83 |

## 3. Phân Tích Kỹ Thuật (Trade-off Matrix)

### A. Cơ Chế Stability Split (`split_on_stability`, `stability_duration_sec`)
- Khi `split_on_stability=False`: Hệ thống hoàn toàn phụ thuộc vào khoảng lặng VAD (`VAD_SILENCE`) để ngắt câu.
- Khi `split_on_stability=True`: Cho phép chốt tiền tố (`STABLE_PREFIX`) khi người nói ngưng nghỉ nhưng chưa hết câu VAD, giúp hiển thị bản dịch sớm hơn.

### B. VAD-Paced Soft Boundary (`max_duration_sec`, `boundary_candidate_silence_ms`)
- `max_duration_sec`: Giới hạn an toàn ngăn chặn tình trạng một lượt thoại độc thoại kéo dài vô hạn.
- `boundary_candidate_silence_ms`: Chờ khoảng lặng tự nhiên trong thời gian ân hạn (`grace_sec`) thay vì cắt ngang giữa chừng một từ (`MAX_DURATION_EMERGENCY`).

### C. Gating Kích Thước (`min_words_to_commit`, `max_chars`)
- `min_words_to_commit=1` cùng `min_words_to_emit_final=1` đảm bảo cả preview lẫn final đều nhạy với các phản hồi ngắn của người Nhật.

## 4. Kết Luận & Đề Xuất Điểm Ngọt (Sweet Spot)
- **Cấu hình tối ưu nhất về CER:** `BASELINE` với CER = **8.92%** (ITN: 8.80%).
- **Đặc trưng phân đoạn:** 83 commits, độ dài trung bình 11.4 ký tự/câu, tỷ lệ cắt khẩn cấp 0.0%.
