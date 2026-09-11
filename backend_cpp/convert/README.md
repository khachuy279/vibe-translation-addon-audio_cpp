# Transcribe.cpp ASR Model Conversion Pipeline

Công cụ tự động hóa toàn bộ quy trình chuyển đổi (convert) các model Speech-to-Text (ASR) từ Hugging Face sang định dạng GGUF tương thích với `transcribe.cpp`, hỗ trợ lượng tử hóa (quantization), kiểm thử tự động (live engine validation), và tự động đăng ký vào `models.yaml`.

---

## 🚀 Cấu trúc thư mục

```
backend_cpp/convert/
├── convert_pipeline.py     # Script điều phối chính (Interactive CLI + One-line command)
├── sync_scripts.py         # Adapter layer & đồng bộ scripts/thư viện từ transcribe.cpp
├── validator.py            # Live engine validation (đo RTF, latency, backend, transcript)
├── models_yaml_updater.py  # Cập nhật an toàn models.yaml (atomic write + auto-backup .bak)
├── bin/                    # Nơi đặt transcribe-quantize.exe (tuỳ chọn)
├── scripts/                # Chứa các script convert theo family (qwen3_asr, sensevoice...)
│   └── lib/                # Thư viện dùng chung (gguf_common, hf_source, quant_policy...)
├── temp/                   # Thư mục tạm lưu tải model từ Hugging Face (kèm last_convert.log)
└── config.json             # Lưu cấu hình tái sử dụng (ví dụ đường dẫn quantize tool)
```

---

## 🛠️ Cài đặt yêu cầu

Tất cả dependencies cần thiết:
```bash
pip install -r backend_cpp/requirements.txt
```
Bao gồm:
- `gguf>=0.10.0`
- `transcribe-cpp>=0.2.3`
- `huggingface_hub>=0.20.0`
- `torch>=2.0.0`, `safetensors`, `librosa`, `soundfile`

---

## 📖 Hướng dẫn sử dụng

### 1. Chạy Interactive Mode (Tương tác từng bước)
Chỉ cần chạy:
```bash
python backend_cpp/convert/convert_pipeline.py
```
Script sẽ:
1. Nhập HF Repo ID (ví dụ: `neosophie/Qwen3-ASR-1.7B-JA`).
2. Tự động truy vấn Hugging Face API để lấy kích thước siblings và gợi ý Family.
3. Cho phép chọn Family (nếu cần chọn thủ công).
4. Tải model về thư mục tạm `backend_cpp/convert/temp/` (hoặc để converter tự xử lý).
5. Tự động chuyển đổi sang GGUF thông qua Adapter Layer theo đúng quy tắc CLI của family.
6. Lựa chọn preset lượng tử hóa (`Q8_0`, `Q4_K_M`, `BF16`...).
7. Chạy Live Validation bằng `transcribe_cpp` trên audio mẫu thực tế.
8. Copy file `.gguf` vào `backend_cpp/models/` và cập nhật `backend_cpp/models.yaml` (key luôn mang hậu tố quant, ví dụ: `qwen3-asr-1.7b-ja-bf16`).
9. Tùy chọn dọn dẹp thư mục tạm để giải phóng ổ cứng.

---

### 2. Chạy Command-Line (Một lệnh duy nhất)

```bash
# Convert model Qwen3-ASR fine-tune, giữ chất lượng gốc BF16 (chạy trực tiếp rất mượt trên GPU)
python backend_cpp/convert/convert_pipeline.py --family whisper --repo-id kotoba-tech/kotoba-whisper-v2.2 --quant BF16

# Convert và lượng tử hóa Q8_0 (nếu đã có transcribe-quantize binary)
python backend_cpp/convert/convert_pipeline.py --family whisper --repo-id kotoba-tech/kotoba-whisper-v2.2 --quant Q8_0 --quantize-bin "D:\vibe-translation-addon-transcribe_cpp\transcribe.cpp\build\bin\transcribe-quantize.exe"

# Chế độ Dry-Run (chỉ in các bước dự kiến mà không tải/convert)
python backend_cpp/convert/convert_pipeline.py \
    --family qwen3_asr \
    --repo-id neosophie/Qwen3-ASR-1.7B-JA \
    --dry-run
```

---

## 🎯 Các tham số dòng lệnh (CLI Flags)

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--family` | Tên family (ví dụ: `qwen3_asr`, `sensevoice`, `whisper`...) | Tự phát hiện / Hỏi |
| `--repo-id` | Hugging Face repo ID (ví dụ: `neosophie/Qwen3-ASR-1.7B-JA`) | Hỏi nếu thiếu |
| `--revision` | Branch / Tag / Commit SHA trên Hugging Face để pin phiên bản | `None` (latest) |
| `--quant` | Preset lượng tử hóa (`Q8_0`, `Q4_K_M`, `Q5_K_M`, `BF16`, `F16`) | `Q8_0` |
| `--quantize-bin` | Đường dẫn tới file thực thi `transcribe-quantize` | Tự tìm trong bin/build |
| `--backend` | Backend cho Live Validation (`auto`, `vulkan`, `cpu`, `cuda`) | `auto` |
| `--audio` | Đường dẫn file WAV test để chạy validation | `wav_test/test_e2e_audio.wav` |
| `--skip-validation` | Bỏ qua bước Live Validation | `False` |
| `--skip-quantize` | Bỏ qua bước quantize, xuất trực tiếp BF16/F16 | `False` |
| `--keep-temp` | Giữ lại thư mục tạm sau khi convert xong | `False` |
| `--dry-run` | Xem trước các bước thực thi mà không tải/convert | `False` |
| `--force` | Ghi đè cấu hình model trong `models.yaml` nếu đã tồn tại | `False` |
| `--offline` | Không tải mới script từ GitHub (dùng script đã lưu) | `False` |
| `--no-download` | Không tải trước HF snapshot; chuyển trực tiếp repo ID cho converter xử lý | `False` |

---

## 🔌 Adapter Layer cho Converter Scripts

Các converter script upstream của `transcribe.cpp` sử dụng một số kiểu tham số khác nhau:
1. **Standard style** (`qwen3_asr`, `whisper`, `sensevoice`, `moonshine`, `voxtral`, `canary`, etc.):
   `python convert-<family>.py <model> <out_path> [--repo-id <id>] [--revision <rev>] [--variant <var>]`
2. **Outdir style** (`granite_nar`, `medasr`):
   `python convert-<family>.py <model> --outdir <dir> [--repo-id <id>] [--revision <rev>]`
3. **OutFlag style** (`sortformer`):
   `python convert-<family>.py <model> --out <file> [--repo-id <id>]`
4. **Variant Flag**: Đa phần dùng `--variant`, riêng `gigaam` dùng `--variant-key`.

Hàm `build_converter_command()` trong `sync_scripts.py` tự động nhận diện `family` và lắp ráp đối số CLI chính xác 100%. Biến môi trường `TRANSCRIBE_MODELS_DIR` được truyền tự động vào tiến trình con để hướng mọi file tải về vào thư mục `backend_cpp/convert/temp/`.

---

## ⚙️ Về công cụ Quantization (`transcribe-quantize`)

`transcribe-quantize` là tool C++ được compile từ repo `transcribe.cpp/tools/transcribe-quantize`.
- **Vị trí tìm kiếm tự động:**
  1. `backend_cpp/convert/bin/transcribe-quantize.exe`
  2. `build/bin/transcribe-quantize.exe`
  3. `transcribe.cpp/build/bin/transcribe-quantize.exe`
  4. Biến môi trường `%PATH%`
- **Nếu chưa có binary:**
  Bạn hoàn toàn có thể chọn xuất định dạng gốc **`BF16`** hoặc **`F16`**. `transcribe.cpp` hỗ trợ trực tiếp BF16 trên GPU qua Vulkan backend (RTX 5060 Ti) với độ trễ cực thấp (RTF < 0.05x).

---

## 🛡️ Đăng ký an toàn vào `models.yaml`

- **Key model**: Luôn tự động gắn hậu tố quant (ví dụ `qwen3-asr-1.7b-ja-bf16`, `qwen3-asr-1.7b-ja-q8-0`) để tránh xung đột khi convert nhiều phiên bản quant của cùng 1 mô hình.
- **Tự động sao lưu**: File sao lưu `models.yaml.bak` được tạo tự động trước mỗi lần ghi.
- **Atomic Write**: Sử dụng cơ chế ghi file tạm rồi replace nguyên tử để chống gián đoạn dữ liệu.
- *Lưu ý*: Quá trình dump YAML an toàn chuẩn hóa lại cấu trúc từ điển; các dòng comment tự do có thể được định dạng lại.

---

## 🧪 Kiểm thử độc lập (Validation)

Có thể chạy thử kiểm tra bất kỳ file GGUF nào độc lập:
```bash
python backend_cpp/convert/validator.py backend_cpp/models/Qwen3-ASR-1.7B-Q8_0.gguf --audio wav_test/test_e2e_audio.wav
```
Hiển thị:
- Backend sử dụng (`Vulkan0`, `CPU`...)
- Độ trễ suy luận (`Infer time`), thời lượng audio, `Real-Time Factor (RTF)`
- Kết quả transcript nhận dạng thực tế
- Tỷ lệ khớp với reference text (nếu có `--reference-text`)
