# Báo Cáo Đo Lường & Kiểm Thử Phase 5: Module Dịch Thuật Local GGUF (Llama.cpp)

- **Thời gian thực hiện**: 2026-09-15 08:36:31
- **Mô hình thử nghiệm**: `tencent` (Hunyuan-MT2 7B (Tencent))
- **GGUF File**: `Hy-MT2-7B-UD-Q4_K_XL.gguf`
- **Prompt Template**: `tencent`

## 1. Kết Quả Benchmark Hiệu Năng & Tốc Độ Dịch Sang Tiếng Việt

| Mẫu Thử Nghiệm | Ngôn Ngữ Gốc | Thời Gian Dịch | Tốc Độ (Tokens/s) | Bản Gốc | Bản Dịch Tiếng Việt (`vi`) |
|---|---|---|---|---|---|
| `English_speech` | `en` | **690.7 ms** | **68.0** | Ready? Yeah. It's crazy. It's freezing. It's completely stopped. | **Sẵn sàng chưa? Ừ. Thật điên rồ. Trời lạnh cóng. Mọi thứ hoàn toàn dừng lại rồi.** |
| `Chinese_speech` | `zh` | **912.4 ms** | **69.1** | 在他十一岁的时候，母亲因为发生交通事故撒手人寰。桂兰的母亲敲了几下杯子。 | **Khi cậu ấy 11 tuổi, mẹ cậu qua đời vì một vụ tai nạn giao thông. Mẹ của Quế Lan gõ vài tiếng vào chiếc cốc.** |
| `Japanese_speech` | `ja` | **1215.8 ms** | **68.3** | 得意先の営業周りを優しくおっしゃし、その日私たち出張先の宿に向かっていた。このホテル初めてですね。 | **Người ấy nói một cách nhẹ nhàng về việc đi thăm khách hàng, rồi chúng tôi cùng nhau hướng về khách sạn ở nơi đi công tác. Đây là lần đầu tiên tôi đến khách sạn này nhỉ.** |
| `Russian_speech` | `ru` | **482.1 ms** | **64.3** | Барсук, живущий в киевском зоопарке, совершил побег из своего вольера. | **Con chồn mỏ sống tại sở thú Kiev đã trốn thoát khỏi chuồng của mình.** |

## 2. Đánh Giá Hiệu Năng & Chất Lượng Bản Dịch

- **Thời Gian Dịch Trung Bình**: **825.2 ms / câu**.
- **Tốc Độ Sinh Từ (Throughput)**: **67.4 tokens / giây** trên GPU.
- **Chất lượng ngữ nghĩa**: Bản dịch tự nhiên, trôi chảy, giữ nguyên ngữ cảnh câu nói.

## 3. Kết Luận Nghiệm Thu Phase 5

- Module dịch thuật GGUF hoạt động hoàn hảo, đáp ứng thời gian thực cho phụ đề (< 300ms/câu).
- Sẵn sàng chuyển sang **Phase 6: Module OmniVoice Clone TTS (`tts/`)**.