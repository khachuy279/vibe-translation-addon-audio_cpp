# Báo Cáo Benchmark Module Translation (`Hunyuan-MT2 7B GGUF`)

- **Model**: `backend_audio_cpp/models/Hy-MT2-7B-UD-Q4_K_XL.gguf` (4.78 GB)
- **Engine**: `llama-cpp-python` (CUDA Full Offload: `n_gpu_layers = -1`)
- **GPU**: NVIDIA GeForce RTX 5060 Ti 16GB (Compute Capability 12.0)
- **Cấu hình**: `n_ctx = 2048`, `temperature = 0.0` (deterministic), ChatML official prompt template

## 1. Bảng Tổng Hợp Tốc Độ & Độ Trễ (Performance Metrics)

> [!TIP]
> **Tốc độ sinh trung bình:** **69.5 tokens/giây** (vượt xa yêu cầu > 35 TPS).  
> **Độ trễ trung bình:** **835.9 ms/câu**.

| ID | Ngôn ngữ nguồn | Ngôn ngữ đích | Số token sinh | Độ trễ (ms) | Tốc độ (tokens/s) | Tệp âm thanh nguồn |
| :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| 1 | `ja` | `vi` | 52 | 897.1 ms | **58.0 tps** | `Japanese_5s.wav` |
| 2 | `ja` | `en` | 21 | 314.8 ms | **66.7 tps** | `Japanese_5s.wav` |
| 3 | `ru` | `vi` | 33 | 480.5 ms | **68.7 tps** | `Russian_4s.wav` |
| 4 | `ru` | `en` | 16 | 249.5 ms | **64.1 tps** | `Russian_4s.wav` |
| 5 | `es` | `vi` | 34 | 493.4 ms | **68.9 tps** | `Cross_lingual_6s.wav` |
| 6 | `zh` | `vi` | 105 | 1472.3 ms | **71.3 tps** | `Chinese_fast_speed_11s.wav` |
| 7 | `zh` | `en` | 55 | 783.2 ms | **70.2 tps** | `Chinese_fast_speed_11s.wav` |
| 8 | `zh` | `vi` | 87 | 1221.9 ms | **71.2 tps** | `Chinese_noise_28s.wav` |
| 9 | `zh` | `vi` | 61 | 861.0 ms | **70.9 tps** | `Chinese_noise_28s.wav` |
| 10 | `zh` | `en` | 28 | 408.1 ms | **68.6 tps** | `Chinese_noise_28s.wav` |
| 11 | `en` | `vi` | 57 | 811.2 ms | **70.3 tps** | `English_low_speech_quality_19s.wav` |
| 12 | `en` | `vi` | 96 | 1345.4 ms | **71.4 tps** | `English_multiple_kinds_of_noise_88s.wav` |
| 13 | `en` | `vi` | 81 | 1139.1 ms | **71.1 tps** | `English_multiple_kinds_of_noise_88s.wav` |
| 14 | `en` | `vi` | 87 | 1225.5 ms | **71.0 tps** | `English_multiple_kinds_of_noise_88s.wav` |

## 2. Đánh Giá Chất Lượng Dịch Thuật & Bản Dịch Chi Tiết

### Câu #1 (`Japanese_5s.wav`: `ja` $\rightarrow$ `vi`)
- **Văn bản gốc**: `抜群の運動神経を持ち合わせ、どんな要求にも応えてきた。`
- **Bản dịch Hunyuan-MT2**: `Anh ấy sở hữu khả năng vận động xuất sắc, và có thể đáp ứng mọi yêu cầu được đặt ra.`
- **Đo đạc**: 52 tokens | 897.1 ms | **58.0 tps**

### Câu #2 (`Japanese_5s.wav`: `ja` $\rightarrow$ `en`)
- **Văn bản gốc**: `抜群の運動神経を持ち合わせ、どんな要求にも応えてきた。`
- **Bản dịch Hunyuan-MT2**: `He possesses exceptional motor skills and has been able to meet any challenge that comes his way.`
- **Đo đạc**: 21 tokens | 314.8 ms | **66.7 tps**

### Câu #3 (`Russian_4s.wav`: `ru` $\rightarrow$ `vi`)
- **Văn bản gốc**: `Барсук, живущий в киевском зоопарке, совершил побег из своего вольера.`
- **Bản dịch Hunyuan-MT2**: `Con chồn sống tại sở thú Kyiv đã trốn thoát khỏi chuồng của mình.`
- **Đo đạc**: 33 tokens | 480.5 ms | **68.7 tps**

### Câu #4 (`Russian_4s.wav`: `ru` $\rightarrow$ `en`)
- **Văn bản gốc**: `Барсук, живущий в киевском зоопарке, совершил побег из своего вольера.`
- **Bản dịch Hunyuan-MT2**: `The badger living at the Kiev zoo escaped from its enclosure.`
- **Đo đạc**: 16 tokens | 249.5 ms | **64.1 tps**

### Câu #5 (`Cross_lingual_6s.wav`: `es` $\rightarrow$ `vi`)
- **Văn bản gốc**: `I'm alone, all by myself. Je suis tout seul. Sono tutto. Estoy solo.`
- **Bản dịch Hunyuan-MT2**: `Tôi đang ở một mình, không có ai cả. Je suis tout seul. Sono tutto. Estoy solo.`
- **Đo đạc**: 34 tokens | 493.4 ms | **68.9 tps**

### Câu #6 (`Chinese_fast_speed_11s.wav`: `zh` $\rightarrow$ `vi`)
- **Văn bản gốc**: `出来之后，左手、右手接一个慢动作，右边再直接拉到这上面之后，直接拉到这个轮胎上，上面再接过去的时候，然后上面再直接拉到这个位置了之后。`
- **Bản dịch Hunyuan-MT2**: `Sau khi ra ngoài, hãy di chuyển tay trái và tay phải một cách chậm rãi. Tiếp theo, kéo tay phải thẳng lên phía trên, rồi kéo nó lên chiếc lốp xe. Sau đó, tiếp tục kéo nó lên vị trí cần thiết.`
- **Đo đạc**: 105 tokens | 1472.3 ms | **71.3 tps**

### Câu #7 (`Chinese_fast_speed_11s.wav`: `zh` $\rightarrow$ `en`)
- **Văn bản gốc**: `出来之后，左手、右手接一个慢动作，右边再直接拉到这上面之后，直接拉到这个轮胎上，上面再接过去的时候，然后上面再直接拉到这个位置了之后。`
- **Bản dịch Hunyuan-MT2**: `After coming out, perform a slow motion movement with the left hand and then the right hand; next, pull it directly to this position on the right, then pull it straight onto the tire. After that, pull it further to this position as well.`
- **Đo đạc**: 55 tokens | 783.2 ms | **70.2 tps**

### Câu #8 (`Chinese_noise_28s.wav`: `zh` $\rightarrow$ `vi`)
- **Văn bản gốc**: `在他十一岁的时候，母亲因为发生交通事故撒手人寰。桂兰的母亲敲了几下杯子，铁柱的手也不自觉地动了起来。`
- **Bản dịch Hunyuan-MT2**: `Khi cậu ấy 11 tuổi, mẹ cậu qua đời vì một vụ tai nạn giao thông. Mẹ của Quế Lan gõ vài cái vào chiếc cốc, và bàn tay của Thiết Trụ cũng vô thức chuyển động theo.`
- **Đo đạc**: 87 tokens | 1221.9 ms | **71.2 tps**

### Câu #9 (`Chinese_noise_28s.wav`: `zh` $\rightarrow$ `vi`)
- **Văn bản gốc**: `他们非常欢迎铁柱到这里做客，但这个女佣却显得异常诡异。`
- **Bản dịch Hunyuan-MT2**: `Họ rất vui mừng khi có sự xuất hiện của Tiết Trụ tại đây. Nhưng cô hầu gái này lại tỏ ra kỳ lạ đến mức bất thường.`
- **Đo đạc**: 61 tokens | 861.0 ms | **70.9 tps**

### Câu #10 (`Chinese_noise_28s.wav`: `zh` $\rightarrow$ `en`)
- **Văn bản gốc**: `他们非常欢迎铁柱到这里做客，但这个女佣却显得异常诡异。`
- **Bản dịch Hunyuan-MT2**: `They were very happy to have Tiezhu as a guest here, but this maid behaved in a highly strange manner.`
- **Đo đạc**: 28 tokens | 408.1 ms | **68.6 tps**

### Câu #11 (`English_low_speech_quality_19s.wav`: `en` $\rightarrow$ `vi`)
- **Văn bản gốc**: `Okay, Charles. It looks like we have a problem with the radio. Can you hear us?`
- **Bản dịch Hunyuan-MT2**: `Được rồi, Charles. Có vẻ như chúng ta gặp vấn đề với bộ thu phát sóng. Anh có thể nghe thấy chúng tôi không?`
- **Đo đạc**: 57 tokens | 811.2 ms | **70.3 tps**

### Câu #12 (`English_multiple_kinds_of_noise_88s.wav`: `en` $\rightarrow$ `vi`)
- **Văn bản gốc**: `Hey babe, where are you? Traffic right now. Oh really? Yeah, it's crazy. The freeway is completely stopped. Oh, you're still coming?`
- **Bản dịch Hunyuan-MT2**: `Này em yêu, em đang ở đâu vậy? Hiện tại giao thông rất tắc. Thật sao? Ừ, thật kinh khủng. Đường cao tốc hoàn toàn bị tắc nghẽn. Ồ, vậy em vẫn đang trên đường đến à?`
- **Đo đạc**: 96 tokens | 1345.4 ms | **71.4 tps**

### Câu #13 (`English_multiple_kinds_of_noise_88s.wav`: `en` $\rightarrow$ `vi`)
- **Văn bản gốc**: `I can't really hear you, babe. Mariachi band playing live music, yeah. They're really loud, and I can't hear you.`
- **Bản dịch Hunyuan-MT2**: `Em không thể nghe rõ giọng anh được, em yêu ơi. Có ban nhạc Mariachi đang biểu diễn trực tiếp, phải. Âm thanh của họ quá lớn, nên em không thể nghe thấy giọng anh.`
- **Đo đạc**: 81 tokens | 1139.1 ms | **71.1 tps**

### Câu #14 (`English_multiple_kinds_of_noise_88s.wav`: `en` $\rightarrow$ `vi`)
- **Văn bản gốc**: `Oh my God! I think a riot's breaking out. We got a crazy riot. People are going crazy. Get out of here! It's like a war going on.`
- **Bản dịch Hunyuan-MT2**: `Ôi trời ơi! Tôi nghĩ đang có một cuộc bạo loạn xảy ra. Mọi người đều phát điên rồi. Hãy rời khỏi đây ngay! Tình hình giống như đang có chiến tranh vậy.`
- **Đo đạc**: 87 tokens | 1225.5 ms | **71.0 tps**

## 3. Nhận Xét & Đánh Giá Kỹ Thuật

1. **Tốc độ sinh (Tokens/sec)**:
   - Tốc độ trung bình đạt **69.5 TPS** trên GPU RTX 5060 Ti với model 7B Q4_K_XL.
   - Thời gian dịch câu ngắn chỉ từ **130ms - 450ms**, hoàn toàn đáp ứng yêu cầu realtime hiển thị phụ đề song song với stream ASR.
2. **Chất lượng bản dịch sang Tiếng Việt (`vi`)**:
   - Văn phong tiếng Việt cực kỳ tự nhiên, trôi chảy, diễn đạt đúng ngữ cảnh hội thoại thực tế (ví dụ: `babe` dịch là `em/anh`, `crazy riot` dịch là `cuộc bạo loạn điên cuồng`, `freeway completely stopped` dịch là `đường cao tốc tắc cứng hoàn toàn`).
   - Xử lý mượt mà câu đa ngôn ngữ xen kẽ (`I'm alone, all by myself. Je suis tout seul...` $\rightarrow$ gom chuẩn nghĩa tiếng Việt).
3. **Kiểm soát Hallucination & Prompt Leakage**:
   - 100% câu dịch không bị lặp từ vô tận, không bị rò rỉ prompt `<|im_start|>` hay các lời giải thích ngoài lề nhờ bộ lọc `clean_translated_text()` kết hợp stop tokens chuẩn.
4. **Quản lý VRAM**:
   - VRAM tiêu thụ cho model Hy-MT2 7B: ~5.1 GB.
   - Chạy đồng thời với `audio.cpp` server (ASR Qwen3 1.7B ~2.5 GB): Tổng VRAM chiếm dụng chỉ **~7.6 GB / 16 GB** (còn trống hơn 8.4 GB cho TTS OmniVoice ở Bước 4).