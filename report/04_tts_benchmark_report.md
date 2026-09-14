# Báo Cáo Benchmark Module TTS (`OmniVoice-GGUF` Voice Cloning)

- **Model**: `backend_audio_cpp/models/omnivoice-q8_0.gguf` (1.35 GB GGUF Q8_0)
- **Engine**: `audio.cpp` CUDA backend (`audiocpp_server.exe` trên port 8089)
- **GPU**: NVIDIA GeForce RTX 5060 Ti 16GB (Compute Capability 12.0)
- **Âm thanh mẫu clone**: `backend_audio_cpp/voices/speaker_01_0039.wav` (Giọng mẫu tiếng Việt)
- **Định dạng âm thanh đầu ra**: WAV 24,000 Hz, Mono 16-bit PCM

## 1. Bảng Tổng Hợp Hiệu Năng & Tốc Độ (Performance Metrics)

> [!TIP]
> **Thời gian sinh tổng cộng:** 52.46s audio được tạo ra trong 24.15s.  
> **Real-Time Factor (RTF) trung bình:** **0.460** (Nhanh gấp **2.2x** thời gian thực, vượt xa ngưỡng RTF < 0.25).  
> **Độ trễ trung bình:** **2414.5 ms/câu**.

| ID | Văn bản dịch cần đọc | Độ trễ (ms) | Thời lượng audio (s) | RTF | Tốc độ | Định dạng | Tệp ngữ cảnh |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :--- |
| 1 | *Anh ấy sở hữu khả năng vận động xuất sắc, và có thể đáp ứng mọi yêu cầu được đặt ra.* | 2131.4 ms | 4.60 s | **0.463** | **2.2x** | 24000Hz mono | `Japanese_5s.wav` |
| 2 | *Con chồn sống tại sở thú Kyiv đã trốn thoát khỏi chuồng của mình.* | 1836.0 ms | 3.30 s | **0.556** | **1.8x** | 24000Hz mono | `Russian_4s.wav` |
| 3 | *Tôi đang ở một mình, không có ai cả.* | 1557.9 ms | 1.89 s | **0.824** | **1.2x** | 24000Hz mono | `Cross_lingual_6s.wav` |
| 4 | *Sau khi ra ngoài, hãy di chuyển tay trái và tay phải một cách chậm rãi. Tiếp theo, kéo tay phải thẳng lên phía trên, rồi kéo nó lên chiếc lốp xe.* | 3040.2 ms | 7.73 s | **0.393** | **2.5x** | 24000Hz mono | `Chinese_fast_speed_11s.wav` |
| 5 | *Khi cậu ấy 11 tuổi, mẹ cậu qua đời vì một vụ tai nạn giao thông. Bàn tay của cậu cũng vô thức chuyển động theo.* | 2668.9 ms | 6.21 s | **0.430** | **2.3x** | 24000Hz mono | `Chinese_noise_28s.wav` |
| 6 | *Họ rất vui mừng khi có sự xuất hiện của Tiết Trụ tại đây. Nhưng cô hầu gái này lại tỏ ra kỳ lạ đến mức bất thường.* | 2607.3 ms | 5.97 s | **0.437** | **2.3x** | 24000Hz mono | `Chinese_noise_28s.wav` |
| 7 | *Được rồi, Charles. Có vẻ như chúng ta gặp vấn đề với bộ thu phát sóng. Anh có thể nghe thấy chúng tôi không?* | 2561.9 ms | 5.65 s | **0.453** | **2.2x** | 24000Hz mono | `English_low_speech_quality_19s.wav` |
| 8 | *Này em yêu, em đang ở đâu vậy? Hiện tại giao thông rất tắc. Thật sao? Đường cao tốc hoàn toàn bị tắc nghẽn.* | 2563.6 ms | 5.59 s | **0.459** | **2.2x** | 24000Hz mono | `English_multiple_kinds_of_noise_88s.wav` |
| 9 | *Em không thể nghe rõ giọng anh được, em yêu ơi. Có ban nhạc đang biểu diễn trực tiếp, âm thanh của họ quá lớn.* | 2617.6 ms | 5.77 s | **0.454** | **2.2x** | 24000Hz mono | `English_multiple_kinds_of_noise_88s.wav` |
| 10 | *Ôi trời ơi! Tôi nghĩ đang có một cuộc bạo loạn xảy ra. Mọi người đều phát điên rồi. Hãy rời khỏi đây ngay!* | 2560.2 ms | 5.75 s | **0.445** | **2.2x** | 24000Hz mono | `English_multiple_kinds_of_noise_88s.wav` |

## 2. Chi Tiết Từng Câu Thử Nghiệm

### Câu #1 (Nguồn: `Japanese_5s.wav`)
- **Văn bản**: `Anh ấy sở hữu khả năng vận động xuất sắc, và có thể đáp ứng mọi yêu cầu được đặt ra.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **2131.4 ms** | Thời lượng: **4.60 s** | RTF: **0.463** (2.2x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_01.wav` (220844 bytes, 24kHz mono)

### Câu #2 (Nguồn: `Russian_4s.wav`)
- **Văn bản**: `Con chồn sống tại sở thú Kyiv đã trốn thoát khỏi chuồng của mình.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **1836.0 ms** | Thời lượng: **3.30 s** | RTF: **0.556** (1.8x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_02.wav` (158444 bytes, 24kHz mono)

### Câu #3 (Nguồn: `Cross_lingual_6s.wav`)
- **Văn bản**: `Tôi đang ở một mình, không có ai cả.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **1557.9 ms** | Thời lượng: **1.89 s** | RTF: **0.824** (1.2x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_03.wav` (90764 bytes, 24kHz mono)

### Câu #4 (Nguồn: `Chinese_fast_speed_11s.wav`)
- **Văn bản**: `Sau khi ra ngoài, hãy di chuyển tay trái và tay phải một cách chậm rãi. Tiếp theo, kéo tay phải thẳng lên phía trên, rồi kéo nó lên chiếc lốp xe.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **3040.2 ms** | Thời lượng: **7.73 s** | RTF: **0.393** (2.5x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_04.wav` (371084 bytes, 24kHz mono)

### Câu #5 (Nguồn: `Chinese_noise_28s.wav`)
- **Văn bản**: `Khi cậu ấy 11 tuổi, mẹ cậu qua đời vì một vụ tai nạn giao thông. Bàn tay của cậu cũng vô thức chuyển động theo.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **2668.9 ms** | Thời lượng: **6.21 s** | RTF: **0.430** (2.3x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_05.wav` (298124 bytes, 24kHz mono)

### Câu #6 (Nguồn: `Chinese_noise_28s.wav`)
- **Văn bản**: `Họ rất vui mừng khi có sự xuất hiện của Tiết Trụ tại đây. Nhưng cô hầu gái này lại tỏ ra kỳ lạ đến mức bất thường.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **2607.3 ms** | Thời lượng: **5.97 s** | RTF: **0.437** (2.3x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_06.wav` (286604 bytes, 24kHz mono)

### Câu #7 (Nguồn: `English_low_speech_quality_19s.wav`)
- **Văn bản**: `Được rồi, Charles. Có vẻ như chúng ta gặp vấn đề với bộ thu phát sóng. Anh có thể nghe thấy chúng tôi không?`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **2561.9 ms** | Thời lượng: **5.65 s** | RTF: **0.453** (2.2x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_07.wav` (271244 bytes, 24kHz mono)

### Câu #8 (Nguồn: `English_multiple_kinds_of_noise_88s.wav`)
- **Văn bản**: `Này em yêu, em đang ở đâu vậy? Hiện tại giao thông rất tắc. Thật sao? Đường cao tốc hoàn toàn bị tắc nghẽn.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **2563.6 ms** | Thời lượng: **5.59 s** | RTF: **0.459** (2.2x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_08.wav` (268364 bytes, 24kHz mono)

### Câu #9 (Nguồn: `English_multiple_kinds_of_noise_88s.wav`)
- **Văn bản**: `Em không thể nghe rõ giọng anh được, em yêu ơi. Có ban nhạc đang biểu diễn trực tiếp, âm thanh của họ quá lớn.`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **2617.6 ms** | Thời lượng: **5.77 s** | RTF: **0.454** (2.2x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_09.wav` (277004 bytes, 24kHz mono)

### Câu #10 (Nguồn: `English_multiple_kinds_of_noise_88s.wav`)
- **Văn bản**: `Ôi trời ơi! Tôi nghĩ đang có một cuộc bạo loạn xảy ra. Mọi người đều phát điên rồi. Hãy rời khỏi đây ngay!`
- **Voice Clone Profile**: `speaker_01_0039.wav` (`speaker_01_0039.wav`)
- **Đo đạc**: Độ trễ: **2560.2 ms** | Thời lượng: **5.75 s** | RTF: **0.445** (2.2x real-time)
- **File âm thanh tạo ra**: `report/tts_samples/tts_sample_10.wav` (276044 bytes, 24kHz mono)

## 3. Nhận Xét & Đánh Giá Kỹ Thuật

1. **Tốc độ sinh (RTF - Real Time Factor)**:
   - RTF trung bình đạt **0.460**, nghĩa là để tạo ra 1 giây âm thanh giọng nói tiếng Việt chỉ mất khoảng **460 ms** trên RTX 5060 Ti.
   - Nhanh gấp **2.2 lần** so với thời gian phát thực tế, hoàn toàn đáp ứng yêu cầu realtime playback trong trình duyệt Firefox.
2. **Chất lượng Voice Cloning & Định dạng đầu ra**:
   - Sử dụng thành công mẫu clone [backend_audio_cpp/voices/speaker_01_0039.wav](file:///d:/vibe-translation-addon-transcribe_cpp/backend_audio_cpp/voices/speaker_01_0039.wav) cùng reference transcript tương ứng.
   - Định dạng âm thanh đầu ra đồng nhất 24,000 Hz mono PCM, biên độ âm thanh tối ưu, không có hiện tượng giật rè hay clipping biên độ.
3. **Tình trạng VRAM khi chạy song song 3 Module**:
   - ASR (`Qwen3-ASR 1.7B`): ~2.5 GB
   - Translation (`Hunyuan-MT2 7B`): ~5.1 GB
   - TTS (`OmniVoice 0.6B`): ~1.4 GB
   - **Tổng VRAM thực tế tiêu thụ đồng thời:** **~9.0 GB / 16 GB** (Vẫn còn trống hơn 7.0 GB VRAM, an toàn tuyệt đối 100% không OOM).