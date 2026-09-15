// AudioWorklet processor for low-latency audio capture
// Chạy trên AUDIO THREAD riêng (không phải main thread), nên không bị jank bởi
// DOM/render/base64. Đây là đường capture CHÍNH của extension (P3.4).
//
// Đặc điểm:
// - Resample từ rate của AudioContext về 16 kHz bằng nội suy tuyến tính có mang pha
//   (`phase`) giữa các block ⇒ không mất mẫu, không cần đệm residual.
// - Gom đủ `chunkSize` mẫu (mặc định 1024 = 64 ms @16 kHz) rồi gửi MỘT message
//   với TRANSFERABLE ArrayBuffer ⇒ zero-copy sang main thread.
// - Cấp phát đúng một Int16Array mỗi chunk (không cấp phát mỗi `process()`),
//   tránh áp lực GC trên audio thread.
// - Mỗi chunk mang `captureTimestamp` (theo `currentTime` của AudioContext) để
//   backend đóng dấu frame VAD chính xác.
//
// LƯU Ý: `sampleRate` và `currentTime` là biến toàn cục do AudioWorkletGlobalScope cung cấp.

const DEFAULT_TARGET_RATE = 16000;
const DEFAULT_CHUNK_SIZE = 1024;

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetSampleRate || DEFAULT_TARGET_RATE;
    this.chunkSize = opts.chunkSize || DEFAULT_CHUNK_SIZE;
    // `sampleRate` = rate của AudioContext (thường 48000). Tỉ lệ resample.
    this.ratio = sampleRate / this.targetRate;

    // Pha dư của bộ resample (chỉ số nguồn dạng phân số còn nợ cho block kế tiếp).
    this.phase = 0;
    this.buffer = new Int16Array(this.chunkSize);
    this.bufferIndex = 0;

    // Mốc thời gian của block ĐẦU TIÊN (theo timeline context). Timestamp của mỗi
    // chunk được suy ra từ mốc này + số mẫu đã phát ⇒ không trôi theo thời gian dài.
    this.firstBlockTime = 0;
    // Tổng số mẫu đã phát ra (ở targetRate).
    this.emittedSamples = 0;
    this.started = false;
    this.framesEmitted = 0;
    this.resampleBuffer = null;

    this.port.onmessage = (ev) => {
      // Cho phép main thread hỏi trạng thái (dùng cho watchdog fallback).
      if (ev.data && ev.data.type === "ping") {
        this.port.postMessage({ type: "pong", framesEmitted: this.framesEmitted });
      }
      // F-44: video bị TUA ⇒ bỏ toàn bộ mẫu đang đệm + pha resample + mốc thời gian.
      // Không reset thì audio thu trước khi tua sẽ lẫn vào câu đầu tiên sau khi tua.
      if (ev.data && ev.data.type === "reset") {
        this.phase = 0;
        this.bufferIndex = 0;
        this.buffer = new Int16Array(this.chunkSize);
        this.started = false;
        this.firstBlockTime = 0;
        this.emittedSamples = 0;
        this.port.postMessage({ type: "reset_done", framesEmitted: this.framesEmitted });
      }
    };
  }

  _resample(input) {
    if (this.ratio === 1) {
      return input;
    }
    const maxOut = Math.floor((input.length - this.phase) / this.ratio) + 2;
    if (maxOut <= 0) return input.subarray(0, 0);
    if (!this.resampleBuffer || this.resampleBuffer.length < maxOut) {
      this.resampleBuffer = new Float32Array(maxOut);
    }
    const out = this.resampleBuffer;
    let src = this.phase;
    let n = 0;
    while (src < input.length) {
      const i = Math.floor(src);
      const frac = src - i;
      const s0 = input[i];
      // Mẫu cuối block không có "i+1": kẹp về s0 (sai số 1 mẫu ở biên block).
      const s1 = i + 1 < input.length ? input[i + 1] : s0;
      out[n++] = s0 + frac * (s1 - s0);
      src += this.ratio;
    }
    this.phase = src - input.length;
    return n === out.length ? out : out.subarray(0, n);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;
    const channel = input[0];

    // Thời điểm bắt đầu của block input này trong timeline của AudioContext.
    const blockStartTime = currentTime - channel.length / sampleRate;
    if (!this.started) {
      this.started = true;
      this.firstBlockTime = blockStartTime;
    }

    const resampled = this._resample(channel);
    let offset = 0;

    while (offset < resampled.length) {
      const room = this.chunkSize - this.bufferIndex;
      const take = Math.min(room, resampled.length - offset);

      for (let i = 0; i < take; i++) {
        const s = resampled[offset + i];
        // Chuyển float32 [-1,1] -> Int16 (giống hệt đường ScriptProcessor cũ)
        this.buffer[this.bufferIndex + i] = Math.max(-32768, Math.min(32767, Math.round(s * 32767)));
      }
      this.bufferIndex += take;
      offset += take;

      if (this.bufferIndex >= this.chunkSize) {
        // Timestamp suy ra từ mốc đầu + số mẫu đã phát ⇒ chính xác và không trôi.
        const ts = this.firstBlockTime + this.emittedSamples / this.targetRate;
        this.port.postMessage(
          {
            type: "audio_chunk",
            buffer: this.buffer.buffer,
            captureTimestamp: ts,
            sampleRate: this.targetRate,
            samples: this.chunkSize,
          },
          [this.buffer.buffer]
        );
        this.framesEmitted++;
        this.emittedSamples += this.chunkSize;

        // Cấp phát mới cho chunk kế tiếp (buffer cũ đã được transfer).
        this.buffer = new Int16Array(this.chunkSize);
        this.bufferIndex = 0;
      }
    }

    return true; // giữ processor sống
  }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
