// AudioWorklet processor for low-latency audio capture
// Runs in a dedicated audio thread, avoiding main thread jank.

class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 1024 samples @ 16kHz = 64ms chunk (matching backend framing)
    this.bufferSize = 1024;
    this.buffer = new Int16Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channel = input[0];
    for (let i = 0; i < channel.length; i++) {
      // Convert float32 [-1,1] → Int16
      const sample = Math.max(-32768, Math.min(32767, Math.round(channel[i] * 32767)));
      this.buffer[this.bufferIndex++] = sample;

      if (this.bufferIndex >= this.bufferSize) {
        // Send buffer to main thread using transferable buffer
        this.port.postMessage(
          { type: "audio_chunk", buffer: this.buffer.buffer },
          [this.buffer.buffer]
        );
        this.buffer = new Int16Array(this.bufferSize);
        this.bufferIndex = 0;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
