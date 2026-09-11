// Audio capture module
// Processes a MediaStream into 16kHz mono Int16 PCM chunks.

class AudioCapture {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.isCapturing = false;
    this.sampleRate = 16000;
    this.chunkIndex = 0;
    this.onChunk = null; // callback(pcmData: ArrayBuffer, captureTimestamp: number)
    this.onError = null;
    this._useAudioWorklet = true;
  }

  /**
   * Start capturing from a MediaStream.
   * @param {MediaStream} stream - The audio stream (from tabCapture or getUserMedia)
   */
  async start(stream) {
    if (this.isCapturing) return;
    this.chunkIndex = 0;
    this.mediaStream = stream;

    try {
      // Create AudioContext with desired sample rate
      this.audioContext = new AudioContext({
        sampleRate: this.sampleRate,
        latencyHint: "interactive",
      });

      // Disable audio processing to avoid distortion
      const audioTracks = this.mediaStream.getAudioTracks();
      for (const track of audioTracks) {
        try {
          await track.applyConstraints({
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          });
        } catch (e) {
          // Ignore if not supported
        }
      }

      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      // Try AudioWorklet first
      const useWorklet = await this._tryAudioWorklet();
      
      if (useWorklet) {
        this._useAudioWorklet = true;
      } else {
        this._useScriptProcessor();
      }

      this.isCapturing = true;
      console.log("[AudioCapture] Started", this._useAudioWorklet ? "(AudioWorklet)" : "(ScriptProcessor)");

    } catch (e) {
      console.error("[AudioCapture] Start error:", e);
      if (this.onError) this.onError(e);
      throw e;
    }
  }

  async _tryAudioWorklet() {
    try {
      const workletUrl = browser.runtime.getURL("lib/audio-processor.js");
      await this.audioContext.audioWorklet.addModule(workletUrl);

      this.processorNode = new AudioWorkletNode(
        this.audioContext,
        "audio-capture-processor"
      );

      this.processorNode.port.onmessage = (event) => {
        if (event.data.type === "audio_chunk" && this.onChunk) {
          const timestamp = this.audioContext.currentTime;
          this.onChunk(event.data.buffer, timestamp, this.chunkIndex++);
        }
      };

      this.sourceNode.connect(this.processorNode);
      return true;
    } catch (e) {
      console.warn("[AudioCapture] AudioWorklet not available, falling back to ScriptProcessorNode:", e.message);
      return false;
    }
  }

  _useScriptProcessor() {
    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

    this.processorNode.onaudioprocess = (event) => {
      if (!this.isCapturing || !this.onChunk) return;

      const inputBuffer = event.inputBuffer.getChannelData(0);
      const pcmData = new Int16Array(inputBuffer.length);

      for (let i = 0; i < inputBuffer.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, Math.round(inputBuffer[i] * 32767)));
      }

      // ScriptProcessorNode gives larger buffers (~4096 samples)
      // Split into 20ms chunks (320 samples @ 16kHz)
      const chunkSize = 320;
      for (let offset = 0; offset < pcmData.length; offset += chunkSize) {
        const chunk = pcmData.slice(offset, offset + chunkSize);
        if (chunk.length < chunkSize) break;

        const timestamp = this.audioContext.currentTime;
        this.onChunk(chunk.buffer, timestamp, this.chunkIndex++);
      }
    };

    this.sourceNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }

  stop() {
    this.isCapturing = false;

    if (this.processorNode) {
      try { this.processorNode.disconnect(); } catch (e) {}
      this.processorNode = null;
    }
    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch (e) {}
      this.sourceNode = null;
    }
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }

    console.log("[AudioCapture] Stopped");
  }
}

if (typeof self !== "undefined") self.AudioCapture = AudioCapture;
if (typeof window !== "undefined") window.AudioCapture = AudioCapture;
