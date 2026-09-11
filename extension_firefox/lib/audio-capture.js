// Audio capture module for Firefox & Chrome Content Scripts
// Processes HTMLVideoElement or MediaStream into 16kHz mono Int16 PCM chunks.

class AudioCapture {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.videoElement = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.isCapturing = false;
    this.sampleRate = 16000;
    this.chunkIndex = 0;
    // 1024 samples @ 16kHz = 64ms chunk (divides 4096 buffer exactly by 4, 0 lost samples)
    this.chunkSize = 1024;
    this.onChunk = null; // callback(pcmData, captureTimestamp, chunkIndex, chunkStartMediaTime, chunkEndMediaTime, epoch, playbackRate)
    this.onError = null;
    this.residualSamples = new Float32Array(0);
    this.resamplePhase = 0.0;
    this.epoch = 0;
  }

  setEpoch(epoch) {
    this.epoch = epoch;
  }


  /**
   * Start capturing from a MediaStream or HTMLVideoElement.
   * @param {MediaStream|HTMLMediaElement} target - Audio source
   */
  async start(target) {
    if (this.isCapturing) return;
    this.chunkIndex = 0;

    try {
      if (target instanceof HTMLMediaElement || (target && typeof target.play === "function")) {
        this.videoElement = target;

        // Try MediaElementSource first (direct HTMLMediaElement audio pipeline)
        if (!target.__bsAudioCtx || target.__bsAudioCtx.state === "closed") {
          const AudioCtx = window.AudioContext || window.webkitAudioContext;
          target.__bsAudioCtx = new AudioCtx({ sampleRate: this.sampleRate });
          try {
            target.__bsSourceNode = target.__bsAudioCtx.createMediaElementSource(target);
            // Route to speakers so the user still hears the video
            target.__bsSourceNode.connect(target.__bsAudioCtx.destination);
          } catch (elemErr) {
            console.warn("[AudioCapture] createMediaElementSource failed, trying stream capture:", elemErr);
          }
        }

        if (target.__bsSourceNode) {
          this.audioContext = target.__bsAudioCtx;
          this.sourceNode = target.__bsSourceNode;
          console.log("[AudioCapture] Source: MediaElementSource (direct video audio, routed to speakers)");
        } else {
          // Fallback: captureStream / mozCaptureStream
          let stream = null;
          try {
            if (typeof target.captureStream === "function") {
              stream = target.captureStream();
            } else if (typeof target.mozCaptureStream === "function") {
              stream = target.mozCaptureStream();
            }
          } catch (e) {}

          if (stream && stream.getAudioTracks().length > 0) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioCtx({ sampleRate: this.sampleRate });
            this.mediaStream = stream;
            this.sourceNode = this.audioContext.createMediaStreamSource(stream);
            this.sourceNode.connect(this.audioContext.destination);
            console.log("[AudioCapture] Source: captureStream (routed to speakers)");
          } else {
            throw new Error("Không thể trích xuất âm thanh từ thẻ video");
          }
        }
      } else if (target && typeof target.getAudioTracks === "function") {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioCtx({ sampleRate: this.sampleRate });
        this.mediaStream = target;
        this.sourceNode = this.audioContext.createMediaStreamSource(target);
        console.log("[AudioCapture] Source: direct MediaStream");
      } else {
        throw new Error("Invalid audio target: expected HTMLMediaElement or MediaStream");
      }

      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      // Setup reliable ScriptProcessor pipeline
      this._setupScriptProcessor();

      this.isCapturing = true;
      console.log(`[AudioCapture] Started capturing (Rate: ${this.audioContext.sampleRate}Hz -> 16000Hz, chunk: ${this.chunkSize} samples)`);
    } catch (e) {
      console.error("[AudioCapture] Start error:", e);
      this.stop();
      if (this.onError) this.onError(e);
      throw e;
    }
  }

  _setupScriptProcessor() {
    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    const chunkSize = this.chunkSize || 1024;
    const actualSampleRate = this.audioContext.sampleRate;

    this.processorNode.onaudioprocess = (event) => {
      if (!this.isCapturing || !this.onChunk) return;

      const inputBuffer = event.inputBuffer.getChannelData(0);
      let float16k;

      if (actualSampleRate === 16000) {
        float16k = inputBuffer;
      } else {
        // Continuous phase linear interpolation downsampling to 16kHz
        const ratio = actualSampleRate / 16000.0;
        let srcIdx = this.resamplePhase || 0.0;
        const maxOut = Math.floor((inputBuffer.length - srcIdx) / ratio) + 2;
        const outArray = new Float32Array(Math.max(0, maxOut));
        let outCount = 0;

        while (srcIdx < inputBuffer.length) {
          const i = Math.floor(srcIdx);
          const frac = srcIdx - i;
          const s0 = inputBuffer[i];
          const s1 = (i + 1 < inputBuffer.length) ? inputBuffer[i + 1] : s0;
          outArray[outCount++] = s0 + frac * (s1 - s0);
          srcIdx += ratio;
        }

        this.resamplePhase = srcIdx - inputBuffer.length;
        float16k = (outCount === outArray.length) ? outArray : outArray.subarray(0, outCount);
      }

      // Concatenate residual carryover samples from previous cycle to guarantee ZERO lost audio
      let combined;
      if (this.residualSamples && this.residualSamples.length > 0) {
        combined = new Float32Array(this.residualSamples.length + float16k.length);
        combined.set(this.residualSamples, 0);
        combined.set(float16k, this.residualSamples.length);
      } else {
        combined = float16k;
      }

      const baseTimestamp = this.audioContext ? this.audioContext.currentTime : 0;
      const baseMediaTime = this.videoElement ? this.videoElement.currentTime : 0.0;
      const playbackRate = this.videoElement ? (this.videoElement.playbackRate || 1.0) : 1.0;
      let offset = 0;

      while (combined.length - offset >= chunkSize) {
        const slice = combined.subarray(offset, offset + chunkSize);
        const pcmData = new Int16Array(chunkSize);
        for (let i = 0; i < chunkSize; i++) {
          pcmData[i] = Math.max(-32768, Math.min(32767, Math.round(slice[i] * 32767)));
        }

        // Monotonic timestamp offset per chunk
        const chunkTime = baseTimestamp + (offset / 16000.0);
        const chunkDurationSec = chunkSize / 16000.0;
        const chunkStartMediaTime = baseMediaTime + (offset / 16000.0) * playbackRate;
        const chunkEndMediaTime = chunkStartMediaTime + chunkDurationSec * playbackRate;

        const idx = this.chunkIndex++;
        if (idx === 0 || idx % 200 === 0) {
          console.log(`[AudioCapture] Emitted audio chunk #${idx} (${pcmData.byteLength} bytes, rate: ${actualSampleRate}Hz->16kHz, media: ${chunkStartMediaTime.toFixed(3)}s->${chunkEndMediaTime.toFixed(3)}s, epoch: ${this.epoch})`);
        }
        this.onChunk(pcmData.buffer, chunkTime, idx, chunkStartMediaTime, chunkEndMediaTime, this.epoch, playbackRate);
        offset += chunkSize;
      }

      // Preserve unchunked residual samples for next cycle (0% sample loss)
      if (offset < combined.length) {
        this.residualSamples = combined.slice(offset);
      } else {
        this.residualSamples = new Float32Array(0);
      }
    };

    this.sourceNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
  }

  resumeAudioContext() {
    if (this.audioContext && this.audioContext.state === "suspended") {
      return this.audioContext.resume().catch(() => {});
    }
    return Promise.resolve();
  }

  stop() {
    this.isCapturing = false;
    this.residualSamples = new Float32Array(0);
    this.resamplePhase = 0.0;


    if (this.processorNode) {
      try {
        if (this.processorNode.port) {
          this.processorNode.port.onmessage = null;
        }
        this.processorNode.onaudioprocess = null;
        this.processorNode.disconnect();
      } catch (e) {}
      this.processorNode = null;
    }

    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch (e) {}
      // If sourceNode was cached on video, re-route it to destination so video audio still plays
      if (this.videoElement && this.videoElement.__bsSourceNode && this.audioContext) {
        try { this.videoElement.__bsSourceNode.connect(this.audioContext.destination); } catch (e) {}
      }
      this.sourceNode = null;
    }

    // Clean up AudioContext if it belongs to a disconnected video element (preventing memory leak)
    if (this.videoElement && !this.videoElement.isConnected && this.videoElement.__bsAudioCtx) {
      try { this.videoElement.__bsAudioCtx.close(); } catch (e) {}
      delete this.videoElement.__bsAudioCtx;
      delete this.videoElement.__bsSourceNode;
    }

    // Do NOT close cached target.__bsAudioCtx if video is still active in DOM
    if (this.audioContext && (!this.videoElement || this.audioContext !== this.videoElement.__bsAudioCtx)) {
      try { this.audioContext.close(); } catch (e) {}
    }
    this.audioContext = null;

    if (this.mediaStream) {
      try {
        this.mediaStream.getTracks().forEach((t) => t.stop());
      } catch (e) {}
      this.mediaStream = null;
    }

    this.videoElement = null;
    this.chunkIndex = 0;
    console.log("[AudioCapture] Stopped and cleaned up");
  }
}

if (typeof self !== "undefined") self.AudioCapture = AudioCapture;
if (typeof window !== "undefined") window.AudioCapture = AudioCapture;
