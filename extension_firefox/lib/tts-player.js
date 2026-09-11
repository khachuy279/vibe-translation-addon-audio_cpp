// TTS Audio Player (Queue + Continuous Auto-Ducking + Blob URL lifecycle)
(function (global) {
  class TTSAudioPlayer {
    constructor() {
      this.queue = [];
      this.isPlaying = false;
      this.currentAudio = null;
      this.currentBlobUrl = null;
      this.ttsEnabled = false;
      this.autoDucking = true;
      this.duckingLevel = 0.25;
      this.targetVideo = null;
      this.originalVideoVolume = 1.0;
      this.isDucked = false;
      this.playedIds = new Set();
    }

    setTargetVideo(video, autoDucking = true, duckingLevel = 0.25, ttsEnabled = false) {
      if (this.targetVideo && this.targetVideo !== video && this.isDucked) {
        this._restoreVideoVolume();
      }
      this.targetVideo = video;
      this.autoDucking = autoDucking;
      this.ttsEnabled = ttsEnabled;
      if (duckingLevel !== undefined) this.duckingLevel = duckingLevel;
      if (video && video.volume !== undefined && !this.isDucked) {
        this.originalVideoVolume = video.volume > 0 ? video.volume : 1.0;
      }
      this._applyContinuousDucking();
    }

    applySettings(autoDucking, duckingLevel, ttsEnabled) {
      if (autoDucking !== undefined) this.autoDucking = autoDucking;
      if (duckingLevel !== undefined) this.duckingLevel = duckingLevel;
      if (ttsEnabled !== undefined) this.ttsEnabled = ttsEnabled;
      this._applyContinuousDucking();
    }

    _applyContinuousDucking() {
      if (!this.targetVideo) return;
      const shouldDuck = !!this.ttsEnabled && !!this.autoDucking;
      if (shouldDuck) {
        try {
          if (!this.isDucked) {
            this.originalVideoVolume = this.targetVideo.volume > 0 ? this.targetVideo.volume : 1.0;
          }
          this.targetVideo.volume = Math.max(0, Math.min(1.0, this.originalVideoVolume * this.duckingLevel));
          this.isDucked = true;
        } catch (e) {
          console.warn("[BS TTS] Constant ducking error:", e);
        }
      } else if (this.isDucked) {
        this._restoreVideoVolume();
      }
    }

    _restoreVideoVolume() {
      if (!this.targetVideo || !this.isDucked) return;
      try {
        this.targetVideo.volume = Math.max(0, Math.min(1.0, this.originalVideoVolume));
        this.isDucked = false;
      } catch (e) {
        console.warn("[BS TTS] Restore volume error:", e);
      }
    }

    enqueue(item) {
      if (!item || !item.audioBase64) return;
      if (item.id) {
        if (this.playedIds.has(item.id)) return;
        this.playedIds.add(item.id);
        if (this.playedIds.size > 300) {
          const oldest = this.playedIds.values().next().value;
          this.playedIds.delete(oldest);
        }
      }
      this.queue.push(item);
      if (!this.isPlaying) {
        this._playNext();
      }
    }

    _base64ToBlobUrl(base64Str) {
      const binary = atob(base64Str);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "audio/wav" });
      return URL.createObjectURL(blob);
    }

    _playNext() {
      if (this.queue.length === 0) {
        this.isPlaying = false;
        this.currentAudio = null;
        this.currentBlobUrl = null;
        return;
      }

      this.isPlaying = true;
      const item = this.queue.shift();

      try {
        const audioUrl = this._base64ToBlobUrl(item.audioBase64);
        const audio = new Audio(audioUrl);
        this.currentAudio = audio;
        this.currentBlobUrl = audioUrl;

        const onDone = () => {
          if (this.currentAudio === audio) {
            try {
              audio.onended = null;
              audio.onerror = null;
              audio.pause();
              audio.src = "";
            } catch (e) {}
            if (audioUrl) {
              try { URL.revokeObjectURL(audioUrl); } catch (e) {}
            }
            this.currentAudio = null;
            this.currentBlobUrl = null;
            this._playNext();
          }
        };

        audio.onended = onDone;
        audio.onerror = (e) => {
          console.warn("[BS TTS] Playback error on item:", item.id, e);
          onDone();
        };

        audio.play().catch((err) => {
          console.warn("[BS TTS] Play error (autoplay/seek):", err);
          onDone();
        });
      } catch (err) {
        console.error("[BS TTS] Failed to play audio chunk:", err);
        this.currentAudio = null;
        this.currentBlobUrl = null;
        this._playNext();
      }
    }

    clear() {
      this.queue = [];
      if (this.currentAudio) {
        try {
          this.currentAudio.onended = null;
          this.currentAudio.onerror = null;
          this.currentAudio.pause();
          this.currentAudio.src = "";
        } catch (e) {}
        this.currentAudio = null;
      }
      if (this.currentBlobUrl) {
        try { URL.revokeObjectURL(this.currentBlobUrl); } catch (e) {}
        this.currentBlobUrl = null;
      }
      this.isPlaying = false;
    }

    destroy() {
      this.clear();
      this.playedIds.clear();
      this._restoreVideoVolume();
      this.targetVideo = null;
    }
  }

  global.TTSAudioPlayer = TTSAudioPlayer;
})(typeof globalThis !== "undefined" ? globalThis : this);
