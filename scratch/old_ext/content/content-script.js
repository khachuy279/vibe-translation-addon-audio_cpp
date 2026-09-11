// Content Script - Full Pipeline: audio capture + WebSocket + overlay
(function () {
  if (window.__bsContentScriptLoaded) return;
  window.__bsContentScriptLoaded = true;
  const api = typeof browser !== "undefined" ? browser : chrome;

  // ── TTS Audio Player (Queue + Continuous Auto-Ducking) ─────────────────────
  class TTSAudioPlayer {
    constructor() {
      this.queue = [];
      this.isPlaying = false;
      this.currentAudio = null;
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
      // Auto-Ducking only activates if BOTH TTS is enabled AND Auto-Ducking is turned on
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
      // Deduplicate by utterance ID to prevent double-enqueuing
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

    _playNext() {
      if (this.queue.length === 0) {
        this.isPlaying = false;
        this.currentAudio = null;
        return;
      }

      this.isPlaying = true;
      const item = this.queue.shift();

      try {
        const audioUrl = `data:audio/wav;base64,${item.audioBase64}`;
        const audio = new Audio(audioUrl);
        this.currentAudio = audio;

        const onDone = () => {
          if (this.currentAudio === audio) {
            this.currentAudio = null;
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
        this._playNext();
      }
    }

    clear() {
      this.queue = [];
      if (this.currentAudio) {
        try {
          this.currentAudio.pause();
          this.currentAudio.src = "";
        } catch (e) {}
        this.currentAudio = null;
      }
      this.isPlaying = false;
    }

    destroy() {
      this.clear();
      this._restoreVideoVolume();
    }
  }

  const ttsPlayer = new TTSAudioPlayer();

  let wsClient = null, audioContext = null, sourceNode = null;
  let scriptProcessor = null, overlayManager = null;
  let isCapturing = false, chunkIndex = 0;
  let settings = { targetLang: "vi", subPosY: 10, subWidth: 80 };

  function ensureOverlay(targetVideo = null) {
    const video = targetVideo || findVideo();
    if (!overlayManager) {
      overlayManager = new OverlayManager();
      overlayManager.init(video);
      overlayManager.applySettings(settings);
    } else if (video) {
      overlayManager.attachToVideo(video);
    }
    return overlayManager;
  }

  function handleSubtitleEvent(eventType, payload) {
    if (eventType === "tts_audio") {
      // Only Top frame or active capturing frame should play audio
      if (window === window.top || isCapturing) {
        const audioB64 = payload?.audio || payload?.audio_base64 || (typeof payload === "string" ? payload : null);
        if (audioB64) {
          ttsPlayer.enqueue({
            id: payload?.utterance_id || payload?.sentence_id || payload?.utteranceId,
            audioBase64: audioB64,
            durationSec: payload?.duration_sec || payload?.durationSec,
            text: payload?.text
          });
        }
      }
      return;
    }

    const video = findVideo();
    // 1. If this frame HAS the video element, render overlay directly inside this frame
    if (video) {
      const om = ensureOverlay(video);
      if (eventType === "partial_transcript") om.onPartialTranscript(payload);
      else if (eventType === "utterance_update") om.onUtteranceUpdate(payload);
      else if (eventType === "sentence_complete") om.onSentenceComplete(payload);
      else if (eventType === "translation") om.onTranslation(payload);
      return;
    }

    // 2. If this frame is actively capturing audio (even if video ref is temporary null)
    if (isCapturing) {
      const om = ensureOverlay(null);
      if (eventType === "partial_transcript") om.onPartialTranscript(payload);
      else if (eventType === "utterance_update") om.onUtteranceUpdate(payload);
      else if (eventType === "sentence_complete") om.onSentenceComplete(payload);
      else if (eventType === "translation") om.onTranslation(payload);
      return;
    }

    // If this frame has no video and is not capturing (e.g. top window hosting an iframe player),
    // do not render a fallback overlay on body to avoid duplicate or misplaced subtitles.
  }

  api.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    switch (msg.type || msg.action) {
      case "START_TRANSLATION": case "start_capture":
        if (msg.settings) Object.assign(settings, msg.settings);
        const v = findVideo();
        if (v) {
          ensureOverlay(v);
        }
        startCapture(msg).then(r => sendResponse?.(r));
        return true;
      case "STOP_TRANSLATION": case "stop_capture":
        stopCapture();
        if (overlayManager) {
          overlayManager.destroy();
          overlayManager = null;
        }
        sendResponse?.({ success: true });
        break;
      case "SUBTITLE_RENDER":
        handleSubtitleEvent(msg.eventType, msg.payload);
        break;
      case "GET_STATUS": case "content_status":
        sendResponse?.({ isCapturing, hasVideo: !!findVideo(), overlayActive: !!overlayManager });
        break;
      case "set_overlay_mode":
        if (msg.payload?.mode) settings.overlayStyle = msg.payload.mode;
        if (overlayManager) overlayManager.setMode(settings.overlayStyle);
        break;
      case "update_settings":
        if (msg.settings) {
          Object.assign(settings, msg.settings);
          if (overlayManager) overlayManager.applySettings(settings);
          if (ttsPlayer) {
            ttsPlayer.applySettings(
              settings.ttsDucking !== false,
              settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
              !!settings.ttsEnabled
            );
          }
        }
        if (wsClient && wsClient.isConnected && msg.settings) {
          wsClient.sendJSON({
            type: "set_config",
            targetLang: settings.targetLang,
            sourceLang: settings.sourceLanguage || settings.sourceLang || "auto",
            silenceDurationMs: settings.silenceDurationMs || settings.vadSilenceDurationMs || settings.silence_duration_ms || 300,
            silence_duration_ms: settings.silenceDurationMs || settings.vadSilenceDurationMs || settings.silence_duration_ms || 300,
            tts_enabled: !!settings.ttsEnabled,
            tts_speed: settings.ttsSpeed || 1.0,
            tts_instruct: settings.ttsInstruct || "",
            tts_ref_audio: settings.ttsRefAudio || "",
            tts_ref_text: settings.ttsRefText || "",
          });
          console.log("[BS] Settings updated live:", msg.settings);
        }
        sendResponse?.({ success: true, settings });
        break;
    }
  });

  async function startCapture(msg) {
    if (isCapturing) return { success: false, error: "Already capturing" };
    try {
      if (msg.settings) Object.assign(settings, msg.settings);
      let video = findVideo();
      if (!video) {
        // Retry for up to ~1s if video element is lazily loaded upon play
        for (let i = 0; i < 4; i++) {
          await new Promise(r => setTimeout(r, 250));
          video = findVideo();
          if (video) break;
        }
      }
      if (!video) return { success: false, error: "No video found" };

      ttsPlayer.setTargetVideo(
        video,
        settings.ttsDucking !== false,
        settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
        !!settings.ttsEnabled
      );
      ttsPlayer.clear();
      video.addEventListener("seeked", () => ttsPlayer.clear());

      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      if (audioContext.state === "suspended") await audioContext.resume();

      // Try captureStream (same-origin), fallback to MediaElementSource (YouTube)
      let stream = null;
      try { stream = (typeof video.captureStream === "function") ? video.captureStream() : (typeof video.mozCaptureStream === "function") ? video.mozCaptureStream() : null; } catch (e) {}

      if (stream && stream.getAudioTracks().length > 0) {
        sourceNode = audioContext.createMediaStreamSource(stream);
        console.log("[BS] Audio: captureStream");
      } else {
        sourceNode = audioContext.createMediaElementSource(video);
        sourceNode.connect(audioContext.destination); // route back to speakers
        console.log("[BS] Audio: MediaElementSource (cross-origin)");
      }

      wsClient = new WSClient("wss://localhost:8765/ws");
      wsClient.on("connected", () => {
        wsClient.sendJSON({
          type: "set_config",
          action: "configure",
          targetLang: settings.targetLang,
          sourceLang: settings.sourceLanguage || settings.sourceLang || "auto",
          silenceDurationMs: settings.silenceDurationMs || settings.vadSilenceDurationMs || settings.silence_duration_ms || 300,
          silence_duration_ms: settings.silenceDurationMs || settings.vadSilenceDurationMs || settings.silence_duration_ms || 300,
          tts_enabled: !!settings.ttsEnabled,
          tts_speed: settings.ttsSpeed || 1.0,
          tts_instruct: settings.ttsInstruct || "",
          tts_ref_audio: settings.ttsRefAudio || "",
          tts_ref_text: settings.ttsRefText || "",
        });
      });

      function emitSubtitleEvent(eventType, payload) {
        // 1. Render locally if eligible
        handleSubtitleEvent(eventType, payload);

        // 2. Broadcast to other frames (Top frame) only if inside a child iframe
        if (window !== window.top) {
          try {
            api.runtime.sendMessage({
              action: "BROADCAST_SUBTITLE",
              eventType,
              payload
            }).catch(() => {});
          } catch (e) {}
        }
      }

      wsClient.on("partial_transcript", p => emitSubtitleEvent("partial_transcript", p));
      wsClient.on("utterance_update", p => emitSubtitleEvent("utterance_update", p));
      wsClient.on("sentence_complete", p => { console.log("[BS] Sentence:", p); emitSubtitleEvent("sentence_complete", p); });
      wsClient.on("translation", p => { console.log("[BS] Translation:", p); emitSubtitleEvent("translation", p); });
      wsClient.on("tts_audio", p => emitSubtitleEvent("tts_audio", p));
      wsClient.on("error", p => console.error("[BS] Backend:", p));
      await wsClient.connect();

      // ScriptProcessorNode for audio chunking
      scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
      scriptProcessor.onaudioprocess = (event) => {
        if (!isCapturing || !wsClient) return;
        const inputData = event.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(inputData[i] * 32767)));
        for (let off = 0; off < pcm.length; off += 320) {
          const chunk = pcm.slice(off, off + 320);
          if (chunk.length < 320) break;
          wsClient.sendBinary(chunk.buffer, audioContext.currentTime, chunkIndex++);
        }
      };
      sourceNode.connect(scriptProcessor);
      scriptProcessor.connect(audioContext.destination);

      // Initialize overlay for Top window or if already in Fullscreen
      if (window === window.top || document.fullscreenElement) {
        ensureOverlay(video);
      }

      // Ensure AudioContext and Fullscreen overlay transitions
      const handleStateKeepAlive = () => {
        if (audioContext && audioContext.state === "suspended") {
          audioContext.resume().catch(() => {});
        }
        if (document.fullscreenElement && !overlayManager) {
          ensureOverlay(video);
        }
      };
      document.addEventListener("fullscreenchange", handleStateKeepAlive);
      document.addEventListener("webkitfullscreenchange", handleStateKeepAlive);
      document.addEventListener("mozfullscreenchange", handleStateKeepAlive);
      if (video) {
        video.addEventListener("play", handleStateKeepAlive);
        video.addEventListener("playing", handleStateKeepAlive);
      }

      isCapturing = true;
      console.log("[BS] Capture started");
      return { success: true };
    } catch (e) {
      console.error("[BS] Start error:", e);
      await cleanup();
      const errText = e?.message || (typeof e === "string" ? e : "Không thể kết nối tới Backend WebSocket (wss://localhost:8765/ws)");
      return { success: false, error: errText };
    }
  }

  function stopCapture() { cleanup(); return { success: true }; }

  async function cleanup() {
    isCapturing = false;
    ttsPlayer.destroy();
    if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
    if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
    if (audioContext) { audioContext.close().catch(()=>{}); audioContext = null; }
    if (wsClient) { wsClient.disconnect(); wsClient = null; }
    if (overlayManager) { overlayManager.destroy(); overlayManager = null; }
    chunkIndex = 0;
  }

  // Global helpers for cross-frame scripting
  window.__bsStartCapture = (msg) => {
    if (msg?.settings) Object.assign(settings, msg.settings);
    const v = findVideo();
    if (v) ensureOverlay(v);
    return startCapture(msg || {});
  };
  window.__bsStopCapture = () => {
    stopCapture();
    if (overlayManager) {
      overlayManager.destroy();
      overlayManager = null;
    }
    return { success: true };
  };
  window.__bsGetStatus = () => ({ isCapturing, hasVideo: !!findVideo(), overlayActive: !!overlayManager });
  window.__bsUpdateSettings = (s) => {
    if (s) {
      Object.assign(settings, s);
      if (overlayManager) overlayManager.applySettings(settings);
      if (ttsPlayer) {
        ttsPlayer.applySettings(
          settings.ttsDucking !== false,
          settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
          !!settings.ttsEnabled
        );
      }
    }
    if (wsClient && wsClient.isConnected && s) {
      wsClient.sendJSON({
        type: "set_config",
        targetLang: settings.targetLang,
        sourceLang: settings.sourceLanguage || settings.sourceLang || "auto",
      });
      console.log("[BS] Settings updated live:", s);
    }
    return { success: true, settings };
  };

  function findVideo() {
    const allVideos = [];

    // 1. Scan direct DOM
    try {
      document.querySelectorAll("video").forEach(v => allVideos.push(v));
    } catch (e) {}

    // 2. Scan Shadow DOMs recursively
    function searchShadowRoots(node) {
      if (!node) return;
      try {
        if (node.shadowRoot) {
          node.shadowRoot.querySelectorAll("video").forEach(v => allVideos.push(v));
          node.shadowRoot.querySelectorAll("*").forEach(searchShadowRoots);
        }
        if (node.querySelectorAll) {
          node.querySelectorAll("*").forEach(n => {
            if (n.shadowRoot) searchShadowRoots(n);
          });
        }
      } catch (e) {}
    }
    searchShadowRoots(document.body || document.documentElement);

    // 3. Scan accessible same-origin/child iframes
    try {
      document.querySelectorAll("iframe").forEach(iframe => {
        try {
          const doc = iframe.contentDocument || iframe.contentWindow?.document;
          if (doc) {
            doc.querySelectorAll("video").forEach(v => allVideos.push(v));
            searchShadowRoots(doc.body || doc.documentElement);
          }
        } catch (e) {}
      });
    } catch (e) {}

    if (!allVideos.length) return null;

    // 3. Rank videos: prioritize actively playing, readyState > 0, valid src, or largest dimensions
    let bestVideo = allVideos[0];
    let maxScore = -1;

    for (const v of allVideos) {
      let score = 0;
      const isPlaying = !v.paused && !v.ended;
      if (isPlaying && v.readyState > 2) score += 1000000;
      else if (isPlaying) score += 500000;
      else if (v.readyState > 0) score += 100000;

      if (v.currentTime > 0) score += 50000;
      if (v.src || v.currentSrc || v.srcObject) score += 20000;

      const rect = v.getBoundingClientRect ? v.getBoundingClientRect() : { width: 0, height: 0 };
      const displayArea = (rect.width * rect.height) || 0;
      const videoArea = (v.videoWidth * v.videoHeight) || 0;
      score += Math.max(displayArea, videoArea);

      if (score > maxScore) {
        maxScore = score;
        bestVideo = v;
      }
    }
    return bestVideo;
  }

  window.__bsFindVideo = findVideo;

  console.log("[BS Content] Ready (Frame: " + (window === window.top ? "Top" : "Iframe") + ")");
})();
