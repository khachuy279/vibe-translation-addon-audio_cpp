// Content Script - Full Pipeline: audio capture + WebSocket + overlay
(function () {
  if (window.__bsContentScriptLoaded) return;
  window.__bsContentScriptLoaded = true;
  const api = typeof browser !== "undefined" ? browser : chrome;

  // ── Unified WebSocket Configuration Builder ─────────────────────────────
  function buildWsConfig(cfg) {
    const silence = cfg.silenceDurationMs || cfg.vadSilenceDurationMs || cfg.silence_duration_ms || 300;
    return {
      type: "set_config",
      action: "configure",
      targetLang: cfg.targetLang || "vi",
      sourceLang: cfg.sourceLanguage || cfg.sourceLang || "auto",
      translationModel: cfg.translationModel || "xiaomi",
      vadEngine: cfg.vadEngine || "fsmn-vad",
      vadThreshold: cfg.vadThreshold !== undefined ? cfg.vadThreshold : (cfg.vad_threshold !== undefined ? cfg.vad_threshold : (cfg.threshold !== undefined ? cfg.threshold : 0.5)),
      threshold: cfg.vadThreshold !== undefined ? cfg.vadThreshold : (cfg.vad_threshold !== undefined ? cfg.vad_threshold : (cfg.threshold !== undefined ? cfg.threshold : 0.5)),
      silenceDurationMs: silence,
      minWordsToCommit: cfg.minWordsToCommit !== undefined && !isNaN(parseInt(cfg.minWordsToCommit, 10)) ? Math.max(0, parseInt(cfg.minWordsToCommit, 10)) : 2,
      ttsEnabled: !!cfg.ttsEnabled,
      ttsVoice: cfg.ttsVoice || "speaker_01_0039.wav",
      ttsSpeed: parseFloat(cfg.ttsSpeed || 1.0),
      ttsInstruct: cfg.ttsInstruct || "",
      ttsRefAudio: cfg.ttsRefAudio || "",
      ttsRefText: cfg.ttsRefText || "",
    };
  }

  const ttsPlayer = new TTSAudioPlayer();

  let wsClient = null, audioCapture = null, overlayManager = null;
  let captureAbortController = null;
  let isCapturing = false;
  let settings = { targetLang: "vi", subPosY: 10, subWidth: 80 };
  let cachedVideo = null;

  function getVideo(forceRefresh = false) {
    if (!forceRefresh && cachedVideo && cachedVideo.isConnected && !cachedVideo.ended) {
      return cachedVideo;
    }
    cachedVideo = findVideo();
    return cachedVideo;
  }

  function ensureOverlay(targetVideo = null) {
    const video = targetVideo || getVideo();
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
          console.log("[BS TTS] 🔊 Received synthesized audio chunk:", payload?.utterance_id, payload?.text, `(${payload?.duration_sec}s)`);
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

    const video = getVideo();
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
          wsClient.sendJSON(buildWsConfig(settings));
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

      // Initialize AbortController for clean listener lifecycle management
      captureAbortController = new AbortController();
      const { signal } = captureAbortController;

      ttsPlayer.setTargetVideo(
        video,
        settings.ttsDucking !== false,
        settings.duckingLevel !== undefined ? settings.duckingLevel : 0.25,
        !!settings.ttsEnabled
      );
      ttsPlayer.clear();
      video.addEventListener("seeked", () => ttsPlayer.clear(), { signal });

      // Connect to WebSocket Backend
      wsClient = new WSClient("wss://localhost:8765/ws");
      wsClient.on("connected", () => {
        wsClient.sendJSON(buildWsConfig(settings));
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

      isCapturing = true;

      // Start Audio Capture module
      audioCapture = new AudioCapture();
      audioCapture.onChunk = (pcmBuffer, timestamp, chunkIdx) => {
        if (wsClient && wsClient.isConnected) {
          wsClient.sendBinary(pcmBuffer, timestamp, chunkIdx);
        }
      };
      await audioCapture.start(video);

      // Initialize overlay for Top window or if already in Fullscreen
      if (window === window.top || document.fullscreenElement) {
        ensureOverlay(video);
      }

      // Ensure AudioContext and Fullscreen overlay transitions
      const handleStateKeepAlive = () => {
        if (audioCapture) {
          audioCapture.resumeAudioContext();
        }
        if (document.fullscreenElement && !overlayManager) {
          ensureOverlay(video);
        }
      };

      document.addEventListener("fullscreenchange", handleStateKeepAlive, { signal });
      document.addEventListener("webkitfullscreenchange", handleStateKeepAlive, { signal });
      document.addEventListener("mozfullscreenchange", handleStateKeepAlive, { signal });
      if (video) {
        video.addEventListener("play", handleStateKeepAlive, { signal });
        video.addEventListener("playing", handleStateKeepAlive, { signal });
      }

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
    if (captureAbortController) {
      try { captureAbortController.abort(); } catch (e) {}
      captureAbortController = null;
    }
    ttsPlayer.destroy();
    if (audioCapture) {
      try { audioCapture.stop(); } catch (e) {}
      audioCapture = null;
    }
    if (wsClient) {
      try { wsClient.disconnect(); } catch (e) {}
      wsClient = null;
    }
    if (overlayManager) {
      try { overlayManager.destroy(); } catch (e) {}
      overlayManager = null;
    }
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
      wsClient.sendJSON(buildWsConfig(settings));
      console.log("[BS] Settings updated live:", s);
    }
    return { success: true, settings };
  };

  function findVideo() {
    if (cachedVideo && cachedVideo.isConnected && !cachedVideo.ended) return cachedVideo;

    const allVideos = [];

    // 1. Scan direct DOM
    try {
      document.querySelectorAll("video").forEach(v => allVideos.push(v));
    } catch (e) {}

    // 2. Scan Shadow DOMs recursively (linear element traversal)
    function collectVideosAndShadowRoots(root) {
      if (!root) return;
      try {
        if (root.querySelectorAll) {
          root.querySelectorAll("video").forEach(v => allVideos.push(v));
          root.querySelectorAll("*").forEach(el => {
            if (el.shadowRoot) {
              collectVideosAndShadowRoots(el.shadowRoot);
            }
          });
        }
      } catch (e) {}
    }
    collectVideosAndShadowRoots(document.body || document.documentElement);

    // 3. Scan accessible same-origin/child iframes
    try {
      document.querySelectorAll("iframe").forEach(iframe => {
        try {
          const doc = iframe.contentDocument || iframe.contentWindow?.document;
          if (doc) {
            collectVideosAndShadowRoots(doc.body || doc.documentElement);
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
