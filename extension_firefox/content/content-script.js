// Content Script - Full Pipeline: audio capture + WebSocket + overlay
//
//  ╔════════════════════════════════════════════════════════════════════════════════╗
//  ║  ⚠️  ĐỌC TRƯỚC KHI SỬA: extension_firefox/CAPTURE_FRAME_NOTES.md                  ║
//  ║                                                                                ║
//  ║  Script này chạy trong MỌI frame (manifest: all_frames: true). Trên các trang   ║
//  ║  streaming, thẻ <video> nằm trong iframe CROSS-ORIGIN, không phải top frame.    ║
//  ║                                                                                ║
//  ║  INV-7  Frame KHÔNG có video phải `return false` (không gọi sendResponse) khi    ║
//  ║         nhận broadcast START — vì broadcast chỉ giao về response ĐẦU TIÊN, nếu   ║
//  ║         không frame top sẽ trả lời trước và CHE MẤT frame thật sự start.         ║
//  ║  INV-8  Phải giữ handler DISCOVER_CAPTURE / RELEASE_CAPTURE_OWNER và việc tự     ║
//  ║         claim `__ownerToken` trong message START.                               ║
//  ║                                                                                ║
//  ║  Guard tự động: backend_cpp/tests/test_extension_invariants.py                 ║
//  ╚════════════════════════════════════════════════════════════════════════════════╝
(function () {
  if (window.__bsContentScriptLoaded) return;
  window.__bsContentScriptLoaded = true;
  const api = typeof browser !== "undefined" ? browser : chrome;

  // ── Unified WebSocket Configuration Builder ─────────────────────────────
  function buildWsConfig(cfg) {
    const silence = cfg.silenceDurationMs || cfg.vadSilenceDurationMs || cfg.silence_duration_ms || 150;
    return {
      type: "set_config",
      action: "configure",
      epoch: streamEpoch,
      targetLang: cfg.targetLang || "vi",
      sourceLang: cfg.sourceLanguage || cfg.sourceLang || "auto",
      translationModel: cfg.translationModel || "tencent",
      vadEngine: cfg.vadEngine || "fsmn-vad",
      vadThreshold: cfg.vadThreshold !== undefined ? cfg.vadThreshold : (cfg.vad_threshold !== undefined ? cfg.vad_threshold : (cfg.threshold !== undefined ? cfg.threshold : 0.20)),
      threshold: cfg.vadThreshold !== undefined ? cfg.vadThreshold : (cfg.vad_threshold !== undefined ? cfg.vad_threshold : (cfg.threshold !== undefined ? cfg.threshold : 0.20)),
      hangoverMs: cfg.hangoverMs || 250,
      silenceDurationMs: silence,
      minWordsToCommit: cfg.minWordsToCommit !== undefined && !isNaN(parseInt(cfg.minWordsToCommit, 10)) ? Math.max(0, parseInt(cfg.minWordsToCommit, 10)) : 4,
      maxDurationSec: parseFloat(cfg.maxDurationSec || 15.0),
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
  let streamEpoch = 0;
  // Single-capture-owner coordination (audit finding P1-04).
  //
  // The manifest injects this content script into EVERY frame (`all_frames: true`).
  // A page can therefore contain several frames with a <video> element, and a START
  // broadcast to all frames would open one backend WebSocket session per frame.
  // The popup must instead designate exactly one owner; frames that were not
  // designated refuse to start, which makes the "one session" invariant enforceable.
  let captureOwnerToken = null;
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
        // The popup may designate this frame as the capture owner in the same message.
        if (msg.__ownerToken) captureOwnerToken = msg.__ownerToken;
        // A broadcast (tabs.sendMessage without frameId) returns only the FIRST response
        // to the popup, so a frame with nothing to capture must stay SILENT rather than
        // answer "no video found" and mask the frame that actually started.
        if (!captureOwnerToken && !findVideo()) {
          return false;
        }
        {
          const video = findVideo();
          if (video) ensureOverlay(video);
        }
        startCapture(msg).then(r => sendResponse?.(r));
        return true;
      case "DISCOVER_CAPTURE":
        // Answer a frame-targeted candidacy query (popup enumerates frames via
        // webNavigation and messages each one, which is the only reliable way to reach
        // a cross-origin player iframe).
        sendResponse?.(window.__bsDiscoverCapture());
        break;
      case "RELEASE_CAPTURE_OWNER":
        captureOwnerToken = null;
        sendResponse?.({ ok: true });
        break;
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
    // Ownership gate (audit finding P1-04): only the frame the popup designated may
    // open a backend session. Because this script runs in every frame, refusing
    // non-designated frames is what stops a START broadcast from opening one backend
    // WebSocket session per frame that happens to contain a <video>.
    //
    // Accepted when:
    //   1. Token match -- the popup targeted this exact frame.
    //   2. Explicit fallback -- the popup could not target a single frame. This frame
    //      still has to actually hold a video, and the backend's admission control
    //      (config.ws.max_sessions) arbitrates down to a single live session.
    //   3. Legacy callers that send no token, accepted only in the top frame AND only
    //      when that frame really has a video (otherwise a video-less top frame would
    //      waste ~1s and report a misleading "No video found").
    const tokenMatches = !!captureOwnerToken && !!msg && msg.__ownerToken === captureOwnerToken;
    const explicitFallback = !!msg && msg.__allowMultiFrameFallback === true && findVideo() !== null;
    const legacyTopFrame = (!msg || msg.__ownerToken === undefined)
      && window === window.top
      && findVideo() !== null;
    if (!tokenMatches && !explicitFallback && !legacyTopFrame) {
      return { success: false, error: "Not the designated capture owner" };
    }
    try {
      streamEpoch = 0;
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
      function triggerStreamReset(reason) {
        if (!isCapturing) return;
        streamEpoch++;
        const mediaTime = video ? video.currentTime : 0.0;
        console.log(`[BS] Stream Reset triggered: reason=${reason}, epoch=${streamEpoch}, mediaTime=${mediaTime.toFixed(3)}s`);
        if (wsClient && wsClient.isConnected) {
          wsClient.sendJSON({
            type: "stream_reset",
            epoch: streamEpoch,
            reason: reason,
            media_time: mediaTime,
            mediaTime: mediaTime,
            wall_time: performance.now()
          });
        }
        if (overlayManager) {
          overlayManager.clear();
        }
        ttsPlayer.clear();
        if (audioCapture) {
          audioCapture.setEpoch(streamEpoch);
        }
        renderedAcks.clear();
      }

      video.addEventListener("seeking", () => triggerStreamReset("seek"), { signal });
      video.addEventListener("pause", () => triggerStreamReset("pause"), { signal });

      // Connect to WebSocket Backend
      wsClient = new WSClient("wss://localhost:8765/ws");
      wsClient.on("connected", () => {
        wsClient.sendJSON(buildWsConfig(settings));
      });

      const renderedAcks = new Set();
      function handleRenderAck(payload, eventType) {
        if (!payload || !payload.utterance_id || payload.epoch === undefined) return;
        // Only ACK when the translated subtitle is updated/rendered (or final ASR update if translation is disabled)
        const isTargetEvent = eventType === "translation" || (payload.is_final && !settings.targetLang);
        if (!isTargetEvent) return;

        const rev = payload.render_revision || 1;
        const ackKey = `${payload.epoch}_${payload.utterance_id}_${rev}`;
        if (renderedAcks.has(ackKey)) return;
        renderedAcks.add(ackKey);

        const rxPerf = performance.now();
        requestAnimationFrame(() => {
          const renderPerf = performance.now();
          const v = getVideo() || findVideo();
          const videoTime = v ? v.currentTime : 0.0;
          const mediaEnd = payload.media_end_time || 0.0;
          const speechOffsetLag = Math.max(0.0, videoTime - mediaEnd);
          const renderCostMs = Math.max(0.0, renderPerf - rxPerf);

          if (wsClient && wsClient.isConnected) {
            wsClient.sendJSON({
              type: "client_render_ack",
              epoch: payload.epoch,
              utterance_id: payload.utterance_id,
              render_revision: rev,
              media_end_time: mediaEnd,
              video_current_time: videoTime,
              speech_offset_to_visible_lag_sec: speechOffsetLag,
              client_render_cost_ms: renderCostMs,
              asr_commit_wall_time: payload.asr_commit_wall_time || 0.0
            });
          }
        });
      }

      function emitSubtitleEvent(eventType, payload) {
        if (payload && payload.epoch !== undefined && payload.epoch < streamEpoch) {
          console.log(`[BS] Generation barrier: Dropped stale ${eventType} from epoch ${payload.epoch} (current=${streamEpoch})`);
          return;
        }
        // 1. Render locally if eligible
        handleSubtitleEvent(eventType, payload);

        // 2. Trigger Client Render Acknowledgement
        handleRenderAck(payload, eventType);

        // 3. Broadcast to other frames (Top frame) only if inside a child iframe
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
      audioCapture.setEpoch(streamEpoch);
      audioCapture.onChunk = (pcmBuffer, timestamp, chunkIdx, chunkStartMediaTime, chunkEndMediaTime, epoch, playbackRate) => {
        if (wsClient && wsClient.isConnected) {
          wsClient.sendBinary(pcmBuffer, timestamp, chunkIdx, false, chunkStartMediaTime, chunkEndMediaTime, epoch, playbackRate);
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
    streamEpoch = 0;
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
    if (msg?.__ownerToken) captureOwnerToken = msg.__ownerToken;
    const v = findVideo();
    if (v) ensureOverlay(v);
    return startCapture(msg || {});
  };
  window.__bsStopCapture = () => {
    captureOwnerToken = null;
    stopCapture();
    if (overlayManager) {
      overlayManager.destroy();
      overlayManager = null;
    }
    return { success: true };
  };
  window.__bsGetStatus = () => ({ isCapturing, hasVideo: !!findVideo(), overlayActive: !!overlayManager });

  // ── Single-capture-owner handshake (audit finding P1-04) ────────────────
  /**
   * Report this frame's video candidacy WITHOUT starting anything, so the popup can
   * pick exactly one capture owner across all frames.
   */
  window.__bsDiscoverCapture = () => {
    const v = findVideo();
    const isTop = window === window.top;
    if (!v) {
      return { hasVideo: false, isCapturing, isTop, isPlaying: false, area: 0, score: -1 };
    }
    const rect = v.getBoundingClientRect ? v.getBoundingClientRect() : { width: 0, height: 0 };
    const area = Math.max((rect.width * rect.height) || 0, (v.videoWidth * v.videoHeight) || 0);
    return {
      hasVideo: true,
      isCapturing,
      isTop,
      isPlaying: !v.paused && !v.ended,
      area,
      score: scoreVideo(v)
    };
  };

  /** Designate (or clear) this frame as the capture owner. */
  window.__bsClaimCaptureOwner = (token) => {
    captureOwnerToken = token || null;
    return { ok: true, isTop: window === window.top, token: captureOwnerToken };
  };

  /** Release ownership so a later START can be claimed again. */
  window.__bsReleaseCaptureOwner = () => {
    captureOwnerToken = null;
    return { ok: true };
  };
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
      const score = scoreVideo(v);
      if (score > maxScore) {
        maxScore = score;
        bestVideo = v;
      }
    }
    return bestVideo;
  }

  /** Ranking score for candidate videos. Shared by findVideo() and __bsDiscoverCapture(). */
  function scoreVideo(v) {
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

    return score;
  }

  window.__bsFindVideo = findVideo;

  console.log("[BS Content] Ready (Frame: " + (window === window.top ? "Top" : "Iframe") + ")");
})();
