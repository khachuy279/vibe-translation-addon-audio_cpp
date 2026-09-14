// Popup - Sends START/STOP to content script and manages persisted preferences & dynamic engine switching
//
//  ╔════════════════════════════════════════════════════════════════════════════════╗
//  ║  ⚠️  ĐỌC TRƯỚC KHI SỬA: extension_firefox/CAPTURE_FRAME_NOTES.md                  ║
//  ║                                                                                ║
//  ║  File này chứa logic chọn "capture owner" cho player nằm trong iframe          ║
//  ║  cross-origin. Phá vỡ các invariant dưới đây sẽ gây ra triệu chứng:            ║
//  ║      "Không tìm thấy video nào (Hãy bấm Play video trước)"                      ║
//  ║  …dù trang đang phát video bình thường.                                        ║
//  ║                                                                                ║
//  ║  INV-4  ensureHostAccess() phải là `await` ĐẦU TIÊN trong handler START        ║
//  ║         (Firefox chỉ hiện prompt cấp quyền trong user-gesture handler).        ║
//  ║  INV-5  Popup phải tự chèn content script khi các frame không phản hồi.        ║
//  ║  INV-6  CONTENT_SCRIPT_FILES phải KHỚP manifest.json (đúng thứ tự).            ║
//  ║  INV-7  Frame không có video phải IM LẶNG khi nhận broadcast START.            ║
//  ║  INV-8  Kênh webNavigation + sendMessage({frameId}) phải được giữ.             ║
//  ║                                                                                ║
//  ║  Guard tự động: backend_cpp/tests/test_extension_invariants.py                 ║
//  ╚════════════════════════════════════════════════════════════════════════════════╝
const api = typeof browser !== "undefined" ? browser : chrome;

(function () {
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnTest = document.getElementById("btnTest");
  const btnReset = document.getElementById("btnReset");
  const statusBadge = document.getElementById("statusBadge");
  const messageArea = document.getElementById("messageArea");

  // Default settings aligned with backend_cpp/config.py
  const DEFAULT_SETTINGS = {
    asrEngine: "qwen3-asr-1.7b",
    vadEngine: "fsmn-vad",
    vadSilenceDurationMs: 500,
    silenceDurationMs: 500,
    silence_duration_ms: 500,
    vadThreshold: 0.45,
    vad_threshold: 0.45,
    threshold: 0.45,
    minWordsToCommit: 4,
    min_words_to_commit: 4,
    sourceLanguage: "auto",
    targetLang: "vi",
    translationModel: "tencent",
    subPosY: 10,
    subWidth: 80,
    origFontSize: 10,
    transFontSize: 21,
    fontWeight: 600,
    fontFamily: "default",
    maxLines: 2,
    ttsEnabled: false,
    ttsVoice: "speaker_01_0039.wav",
    ttsSpeed: "1.0",
    ttsDucking: true,
    duckingLevel: 0.25,
  };

  // Empirical optimal profiles per VAD engine (from benchmarks/vad_tuning_benchmark.py)
  const VAD_PROFILES = {
    "fsmn-vad": {
      threshold: 0.45,
      silenceDurationMs: 500,
      hangoverMs: 300,
      preSpeechBufferMs: 120,
    },
    "firered-vad": {
      threshold: 0.80,
      silenceDurationMs: 500,
      hangoverMs: 300,
      preSpeechBufferMs: 100,
    },
    "silero-vad": {
      threshold: 0.50,
      silenceDurationMs: 500,
      hangoverMs: 288,
      preSpeechBufferMs: 96,
    },
  };

  const selAsrEngine = document.getElementById("selAsrEngine");
  const selVadEngine = document.getElementById("selVadEngine");
  const rangeVadSilence = document.getElementById("rangeVadSilence");
  const valVadSilence = document.getElementById("valVadSilence");
  const rangeVadThreshold = document.getElementById("rangeVadThreshold");
  const valVadThreshold = document.getElementById("valVadThreshold");
  const rangeMinWords = document.getElementById("rangeMinWords");
  const valMinWords = document.getElementById("valMinWords");
  const lblActiveModel = document.getElementById("lblActiveModel");
  const selSourceLang = document.getElementById("selSourceLang");
  const selTargetLang = document.getElementById("selTargetLang");
  const selTranslationModel = document.getElementById("selTranslationModel");
  const rangeSubPosY = document.getElementById("rangeSubPosY");
  const valSubPosY = document.getElementById("valSubPosY");
  const rangeSubWidth = document.getElementById("rangeSubWidth");
  const valSubWidth = document.getElementById("valSubWidth");
  const rangeOrigSize = document.getElementById("rangeOrigSize");
  const rangeTransSize = document.getElementById("rangeTransSize");
  const valOrigSize = document.getElementById("valOrigSize");
  const valTransSize = document.getElementById("valTransSize");
  const rangeFontWeight = document.getElementById("rangeFontWeight");
  const valFontWeight = document.getElementById("valFontWeight");
  const selFontFamily = document.getElementById("selFontFamily");
  const rangeMaxLines = document.getElementById("rangeMaxLines");
  const valMaxLines = document.getElementById("valMaxLines");

  const chkEnableTts = document.getElementById("chkEnableTts");
  const selTtsVoice = document.getElementById("selTtsVoice");
  const selTtsSpeed = document.getElementById("selTtsSpeed");
  const valTtsSpeed = document.getElementById("valTtsSpeed");
  const selTtsDucking = document.getElementById("selTtsDucking");
  const rangeDuckingLevel = document.getElementById("rangeDuckingLevel");
  const valDuckingLevel = document.getElementById("valDuckingLevel");

  let availableVoices = [];
  let savedPreferredVoiceId = null;
  let savedPreferredLang = null;

  let activeTab = null;
  let isCapturingNow = false;
  let isSwitchingEngine = false;
  let lastActiveAsr = "sensevoice";
  let lastActiveVad = "auto";
  let lastActiveLang = "auto";
  // null = unknown, true/false = whether the extension holds <all_urls> host access.
  // Without it the extension cannot see cross-origin player iframes at all: both
  // `scripting.executeScript({allFrames:true})` and `webNavigation.getAllFrames()`
  // silently filter out frames it has no permission for.
  let hasHostAccess = null;

  // Keep in sync with manifest.json -> content_scripts[0].js (same order).
  const CONTENT_SCRIPT_FILES = [
    "lib/frame-builder.js",
    "lib/tts-player.js",
    "lib/ws-client.js",
    "lib/audio-capture.js",
    "lib/subtitle-renderer.js",
    "content/overlay-manager.js",
    "content/content-script.js"
  ];

  const HOST_ACCESS_HINT = "Extension chưa được cấp quyền truy cập trang web. Mở about:addons → Bilingual Subtitle → Quyền → bật \"Truy cập dữ liệu của bạn trên mọi website\", rồi F5 lại trang.";

  /**
   * Ask for host access to pages.
   *
   * Declarative content scripts are injected into EVERY frame only when the extension
   * holds host permission for that frame's origin. With just `activeTab` only the active
   * tab's TOP-LEVEL document is covered -- child (and especially cross-origin) frames are
   * not. That is why a cross-origin player iframe is invisible to the popup and never
   * answers: no content script was injected there.
   *
   * MUST be called from a user gesture (the START click); Firefox refuses to show the
   * prompt otherwise.
   *
   * @returns {Promise<boolean>} whether host access is available after the call.
   */
  async function ensureHostAccess() {
    if (!api.permissions) return true;

    try {
      if (api.permissions.contains) {
        hasHostAccess = await api.permissions.contains({ origins: ["<all_urls>"] });
        if (hasHostAccess) return true;
      }
      if (!api.permissions.request) return hasHostAccess === true;

      hasHostAccess = !!(await api.permissions.request({ origins: ["<all_urls>"] }));
      console.log("[Popup] host permission request result:", hasHostAccess);
      return hasHostAccess;
    } catch (e) {
      console.warn("[Popup] host permission request failed:", e);
      return hasHostAccess === true;
    }
  }

  function renderVoiceOptions(voicesList, currentVoiceId) {
    if (!selTtsVoice || !voicesList || voicesList.length === 0) return;
    const prev = currentVoiceId || savedPreferredVoiceId || selTtsVoice.value;
    selTtsVoice.textContent = "";

    voicesList.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = item.name || item.id;
      selTtsVoice.appendChild(opt);
    });

    if (prev && voicesList.some((item) => item.id === prev)) {
      selTtsVoice.value = prev;
    } else {
      selTtsVoice.value = voicesList[0].id;
    }
  }

  function getSettings() {
    const isTts = chkEnableTts ? chkEnableTts.checked : false;
    const selectedVoiceId = selTtsVoice ? selTtsVoice.value : "audio.wav";
    const voiceObj = availableVoices.find(v => v.id === selectedVoiceId) || availableVoices[0] || null;
    const refAudioPath = voiceObj ? voiceObj.audio : ("backend/voices/" + selectedVoiceId);
    const refAudioText = voiceObj ? (voiceObj.text || "") : "";

    const duckingPercent = parseInt(rangeDuckingLevel ? rangeDuckingLevel.value : 25, 10);
    const speedVal = selTtsSpeed ? (selTtsSpeed.value || DEFAULT_SETTINGS.ttsSpeed) : DEFAULT_SETTINGS.ttsSpeed;
    const vadSilenceMs = parseInt(rangeVadSilence ? rangeVadSilence.value : DEFAULT_SETTINGS.silenceDurationMs, 10) || DEFAULT_SETTINGS.silenceDurationMs;
    const rawThresh = rangeVadThreshold ? parseFloat(rangeVadThreshold.value) : DEFAULT_SETTINGS.vadThreshold;
    const vadThresholdVal = isNaN(rawThresh) ? DEFAULT_SETTINGS.vadThreshold : rawThresh;
    const rawMinWords = rangeMinWords ? parseInt(rangeMinWords.value, 10) : DEFAULT_SETTINGS.minWordsToCommit;
    const minWords = isNaN(rawMinWords) ? DEFAULT_SETTINGS.minWordsToCommit : Math.max(0, rawMinWords);
    const cfg = {
      asrEngine: selAsrEngine ? selAsrEngine.value : DEFAULT_SETTINGS.asrEngine,
      vadEngine: selVadEngine ? selVadEngine.value : DEFAULT_SETTINGS.vadEngine,
      vadSilenceDurationMs: vadSilenceMs,
      silenceDurationMs: vadSilenceMs,
      silence_duration_ms: vadSilenceMs,
      vadThreshold: vadThresholdVal,
      vad_threshold: vadThresholdVal,
      threshold: vadThresholdVal,
      minWordsToCommit: minWords,
      min_words_to_commit: minWords,
      sourceLanguage: selSourceLang ? selSourceLang.value : DEFAULT_SETTINGS.sourceLanguage,
      targetLang: selTargetLang ? selTargetLang.value : DEFAULT_SETTINGS.targetLang,
      translationModel: selTranslationModel ? selTranslationModel.value : DEFAULT_SETTINGS.translationModel,
      subPosY: parseInt(rangeSubPosY ? rangeSubPosY.value : DEFAULT_SETTINGS.subPosY, 10) || DEFAULT_SETTINGS.subPosY,
      subWidth: parseInt(rangeSubWidth ? rangeSubWidth.value : DEFAULT_SETTINGS.subWidth, 10) || DEFAULT_SETTINGS.subWidth,
      origFontSize: parseInt(rangeOrigSize ? rangeOrigSize.value : DEFAULT_SETTINGS.origFontSize, 10) || DEFAULT_SETTINGS.origFontSize,
      transFontSize: parseInt(rangeTransSize ? rangeTransSize.value : DEFAULT_SETTINGS.transFontSize, 10) || DEFAULT_SETTINGS.transFontSize,
      fontWeight: parseInt(rangeFontWeight ? rangeFontWeight.value : DEFAULT_SETTINGS.fontWeight, 10) || DEFAULT_SETTINGS.fontWeight,
      fontFamily: selFontFamily ? selFontFamily.value : DEFAULT_SETTINGS.fontFamily,
      maxLines: parseInt(rangeMaxLines ? rangeMaxLines.value : DEFAULT_SETTINGS.maxLines, 10) || DEFAULT_SETTINGS.maxLines,
      ttsEnabled: isTts,
      ttsVoice: selectedVoiceId,
      ttsInstruct: "",
      ttsRefAudio: refAudioPath,
      ttsRefText: refAudioText,
      ttsSpeed: speedVal,
      ttsDucking: selTtsDucking ? (selTtsDucking.value === "true") : DEFAULT_SETTINGS.ttsDucking,
      duckingLevel: isNaN(duckingPercent) ? DEFAULT_SETTINGS.duckingLevel : duckingPercent / 100,
    };
    return cfg;
  }

  function updateVadControlsUI() {
    // VAD is mandatory in streaming pipeline
  }

  function updateRangeLabels() {
    if (selVadEngine) updateVadControlsUI(selVadEngine.value);
    if (valVadSilence && rangeVadSilence) valVadSilence.textContent = rangeVadSilence.value;
    if (valVadThreshold && rangeVadThreshold) valVadThreshold.textContent = parseFloat(rangeVadThreshold.value).toFixed(2);
    if (valMinWords && rangeMinWords) {
      const mw = parseInt(rangeMinWords.value, 10);
      valMinWords.textContent = isNaN(mw) ? String(DEFAULT_SETTINGS.minWordsToCommit) : mw;
    }
    if (valSubPosY && rangeSubPosY) valSubPosY.textContent = rangeSubPosY.value;
    if (valSubWidth && rangeSubWidth) valSubWidth.textContent = rangeSubWidth.value;
    if (valOrigSize && rangeOrigSize) valOrigSize.textContent = rangeOrigSize.value;
    if (valTransSize && rangeTransSize) valTransSize.textContent = rangeTransSize.value;
    if (valFontWeight && rangeFontWeight) valFontWeight.textContent = rangeFontWeight.value;
    if (valMaxLines && rangeMaxLines) valMaxLines.textContent = rangeMaxLines.value;
    if (valTtsSpeed && selTtsSpeed) valTtsSpeed.textContent = selTtsSpeed.value || DEFAULT_SETTINGS.ttsSpeed;
    if (valDuckingLevel && rangeDuckingLevel) valDuckingLevel.textContent = rangeDuckingLevel.value;
  }

  async function fetchActiveTab() {
    if (activeTab) return activeTab;
    try {
      let [tab] = await api.tabs.query({ active: true, lastFocusedWindow: true });
      if (!tab) [tab] = await api.tabs.query({ active: true, currentWindow: true });
      if (!tab) {
        const tabs = await api.tabs.query({ active: true });
        if (tabs && tabs.length > 0) tab = tabs[0];
      }
      activeTab = tab || null;
    } catch (e) {
      console.warn("[Popup] Tab query error:", e);
    }
    return activeTab;
  }

  function renderAsrEngineOptions(availableModels, currentActiveId) {
    if (!selAsrEngine || !availableModels || availableModels.length === 0) return;
    const previousSelection = currentActiveId || selAsrEngine.value;
    selAsrEngine.textContent = "";

    availableModels.forEach((item) => {
      const opt = document.createElement("option");
      const id = typeof item === "string" ? item : item.id;
      let name = typeof item === "string" ? item : (item.name || item.id);
      if (typeof item === "object" && item.is_downloaded) {
        name += " ⚡";
      }
      opt.value = id;
      opt.textContent = name;
      if (typeof item === "object" && item.description) {
        opt.title = item.description;
      }
      selAsrEngine.appendChild(opt);
    });

    if (availableModels.some((item) => (typeof item === "string" ? item : item.id) === previousSelection)) {
      selAsrEngine.value = previousSelection;
    } else {
      selAsrEngine.value = typeof availableModels[0] === "string" ? availableModels[0] : availableModels[0].id;
    }
  }

  function renderTranslationModelOptions(availableModels, currentActiveId) {
    if (!selTranslationModel || !availableModels || !Array.isArray(availableModels) || availableModels.length === 0) return;
    const previousSelection = currentActiveId || selTranslationModel.value;
    selTranslationModel.textContent = "";

    availableModels.forEach((item) => {
      const opt = document.createElement("option");
      const id = typeof item === "string" ? item : item.id;
      let name = typeof item === "string" ? item : (item.name || item.id);
      if (typeof item === "object" && item.is_downloaded) {
        name += " ⚡";
      }
      opt.value = id;
      opt.textContent = name;
      if (typeof item === "object" && (item.desc || item.description)) {
        opt.title = item.desc || item.description;
      }
      selTranslationModel.appendChild(opt);
    });

    if (availableModels.some((item) => (typeof item === "string" ? item : item.id) === previousSelection)) {
      selTranslationModel.value = previousSelection;
    } else {
      selTranslationModel.value = typeof availableModels[0] === "string" ? availableModels[0] : availableModels[0].id;
    }
  }

  function renderLanguageOptions(supportedLanguages, currentLangCode) {
    if (!selSourceLang || !supportedLanguages || supportedLanguages.length === 0) return;
    const targetSelection = currentLangCode || savedPreferredLang || selSourceLang.value || "auto";
    selSourceLang.textContent = "";

    supportedLanguages.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.code;
      opt.textContent = item.name;
      selSourceLang.appendChild(opt);
    });

    if (supportedLanguages.some((item) => item.code === targetSelection)) {
      selSourceLang.value = targetSelection;
    } else {
      selSourceLang.value = supportedLanguages[0].code || "auto";
    }
  }

  let cachedBackendScheme = (typeof sessionStorage !== "undefined" && sessionStorage.getItem("bs_preferred_scheme")) || "https";

  async function fetchBackend(path, options = {}) {
    const timeoutMs = options.timeoutMs || options.timeout || (
      options.method === "POST" ? 60000 : 8000
    );
    const { timeout, timeoutMs: _, ...fetchOpts } = options;
    const schemes = cachedBackendScheme === "http" ? ["http", "https"] : ["https", "http"];
    let lastRes = null;

    for (const scheme of schemes) {
      try {
        const url = `${scheme}://localhost:8765${path}`;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        const res = await fetch(url, { ...fetchOpts, signal: controller.signal });
        clearTimeout(timer);
        if (res) {
          if (res.ok && cachedBackendScheme !== scheme) {
            cachedBackendScheme = scheme;
            if (typeof sessionStorage !== "undefined") {
              sessionStorage.setItem("bs_preferred_scheme", scheme);
            }
          }
          return res;
        }
      } catch (e) {
        // Only retry fallback scheme on network-level failure
      }
    }
    return lastRes;
  }

  async function fetchBackendEngineConfig() {
    let data = null;
    try {
      const res = await fetchBackend("/api/config");
      if (res && res.ok) {
        data = await res.json();
      }
    } catch (e) {
      console.warn("[Popup] Could not fetch backend config:", e);
    }

    if (!data || data.status !== "ok") {
      statusBadge.textContent = "Server Offline";
      statusBadge.className = "badge badge-disconnected";
      showMsg("⚠️ Không kết nối được Backend! Nếu đã chạy 'python main.py', bấm vào đây để mở https://localhost:8765 và chọn 'Nâng cao -> Tiếp tục' để chấp nhận chứng chỉ SSL.", "error");
      if (messageArea) {
        messageArea.style.cursor = "pointer";
        messageArea.onclick = () => {
          api.tabs.create({ url: "https://localhost:8765" });
        };
      }
      btnStart.disabled = true;
      return false;
    } else {
      if (messageArea) {
        messageArea.style.cursor = "default";
        messageArea.onclick = null;
      }
    }

    // Sync ASR & VAD engines from backend
    const activeAsr = data.active_model || data.asr_engine || data.engine || "sensevoice";
    const activeVad = data.vad_engine || "fsmn-vad";
    lastActiveAsr = activeAsr;
    lastActiveVad = activeVad;

    if (lblActiveModel) {
      lblActiveModel.textContent = data.loaded_model || data.model_size || activeAsr;
      lblActiveModel.title = `Mô hình ASR đang nạp: ${data.loaded_model || data.model_size || activeAsr}`;
    }

    // Dynamically populate available ASR engines from backend catalog
    if (data.available_models || data.available_asr_engines) {
      renderAsrEngineOptions(data.available_models || data.available_asr_engines, activeAsr);
    } else if (selAsrEngine) {
      selAsrEngine.value = activeAsr;
    }

    if (data.vad_profiles) {
      for (const [k, prof] of Object.entries(data.vad_profiles)) {
        VAD_PROFILES[k] = {
          threshold: prof.threshold,
          silenceDurationMs: prof.silence_duration_ms,
          hangoverMs: prof.hangover_ms,
          preSpeechBufferMs: prof.pre_speech_buffer_ms,
        };
      }
    }

    if (selVadEngine) selVadEngine.value = activeVad;
    if (rangeVadSilence && (data.vad_silence_duration_ms || data.silence_duration_ms) && !rangeVadSilence.dataset.userEdited) {
      rangeVadSilence.value = data.vad_silence_duration_ms || data.silence_duration_ms;
      updateRangeLabels();
    }
    if (rangeVadThreshold && data.vad_threshold !== undefined && !rangeVadThreshold.dataset.userEdited) {
      rangeVadThreshold.value = data.vad_threshold;
      updateRangeLabels();
    }
    if (rangeMinWords && data.min_words_to_commit !== undefined && !rangeMinWords.dataset.userEdited) {
      rangeMinWords.value = data.min_words_to_commit;
      updateRangeLabels();
    }

    if (!isCapturingNow) {
      statusBadge.textContent = `${activeAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      const vadDisplay = data.resolved_vad || activeVad;
      showMsg(`✅ Server Online (ASR: ${activeAsr.toUpperCase()} | VAD: ${vadDisplay})`, "success");
      btnStart.disabled = false;
    }

    // Populate source languages
    if (data.supported_languages) {
      renderLanguageOptions(data.supported_languages, savedPreferredLang || selSourceLang.value);
    }

    // Populate voice clone profiles
    if (data.tts && data.tts.voices && data.tts.voices.length > 0) {
      availableVoices = data.tts.voices;
      renderVoiceOptions(availableVoices, selTtsVoice ? selTtsVoice.value : null);
    }

    // Sync active translation model from backend & populate dynamic options
    if (data.translation) {
      const activeTrans = data.translation.base || data.translation.translation_model;
      if (data.translation.available_models && data.translation.available_models.length > 0) {
        renderTranslationModelOptions(data.translation.available_models, activeTrans);
      } else if (activeTrans && selTranslationModel) {
        selTranslationModel.value = activeTrans;
      }
    }

    return true;
  }

  // ── Hot-swap ASR / VAD Engine on Backend ───────────────────
  async function handleEngineSwitch(force = false) {
    if (isSwitchingEngine) return;

    const newAsr = selAsrEngine ? selAsrEngine.value : lastActiveAsr;
    const newVad = selVadEngine ? selVadEngine.value : lastActiveVad;
    const newLang = selSourceLang ? selSourceLang.value : "auto";

    if (!force && newAsr === lastActiveAsr && newVad === lastActiveVad && (newAsr !== "whisper" || newLang === lastActiveLang)) return;

    isSwitchingEngine = true;
    if (selAsrEngine) selAsrEngine.disabled = true;
    if (selVadEngine) selVadEngine.disabled = true;
    btnStart.disabled = true;

    const labelDesc = newAsr === "whisper" ? `Whisper (${newLang})` : newAsr.toUpperCase();
    statusBadge.textContent = `Nạp ${labelDesc}...`;
    statusBadge.className = "badge badge-reconnecting";
    if (lblActiveModel) {
      lblActiveModel.textContent = `Đang nạp ${labelDesc}...`;
      lblActiveModel.title = `Đang nạp mô hình ${labelDesc}...`;
    }
    showMsg(`⏳ Đang chuyển đổi sang ${labelDesc} & nạp model Finetunes... Vui lòng đợi.`, "info");

    try {
      const vadProfile = VAD_PROFILES[newVad];
      const payload = {
        asr_engine: newAsr,
        vad_engine: newVad,
        silence_duration_ms: parseInt(rangeVadSilence ? rangeVadSilence.value : DEFAULT_SETTINGS.silenceDurationMs, 10) || DEFAULT_SETTINGS.silenceDurationMs,
        vad_threshold: !isNaN(parseFloat(rangeVadThreshold?.value)) ? parseFloat(rangeVadThreshold.value) : DEFAULT_SETTINGS.vadThreshold,
        hangover_ms: vadProfile ? vadProfile.hangoverMs : undefined,
        pre_speech_buffer_ms: vadProfile ? vadProfile.preSpeechBufferMs : undefined,
        min_words_to_commit: !isNaN(parseInt(rangeMinWords?.value, 10)) ? Math.max(0, parseInt(rangeMinWords.value, 10)) : DEFAULT_SETTINGS.minWordsToCommit,
        source_lang: newLang,
        tts_enabled: chkEnableTts ? chkEnableTts.checked : false,
        tts_voice: selTtsVoice ? selTtsVoice.value : undefined,
        tts_speed: selTtsSpeed ? parseFloat(selTtsSpeed.value || 1.0) : 1.0,
      };

      const res = await fetchBackend("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        timeout: 120000,
      });

      if (!res || !res.ok) {
        const errorDetail = res ? await res.text() : "Network error";
        throw new Error(errorDetail);
      }

      const result = await res.json();
      lastActiveAsr = result.engine || newAsr;
      lastActiveVad = result.vad_engine || newVad;
      lastActiveLang = result.source_lang || newLang;

      if (lblActiveModel) {
        lblActiveModel.textContent = result.loaded_model || result.model_size || lastActiveAsr;
        lblActiveModel.title = `Mô hình ASR đang nạp: ${result.loaded_model || result.model_size || lastActiveAsr}`;
      }

      // Update supported languages dropdown
      if (result.supported_languages) {
        renderLanguageOptions(result.supported_languages, savedPreferredLang || result.source_lang || selSourceLang.value);
      }

      // Save settings
      const cfg = getSettings();
      api.storage.local.set({ bs_settings: cfg });

      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = isCapturingNow ? "badge badge-connected" : "badge badge-ready";
      showMsg(`✅ Đã chuyển sang ASR: ${lastActiveAsr.toUpperCase()} (${lastActiveLang}) | VAD: ${result.resolved_vad || lastActiveVad}`, "success");

      // Notify content script of active changes
      const tab = await fetchActiveTab();
      if (tab && isCapturingNow) {
        await broadcastToFrames("update_settings", { settings: cfg });
      }
    } catch (err) {
      console.error("[Popup] Engine switch error:", err);
      // Rollback dropdown selections
      if (selAsrEngine) selAsrEngine.value = lastActiveAsr;
      if (selVadEngine) selVadEngine.value = lastActiveVad;
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`❌ Lỗi nạp model: ${err.message || "Không thể chuyển engine"}`, "error");
    } finally {
      isSwitchingEngine = false;
      if (selAsrEngine) selAsrEngine.disabled = isCapturingNow;
      if (selVadEngine) selVadEngine.disabled = isCapturingNow;
      btnStart.disabled = isCapturingNow || isSwitchingTranslationModel;
    }
  }

  // ── Hot-swap Translation Model on Backend ─────────────────
  let isSwitchingTranslationModel = false;

  async function handleTranslationModelSwitch() {
    if (isCapturingNow) {
      showMsg("⚠️ Không thể đổi model dịch khi đang dịch! Hãy bấm 'Dừng dịch' trước.", "warning");
      return;
    }
    if (isSwitchingTranslationModel) return;

    const newModel = selTranslationModel ? selTranslationModel.value : "tencent";
    const selectedOption = selTranslationModel && selTranslationModel.selectedOptions ? selTranslationModel.selectedOptions[0] : null;
    const modelDesc = selectedOption ? selectedOption.textContent.replace("⚡", "").trim() : newModel;
    const shortDesc = modelDesc.split("(")[0].trim() || newModel;

    isSwitchingTranslationModel = true;
    if (selTranslationModel) selTranslationModel.disabled = true;
    btnStart.disabled = true;

    statusBadge.textContent = `Nạp ${shortDesc}...`;
    statusBadge.className = "badge badge-reconnecting";
    showMsg(`⏳ Đang chuyển đổi sang model dịch ${modelDesc}... (Nếu chưa có, model sẽ tự tải về)`, "info");

    try {
      const res = await fetchBackend("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ translation_model: newModel }),
        timeout: 180000,
      });

      if (!res || !res.ok) {
        const errorDetail = res ? await res.text() : "Network error";
        throw new Error(errorDetail);
      }

      const cfg = getSettings();
      cfg.translationModel = newModel;
      await api.storage.local.set({ bs_settings: cfg });

      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Đã chuyển sang model dịch: ${modelDesc}`, "success");
    } catch (err) {
      console.error("[Popup] Translation model switch error:", err);
      showMsg(`❌ Lỗi nạp model dịch: ${err.message || "Không thể chuyển model"}`, "error");
      statusBadge.textContent = `${lastActiveAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
    } finally {
      isSwitchingTranslationModel = false;
      if (selTranslationModel) selTranslationModel.disabled = isCapturingNow;
      btnStart.disabled = isCapturingNow || isSwitchingEngine;
    }
  }

  // ── Multi-Frame Broadcast Helper ──────────────────
  async function broadcastToFrames(action, payload = {}) {
    const tab = await fetchActiveTab();
    if (!tab) return null;

    if (api.scripting && api.scripting.executeScript) {
      try {
        const results = await api.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true },
          func: (act, p) => {
            try {
              if (act === "START_TRANSLATION" && typeof window.__bsStartCapture === "function") {
                return window.__bsStartCapture(p);
              }
              if (act === "STOP_TRANSLATION" && typeof window.__bsStopCapture === "function") {
                return window.__bsStopCapture();
              }
              if (act === "GET_STATUS" && typeof window.__bsGetStatus === "function") {
                return window.__bsGetStatus();
              }
              if (act === "update_settings" && typeof window.__bsUpdateSettings === "function") {
                return window.__bsUpdateSettings(p.settings || p);
              }
            } catch (e) {
              return { success: false, error: e.message };
            }
            return null;
          },
          args: [action, payload]
        });
        if (results && results.length > 0) {
          return results.filter(r => r != null && r.result !== undefined).map(r => r.result).filter(Boolean);
        }
      } catch (e) {
        console.warn("[Popup] Scripting broadcast fallback to tabs.sendMessage:", e);
      }
    }

    return new Promise((resolve) => {
      api.tabs.sendMessage(tab.id, { action, ...payload }, (r) => {
        if (api.runtime.lastError) {
          resolve(null);
        } else {
          resolve(r ? [r] : []);
        }
      });
    });
  }

  // ── Single capture owner selection (audit finding P1-04) ───────────────
  //
  // The content script is injected into every frame (`all_frames: true`), and a page can
  // contain several frames holding a <video> -- very common on streaming sites where the
  // real player lives in a CROSS-ORIGIN iframe (e.g. `play.example.net/embed/...`). In
  // that layout the top frame's findVideo() sees nothing, because it cannot read the
  // iframe's cross-origin document. Broadcasting START to all frames would then open one
  // backend WebSocket session per video-bearing frame, which is what P1-04 asked us to
  // stop.
  //
  // Strategy, most precise first:
  //   1. Discover each frame's video candidacy, pick the best, START only that frame.
  //   2. If the browser cannot target a single frame (e.g. `frameIds` unsupported, or the
  //      frame is not injectable), broadcast START with `__allowMultiFrameFallback`, and
  //      let the backend admission control (config.ws.max_sessions) guarantee that only
  //      one session actually survives.
  //
  // Discovery is retried because players commonly create their <video> element only after
  // the iframe has finished bootstrapping -- the old single-frame start path had the same
  // retry, and losing it made iframe-based players fail with "no video found".

  const DISCOVERY_ATTEMPTS = 5;
  const DISCOVERY_RETRY_MS = 300;

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function makeOwnerToken() {
    try {
      if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
    } catch (e) {}
    return `own-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  function pickCaptureOwner(candidates) {
    const eligible = (candidates || []).filter(c => c.info && c.info.hasVideo && !c.info.isCapturing);
    if (!eligible.length) return null;

    eligible.sort((a, b) => {
      if (b.info.score !== a.info.score) return b.info.score - a.info.score;
      if (a.info.isTop !== b.info.isTop) return a.info.isTop ? -1 : 1;
      return (a.frameId ?? 0) - (b.frameId ?? 0);
    });
    return eligible[0];
  }

  function summarizeCandidates(discovery) {
    return (discovery || []).map(r => ({
      frameId: r.frameId,
      hasVideo: !!(r.result && r.result.hasVideo),
      isTop: r.result ? r.result.isTop : null,
      score: r.result ? r.result.score : null
    }));
  }

  /** Ask every frame whether it holds a candidate video, retrying for late players. */
  async function discoverFrames(tab) {
    let last = [];
    for (let attempt = 0; attempt < DISCOVERY_ATTEMPTS; attempt++) {
      try {
        const res = await api.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true },
          func: () => (typeof window.__bsDiscoverCapture === "function" ? window.__bsDiscoverCapture() : null)
        });
        last = res || [];
        if (last.some(r => r && r.result && r.result.hasVideo && !r.result.isCapturing)) {
          return last;
        }
      } catch (e) {
        console.warn("[Popup] discovery attempt failed:", e);
      }
      if (attempt < DISCOVERY_ATTEMPTS - 1) await delay(DISCOVERY_RETRY_MS);
    }
    return last;
  }

  /** True when the extension can enumerate frames and message a specific one. */
  function canTargetFrames() {
    return !!(api.webNavigation && api.webNavigation.getAllFrames
      && api.tabs && api.tabs.sendMessage);
  }

  /**
   * Send a message to a tab, optionally targeting one frame, and resolve with the first
   * response (or null). Works with both the promise-based `browser.*` API and the
   * callback-based `chrome.*`.
   *
   * Omitting `options.frameId` broadcasts to EVERY frame the extension can access, which
   * is the only channel that reaches a cross-origin player iframe.
   */
  function sendMessageToTab(tabId, message, options) {
    return new Promise((resolve) => {
      let settled = false;
      const done = (value) => {
        if (!settled) {
          settled = true;
          resolve(value ?? null);
        }
      };
      const onResponse = (response) => {
        if (api.runtime.lastError) done(null);
        else done(response);
      };
      try {
        const maybePromise = options
          ? api.tabs.sendMessage(tabId, message, options, onResponse)
          : api.tabs.sendMessage(tabId, message, onResponse);
        if (maybePromise && typeof maybePromise.then === "function") {
          maybePromise.then(done).catch(() => done(null));
        }
      } catch (e) {
        done(null);
      }
    });
  }

  /** Send a message to ONE frame and resolve with its response (or null). */
  function sendToFrame(tabId, frameId, message) {
    return sendMessageToTab(tabId, message, { frameId });
  }

  /**
   * Log what the extension is actually allowed to touch.
   *
   * This matters because BOTH `scripting.executeScript({allFrames:true})` and
   * `webNavigation.getAllFrames()` silently filter out frames the extension has no host
   * permission for. When a page's real player lives in a cross-origin iframe that the
   * extension cannot access, the iframe simply never appears -- which looks exactly like
   * "no video found".
   */
  async function logHostPermissions() {
    try {
      if (!api.permissions || !api.permissions.getAll) return;
      const all = await api.permissions.getAll();
      console.log("[Popup] granted origins:", JSON.stringify(all.origins || []));
      if (api.permissions.contains) {
        hasHostAccess = await api.permissions.contains({ origins: ["<all_urls>"] });
        console.log("[Popup] host permission <all_urls>:", hasHostAccess);
      }
    } catch (e) {
      console.warn("[Popup] permission probe failed:", e);
    }
  }

  /**
   * Enumerate the tab's frames with webNavigation and ask each one for its video
   * candidacy.
   *
   * This is the layer that actually works on Firefox for a cross-origin player iframe:
   * `scripting.executeScript({allFrames: true})` only reaches the top frame there, so a
   * page whose real player lives in an iframe (the common streaming-site layout) never
   * produced a candidate. The content script IS present in every frame thanks to the
   * declarative `all_frames: true` manifest entry -- the popup just has to talk to each
   * frame individually.
   *
   * @returns {Promise<Array<{frameId: number, url: string, info: object}>|null>}
   *          `null` when the required APIs are unavailable.
   */
  async function discoverFramesViaMessaging(tab) {
    if (!canTargetFrames()) return null;

    let frames = [];
    let candidates = [];

    for (let attempt = 0; attempt < DISCOVERY_ATTEMPTS; attempt++) {
      frames = await getAllFramesSafe(tab);

      if (attempt === 0) {
        // Log the RAW frame list, including frames that never answer. This is what
        // separates "the iframe is not in the tab at all" (timing / page layout) from
        // "the iframe exists but its content script did not answer" (injection or
        // permission problem) -- two very different fixes.
        console.log("[Popup] frames in tab:", JSON.stringify(
          frames.map(f => ({ frameId: f.frameId, parentFrameId: f.parentFrameId, url: f.url }))
        ));
      }

      candidates = [];
      for (const frame of frames) {
        const info = await sendToFrame(tab.id, frame.frameId, { action: "DISCOVER_CAPTURE" });
        if (info) {
          candidates.push({ frameId: frame.frameId, url: frame.url, info });
        }
      }

      if (candidates.some(c => c.info.hasVideo && !c.info.isCapturing)) break;
      if (attempt < DISCOVERY_ATTEMPTS - 1) await delay(DISCOVERY_RETRY_MS);
    }

    console.log("[Popup] frame candidates (messaging):", JSON.stringify(
      candidates.map(c => ({
        frameId: c.frameId,
        hasVideo: !!c.info.hasVideo,
        isTop: !!c.info.isTop,
        score: c.info.score,
        url: c.url
      }))
    ));

    return candidates;
  }

  /**
   * Start capture in exactly one frame.
   * @returns {Promise<null|{started: boolean, mode: string, results: Array, ownerFrameId?: number}>}
   *          `null` when no popup-side API is available (caller should fall back).
   */
  async function startCaptureOnBestFrame(tab, payload) {
    // ── Layer A: per-frame messaging (works for cross-origin player iframes) ──
    let messagingCandidates = null;
    try {
      messagingCandidates = await discoverFramesViaMessaging(tab);
    } catch (e) {
      console.warn("[Popup] per-frame discovery failed:", e);
    }

    if (messagingCandidates) {
      const owner = pickCaptureOwner(messagingCandidates);
      if (owner) {
        const token = makeOwnerToken();
        const result = await sendToFrame(tab.id, owner.frameId, { ...payload, __ownerToken: token });
        if (result && (result.success || result.isCapturing)) {
          return { started: true, mode: "frame-targeted", ownerFrameId: owner.frameId, results: [result] };
        }
        console.warn("[Popup] Frame-targeted start did not succeed:", JSON.stringify(result));
        return {
          started: false,
          mode: "frame-targeted",
          ownerFrameId: owner.frameId,
          results: result ? [result] : []
        };
      }
    }

    // ── Layer B: messaging broadcast ──
    // `tabs.sendMessage` WITHOUT a frameId is delivered to every frame the extension can
    // access, including cross-origin player iframes that neither `scripting.executeScript`
    // nor `webNavigation.getAllFrames` will reveal. Frames that have nothing to capture
    // stay silent on purpose, so the single response we receive comes from a frame that
    // actually tried to start.
    if (canTargetFrames()) {
      const broadcastResult = await sendMessageToTab(
        tab.id, { ...payload, __allowMultiFrameFallback: true }
      );
      if (broadcastResult && (broadcastResult.success || broadcastResult.isCapturing)) {
        return { started: true, mode: "message-broadcast", results: [broadcastResult] };
      }
      if (broadcastResult && broadcastResult.error) {
        console.warn("[Popup] Message broadcast reported:", JSON.stringify(broadcastResult));
      }
    }

    if (!api.scripting || !api.scripting.executeScript) {
      return messagingCandidates ? { started: false, mode: "none", results: [] } : null;
    }

    // ── Layer B: scripting-based discovery + targeted start ──
    let discovery = [];
    try {
      discovery = await discoverFrames(tab);
    } catch (e) {
      console.warn("[Popup] frame discovery failed:", e);
    }

    console.log("[Popup] capture candidates:", JSON.stringify(summarizeCandidates(discovery)));

    const owner = pickCaptureOwner(
      (discovery || []).map(r => ({ frameId: r.frameId, info: r.result }))
    );

    if (owner && owner.frameId !== undefined && owner.frameId !== null) {
      const token = makeOwnerToken();
      try {
        const started = await api.scripting.executeScript({
          target: { tabId: tab.id, frameIds: [owner.frameId] },
          func: (p, t) => {
            if (typeof window.__bsClaimCaptureOwner === "function") window.__bsClaimCaptureOwner(t);
            if (typeof window.__bsStartCapture === "function") return window.__bsStartCapture(p);
            return { success: false, error: "content script not ready" };
          },
          args: [{ ...payload, __ownerToken: token }, token]
        });

        const results = (started || []).map(r => r.result).filter(Boolean);
        if (results.some(r => r && (r.success || r.isCapturing))) {
          return { started: true, mode: "targeted", ownerFrameId: owner.frameId, results };
        }
        console.warn("[Popup] Targeted start did not succeed; falling back to broadcast.", results);
      } catch (e) {
        console.warn("[Popup] Targeted start failed; falling back to broadcast:", e);
      }
    }

    // ── Layer C: broadcast, arbitrated by the backend's admission control ──
    try {
      const broadcast = await api.scripting.executeScript({
        target: { tabId: tab.id, allFrames: true },
        func: (p) => (typeof window.__bsStartCapture === "function"
          ? window.__bsStartCapture(p)
          : { success: false, error: "content script not ready" }),
        args: [{ ...payload, __allowMultiFrameFallback: true }]
      });

      const results = (broadcast || []).map(r => r.result).filter(Boolean);
      const succeeded = results.some(r => r && (r.success || r.isCapturing));
      return { started: succeeded, mode: "broadcast", results };
    } catch (e) {
      console.warn("[Popup] Broadcast start failed:", e);
    }

    return { started: false, mode: "none", results: [] };
  }

  /** Enumerate the tab's frames, tolerating missing APIs / errors. */
  /**
   * Programmatically inject the content scripts into every frame.
   *
   * Declarative content scripts only run in documents that were loaded while the
   * extension held host permission for that frame's origin. A page opened before the
   * permission was granted (or before the extension was reloaded) therefore has NO
   * content script at all -- not even in the top frame -- so every messaging channel
   * silently returns nothing, which looks like "no video found".
   *
   * Re-injecting is safe: content-script.js bails out immediately when
   * `window.__bsContentScriptLoaded` is already set, and the other files only define
   * globals.
   */
  async function injectContentScripts(tab) {
    if (!api.scripting || !api.scripting.executeScript) return false;
    try {
      await api.scripting.executeScript({
        target: { tabId: tab.id, allFrames: true },
        files: CONTENT_SCRIPT_FILES
      });
      console.log("[Popup] injected content scripts into all frames");
      return true;
    } catch (e) {
      console.warn("[Popup] content script injection failed:", e);
      return false;
    }
  }

  async function getAllFramesSafe(tab) {
    if (!api.webNavigation || !api.webNavigation.getAllFrames) return [];
    try {
      return (await api.webNavigation.getAllFrames({ tabId: tab.id })) || [];
    } catch (e) {
      console.warn("[Popup] getAllFrames failed:", e);
      return [];
    }
  }

  /**
   * Clear ownership in every frame (used on STOP).
   *
   * Uses BOTH mechanisms on purpose: `executeScript` reaches whatever it can, and the
   * per-frame message covers frames it cannot reach (cross-origin iframes on Firefox),
   * which is usually exactly where the capture owner lives.
   */
  async function releaseCaptureOwnership(tab) {
    if (api.scripting && api.scripting.executeScript) {
      try {
        await api.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true },
          func: () => (typeof window.__bsReleaseCaptureOwner === "function"
            ? window.__bsReleaseCaptureOwner()
            : null)
        });
      } catch (e) {
        console.warn("[Popup] executeScript ownership release failed:", e);
      }
    }

    if (canTargetFrames()) {
      const frames = await getAllFramesSafe(tab);
      await Promise.all(
        frames.map(f => sendToFrame(tab.id, f.frameId, { action: "RELEASE_CAPTURE_OWNER" }))
      );
    }
  }

  // ── Init (async, non-blocking) ─────────────────────
  (async () => {
    try {
      const stored = await api.storage.local.get("bs_settings");
      const s = stored.bs_settings || {};
      if (s.asrEngine && selAsrEngine) selAsrEngine.value = s.asrEngine;
      if (s.vadEngine && selVadEngine) selVadEngine.value = s.vadEngine;
      if ((s.vadSilenceDurationMs !== undefined || s.silenceDurationMs !== undefined) && rangeVadSilence) {
        rangeVadSilence.value = s.vadSilenceDurationMs || s.silenceDurationMs;
        rangeVadSilence.dataset.userEdited = "true";
      }
      if ((s.vadThreshold !== undefined || s.vad_threshold !== undefined || s.threshold !== undefined) && rangeVadThreshold) {
        rangeVadThreshold.value = s.vadThreshold !== undefined ? s.vadThreshold : (s.vad_threshold !== undefined ? s.vad_threshold : s.threshold);
        rangeVadThreshold.dataset.userEdited = "true";
      }
      if (s.minWordsToCommit !== undefined && rangeMinWords) {
        const mw = parseInt(s.minWordsToCommit, 10);
        rangeMinWords.value = isNaN(mw) ? DEFAULT_SETTINGS.minWordsToCommit : Math.max(0, mw);
        rangeMinWords.dataset.userEdited = "true";
      }
      if (s.sourceLanguage || s.sourceLang) {
        savedPreferredLang = s.sourceLanguage || s.sourceLang;
        if (selSourceLang) selSourceLang.value = savedPreferredLang;
      }
      if (s.targetLang && selTargetLang) selTargetLang.value = s.targetLang;
      if (s.translationModel && selTranslationModel) selTranslationModel.value = s.translationModel;
      if (s.subPosY !== undefined && rangeSubPosY) rangeSubPosY.value = s.subPosY;
      if (s.subWidth !== undefined && rangeSubWidth) rangeSubWidth.value = s.subWidth;
      if (s.origFontSize) rangeOrigSize.value = s.origFontSize;
      if (s.transFontSize) rangeTransSize.value = s.transFontSize;
      if (s.fontWeight && rangeFontWeight) rangeFontWeight.value = s.fontWeight;
      if (s.fontFamily) selFontFamily.value = s.fontFamily;
      if (s.maxLines && rangeMaxLines) rangeMaxLines.value = s.maxLines;

      // Restore TTS Settings
      if (s.ttsEnabled !== undefined && chkEnableTts) {
        chkEnableTts.checked = !!s.ttsEnabled;
      }
      if (s.ttsVoice) {
        savedPreferredVoiceId = s.ttsVoice;
        if (selTtsVoice) selTtsVoice.value = s.ttsVoice;
      }
      if (s.ttsSpeed !== undefined && selTtsSpeed) {
        let spd = String(s.ttsSpeed);
        if (spd === "1") spd = "1.0";
        selTtsSpeed.value = spd;
        if (!selTtsSpeed.value) selTtsSpeed.value = "1.0";
      }
      if (s.ttsDucking !== undefined && selTtsDucking) selTtsDucking.value = s.ttsDucking ? "true" : "false";
      if (s.duckingLevel !== undefined && rangeDuckingLevel) {
        rangeDuckingLevel.value = Math.round(s.duckingLevel * 100);
      }

      updateRangeLabels();
    } catch (e) {}

    await fetchBackendEngineConfig();

    await logHostPermissions();

    const tab = await fetchActiveTab();
    if (!tab) return;

    // Status must be gathered per frame: the capture owner is very often a cross-origin
    // player iframe, and `executeScript({allFrames:true})` does not reliably reach those,
    // so an iframe-owned capture would otherwise look inactive when the popup is reopened.
    const frames = await getAllFramesSafe(tab);
    const perFrameStatuses = await Promise.all(
      frames.map(f => sendToFrame(tab.id, f.frameId, { action: "GET_STATUS", type: "GET_STATUS" }))
    );
    if (perFrameStatuses.some(r => r && r.isCapturing)) {
      setUI(true);
      return;
    }

    const statuses = await broadcastToFrames("GET_STATUS");
    if (statuses && statuses.some(s => s?.isCapturing)) {
      setUI(true);
    } else {
      api.tabs.sendMessage(tab.id, { action: "GET_STATUS", type: "GET_STATUS" }, (r) => {
        if (!api.runtime.lastError && r?.isCapturing) setUI(true);
      });
    }
  })();

  btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;

    // Ask for host access FIRST, while the user gesture is still fresh: Firefox only
    // allows permissions.request() from a user input handler, and the awaits below would
    // consume that window.
    if (hasHostAccess !== true) {
      const granted = await ensureHostAccess();
      if (!granted) {
        showMsg("⚠️ " + HOST_ACCESS_HINT, "error");
        btnStart.disabled = false;
        return;
      }
    }

    const tab = await fetchActiveTab();
    if (!tab) { showMsg("No active tab found", "error"); btnStart.disabled = false; return; }

    // Make sure the content scripts are actually present. A page loaded before host access
    // was granted has none at all, and then every channel returns nothing.
    if (canTargetFrames()) {
      const frames = await getAllFramesSafe(tab);
      if (frames.length > 0) {
        const responses = await Promise.all(
          frames.map(f => sendToFrame(tab.id, f.frameId, { action: "DISCOVER_CAPTURE" }))
        );
        if (responses.every(r => r === null)) {
          console.log("[Popup] no frame answered -- content scripts are missing, injecting");
          if (!(await injectContentScripts(tab))) {
            showMsg("⚠️ Không chèn được content script. Hãy F5 (tải lại) trang rồi thử lại.", "error");
            btnStart.disabled = false;
            return;
          }
          await delay(200);
        }
      }
    }

    const cfg = getSettings();
    api.storage.local.set({ bs_settings: cfg });

    showMsg("🔍 Đang tìm kiếm video player...", "info");

    const payload = { action: "START_TRANSLATION", sourceLanguage: selSourceLang.value, settings: cfg };

    // Preferred path: designate exactly one capture owner across all frames.
    let outcome = null;
    try {
      outcome = await startCaptureOnBestFrame(tab, payload);
    } catch (e) {
      console.warn("[Popup] Single-owner start failed, falling back to broadcast:", e);
    }

    if (outcome) {
      console.log("[Popup] start outcome:", outcome.mode, outcome.ownerFrameId, JSON.stringify(outcome.results));

      const succeeded = outcome.results.some(r => r && (r.success || r.isCapturing));
      if (succeeded) {
        setUI(true);
        showMsg("✅ Đã tìm thấy video và bắt đầu dịch!", "success");
        return;
      }

      // Only "No video found" is treated as the generic case. Anything else (backend
      // rejected the socket, captureStream failed, ...) is surfaced verbatim so it can
      // actually be diagnosed instead of being hidden behind a friendly message.
      const errors = outcome.results.map(r => r && r.error).filter(Boolean);
      const onlyMissingVideo = errors.length === 0 || errors.every(e => e === "No video found");

      if (onlyMissingVideo && hasHostAccess === false) {
        showMsg("⚠️ " + HOST_ACCESS_HINT, "error");
        btnStart.disabled = false;
        return;
      }

      showMsg(
        "❌ " + (onlyMissingVideo ? "Không tìm thấy video nào (Hãy bấm Play video trước)" : errors[0]),
        "error"
      );
      btnStart.disabled = false;
      return;
    }

    // Legacy fallback when the scripting API is unavailable.
    const results = await broadcastToFrames("START_TRANSLATION", payload);

    if (!results || results.length === 0) {
      api.tabs.sendMessage(tab.id, payload, (r) => {
        if (api.runtime.lastError) {
          showMsg("⚠️ Hãy F5 lại trang và bấm Play video trước!", "error");
          btnStart.disabled = false;
          return;
        }
        if (r?.success || r?.isCapturing) {
          setUI(true);
          showMsg("✅ Đã tìm thấy video và bắt đầu dịch!", "success");
        } else {
          showMsg("❌ " + (r?.error || "Không tìm thấy video nào (Hãy bấm Play video trước)"), "error");
          btnStart.disabled = false;
        }
      });
      return;
    }

    const successResult = results.find(r => r && (r.success || r.isCapturing));
    if (successResult) {
      setUI(true);
      showMsg("✅ Đã tìm thấy video và bắt đầu dịch!", "success");
    } else {
      const specificError = results.find(r => r?.error && r.error !== "No video found");
      const errorMsg = specificError?.error || results.map(r => r?.error).filter(Boolean)[0] || "Không tìm thấy video nào (Hãy bấm Play video trước)";
      showMsg("❌ " + errorMsg, "error");
      btnStart.disabled = false;
    }
  });

  btnStop.addEventListener("click", async () => {
    const tab = await fetchActiveTab();
    if (!tab) return;
    // STOP must reach EVERY frame, including the cross-origin player iframe that the
    // capture owner usually lives in (executeScript alone cannot reach it on Firefox).
    const frames = await getAllFramesSafe(tab);
    await Promise.all(
      frames.map(f => sendToFrame(tab.id, f.frameId, { action: "STOP_TRANSLATION" }))
    );
    await broadcastToFrames("STOP_TRANSLATION");
    // Release ownership so the next START can be claimed again.
    await releaseCaptureOwnership(tab);
    setUI(false);
    showMsg("⏹️ Đã dừng dịch", "info");
  });

  btnTest.addEventListener("click", () => {
    btnTest.disabled = true; btnTest.textContent = "Testing...";
    const ws = new WebSocket("wss://localhost:8765/ws");
    ws.onopen = () => { ws.close(); showMsg("✅ Backend reachable!", "success"); btnTest.disabled = false; btnTest.textContent = "🔌 Test"; };
    ws.onerror = () => { showMsg("❌ Cannot reach backend", "error"); btnTest.disabled = false; btnTest.textContent = "🔌 Test"; };
    setTimeout(() => { if (ws.readyState === 0) { ws.close(); showMsg("❌ Timeout", "error"); btnTest.disabled = false; btnTest.textContent = "🔌 Test"; } }, 5000);
  });

  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      // 1. Remove persisted settings from storage
      await api.storage.local.remove("bs_settings");

      // 2. Reset UI controls to defaults
      if (selAsrEngine) selAsrEngine.value = DEFAULT_SETTINGS.asrEngine;
      if (selVadEngine) selVadEngine.value = DEFAULT_SETTINGS.vadEngine;
      if (rangeVadSilence) {
        rangeVadSilence.value = DEFAULT_SETTINGS.silenceDurationMs;
        delete rangeVadSilence.dataset.userEdited;
      }
      if (rangeVadThreshold) {
        rangeVadThreshold.value = DEFAULT_SETTINGS.vadThreshold;
        delete rangeVadThreshold.dataset.userEdited;
      }
      if (rangeMinWords) {
        rangeMinWords.value = DEFAULT_SETTINGS.minWordsToCommit;
        delete rangeMinWords.dataset.userEdited;
      }
      if (selSourceLang) {
        selSourceLang.value = DEFAULT_SETTINGS.sourceLanguage;
        savedPreferredLang = DEFAULT_SETTINGS.sourceLanguage;
      }
      if (selTargetLang) selTargetLang.value = DEFAULT_SETTINGS.targetLang;
      if (selTranslationModel) selTranslationModel.value = DEFAULT_SETTINGS.translationModel;

      if (chkEnableTts) chkEnableTts.checked = DEFAULT_SETTINGS.ttsEnabled;
      if (selTtsSpeed) selTtsSpeed.value = DEFAULT_SETTINGS.ttsSpeed;
      if (selTtsDucking) selTtsDucking.value = DEFAULT_SETTINGS.ttsDucking ? "true" : "false";
      if (rangeDuckingLevel) rangeDuckingLevel.value = Math.round(DEFAULT_SETTINGS.duckingLevel * 100);

      if (rangeSubPosY) rangeSubPosY.value = DEFAULT_SETTINGS.subPosY;
      if (rangeSubWidth) rangeSubWidth.value = DEFAULT_SETTINGS.subWidth;
      if (rangeOrigSize) rangeOrigSize.value = DEFAULT_SETTINGS.origFontSize;
      if (rangeTransSize) rangeTransSize.value = DEFAULT_SETTINGS.transFontSize;
      if (rangeFontWeight) rangeFontWeight.value = DEFAULT_SETTINGS.fontWeight;
      if (selFontFamily) selFontFamily.value = DEFAULT_SETTINGS.fontFamily;
      if (rangeMaxLines) rangeMaxLines.value = DEFAULT_SETTINGS.maxLines;

      updateRangeLabels();

      // 3. Save clean defaults
      await api.storage.local.set({ bs_settings: DEFAULT_SETTINGS });

      // 4. Propagate to active capturing frames if any
      const tab = await fetchActiveTab();
      if (tab && isCapturingNow) {
        await broadcastToFrames("update_settings", { settings: DEFAULT_SETTINGS });
      }

      // 5. Sync to backend configuration
      try {
        await fetchBackend("/api/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            asr_engine: DEFAULT_SETTINGS.asrEngine,
            vad_engine: DEFAULT_SETTINGS.vadEngine,
            silence_duration_ms: DEFAULT_SETTINGS.silenceDurationMs,
            vad_threshold: DEFAULT_SETTINGS.vadThreshold,
            min_words_to_commit: DEFAULT_SETTINGS.minWordsToCommit,
            source_lang: DEFAULT_SETTINGS.sourceLanguage,
            target_lang: DEFAULT_SETTINGS.targetLang,
            tts_enabled: DEFAULT_SETTINGS.ttsEnabled,
          }),
        });
      } catch (e) {}

      showMsg("↺ Đã đưa toàn bộ cài đặt về mặc định", "success");
    });
  }

  // ── Live update settings with 150ms debounce for sliders ──────
  let _settingDebounceTimer = null;

  async function liveUpdateSettings(immediate = false) {
    updateRangeLabels();
    clearTimeout(_settingDebounceTimer);

    const apply = async () => {
      const cfg = getSettings();
      api.storage.local.set({ bs_settings: cfg });
      const tab = await fetchActiveTab();
      if (tab && isCapturingNow) {
        await broadcastToFrames("update_settings", { settings: cfg });
        showMsg("⚡ Settings applied live", "info");
      }
      fetchBackend("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          min_words_to_commit: cfg.minWordsToCommit,
          silence_duration_ms: cfg.silenceDurationMs,
          vad_threshold: cfg.vadThreshold,
        }),
      }).catch(() => {});
    };

    if (immediate) {
      await apply();
    } else {
      _settingDebounceTimer = setTimeout(apply, 150);
    }
  }

  function onSettingChange(immediate = false) {
    liveUpdateSettings(immediate);
  }

  if (selAsrEngine) selAsrEngine.onchange = () => handleEngineSwitch();
  if (selVadEngine) {
    selVadEngine.onchange = () => {
      const selectedVad = selVadEngine.value;
      updateVadControlsUI(selectedVad);
      const profile = VAD_PROFILES[selectedVad];
      if (profile) {
        if (rangeVadSilence) {
          rangeVadSilence.value = profile.silenceDurationMs;
          delete rangeVadSilence.dataset.userEdited;
        }
        if (rangeVadThreshold) {
          rangeVadThreshold.value = profile.threshold;
          delete rangeVadThreshold.dataset.userEdited;
        }
        updateRangeLabels();
      }
      handleEngineSwitch();
    };
  }
  if (rangeVadSilence) {
    rangeVadSilence.oninput = () => {
      rangeVadSilence.dataset.userEdited = "true";
      onSettingChange();
    };
  }
  if (rangeVadThreshold) {
    rangeVadThreshold.oninput = () => {
      rangeVadThreshold.dataset.userEdited = "true";
      onSettingChange();
    };
  }
  if (rangeMinWords) {
    rangeMinWords.oninput = () => {
      rangeMinWords.dataset.userEdited = "true";
      onSettingChange();
    };
  }

  if (selSourceLang) {
    selSourceLang.onchange = () => {
      savedPreferredLang = selSourceLang.value;
      if (selAsrEngine && selAsrEngine.value === "whisper") {
        handleEngineSwitch(true);
      } else {
        onSettingChange();
      }
    };
  }
  if (selTargetLang) selTargetLang.onchange = onSettingChange;
  if (selTranslationModel) selTranslationModel.onchange = handleTranslationModelSwitch;
  if (rangeSubPosY) rangeSubPosY.oninput = onSettingChange;
  if (rangeSubWidth) rangeSubWidth.oninput = onSettingChange;
  if (rangeOrigSize) rangeOrigSize.oninput = onSettingChange;
  if (rangeTransSize) rangeTransSize.oninput = onSettingChange;
  if (rangeFontWeight) rangeFontWeight.oninput = onSettingChange;
  if (selFontFamily) selFontFamily.onchange = onSettingChange;
  if (rangeMaxLines) rangeMaxLines.oninput = onSettingChange;

  function syncTtsConfig(enabled) {
    const payload = {
      tts_enabled: enabled,
      tts_voice: selTtsVoice ? selTtsVoice.value : undefined,
      tts_speed: selTtsSpeed ? parseFloat(selTtsSpeed.value || 1.0) : 1.0,
    };
    fetchBackend("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).catch(() => {});
  }

  function prewarmTTS() {
    syncTtsConfig(true);
    fetchBackend("/api/tts/prewarm", { method: "POST" }).catch(() => {});
  }

  if (chkEnableTts) {
    chkEnableTts.addEventListener("change", () => {
      onSettingChange();
      syncTtsConfig(chkEnableTts.checked);
      if (chkEnableTts.checked) {
        prewarmTTS();
      }
    });
  }

  const ttsToggleRow = document.getElementById("ttsToggleRow");
  if (ttsToggleRow && chkEnableTts) {
    ttsToggleRow.addEventListener("click", (e) => {
      if (e.target.closest(".switch")) {
        return;
      }
      chkEnableTts.checked = !chkEnableTts.checked;
      chkEnableTts.dispatchEvent(new Event("change"));
    });
  }

  if (selTtsVoice) {
    selTtsVoice.onchange = () => {
      savedPreferredVoiceId = selTtsVoice.value;
      onSettingChange();
    };
  }
  if (selTtsSpeed) selTtsSpeed.onchange = onSettingChange;
  if (selTtsDucking) selTtsDucking.onchange = onSettingChange;
  if (rangeDuckingLevel) rangeDuckingLevel.oninput = onSettingChange;

  function setUI(active) {
    isCapturingNow = active;
    btnStart.disabled = active || isSwitchingEngine || isSwitchingTranslationModel;
    btnStop.disabled = !active;
    if (selTranslationModel) {
      selTranslationModel.disabled = active || isSwitchingTranslationModel;
    }
    if (selAsrEngine) {
      selAsrEngine.disabled = active || isSwitchingEngine;
    }
    if (selVadEngine) {
      selVadEngine.disabled = active || isSwitchingEngine;
    }
    statusBadge.textContent = active ? "Capturing" : `${lastActiveAsr.toUpperCase()}`;
    statusBadge.className = active ? "badge badge-connected" : "badge badge-ready";
  }

  function showMsg(t, tp) {
    messageArea.textContent = t;
    messageArea.className = "message-area message-" + tp;
  }
})();
