// Popup - Sends START/STOP to content script and manages persisted preferences & dynamic engine switching
const api = typeof browser !== "undefined" ? browser : chrome;

(function () {
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnTest = document.getElementById("btnTest");
  const statusBadge = document.getElementById("statusBadge");
  const messageArea = document.getElementById("messageArea");

  const selAsrEngine = document.getElementById("selAsrEngine");
  const selVadEngine = document.getElementById("selVadEngine");
  const rangeVadSilence = document.getElementById("rangeVadSilence");
  const valVadSilence = document.getElementById("valVadSilence");
  const lblActiveModel = document.getElementById("lblActiveModel");
  const selSourceLang = document.getElementById("selSourceLang");
  const selTargetLang = document.getElementById("selTargetLang");
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
  const ttsOptionsPanel = document.getElementById("ttsOptionsPanel");
  const selTtsVoice = document.getElementById("selTtsVoice");
  const selTtsSpeed = document.getElementById("selTtsSpeed");
  const valTtsSpeed = document.getElementById("valTtsSpeed");
  const selTtsDucking = document.getElementById("selTtsDucking");
  const rangeDuckingLevel = document.getElementById("rangeDuckingLevel");
  const valDuckingLevel = document.getElementById("valDuckingLevel");

  let availableVoices = [];
  let savedPreferredVoiceId = null;

  let activeTab = null;
  let isCapturingNow = false;
  let isSwitchingEngine = false;
  let lastActiveAsr = "sensevoice";
  let lastActiveVad = "auto";
  let lastActiveLang = "auto";

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
    const speedVal = selTtsSpeed ? (selTtsSpeed.value || "1.0") : "1.0";
    const vadSilenceMs = parseInt(rangeVadSilence ? rangeVadSilence.value : 300, 10) || 300;
    const cfg = {
      asrEngine: selAsrEngine ? selAsrEngine.value : undefined,
      vadEngine: selVadEngine ? selVadEngine.value : undefined,
      vadSilenceDurationMs: vadSilenceMs,
      silenceDurationMs: vadSilenceMs,
      silence_duration_ms: vadSilenceMs,
      sourceLanguage: selSourceLang ? selSourceLang.value : "auto",
      targetLang: selTargetLang ? selTargetLang.value : "vi",
      subPosY: parseInt(rangeSubPosY ? rangeSubPosY.value : 10, 10) || 10,
      subWidth: parseInt(rangeSubWidth ? rangeSubWidth.value : 80, 10) || 80,
      origFontSize: parseInt(rangeOrigSize ? rangeOrigSize.value : 13, 10) || 13,
      transFontSize: parseInt(rangeTransSize ? rangeTransSize.value : 17, 10) || 17,
      fontWeight: parseInt(rangeFontWeight ? rangeFontWeight.value : 600, 10) || 600,
      fontFamily: selFontFamily ? selFontFamily.value : "default",
      maxLines: parseInt(rangeMaxLines ? rangeMaxLines.value : 3, 10) || 3,
      ttsEnabled: isTts,
      ttsVoice: selectedVoiceId,
      ttsInstruct: "",
      ttsRefAudio: refAudioPath,
      ttsRefText: refAudioText,
      ttsSpeed: speedVal,
      ttsDucking: selTtsDucking ? (selTtsDucking.value === "true") : true,
      duckingLevel: isNaN(duckingPercent) ? 0.25 : duckingPercent / 100,
    };
    return cfg;
  }

  function updateRangeLabels() {
    if (valVadSilence && rangeVadSilence) valVadSilence.textContent = rangeVadSilence.value;
    if (valSubPosY && rangeSubPosY) valSubPosY.textContent = rangeSubPosY.value;
    if (valSubWidth && rangeSubWidth) valSubWidth.textContent = rangeSubWidth.value;
    if (valOrigSize && rangeOrigSize) valOrigSize.textContent = rangeOrigSize.value;
    if (valTransSize && rangeTransSize) valTransSize.textContent = rangeTransSize.value;
    if (valFontWeight && rangeFontWeight) valFontWeight.textContent = rangeFontWeight.value;
    if (valMaxLines && rangeMaxLines) valMaxLines.textContent = rangeMaxLines.value;
    if (valTtsSpeed && selTtsSpeed) valTtsSpeed.textContent = selTtsSpeed.value || "1.0";
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

  function renderLanguageOptions(supportedLanguages, currentLangCode) {
    if (!selSourceLang || !supportedLanguages || supportedLanguages.length === 0) return;
    const previousSelection = currentLangCode || selSourceLang.value;
    selSourceLang.textContent = "";

    supportedLanguages.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.code;
      opt.textContent = item.name;
      selSourceLang.appendChild(opt);
    });

    if (supportedLanguages.some((item) => item.code === previousSelection)) {
      selSourceLang.value = previousSelection;
    } else if (supportedLanguages.some((item) => item.code === "ja")) {
      selSourceLang.value = "ja";
    } else {
      selSourceLang.value = supportedLanguages[0].code;
    }
  }

  async function fetchBackendEngineConfig() {
    let data = null;
    try {
      let res = await fetch("https://localhost:8765/api/config").catch(() => null);
      if (!res || !res.ok) {
        res = await fetch("http://localhost:8765/api/config").catch(() => null);
      }
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

    if (selVadEngine) selVadEngine.value = activeVad;
    if (rangeVadSilence && (data.vad_silence_duration_ms || data.silence_duration_ms) && !rangeVadSilence.dataset.userEdited) {
      rangeVadSilence.value = data.vad_silence_duration_ms || data.silence_duration_ms;
      updateRangeLabels();
    }

    if (!isCapturingNow) {
      statusBadge.textContent = `${activeAsr.toUpperCase()}`;
      statusBadge.className = "badge badge-ready";
      showMsg(`✅ Server Online (ASR: ${activeAsr.toUpperCase()} | VAD: ${data.resolved_vad || activeVad})`, "success");
      btnStart.disabled = false;
    }

    // Populate source languages
    if (data.supported_languages) {
      renderLanguageOptions(data.supported_languages, selSourceLang.value);
    }

    // Populate voice clone profiles
    if (data.tts && data.tts.voices && data.tts.voices.length > 0) {
      availableVoices = data.tts.voices;
      renderVoiceOptions(availableVoices, selTtsVoice ? selTtsVoice.value : null);
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
      const payload = {
        asr_engine: newAsr,
        vad_engine: newVad,
        silence_duration_ms: parseInt(rangeVadSilence ? rangeVadSilence.value : 300, 10) || 300,
        source_lang: newLang,
      };

      let res = await fetch("https://localhost:8765/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).catch(() => null);

      if (!res || !res.ok) {
        res = await fetch("http://localhost:8765/api/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }).catch(() => null);
      }

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
        renderLanguageOptions(result.supported_languages, result.source_lang || selSourceLang.value);
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
        api.tabs.sendMessage(tab.id, { action: "update_settings", settings: cfg }, () => {});
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
      if (selAsrEngine) selAsrEngine.disabled = false;
      if (selVadEngine) selVadEngine.disabled = false;
      btnStart.disabled = isCapturingNow;
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
          return results.map(r => r.result).filter(Boolean);
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
      if ((s.sourceLanguage || s.sourceLang) && selSourceLang) selSourceLang.value = s.sourceLanguage || s.sourceLang;
      if (s.targetLang && selTargetLang) selTargetLang.value = s.targetLang;
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
        if (ttsOptionsPanel) ttsOptionsPanel.style.display = chkEnableTts.checked ? "flex" : "none";
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

    const tab = await fetchActiveTab();
    if (!tab) return;

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
    const tab = await fetchActiveTab();
    if (!tab) { showMsg("No active tab found", "error"); return; }
    const cfg = getSettings();
    api.storage.local.set({ bs_settings: cfg });

    btnStart.disabled = true;
    showMsg("🔍 Đang tìm kiếm video player...", "info");

    const payload = { action: "START_TRANSLATION", sourceLanguage: selSourceLang.value, settings: cfg };
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
    await broadcastToFrames("STOP_TRANSLATION");
    api.tabs.sendMessage(tab.id, { action: "STOP_TRANSLATION" }, () => {});
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

  // ── Live update settings when changed mid-capture ──────
  async function liveUpdateSettings() {
    updateRangeLabels();
    const cfg = getSettings();
    api.storage.local.set({ bs_settings: cfg });
    const tab = await fetchActiveTab();
    if (tab && isCapturingNow) {
      await broadcastToFrames("update_settings", { settings: cfg });
      api.tabs.sendMessage(tab.id, { action: "update_settings", settings: cfg }, () => {});
      showMsg("⚡ Settings applied live", "info");
    }
  }

  function onSettingChange() {
    liveUpdateSettings();
  }

  if (selAsrEngine) selAsrEngine.onchange = handleEngineSwitch;
  if (selVadEngine) selVadEngine.onchange = handleEngineSwitch;
  if (rangeVadSilence) {
    rangeVadSilence.oninput = () => {
      rangeVadSilence.dataset.userEdited = "true";
      onSettingChange();
    };
  }

  if (selSourceLang) {
    selSourceLang.onchange = () => {
      if (selAsrEngine && selAsrEngine.value === "whisper") {
        handleEngineSwitch(true);
      } else {
        onSettingChange();
      }
    };
  }
  if (selTargetLang) selTargetLang.onchange = onSettingChange;
  if (rangeSubPosY) rangeSubPosY.oninput = onSettingChange;
  if (rangeSubWidth) rangeSubWidth.oninput = onSettingChange;
  if (rangeOrigSize) rangeOrigSize.oninput = onSettingChange;
  if (rangeTransSize) rangeTransSize.oninput = onSettingChange;
  if (rangeFontWeight) rangeFontWeight.oninput = onSettingChange;
  if (selFontFamily) selFontFamily.onchange = onSettingChange;
  if (rangeMaxLines) rangeMaxLines.oninput = onSettingChange;

  function updateTtsPanel() {
    if (ttsOptionsPanel && chkEnableTts) {
      ttsOptionsPanel.style.display = chkEnableTts.checked ? "flex" : "none";
    }
  }

  if (chkEnableTts) {
    chkEnableTts.addEventListener("change", () => {
      updateTtsPanel();
      onSettingChange();
    });
  }

  const ttsToggleRow = document.getElementById("ttsToggleRow");
  if (ttsToggleRow && chkEnableTts) {
    ttsToggleRow.addEventListener("click", (e) => {
      if (e.target !== chkEnableTts) {
        chkEnableTts.checked = !chkEnableTts.checked;
        updateTtsPanel();
        onSettingChange();
      }
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
    btnStart.disabled = active || isSwitchingEngine;
    btnStop.disabled = !active;
    statusBadge.textContent = active ? "Capturing" : `${lastActiveAsr.toUpperCase()}`;
    statusBadge.className = active ? "badge badge-connected" : "badge badge-ready";
  }

  function showMsg(t, tp) {
    messageArea.textContent = t;
    messageArea.className = "message-area message-" + tp;
  }
})();
