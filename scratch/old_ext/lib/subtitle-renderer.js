// Subtitle Renderer Engine — 3-Layer Anchor Focus Architecture
// Layer 1: [Lịch sử cũ + Bản dịch] — Cuộn lên trên (dimmed, font nhỏ hơn một chút)
// Layer 2: [Phụ đề mới nhất + Bản dịch] — CỐ ĐỊNH LÀM TRUNG TÂM TIÊU ĐIỂM (100% focus)
// Layer 3: [Phụ đề đang nhận diện / chưa chốt] — Luôn ở dưới cùng (live typing)

class SubtitleRenderer {
  constructor(container) {
    this.container = container;
    this.mode = "bottom_bar"; // "bottom_bar" | "floating"

    // Completed translated sentences pool (CHỈ CHỨA CÁC CÂU ĐÃ CÓ BẢN DỊCH HOÀN CHỈNH)
    this.completedSentences = [];
    this.maxLines = 3; // Tổng số câu hoàn chỉnh hiển thị (mặc định 3)
    this.maxHistorySentences = 2; // Số câu lịch sử tối đa ở Layer 1 (= maxLines - 1)

    // Lưu trữ câu gốc (utterance_id -> { id, originalText })
    this.utteranceStore = new Map();

    // Auto-clear timeout (xóa phụ đề sau 10s im lặng)
    this.autoClearTimeoutMs = 10000;
    this.autoClearTimer = null;

    // Current live draft (Layer 3) — Đang nhận diện realtime (CHƯA chốt VAD)
    this.currentDraft = null; // { id, originalText, isFinal: false }

    // Pending focus (Layer 2) — Câu đã chốt VAD, đang chờ bản dịch
    // Xuất hiện ngay ở Layer 2 với indicator ・・・, bản dịch điền vào sau
    this.pendingFocus = null; // { id, originalText }

    // DOM Elements for 3 distinct layers
    this._buildDOM();

    // Tự động tải cài đặt maxLines từ chrome.storage
    if (typeof chrome !== "undefined" && chrome?.storage?.local) {
      chrome.storage.local.get("bs_settings", (data) => {
        if (data?.bs_settings) {
          this.applySettings(data.bs_settings);
        }
      });
    }
  }

  _resetAutoClearTimer() {
    if (this.autoClearTimer) {
      clearTimeout(this.autoClearTimer);
      this.autoClearTimer = null;
    }
    if (this.autoClearTimeoutMs > 0) {
      this.autoClearTimer = setTimeout(() => {
        this.clear();
      }, this.autoClearTimeoutMs);
    }
  }

  _buildDOM() {
    this.container.textContent = "";

    // TẦNG 1: Lịch sử cũ (Cuộn lên trên)
    this.historyLayer = document.createElement("div");
    this.historyLayer.className = "bs-history-layer";

    // TẦNG 2: Tiêu điểm trung tâm cố định (Anchor Focus)
    this.focusLayer = document.createElement("div");
    this.focusLayer.className = "bs-focus-layer";

    // TẦNG 3: Liveview đang nhận diện / chưa chốt VAD (Cố định ở dưới cùng)
    this.liveLayer = document.createElement("div");
    this.liveLayer.className = "bs-live-layer";

    this.container.appendChild(this.historyLayer);
    this.container.appendChild(this.focusLayer);
    this.container.appendChild(this.liveLayer);
  }

  setMode(mode) {
    this.mode = mode;
    this.container.className = "bs-content-area";
  }

  setMaxLines(maxLines) {
    const val = parseInt(maxLines, 10);
    if (!isNaN(val) && val >= 1) {
      this.maxLines = val;
      this.maxHistorySentences = Math.max(0, val - 1); // Layer 1 chứa (maxLines - 1) câu cũ
      while (this.completedSentences.length > this.maxLines) {
        this.completedSentences.shift();
      }
      this._renderAll();
    }
  }

  applySettings(settings) {
    if (!settings) return;
    if (settings.maxLines !== undefined) {
      this.setMaxLines(settings.maxLines);
    }
  }

  // ── Utterance Update (Realtime Live Stream & Finalization) ─────────────

  onUtteranceUpdate(payload) {
    if (!payload) return;
    const utteranceId = payload.utterance_id || payload.utteranceId || payload.id;

    // Nếu bị lọc hoặc xóa
    if (payload.filtered || payload.is_deleted || payload.deleted) {
      if (this.currentDraft && this.currentDraft.id === utteranceId) {
        this.currentDraft = null;
      }
      if (this.pendingFocus && this.pendingFocus.id === utteranceId) {
        this.pendingFocus = null;
      }
      this.utteranceStore.delete(utteranceId);
      this.completedSentences = this.completedSentences.filter((s) => s.id !== utteranceId);
      this._renderAll();
      return;
    }

    const text = payload.ui_text || payload.original || payload.text || "";
    if (!text || !text.trim()) {
      if (this.currentDraft && this.currentDraft.id === utteranceId) {
        this.currentDraft = null;
        this._renderAll();
      }
      return;
    }

    this._resetAutoClearTimer();

    const isFinal = payload.is_final || payload.isFinal || false;

    // Lưu vào utteranceStore để không bao giờ bị mất câu gốc khi dịch hoàn thành
    this.utteranceStore.set(utteranceId, {
      id: utteranceId,
      originalText: text,
      isFinal: isFinal,
    });

    // Giới hạn kích thước utteranceStore tránh phình bộ nhớ
    if (this.utteranceStore.size > 50) {
      const oldestKey = this.utteranceStore.keys().next().value;
      this.utteranceStore.delete(oldestKey);
    }

    // 1. Nếu câu này đã nằm trong completedSentences (đã có bản dịch) -> cập nhật text
    let existingCompleted = this.completedSentences.find((s) => s.id === utteranceId);
    if (existingCompleted) {
      existingCompleted.originalText = text;
      this._renderAll();
      return;
    }

    // 2. Nếu câu này đang ở pendingFocus (chờ dịch ở Layer 2) -> cập nhật text
    if (this.pendingFocus && this.pendingFocus.id === utteranceId) {
      this.pendingFocus.originalText = text;
      this._renderAll();
      return;
    }

    if (isFinal) {
      // 3a. Câu vừa CHỐT VAD -> đẩy lên Layer 2 ngay (pendingFocus), xóa Layer 3
      // Nếu đã có pendingFocus cũ (bị timeout không có dịch) -> đẩy xuống completedSentences không có dịch
      if (this.pendingFocus) {
        this._promotePendingToCompleted(null);
      }
      this.pendingFocus = { id: utteranceId, originalText: text };
      this.currentDraft = null;
    } else {
      // 3b. Câu đang nhận diện dở (chưa chốt) -> Layer 3
      this.currentDraft = { id: utteranceId, originalText: text, isFinal: false };
    }

    this._renderAll();
  }

  // ── Translation Update ──────────────────────────────────────────

  onTranslation(sentenceId, translatedText, status) {
    this._resetAutoClearTimer();
    let id = sentenceId;
    let text = translatedText;
    let st = status || "ok";

    if (typeof sentenceId === "object" && sentenceId !== null) {
      id = sentenceId.sentence_id || sentenceId.sentenceId || sentenceId.utterance_id || sentenceId.utteranceId;
      text = sentenceId.translated || sentenceId.text || (sentenceId.payload && (sentenceId.payload.text || sentenceId.payload.translated)) || translatedText;
      st = sentenceId.status || (sentenceId.payload && sentenceId.payload.status) || status || "ok";
    }

    if (!id) return;

    const cleanTranslation = (st === "ok" && text) ? text : null;

    // Lấy câu gốc đã lưu trong utteranceStore
    const stored = this.utteranceStore.get(id) || {};

    // 1. Nếu câu đang ở pendingFocus (Layer 2, chờ dịch) -> điền bản dịch vào, chuyển sang completedSentences
    if (this.pendingFocus && this.pendingFocus.id === id) {
      this._promotePendingToCompleted(cleanTranslation);
      this._renderAll();
      return;
    }

    const originalText = stored.originalText || "";

    // 2. Kiểm tra xem câu đã có trong completedSentences chưa
    const existing = this.completedSentences.find((s) => s.id === id);
    if (existing) {
      existing.translatedText = cleanTranslation;
      if (!existing.originalText && originalText) {
        existing.originalText = originalText;
      }
      existing.isFinal = true;
    } else {
      // 3. Tạo completedItem đầy đủ cả originalText và translatedText
      const completedItem = {
        id: id,
        originalText: originalText,
        translatedText: cleanTranslation,
        isFinal: true,
      };
      this.completedSentences.push(completedItem);
    }

    // Nếu câu vừa dịch trùng với currentDraft ở Layer 3 -> Xóa Layer 3
    if (this.currentDraft && this.currentDraft.id === id) {
      this.currentDraft = null;
    }

    // Giới hạn số lượng câu lưu trữ theo Max Lines (Focus + History = maxLines)
    const maxTotal = this.maxLines || (this.maxHistorySentences + 1) || 3;
    while (this.completedSentences.length > maxTotal) {
      this.completedSentences.shift();
    }

    this._renderAll();
  }

  // Chuyển pendingFocus thành completedSentence với bản dịch cho sẵn (hoặc null nếu không có)
  _promotePendingToCompleted(translatedText) {
    if (!this.pendingFocus) return;
    const item = {
      id: this.pendingFocus.id,
      originalText: this.pendingFocus.originalText,
      translatedText: translatedText,
      isFinal: true,
    };
    this.completedSentences.push(item);
    this.pendingFocus = null;

    // Giới hạn số lượng câu lưu trữ
    const maxTotal = this.maxLines || (this.maxHistorySentences + 1) || 3;
    while (this.completedSentences.length > maxTotal) {
      this.completedSentences.shift();
    }
  }

  // ── Render 3 Layers ───────────────────────────────────────────────

  _renderAll() {
    this._renderHistoryAndFocusLayers();
    this._renderLiveLayer();
  }

  // TẦNG 1 & 2: Lịch sử cũ (Tầng 1) và Tiêu điểm mới nhất (Tầng 2)
  _renderHistoryAndFocusLayers() {
    const historyFragment = document.createDocumentFragment();
    const focusFragment = document.createDocumentFragment();
    const len = this.completedSentences.length;

    // Nếu có pendingFocus (câu chốt đang chờ dịch) -> ưu tiên hiển thị ở Layer 2
    if (this.pendingFocus) {
      // Layer 2: pendingFocus với indicator ・・・
      const focusEl = this._createPendingFocusElement(this.pendingFocus);
      focusFragment.appendChild(focusEl);

      // Layer 1: completedSentences hiển thị trong history (capped by maxHistorySentences)
      const histStart = Math.max(0, len - this.maxHistorySentences);
      for (let i = histStart; i < len; i++) {
        const historyEl = this._createSentenceElement(this.completedSentences[i], "bs-sentence bs-history-item");
        historyFragment.appendChild(historyEl);
      }
    } else if (len > 0) {
      // Không có pendingFocus -> hiển thị câu completed mới nhất ở Layer 2
      const latestItem = this.completedSentences[len - 1];
      const focusEl = this._createSentenceElement(latestItem, "bs-sentence bs-focus-item");
      focusFragment.appendChild(focusEl);

      // Các câu cũ trước đó vào Layer 1
      const histStart = Math.max(0, len - 1 - this.maxHistorySentences);
      for (let i = histStart; i < len - 1; i++) {
        const historyEl = this._createSentenceElement(this.completedSentences[i], "bs-sentence bs-history-item");
        historyFragment.appendChild(historyEl);
      }
    }

    this.historyLayer.replaceChildren(historyFragment);
    this.focusLayer.replaceChildren(focusFragment);
  }

  // TẦNG 3: Liveview nhận diện chưa chốt (Dưới cùng)
  _renderLiveLayer() {
    const fragment = document.createDocumentFragment();

    // Layer 3 chỉ hiển thị khi đang nhận diện dở (chưa chốt VAD)
    // Câu đã chốt VAD đã được đẩy lên Layer 2 (pendingFocus)
    if (this.currentDraft && this.currentDraft.originalText && !this.currentDraft.isFinal) {
      const sentenceDiv = document.createElement("div");
      sentenceDiv.className = "bs-sentence bs-live-item";

      const originalSpan = document.createElement("span");
      originalSpan.className = "bs-original";
      originalSpan.textContent = this.currentDraft.originalText;
      sentenceDiv.appendChild(originalSpan);

      fragment.appendChild(sentenceDiv);
    }

    this.liveLayer.replaceChildren(fragment);
  }

  // Tạo element cho pendingFocus (Layer 2, chờ dịch)
  _createPendingFocusElement(item) {
    const sentenceDiv = document.createElement("div");
    sentenceDiv.className = "bs-sentence bs-focus-item bs-focus-pending";
    sentenceDiv.setAttribute("data-sentence-id", item.id);

    const originalDiv = document.createElement("div");
    originalDiv.className = "bs-original";
    originalDiv.textContent = item.originalText || "";
    sentenceDiv.appendChild(originalDiv);

    // Indicator chờ dịch
    const translatingDiv = document.createElement("div");
    translatingDiv.className = "bs-translating";
    translatingDiv.textContent = "・・・";
    sentenceDiv.appendChild(translatingDiv);

    return sentenceDiv;
  }

  _createSentenceElement(item, className) {
    const sentenceDiv = document.createElement("div");
    sentenceDiv.className = className;
    sentenceDiv.setAttribute("data-sentence-id", item.id);
    sentenceDiv.setAttribute("data-utterance-id", item.id);

    const originalDiv = document.createElement("div");
    originalDiv.className = "bs-original";
    originalDiv.textContent = item.originalText || "";
    sentenceDiv.appendChild(originalDiv);

    // Bản dịch
    if (item.translatedText) {
      const translatedDiv = document.createElement("div");
      translatedDiv.className = "bs-translated";
      translatedDiv.textContent = item.translatedText;
      sentenceDiv.appendChild(translatedDiv);
    }

    return sentenceDiv;
  }

  // ── Legacy Handlers ───────────────────────────────────────

  onPartialTranscript(tokens) {
    if (!tokens || tokens.length === 0) return;
    const text = tokens.map((t) => t.text).join("");
    if (text) {
      this.onUtteranceUpdate({
        utterance_id: "legacy-draft",
        ui_text: text,
        is_final: false,
      });
    }
  }

  onSentenceComplete(sentence) {
    if (!sentence) return;
    this.onUtteranceUpdate({
      utterance_id: sentence.sentence_id || sentence.id,
      ui_text: sentence.ui_text || sentence.text || "",
      is_final: true,
    });
  }

  clear() {
    if (this.autoClearTimer) {
      clearTimeout(this.autoClearTimer);
      this.autoClearTimer = null;
    }
    this.completedSentences = [];
    this.utteranceStore.clear();
    this.currentDraft = null;
    this.pendingFocus = null;
    if (this.historyLayer) this.historyLayer.textContent = "";
    if (this.focusLayer) this.focusLayer.textContent = "";
    if (this.liveLayer) this.liveLayer.textContent = "";
  }
}

// Attach to global for content script use
if (typeof window !== "undefined") {
  window.SubtitleRenderer = SubtitleRenderer;
}
