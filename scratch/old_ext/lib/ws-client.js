// WebSocket client for Extension (Content Script & Popup)
// Uses Background Port Bridge to bypass iframe sandbox / cross-origin CSP restrictions.

class WSClient {
  constructor(url = "wss://localhost:8765/ws") {
    this.url = url;
    this.port = null;
    this.ws = null;
    this.useBridge = typeof chrome !== "undefined" && !!chrome.runtime?.connect;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectDelay = 30000;
    this.baseReconnectDelay = 1000;
    this.pingInterval = null;
    this.listeners = new Map();
  }

  // ── Connection ──────────────────────────────────────────

  async connect() {
    if (this.isConnected) return;

    if (this.useBridge) {
      return this._connectViaBridge();
    } else {
      return this._connectDirect();
    }
  }

  _connectViaBridge() {
    const api = typeof browser !== "undefined" ? browser : chrome;

    return new Promise((resolve, reject) => {
      let settled = false;

      try {
        this.port = api.runtime.connect({ name: "bs-ws-bridge" });

        this.port.onMessage.addListener((msg) => {
          if (!msg) return;

          if (msg.type === "connected") {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this._startPing();
            this._emit("connected");
            if (!settled) {
              settled = true;
              resolve();
            }
          } else if (msg.type === "ws_json") {
            const data = msg.data;
            const payload = data.payload !== undefined ? data.payload : data;
            this._emit(data.type, payload);
            this._emit("message", data);
          } else if (msg.type === "disconnected") {
            this.isConnected = false;
            this._stopPing();
            this._emit("disconnected", { code: msg.code, reason: msg.reason });
            if (!settled) {
              settled = true;
              reject(new Error(`WebSocket connection closed (code: ${msg.code})`));
            } else {
              this._scheduleReconnect();
            }
          } else if (msg.type === "error") {
            const errMsg = msg.error || "WebSocket connection error";
            this._emit("error", errMsg);
            if (!settled) {
              settled = true;
              reject(new Error(errMsg));
            }
          }
        });

        this.port.onDisconnect.addListener(() => {
          this.isConnected = false;
          this._stopPing();
          if (!settled) {
            settled = true;
            reject(new Error("Background bridge disconnected"));
          }
        });

        this.port.postMessage({ action: "CONNECT", url: this.url });
      } catch (e) {
        if (!settled) {
          settled = true;
          reject(e);
        }
      }
    });
  }

  _connectDirect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = "arraybuffer";
        let settled = false;

        this.ws.onopen = () => {
          settled = true;
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this._startPing();
          this._emit("connected");
          resolve();
        };

        this.ws.onmessage = (event) => {
          this._handleMessage(event);
        };

        this.ws.onclose = (event) => {
          this.isConnected = false;
          this._stopPing();
          this._emit("disconnected", { code: event.code, reason: event.reason });
          if (!settled) {
            settled = true;
            reject(new Error(`WebSocket connection closed (code ${event.code})`));
          } else {
            this._scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          this._emit("error", error);
          if (!settled) {
            settled = true;
            reject(new Error("Lỗi kết nối tới " + this.url));
          }
        };
      } catch (e) {
        reject(e);
      }
    });
  }

  disconnect() {
    this._stopPing();
    if (this.port) {
      try {
        this.port.postMessage({ action: "DISCONNECT" });
        this.port.disconnect();
      } catch (e) {}
      this.port = null;
    }
    if (this.ws) {
      try {
        this.ws.onclose = null;
        this.ws.close();
      } catch (e) {}
      this.ws = null;
    }
    this.isConnected = false;
  }

  _scheduleReconnect() {
    if (this.reconnectAttempts > 10) return;
    const delay = Math.min(
      this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );
    this.reconnectAttempts++;
    this._emit("reconnecting", { attempt: this.reconnectAttempts, delayMs: delay });

    setTimeout(() => {
      if (!this.isConnected) {
        this.connect().catch(() => {});
      }
    }, delay);
  }

  _startPing() {
    this._stopPing();
    this.pingInterval = setInterval(() => {
      this.sendJSON({ type: "ping", timestamp: performance.now() });
    }, 30000);
  }

  _stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  sendBinary(pcmData, captureTimestamp, chunkIndex, isPreSpeech = false) {
    if (!this.isConnected || !pcmData) return;

    let byteLength = 0;
    let rawBuffer = null;

    if (pcmData instanceof ArrayBuffer) {
      byteLength = pcmData.byteLength;
      rawBuffer = pcmData;
    } else if (ArrayBuffer.isView(pcmData)) {
      byteLength = pcmData.byteLength;
      rawBuffer = pcmData.buffer.slice(pcmData.byteOffset, pcmData.byteOffset + pcmData.byteLength);
    } else if (pcmData.buffer instanceof ArrayBuffer) {
      byteLength = pcmData.byteLength || pcmData.buffer.byteLength;
      rawBuffer = pcmData.buffer;
    }

    if (!rawBuffer || byteLength === 0) return;

    const header = {
      type: "audio_chunk",
      captureTimestamp,
      chunkIndex,
      sampleRate: 16000,
      channels: 1,
      bitDepth: 16,
      chunkDurationMs: 20,
      samples: Math.floor(byteLength / 2),
      isPreSpeech,
    };

    if (this.port) {
      // Send via port bridge
      this.port.postMessage({
        action: "SEND_BINARY",
        header,
        pcmBuffer: rawBuffer
      });
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const headerStr = JSON.stringify(header);
      const headerBytes = new TextEncoder().encode(headerStr);
      const totalSize = 4 + headerBytes.length + byteLength;
      const buffer = new ArrayBuffer(totalSize);
      const view = new DataView(buffer);
      view.setUint32(0, headerBytes.length, true);
      new Uint8Array(buffer, 4, headerBytes.length).set(headerBytes);
      new Uint8Array(buffer, 4 + headerBytes.length).set(new Uint8Array(rawBuffer));

      try {
        this.ws.send(buffer);
      } catch (e) {
        console.error("[WSClient] Send error:", e);
      }
    }
  }

  sendJSON(data) {
    if (!this.isConnected) return;
    if (this.port) {
      this.port.postMessage({ action: "SEND_JSON", data });
      return;
    }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
      } catch (e) {
        console.error("[WSClient] JSON send error:", e);
      }
    }
  }

  _handleMessage(event) {
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        const payload = msg.payload !== undefined ? msg.payload : msg;
        this._emit(msg.type, payload);
        this._emit("message", msg);
      } catch (e) {
        console.error("[WSClient] Parse error:", e);
      }
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  off(event, callback) {
    const cbs = this.listeners.get(event);
    if (cbs) cbs.delete(callback);
  }

  _emit(event, data) {
    const cbs = this.listeners.get(event);
    if (cbs) {
      cbs.forEach((cb) => {
        try { cb(data); } catch (e) { console.error("[WSClient] Listener error:", e); }
      });
    }
  }
}

if (typeof self !== "undefined") self.WSClient = WSClient;
if (typeof window !== "undefined") window.WSClient = WSClient;
