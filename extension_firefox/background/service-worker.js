// Background Service Worker
// Provides a WebSocket Bridge so content scripts inside cross-origin iframes
// can reliably connect to the backend (wss://localhost:8765/ws) without being
// blocked by iframe CSP, sandbox, or Private Network Access restrictions.

const api = typeof browser !== "undefined" ? browser : chrome;
const textEncoder = new TextEncoder();

// P3.2: NGƯỠNG BACKPRESSURE. Trước đây `bufferedAmount` không được đọc ở đâu cả
// (0 lần trong toàn bộ extension), nên khi backend nghẽn thì client vẫn bơm
// 32 KB/s audio vào buffer gửi của WebSocket mà không có giới hạn => RAM tăng và
// độ trễ KHÔNG BAO GIỜ hồi phục (audio được gửi là audio của hàng chục giây trước).
//   - vượt SOFT: bỏ frame audio cũ đang xếp (chỉ giữ audio mới nhất) + báo content script
//   - vượt HARD: ngừng gửi audio, báo content script tạm dừng capture
const SEND_BUFFER_SOFT_LIMIT = 128 * 1024;   // 128 KB ~ 4 giây audio 16kHz PCM16
const SEND_BUFFER_HARD_LIMIT = 512 * 1024;   // 512 KB ~ 16 giây

api.runtime.onConnect.addListener((port) => {
  if (port.name !== "bs-ws-bridge") return;

  let ws = null;
  let isClosed = false;
  let droppedFrames = 0;
  let pausedByBackpressure = false;
  // Chỉ giữ frame audio MỚI NHẤT khi socket nghẽn: audio cũ đã vô dụng cho phụ đề realtime.
  let pendingAudio = null;

  function cleanup() {
    isClosed = true;
    pendingAudio = null;
    if (ws) {
      try {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close();
      } catch (e) {}
      ws = null;
    }
  }

  function notifyBackpressure(state) {
    try {
      port.postMessage({
        type: "backpressure",
        state: state,                 // "ok" | "dropping" | "paused"
        bufferedAmount: ws ? ws.bufferedAmount : 0,
        droppedFrames: droppedFrames
      });
    } catch (e) {}
  }

  function sendOrQueue(buffer) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      pendingAudio = null;
      return;
    }
    const buffered = ws.bufferedAmount || 0;

    if (buffered >= SEND_BUFFER_HARD_LIMIT) {
      // Quá tải nặng: bỏ frame, tạm dừng capture cho tới khi rút hết hàng đợi.
      droppedFrames++;
      pendingAudio = null;
      if (!pausedByBackpressure) {
        pausedByBackpressure = true;
        console.warn("[BS Background] Backpressure HARD: tạm dừng gửi audio.");
        notifyBackpressure("paused");
      }
      return;
    }

    if (buffered >= SEND_BUFFER_SOFT_LIMIT) {
      // Đang tắc: chỉ giữ frame mới nhất, bỏ frame đang chờ.
      droppedFrames++;
      pendingAudio = buffer;
      notifyBackpressure("dropping");
      return;
    }

    if (pausedByBackpressure) {
      pausedByBackpressure = false;
      notifyBackpressure("ok");
    }
    pendingAudio = null;
    try {
      ws.send(buffer);
    } catch (e) {
      console.error("[BS Background] Send binary error:", e);
    }
  }

  port.onDisconnect.addListener(() => {
    cleanup();
  });

  port.onMessage.addListener((msg) => {
    if (!msg) return;

    if (msg.action === "CONNECT") {
      const url = msg.url || "wss://localhost:8765/ws";
      try {
        ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
          if (!isClosed) port.postMessage({ type: "connected" });
        };

        ws.onmessage = (event) => {
          if (isClosed) return;
          if (typeof event.data === "string") {
            try {
              const parsed = JSON.parse(event.data);
              port.postMessage({ type: "ws_json", data: parsed });
            } catch (e) {
              port.postMessage({ type: "ws_json_raw", data: event.data });
            }
          } else {
            // P3.1: audio TTS có thể tới dưới dạng binary frame (không base64).
            port.postMessage({ type: "ws_binary", data: event.data });
          }
        };

        ws.onerror = (err) => {
          if (!isClosed) {
            console.error("[BS Background] WS error:", err);
            port.postMessage({
              type: "error",
              error: "Lỗi kết nối tới " + url + ". Vui lòng kiểm tra backend đang chạy."
            });
          }
        };

        ws.onclose = (event) => {
          if (!isClosed) {
            port.postMessage({ type: "disconnected", code: event.code, reason: event.reason });
          }
        };
      } catch (e) {
        if (!isClosed) {
          port.postMessage({ type: "error", error: e.message });
        }
      }
    } else if (msg.action === "SEND_JSON") {
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify(msg.data));
        } catch (e) {
          console.error("[BS Background] Send JSON error:", e);
        }
      }
    } else if (msg.action === "SEND_BINARY") {
      const buffer = (typeof buildBinaryAudioPacket === "function")
        ? buildBinaryAudioPacket(msg.header, msg.pcmBuffer, textEncoder)
        : (() => {
            const headerBytes = textEncoder.encode(JSON.stringify(msg.header));
            const pcmBytes = new Uint8Array(msg.pcmBuffer);
            const totalSize = 4 + headerBytes.length + pcmBytes.byteLength;
            const buf = new ArrayBuffer(totalSize);
            const view = new DataView(buf);
            view.setUint32(0, headerBytes.length, true);
            new Uint8Array(buf, 4, headerBytes.length).set(headerBytes);
            new Uint8Array(buf, 4 + headerBytes.length).set(pcmBytes);
            return buf;
          })();

      sendOrQueue(buffer);
    } else if (msg.action === "FLUSH_PENDING") {
      // Được gọi khi socket đã rút hết hàng đợi.
      if (pendingAudio && ws && ws.readyState === WebSocket.OPEN
          && (ws.bufferedAmount || 0) < SEND_BUFFER_SOFT_LIMIT) {
        const buf = pendingAudio;
        pendingAudio = null;
        try { ws.send(buf); } catch (e) {}
        notifyBackpressure("ok");
      }
    } else if (msg.action === "DISCONNECT") {
      cleanup();
    }
  });
});

// Broadcast subtitle events from capturing iframe to Top frame / all frames
api.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.action === "BROADCAST_SUBTITLE" && sender.tab?.id) {
    api.tabs.sendMessage(sender.tab.id, {
      action: "SUBTITLE_RENDER",
      eventType: msg.eventType,
      payload: msg.payload
    }).catch(() => {});
  }
});

console.log("[BS Background] Service worker & WebSocket bridge ready");
