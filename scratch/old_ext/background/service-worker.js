// Background Service Worker
// Provides a WebSocket Bridge so content scripts inside cross-origin iframes
// can reliably connect to the backend (wss://localhost:8765/ws) without being
// blocked by iframe CSP, sandbox, or Private Network Access restrictions.

const api = typeof browser !== "undefined" ? browser : chrome;

api.runtime.onConnect.addListener((port) => {
  if (port.name !== "bs-ws-bridge") return;

  let ws = null;
  let isClosed = false;

  function cleanup() {
    isClosed = true;
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
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          const header = JSON.stringify(msg.header);
          const headerBytes = new TextEncoder().encode(header);
          const pcmBytes = new Uint8Array(msg.pcmBuffer);
          const totalSize = 4 + headerBytes.length + pcmBytes.byteLength;
          const buffer = new ArrayBuffer(totalSize);
          const view = new DataView(buffer);
          view.setUint32(0, headerBytes.length, true);
          new Uint8Array(buffer, 4, headerBytes.length).set(headerBytes);
          new Uint8Array(buffer, 4 + headerBytes.length).set(pcmBytes);

          ws.send(buffer);
        } catch (e) {
          console.error("[BS Background] Send binary error:", e);
        }
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
