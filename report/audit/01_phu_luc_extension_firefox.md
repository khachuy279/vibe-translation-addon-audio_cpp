# PHỤ LỤC 01 — AUDIT CHI TIẾT `extension_firefox/**`

> Phụ lục của `report/audit/00_BAO_CAO_AUDIT_HIEU_NANG.md`
> Phạm vi đọc **toàn bộ**: `manifest.json`, `background/service-worker.js`, `content/content-script.js`, `content/overlay-manager.js`, `lib/audio-capture.js`, `lib/audio-processor.js`, `lib/frame-builder.js`, `lib/ws-client.js`, `lib/subtitle-renderer.js`, `lib/tts-player.js`, `popup/popup.js`
> Đối chiếu consumer phía backend: `backend/ws/protocol.py`, `backend/vad/processor.py`
> **Read-only. Không sửa file nào.**

---

## 0. Tóm tắt

Các quyết định về *hình dạng dữ liệu* phần lớn đúng (binary framing thay vì base64 trên đường gửi, transferable buffer trong worklet, cập nhật subtitle tại chỗ, các store có bound). Vấn đề nằm ở **NƠI công việc được thực thi**, không phải ở cách bố trí byte:

1. AudioWorklet mà thiết kế giả định **không tồn tại ở runtime** — capture thực tế là `ScriptProcessorNode` (deprecated) chạy trên **main thread**.
2. Mọi thứ đắt đỏ — decode base64 WAV, parse JSON payload lớn, DOM, quét video toàn tài liệu — chạy trên **cùng main thread đó**.
3. **Không có flow control ở bất kỳ hop nào** (content script → port → service worker → WebSocket).

### Bảng xếp hạng

| # | Mức độ | Phát hiện | Bằng chứng | Tác động |
| --- | --- | --- | --- | --- |
| 1 | **CRITICAL** | AudioWorklet là dead code; capture thật là `ScriptProcessorNode(4096)` trên main thread | `audio-capture.js:102-103`; `audio-processor.js` không được tham chiếu (chỉ `manifest.json:73`) | +170 ms (48kHz) đến +512 ms (16kHz) latency capture; main-thread jank ⇒ audio glitch/dropout |
| 2 | **CRITICAL** | WAV TTS decode bằng `atob` + vòng lặp per-char + `new Audio()` mỗi câu, trên main thread | `tts-player.js:83-92, 107` | Hàng trăm KB công việc JS char-by-char chặn thread nuôi audio; decoder mới mỗi câu |
| 3 | **HIGH** | Không backpressure; `bufferedAmount` không được đọc lần nào | `ws-client.js:226-250`; `service-worker.js:88-109`; 0 kết quả grep cho `bufferedAmount` | Queue tăng vô hạn + trôi latency khi backend nghẽn |
| 4 | **HIGH** | 3 bản copy mỗi byte PCM; `port.postMessage` không truyền transfer list | `ws-client.js:228-232`; `service-worker.js:91-105`; `frame-builder.js:39-40` | CPU + GC mỗi frame ở 15.6 frame/s |
| 5 | **HIGH** | `findVideo()` quét toàn tài liệu (`querySelectorAll("*")`) mỗi sự kiện mỗi frame; kết quả âm không được cache | `content-script.js:42-43, 76, 311, 326` | CPU ∝ kích thước DOM × số frame × sự kiện/s |
| 6 | **HIGH** | Queue TTS không bound, phát tuần tự, không drop theo timestamp | `tts-player.js:77, 94-144` | Memory growth + desync TTS/video vĩnh viễn |
| 7 | **HIGH** | TTS phát **2 lần** khi player nằm trong iframe (frame capture **và** frame top) | `content-script.js:61, 184-198`; `service-worker.js:117-125` | Gấp đôi decode/CPU + echo nghe được |
| 8 | **HIGH** | `AudioContext` 16kHz route ra `destination` ⇒ band-limit audio video của người dùng xuống 8kHz | `audio-capture.js:38-42, 65-68, 76-78` | Suy giảm chất lượng nghe được `[SUSPECTED]` (routing đã VERIFIED) |
| 9 | **MEDIUM** | `setTimeout` reconnect không bao giờ bị huỷ ⇒ port + WS zombie sau khi stop | `ws-client.js:160-174` vs `141-158` | Rò 1 connection sống mỗi lần stop-sau-khi-drop |
| 10 | **MEDIUM** | Timestamp chunk trôi 21–64 ms do residual carryover | `audio-capture.js:132-133, 146, 157` | Sai timing phụ đề; backend dùng giá trị này (`backend/vad/processor.py:162`) |
| 11 | **MEDIUM** | `MutationObserver(document.body, {subtree:true})` sống suốt phiên capture | `overlay-manager.js:68-78` | Chi phí callback ∝ DOM churn của trang |
| 12 | **MEDIUM** | Transition animate layout (`max-height`/`margin`/`padding`), `bs-pulse` vô hạn, `text-shadow` blur gấp 3 | `overlay-manager.js:424, 445, 453-464, 385, 473, 485` | Relayout+repaint mỗi lần evict; re-raster text blur mỗi preview |
| 13 | **MEDIUM** | `SubtitleRenderer` không có `destroy()`; timer lifecycle/eviction sống lâu hơn overlay | `subtitle-renderer.js:105-141, 470`; `overlay-manager.js:221` | Timer lãng phí, giữ renderer + DOM detached |
| 14 | **MEDIUM** | `AudioContext` của video vẫn connected thì không bao giờ được đóng (1 context / video element) | `audio-capture.js:36-49, 212-221` | Nhiều context 16kHz sống đồng thời |
| 15 | **MEDIUM** | Mỗi sự kiện được render 2 lần ở frame gốc (local + broadcast echo) | `content-script.js:184-198`; `service-worker.js:117-125` | N× công render mỗi sự kiện |
| 16 | **LOW** | `JSON.stringify` + `TextEncoder.encode` + object literal mỗi frame cho header gần như không đổi | `ws-client.js:214-224`; `frame-builder.js:14-15` | GC nhỏ, 15.6/s |
| 17 | **LOW** | `querySelector` bên trong vòng lặp history thay vì dùng children đã có | `subtitle-renderer.js:487-496` | Bị chặn bởi `maxLines` (nhỏ) |
| 18 | **LOW** | `console.log` cả payload mỗi câu dịch | `content-script.js:202-203` | Console Firefox giữ object graph khi mở |
| 19 | **LOW** | `ws_binary` / `ws_json_raw` gửi về client không có handler | `service-worker.js:53, 56` vs `ws-client.js:40-75` | Mất dữ liệu âm thầm, không tốn perf |
| 20 | **LOW** | Không set `latencyHint`; cấp phát nhỏ mỗi callback | `audio-capture.js:65, 76, 120, 139, 170` | Jitter nhỏ |

### Ngân sách latency phía client (trước mọi inference của backend)

| Chặng | Context 48kHz `[SUSPECTED bội số]` | Nếu 16kHz được tôn trọng |
| --- | --- | --- |
| `ScriptProcessorNode(4096)` fill + buffer nội bộ Firefox | ~85 ms (+~85 ms) = **~170 ms** | 256 ms (+256) = **~512 ms** |
| Lượng tử hoá frame: `chunkSize = 1024` (`audio-capture.js:15`) | 0–64 ms | 0–64 ms |
| Clone qua port + SW rebuild packet + `ws.send` | ~1–5 ms | như trên |
| **Tổng client** | **~175–240 ms** | **~515–580 ms** |

→ Hơn một nửa ngân sách 1 giây bị tiêu **trước khi audio rời khỏi browser** trong trường hợp xấu nhất thực tế.

---

## 1. ĐƯỜNG CAPTURE AUDIO

### 1.1 CRITICAL — AudioWorklet không bao giờ được load; capture chạy trên main thread `[VERIFIED]`

`lib/audio-processor.js` có đăng ký worklet processor (`:38`):
```js
registerProcessor("audio-capture-processor", AudioCaptureProcessor);
```
Grep toàn repo cho `addModule|AudioWorkletNode|audio-capture-processor` chỉ trả **2 kết quả**: entry web-accessible-resources trong manifest và chính lệnh `registerProcessor`. **Không có `audioWorklet.addModule(...)` và không có `new AudioWorkletNode(...)` ở đâu cả.**

Đường capture thực tế — `lib/audio-capture.js:101-103`:
```js
_setupScriptProcessor() {
    const bufferSize = 4096;
    this.processorNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
```
`onaudioprocess` (`:107`) chạy trên **main thread**. Mọi tuyên bố "runs in a dedicated audio thread, avoiding main thread jank" (`audio-processor.js:2`) là **không đúng ở runtime**. Và mọi chi phí main-thread khác trong báo cáo này (decode base64, parse JSON TTS lớn, DOM subtitle, quét `findVideo`) **trực tiếp tranh chấp** với việc giao audio đúng hạn. Đây là phát hiện kiến trúc gốc.

### 1.2 CRITICAL/HIGH — `bufferSize` tính theo sample của context, nên 4096 không phải 85 ms

`audio-capture.js:14-15` ghi ý định:
```js
// 1024 samples @ 16kHz = 64ms chunk (divides 4096 buffer exactly by 4, 0 lost samples)
this.chunkSize = 1024;
```
`bufferSize = 4096` tính theo **rate của context**:
* Context 48kHz: 4096 sample = **85.3 ms**/callback, cộng double-buffering ⇒ **~170 ms** `[SUSPECTED]`
* Context 16kHz (rate được yêu cầu): 4096 sample = **256 ms**/callback ⇒ **~512 ms** `[SUSPECTED]`

Thêm nữa, vì đúng 4 chunk được sinh ra mỗi callback (`4096/1024`), frame được gửi theo **burst 4 lần liên tiếp** cách nhau 85–256 ms, **không phải nhịp đều 64 ms**. Lượng tử hoá đó tạo 0–64 ms jitter, và làm cho nhịp "~3 preview/giây" của backend đến không đều. Buffer worklet 512–1024 sample (hoặc tối thiểu `bufferSize = 1024/2048`) sẽ loại bỏ phần lớn vấn đề này.

### 1.3 HIGH — Context 16kHz bị ép buộc resample cả graph playback của người dùng

```js
// audio-capture.js:37-42
const AudioCtx = window.AudioContext || window.webkitAudioContext;
target.__bsAudioCtx = new AudioCtx({ sampleRate: this.sampleRate });
try {
  target.__bsSourceNode = target.__bsAudioCtx.createMediaElementSource(target);
  // Route to speakers so the user still hears the video
  target.__bsSourceNode.connect(target.__bsAudioCtx.destination);
```
Cùng pattern ở `:64-68` (đường `captureStream`) và `:75-78` (đường MediaStream). Hai hệ quả:
* Graph capture được tạo cùng rate với graph playback, và media element được route vào rồi ra (`:42`). Nếu 16kHz được tôn trọng, audio người dùng nghe bị resample 48k → 16k → rate phần cứng, tức **band-limit xuống 8 kHz** với một resampler nằm trên đường nghe. `[SUSPECTED về thính giác, VERIFIED về routing]`
* Chính code tác giả thừa nhận rate yêu cầu có thể không được đáp ứng — `:105` đọc `this.audioContext.sampleRate` và `:92` log giá trị đó.

Hình dạng đúng: **một** context ở rate thiết bị, việc chuyển đổi sang 16kHz làm **một lần, off-main-thread** (trong worklet), thay vì ép graph *output* về 16kHz.

### 1.4 MEDIUM — Resampler JS là vòng lặp thông dịch per-sample + cấp phát mỗi callback

`audio-capture.js:117-133`:
```js
const ratio = actualSampleRate / 16000.0;
let srcIdx = this.resamplePhase || 0.0;
const maxOut = Math.floor((inputBuffer.length - srcIdx) / ratio) + 2;
const outArray = new Float32Array(Math.max(0, maxOut));
let outCount = 0;
while (srcIdx < inputBuffer.length) {
  const i = Math.floor(srcIdx);
  const frac = srcIdx - i;
  const s0 = inputBuffer[i];
  const s1 = (i + 1 < inputBuffer.length) ? inputBuffer[i + 1] : s0;
  outArray[outCount++] = s0 + frac * (s1 - s0);
  srcIdx += ratio;
}
```
Nhánh này chỉ chạy khi `actualSampleRate !== 16000` (`:113`). Chi phí: **4096 vòng lặp thông dịch mỗi callback (~48k vòng/giây ở 48kHz)** với `Math.floor` và truy cập typed array có bounds-check, cộng một `Float32Array` mới (`:120`). Nội suy tuyến tính **không có anti-alias pre-filter** cũng có nghĩa front-end ASR nhận năng lượng aliased — đây là vấn đề độ chính xác model chứ không phải latency, nhưng đáng ghi nhận.

Xử lý `resamplePhase` đúng về nguyên tắc (carryover phân số, `:132`), phase được reset trong `stop()` (`:188`).

### 1.5 MEDIUM — Trôi timestamp: sample residual được đóng dấu bằng thời gian của callback *hiện tại*

```js
// audio-capture.js:136-146
// Concatenate residual carryover samples from previous cycle to guarantee ZERO lost audio
let combined;
if (this.residualSamples && this.residualSamples.length > 0) {
  combined = new Float32Array(this.residualSamples.length + float16k.length);
  combined.set(this.residualSamples, 0);
  combined.set(float16k, this.residualSamples.length);
} else {
  combined = float16k;
}
const baseTimestamp = this.audioContext ? this.audioContext.currentTime : 0;
```
```js
// audio-capture.js:157
const chunkTime = baseTimestamp + (offset / 16000.0);
```
`combined[0]` là audio residual **cũ nhất**, được capture **trước** `baseTimestamp`, nhưng bị đóng dấu `baseTimestamp + 0`. Sai số bằng `residualSamples.length / 16000` giây và áp cho mọi chunk trong callback. Với `bufferSize = 4096` ở 48kHz, output mỗi callback là 1365.33 sample @16kHz, nên residual xoay vòng 341 → 682 → 1023 → 340…, tạo **răng cưa trễ hệ thống ~21 ms đến ~64 ms**.

`captureTimestamp` không phải chỉ để trang trí: backend đóng dấu mỗi frame VAD từ nó (`backend/vad/processor.py:162` — `frame_ts = capture_timestamp + (offset / (self.sample_rate * 2.0))`) và nó được parse từ header frame (`backend/ws/protocol.py:39`). Với mục tiêu <1s và phụ đề cần đồng bộ A/V, đây là lỗi thật và sửa được (trừ `residualLen/16000`).

### 1.6 Kiểm kê cấp phát trên thread nuôi audio

Mỗi callback `onaudioprocess` (`audio-capture.js:107-172`), ở 48kHz (≈11.7 callback/s):

| Dòng | Cấp phát | Kích thước |
| --- | --- | --- |
| `:120` | `new Float32Array(maxOut)` (chỉ khi resample) | ~5.5 KB |
| `:139` | `new Float32Array(residual + float16k.length)` | ~5.5 KB |
| `:151` | `new Int16Array(chunkSize)` ×1–2 | 2 KB mỗi cái |
| `:168` | `combined.slice(offset)` (copy residual) | ~1–4 KB |
| `:170` | `new Float32Array(0)` khi không có residual | không đáng kể |

≈14–19 KB mỗi 85 ms ⇒ **~170–230 KB/s rác ngắn hạn**, cộng ~33 KB/s từ packet buffer mỗi frame (§2.2). Không phải vấn đề throughput nhưng là **áp lực GC liên tục trên thread phải phục vụ deadline 4096 sample** — đúng rủi ro mà header module cảnh báo (`audio-processor.js:2`). Vòng lặp chuyển Int16 cũng thông dịch per-sample (`:152-154`), trùng lặp với những gì worklet đã làm ở `audio-processor.js:18-21`.

### 1.7 TỐT — Worklet (không dùng) làm đúng pattern transfer

`audio-processor.js:23-31`:
```js
if (this.bufferIndex >= this.bufferSize) {
  // Send buffer to main thread using transferable buffer
  this.port.postMessage(
    { type: "audio_chunk", buffer: this.buffer.buffer },
    [this.buffer.buffer]
  );
  this.buffer = new Int16Array(this.bufferSize);
  this.bufferIndex = 0;
}
```
Đây là hand-off zero-copy đúng: buffer được **transfer** chứ không clone, và việc cấp phát lại xảy ra **một lần mỗi 1024 sample** (mỗi 64 ms @16kHz) thay vì mỗi `process()` — vòng lặp tích luỹ input (`:18-32`) ghi vào `this.buffer` đã cấp phát sẵn (`:9`) và **không cấp phát gì mỗi callback**. Nếu chuyển capture sang processor này, đây là hình dạng cần giữ. Lưu ý nó **không có resampler**, nên chỉ đúng với context 16kHz, và nó không kiểm tra `input.length` cho input đa kênh.

### 1.8 LOW — Không có `latencyHint`; processor nối tới `destination` cho cả hai đường

```js
// audio-capture.js:174-175
this.sourceNode.connect(this.processorNode);
this.processorNode.connect(this.audioContext.destination);
```
Không có `latencyHint` ở bất kỳ `new AudioCtx(...)` nào (`:38, :65, :76`); Firefox mặc định `"interactive"` nên đây là thiếu sót nhỏ. `latencyHint: "interactive"` + buffer ScriptProcessor nhỏ hơn là một cải thiện latency rẻ.

---

## 2. FRAME BUILDING + GỬI WS

### 2.1 Cách một frame được lắp ráp `[VERIFIED]`

```js
// content-script.js:212-216
audioCapture.onChunk = (pcmBuffer, timestamp, chunkIdx) => {
  if (wsClient && wsClient.isConnected) {
    wsClient.sendBinary(pcmBuffer, timestamp, chunkIdx);
  }
};
```
```js
// ws-client.js:226-234
if (this.port) {
  // Send via port bridge
  this.port.postMessage({ action: "SEND_BINARY", header, pcmBuffer: rawBuffer });
  return;
}
```
```js
// service-worker.js:91-105
const buffer = (typeof buildBinaryAudioPacket === "function")
  ? buildBinaryAudioPacket(msg.header, msg.pcmBuffer, textEncoder)
  : (() => { ... })();
ws.send(buffer);
```
```js
// frame-builder.js:33-42
const totalSize = 4 + headerBytes.length + pcmBytes.byteLength;
const buffer = new ArrayBuffer(totalSize);
const view = new DataView(buffer);
view.setUint32(0, headerBytes.length, true);
new Uint8Array(buffer, 4, headerBytes.length).set(headerBytes);
new Uint8Array(buffer, 4 + headerBytes.length).set(pcmBytes);
return buffer;
```
Định dạng wire **tốt**: binary framing, không base64 trên đường gửi, 4-byte length prefix + JSON header + PCM16 thô (`:33-40`), backend parse đúng định dạng này (`backend/ws/protocol.py:30-41`). **Không có overhead JSON/base64 trên đường gửi audio** — JSON duy nhất là header ~190 byte.

### 2.2 MEDIUM — Ba bản copy mỗi byte PCM, và port call không transfer

`ws-client.js:9` chọn bridge khi `chrome.runtime.connect` tồn tại → trong content script Firefox namespace `chrome` có mặt, nên **bridge luôn được dùng**; `_connectDirect()` (`:96-139`) và `_buildPacketFallback` (`:253-264`) là dead path.

Số bản copy mỗi frame:
1. `port.postMessage({action:"SEND_BINARY", ..., pcmBuffer: rawBuffer})` — **không có transfer list** (`ws-client.js:228-232`) ⇒ structured-clone copy `ArrayBuffer` 2 KB qua ranh giới process. So sánh: worklet không dùng thì **có** truyền transfer list (`audio-processor.js:25-28`).
2. `buildBinaryAudioPacket` ⇒ `new Uint8Array(...).set(pcmBytes)` copy vào packet (`frame-builder.js:40`).
3. `ws.send(buffer)` — engine copy nội bộ vào socket buffer.

⇒ ~2 bản copy JS tường minh + 1 transport copy mỗi frame 2 KB, 15.6 frame/s. Hai cách giảm: (a) truyền `[rawBuffer]` làm transfer list (buffer mới cấp phát mỗi chunk, content script không dùng lại) `[SUSPECTED: hỗ trợ transferable của Firefox Port.postMessage]`; (b) nếu không, lắp packet ngay trong content script (`lib/frame-builder.js` đã được inject ở đó, `manifest.json:45`) rồi gửi `ArrayBuffer` hoàn chỉnh cho SW để chỉ `ws.send`, loại bỏ copy #2 và cả `JSON.stringify`/`encode` phía SW.

### 2.3 HIGH — Không có backpressure; queue tăng vô hạn

Grep `bufferedAmount` toàn extension: **0 kết quả**. `sendBinary` chỉ kiểm tra liveness:
```js
// ws-client.js:190-191
sendBinary(pcmData, captureTimestamp, chunkIndex, isPreSpeech = false) {
  if (!this.isConnected || !pcmData) return;
```
```js
// service-worker.js:88-89
} else if (msg.action === "SEND_BINARY") {
  if (ws && ws.readyState === WebSocket.OPEN) {
```
Hệ quả: nếu backend nghẽn (ASR/TTS spike, GC, disk chậm), client vẫn sản xuất 32 KB/s audio realtime (16kHz × 2 B) vào (i) port message queue và (ii) WebSocket send buffer **không bound, không drop policy, không đối soát `chunkIndex`**. Sau 30 s nghẽn là ~1 MB audio cũ trong queue và quan trọng hơn: **latency không bao giờ hồi phục** — audio đang được transcribe là audio của 30 s trước trong khi client vẫn đóng dấu timestamp gốc. Backend che một phần triệu chứng bằng cách cắt raw buffer về 3 s (`backend/vad/processor.py:149-154`, `max_buffer_bytes = int(self.sample_rate * 2 * 3.0)`), nên triệu chứng quan sát được là **mất audio âm thầm + trôi**, chứ không phải RAM server phình. Ngưỡng `ws.bufferedAmount` (drop cũ nhất / dừng capture / hiện trạng thái "lagging") là vòng điều khiển còn thiếu.

### 2.4 LOW/MEDIUM — Bridge biến service worker thành phễu đơn luồng cho mọi tab

Mọi frame audio của mọi tab và mọi frame đều đi qua một event loop của service worker (`service-worker.js:33-113`). Ở 32 KB/s/tab thì rẻ với vài tab, nhưng đây là **điểm serialize chia sẻ** (và là nơi duy nhất có thể đặt chính sách backpressure toàn cục). Bất đối xứng đáng lưu ý: client chỉ xử lý `connected`/`ws_json`/`disconnected`/`error` (`ws-client.js:40-75`), trong khi SW có thể gửi `ws_binary` và `ws_json_raw` (`service-worker.js:53, 56`) — **những message này bị bỏ âm thầm**.

### 2.5 TỐT — Không log trong hot path gửi, encoder chia sẻ

`sendBinary`/`SEND_BINARY` **không có** console call; chỉ log khi lỗi (`ws-client.js:248, 276`; `service-worker.js:85, 107`). Một `TextEncoder` được chia sẻ (`frame-builder.js:3`, `service-worker.js:7`). Log per-chunk duy nhất được throttle 1/200 chunk (`audio-capture.js:159-161`).

---

## 3. ĐƯỜNG PREVIEW / RENDER

### 3.1 LOW (ở 3/s) / MEDIUM (khi burst) — Không coalesce bằng `requestAnimationFrame`

Grep `requestAnimationFrame` toàn extension: **0 kết quả**. Mỗi message preview đồng bộ đi qua cả 3 layer:
```js
// subtitle-renderer.js:440-444
_renderAll() {
  this._renderHistoryLayer();
  this._renderFocusLayer();
  this._renderLiveLayer();
}
```
`onUtteranceUpdate` gọi nó ở mọi nhánh (`:271, 278, 295`, và `:237`), và timer auto-clear được **re-arm mỗi message**:
```js
// subtitle-renderer.js:250
this._resetAutoClearTimer();
```
`_resetAutoClearTimer` (`:51-61`) làm `clearTimeout` + `setTimeout` **mỗi message preview** (≈3 cấp phát/s). Ở nhịp 3/s đây không phải bottleneck; vấn đề là **hành vi burst** — WS handler (`ws-client.js:52-56`) dispatch mỗi message trong task riêng, nên một burst `utterance_update` + `translation` + `tts_audio` liên tiếp tạo N lần render đồng bộ đầy đủ với N lần invalidate style trong cùng một frame. Một `_renderAll` coalesce bằng rAF (dirty-flag) sẽ gộp miễn phí.

### 3.2 HIGH — `findVideo()` quét tài liệu mỗi sự kiện, kết quả âm không được cache `[VERIFIED]`

```js
// content-script.js:38-44
function getVideo(forceRefresh = false) {
  if (!forceRefresh && cachedVideo && cachedVideo.isConnected && !cachedVideo.ended) {
    return cachedVideo;
  }
  cachedVideo = findVideo();
  return cachedVideo;
}
```
Nếu `findVideo()` trả `null`, `cachedVideo` thành `null` và **lần gọi sau quét lại** — guard chỉ short-circuit khi **có** kết quả. Và `handleSubtitleEvent` gọi nó **trước mọi kiểm tra eligibility**:
```js
// content-script.js:76-85
const video = getVideo();
// 1. If this frame HAS the video element, render overlay directly inside this frame
if (video) { ... }
```
Bản thân phép quét rất đắt (`content-script.js:310-346`):
```js
document.querySelectorAll("video").forEach(v => allVideos.push(v));
...
root.querySelectorAll("video").forEach(v => allVideos.push(v));
root.querySelectorAll("*").forEach(el => {
  if (el.shadowRoot) { collectVideosAndShadowRoots(el.shadowRoot); }
});
...
document.querySelectorAll("iframe").forEach(iframe => { ... });
```
`querySelectorAll("*")` cấp phát NodeList của **mọi element trong tài liệu**, `forEach` với closure gọi `shadowRoot` trên từng cái, và `document.querySelectorAll("video")` được gọi **hai lần** (`:317` và `:325`). Nếu tìm thấy video, vòng lặp xếp hạng còn thêm `getBoundingClientRect()` mỗi video (`:364`) — **forced layout**.

Vì background broadcast tới **mọi frame** (`service-worker.js:117-125`), *mọi* frame của tab chạy phép quét này mỗi sự kiện, kể cả frame không có video và sẽ không bao giờ render gì. Với `all_frames: true` và `match_origin_as_fallback: true` (`manifest.json:42-43`), số frame mỗi trang là lớn. Ước lượng bậc độ lớn: top frame với DOM 10k element × ~5 sự kiện/s ≈ 5 lần quét đầy đủ/s; trên trang 20 frame thì nhân lên. Sửa rẻ: cache kết quả âm với TTL ngắn, và bỏ qua phép quét trừ khi `isCapturing` hoặc frame trước đó đã sở hữu một video.

### 3.3 MEDIUM — Mỗi sự kiện được render 2 lần ở frame gốc

```js
// content-script.js:184-198
function emitSubtitleEvent(eventType, payload) {
  // 1. Render locally if eligible
  handleSubtitleEvent(eventType, payload);
  // 2. Broadcast to other frames (Top frame) only if inside a child iframe
  if (window !== window.top) {
    try {
      api.runtime.sendMessage({ action: "BROADCAST_SUBTITLE", eventType, payload }).catch(() => {});
```
```js
// service-worker.js:117-125
api.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.action === "BROADCAST_SUBTITLE" && sender.tab?.id) {
    api.tabs.sendMessage(sender.tab.id, {
      action: "SUBTITLE_RENDER", eventType: msg.eventType, payload: msg.payload
    }).catch(() => {});
```
`tabs.sendMessage(tabId, ...)` không có `frameId` được gửi tới các frame của tab, nên **frame gửi nhận lại chính sự kiện của nó** và chạy `handleSubtitleEvent` lần thứ hai (`content-script.js:119-121`). Renderer idempotent nên thiệt hại hiển thị bị chặn (fingerprint guard + so sánh `textContent`, §3.6), nhưng **gấp đôi công việc DOM** và **gấp đôi số lần quét `findVideo`** từ §3.2. `[SUSPECTED: ngữ nghĩa delivery all-frames chính xác]`

### 3.4 MEDIUM — Lựa chọn CSS gây layout/repaint theo nhịp render

* Transition animate layout mỗi câu:
  ```css
  /* overlay-manager.js:445 */
  transition: transform 0.25s ease, opacity 0.3s ease, max-height 0.35s ease, margin 0.35s ease, padding 0.35s ease;
  ```
  và khi evict (`:463`): tương tự với `!important`. `max-height`, `margin`, `padding` **không compositable**: mỗi frame animate đều relayout subtree overlay (`.bs-overlay` có `max-height: 45%`, `:320`). Vòng lặp mitigation ở `:479-484` xoá node `.bs-evicting` thừa, giới hạn thiệt hại, nhưng mỗi lần evict vẫn tạo 350 ms relayout.
* Animation chạy vĩnh viễn khi có câu đang chờ:
  ```css
  /* overlay-manager.js:489-498 */
  .bs-translating { ... animation: bs-pulse 1.2s ease-in-out infinite; }
  @keyframes bs-pulse { 0%,100%{opacity:0.3} 50%{opacity:1} }
  ```
  Cũng áp cho `.bs-live-layer .bs-translating` (`:424`). Animation opacity được composite nên chi phí vừa phải, nhưng nó **không bao giờ idle** và tạo compositing layer thường trú.
* `text-shadow` blur gấp 3 trên text đổi ~3 lần/s:
  ```css
  /* overlay-manager.js:385 */
  text-shadow: 0 0 6px #000, 0 0 6px #000, 0 2px 4px #000;
  ```
  lặp ở `:473` (0 0 4px ×2) và `:485` (0 0 5px ×2 + 0 2px 3px). Rasterize shadow blur là một trong những hiệu ứng text đắt nhất; re-raster ở nhịp preview trên vùng rộng tới 80% chiều rộng video (`--bs-sub-width: 80%`, `:312`) là chi phí paint đo được, rõ nhất ở fullscreen.
* Riêng Shadow DOM là lựa chọn **tích cực**: `this.shadow = this.host.attachShadow({ mode: "open" })` (`overlay-manager.js:38`) với **một** `<style>` inject (`:52-54`) cô lập ~200 dòng CSS khỏi trang host. **Giữ nguyên.**

### 3.5 MEDIUM — MutationObserver toàn tài liệu chạy suốt phiên

```js
// overlay-manager.js:68-78
this._domObserver = new MutationObserver(() => {
  if (this.isActive && this.host && !this.host.isConnected) {
    this._attachHost();
  }
});
this._domObserver.observe(document.body || document.documentElement, {
  childList: true,
  subtree: true,
});
```
Callback rẻ (2 phép đọc property) và short-circuit đúng trước mọi `getComputedStyle` — nên đây **không** phải pattern layout-thrash. Nhưng **phạm vi quan sát là toàn bộ subtree tài liệu** suốt thời gian capture đang bật, nên trên trang DOM churn (SPA, chat-heavy) browser ghi nhận và giao mutation batch liên tục. Thu hẹp phạm vi về player container đã chọn (có sẵn từ `_findPlayerContainer`, `:91-129`) sẽ loại bỏ toàn bộ chi phí đó. Lưu ý pattern này **tốt hơn** thứ nó thay thế: không có `setInterval` polling để re-attach ở đâu cả (xem §6).

### 3.6 TỐT — Thiết kế incremental-update của renderer là phần được làm tốt nhất

* History layer short-circuit theo content fingerprint:
  ```js
  // subtitle-renderer.js:453-458
  const fp = histItems.map((s) => `${s.id}:${s.originalText || ""}:${s.translatedText || ""}`).join("|");
  if (fp === this._historyFingerprint) {
    return; // Không có thay đổi, giữ nguyên DOM tránh layout thrashing
  }
  ```
* Focus layer cập nhật tại chỗ thay vì destroy/recreate:
  ```js
  // subtitle-renderer.js:526-538
  const currentEl = this.focusLayer.firstElementChild;
  // Nếu cùng câu đang hiển thị -> cập nhật in-place textContent thay vì destroy/recreate
  if (this.currentFocusId === targetItem.id && currentEl) {
    const origEl = currentEl.querySelector(".bs-original");
    if (origEl && origEl.textContent !== targetItem.originalText) {
      origEl.textContent = targetItem.originalText || "";
  ```
* Live layer chỉ ghi khi thực sự đổi:
  ```js
  // subtitle-renderer.js:591-594
  const originalSpan = liveEl.querySelector(".bs-original");
  if (originalSpan && originalSpan.textContent !== text) {
    originalSpan.textContent = text;
  }
  ```
* Node cũ **được xoá** nên DOM subtitle không phình: eviction dùng animation có bound rồi `child.remove()` (`:470-475`), với trần cứng số node evicting đồng thời (`:479-484`).
* State được bound tường minh: `completedSentences` trim theo `maxTotal` (`:381-383, :427-429`), `utteranceStore` cap 50 với FIFO (`:262-265`), `clear()` reset cả fingerprint (`:671-689`).

### 3.7 LOW — Truy vấn DOM lặp lại

`querySelector` **bên trong** vòng lặp mỗi item (`subtitle-renderer.js:487-496`):
```js
for (const item of histItems) {
  let existingEl = this.historyLayer.querySelector(`[data-sentence-id="${item.id}"]:not(.bs-evicting)`);
  if (existingEl) {
    const origEl = existingEl.querySelector(".bs-original");
    const transEl = existingEl.querySelector(".bs-translated");
```
Cũng có `_applyFadeOutToDOM`/`_removeFadeOutFromDOM` query theo attribute selector mỗi lần gọi (`:91, :99`). Bị chặn bởi `maxLines` (mặc định 2–3, `:13`, `popup.js:109`) nên tác động nhỏ — nhưng `Array.from(this.historyLayer.children)` (`:464`) đã tồn tại như một cơ sở lặp rẻ hơn.

### 3.8 LOW — `replaceChildren` mỗi câu vừa finalize

```js
// subtitle-renderer.js:571-574
const focusEl = isPending
  ? this._createPendingFocusElement(targetItem)
  : this._createSentenceElement(targetItem, "bs-sentence bs-focus-item");
this.focusLayer.replaceChildren(focusEl);
```
Một lần mỗi utterance finalize (~1/s). Nhánh tại chỗ (`:528-566`) đã xử lý trường hợp phổ biến, nên chỉ tốn khi chuyển câu; chấp nhận được, nhưng nó vứt và dựng lại subtree focus (và mọi state trên đó) trong khi history layer dùng pattern tái sử dụng.

### 3.9 LOW — Cấp phát mảng mỗi translation trong timing constraint

```js
// subtitle-renderer.js:80-81
const historySentences = this.completedSentences.slice(0, len - 1);
const maxHistoryExpireAt = Math.max(...historySentences.map((s) => s.expireAt));
```
`slice` + `map` + spread mỗi translation. `historySentences.length ≤ 2` nên không đáng kể; ghi lại vì nó được gọi từ cả `onTranslation` (`:386`) và `_promotePendingToCompleted` (`:432`).

---

## 4. PHÁT LẠI TTS

### 4.1 CRITICAL — WAV base64 được decode thủ công trên main thread

```js
// tts-player.js:83-92
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
```
Đây là pattern "atob + vòng lặp byte thủ công" ở **chỗ tệ nhất**. Ba tầng chi phí, tất cả trên main thread:
1. `atob` tạo một **JS string** trung gian (1 ký tự/byte) — với WAV 480 KB là ~960 KB string (UTF-16).
2. Vòng `for` thực hiện 480k lệnh `charCodeAt` thông dịch + store typed array, thường **hàng chục ms** với payload vài trăm KB, và nó **phải xong trước khi playback bắt đầu**.
3. `new Blob([bytes])` copy lần nữa, và `URL.createObjectURL` đăng ký blob.

Vì đây **cùng main thread** phục vụ `ScriptProcessorNode.onaudioprocess` (`audio-capture.js:107`), việc decode này **trực tiếp tranh chấp deadline capture** (§1.2) — đúng cơ chế sinh ra audio glitch và latency phình khi TTS đang bật. `decodeAudioData` (async, decode off-thread vào `AudioBuffer`) + một `AudioContext` sống lâu là cách sửa chuẩn, và còn làm audio schedulable.

### 4.2 HIGH — `Audio` element mới (⇒ media pipeline + decoder mới) mỗi câu

```js
// tts-player.js:106-137
const audioUrl = this._base64ToBlobUrl(item.audioBase64);
const audio = new Audio(audioUrl);
this.currentAudio = audio;
this.currentBlobUrl = audioUrl;
const onDone = () => { ... };
audio.onended = onDone;
audio.onerror = (e) => { ... };
audio.play().catch((err) => { ... });
```
**Không có `AudioContext` và không có `decodeAudioData` ở đâu trong extension** (grep xác nhận: chỉ `atob`, `createObjectURL`, `revokeObjectURL`). Hệ quả riêng cho codebase này:
* Mỗi utterance tạo một `HTMLMediaElement` và một decoder mới ⇒ latency audio đầu tiên bao gồm dựng element + demux + decode + resolve promise `play()`, tất cả **sau** công việc base64 ở §4.1.
* Playback nằm **ngoài** Web Audio graph dùng cho capture (một `AudioContext` 16kHz khác, `audio-capture.js:38`), nên TTS không thể được schedule theo `audioContext.currentTime` và không chia sẻ ducking graph. Ducking thay vào đó bằng cách gán `volume`:
  ```js
  // tts-player.js:44-48
  if (!this.isDucked) {
    this.originalVideoVolume = this.targetVideo.volume > 0 ? this.targetVideo.volume : 1.0;
  }
  this.targetVideo.volume = Math.max(0, Math.min(1.0, this.originalVideoVolume * this.duckingLevel));
  ```
  Đây không phải hot path (gọi từ `setTargetVideo`/`applySettings`, `:29, :36`), nên là ghi chú thiết kế: nó duck **liên tục** khi `ttsEnabled && autoDucking` (`:41`), không chỉ khi TTS đang phát.
* Vì playback chạy trong pipeline element riêng, nó **không đồng bộ với timeline video** đã capture; `seeked` chỉ flush queue (`content-script.js:176`).

### 4.3 HIGH — Queue TTS không bound và không có drop policy ⇒ desync vĩnh viễn

```js
// tts-player.js:67-81
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
```
`this.queue` **không có trần độ dài**, và `_playNext` chỉ dequeue sau khi element trước kết thúc (`:111-126`). Nếu TTS được sản xuất nhanh hơn tiêu thụ (output voice clone dài, backend chậm, socket nghẽn rồi burst), queue tích luỹ `{audioBase64}` — mỗi item giữ **một chuỗi base64 WAV đầy đủ**. Không có khái niệm "utterance này đã cũ X giây, bỏ đi", nên trong phiên dài phần lồng tiếng **trượt xa dần khỏi video** và không có cơ chế bắt kịp. `playedIds` được bound đúng ở 300 (`:72-75`) — tác giả đã nghĩ tới một trong hai vector tăng trưởng nhưng không tới cái còn lại. Lưu ý dedup chỉ áp dụng khi `item.id` tồn tại (`:69`) — message `tts_audio` không có id **bỏ qua hoàn toàn** dedup (và mỗi frame có player riêng, §4.5).

### 4.4 HIGH — WAV base64 bị clone qua ba ranh giới process trước khi decode

Với `tts_audio`, payload đi: SW parse string WS rồi re-post object (`service-worker.js:50-51`) → client re-emit (`ws-client.js:52-56`) → content script kiểm tra (`content-script.js:62-71`) → nếu trong iframe thì clone **lần nữa** qua `runtime.sendMessage` (`content-script.js:191-195`) và **một lần nữa mỗi frame** qua `tabs.sendMessage` (`service-worker.js:119-123`). Mỗi hop là structured clone của một object chứa **chuỗi base64 đầy đủ**. Với utterance 5 s (~240 KB WAV → ~320 KB base64), đó là vài trăm KB copy 3–4 lần mỗi utterance, cộng các bản copy tạm trong `_base64ToBlobUrl` (string `atob`, `Uint8Array`, `Blob`). Tổng bộ nhớ tạm peak mỗi utterance có thể đạt vài MB, và GC phải thu hồi tất cả ngay sau đó — **trên main thread** (§4.1). Gửi TTS dưới dạng `ArrayBuffer` binary (hoặc fetch theo id qua HTTP) sẽ loại bỏ phần lớn.

### 4.5 HIGH — TTS phát hai lần khi video nằm trong iframe

```js
// content-script.js:59-72
if (eventType === "tts_audio") {
  // Only Top frame or active capturing frame should play audio
  if (window === window.top || isCapturing) {
    const audioB64 = payload?.audio || payload?.audio_base64 || (typeof payload === "string" ? payload : null);
    if (audioB64) {
      console.log("[BS TTS] 🔊 Received synthesized audio chunk:", ...);
      ttsPlayer.enqueue({ id: ..., audioBase64: audioB64, ... });
```
Comment nói "Only Top frame **or** active capturing frame", và điều kiện là `||` — nên khi player là iframe nhúng (YouTube embed, phần lớn site streaming), iframe đang capture enqueue (vì `isCapturing`), và broadcast echo từ `emitSubtitleEvent` (`:189-196`) làm **frame top** enqueue cùng utterance (`window === window.top`). `playedIds` **không thể** ngăn: nó là per-`TTSAudioPlayer` instance (`tts-player.js:15`), và mỗi frame có instance riêng (`content-script.js:30`). Kết quả: **cùng một giọng clone được decode và phát hai lần**, lệch nhau — gấp đôi CPU/decode cộng echo nghe được. Điều kiện có lẽ muốn nói "top frame nếu nó sở hữu video, ngược lại frame đang capture".

### 4.6 TỐT — Vòng đời object-URL và element trong player

```js
// tts-player.js:111-126
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
```
Handler được detach, `src` được clear, object URL được revoke, và `clear()` (`:146-162`) / `destroy()` (`:164-169`) theo cùng kỷ luật và phục hồi volume video. Guard re-entrancy `if (this.currentAudio === audio)` đúng. Chỉ có một field `currentBlobUrl` nghĩa là chỉ URL đang bay được revoke, nhưng item trong queue chỉ giữ string nên không rò URL mồ côi.

---

## 5. RÒ RỈ / VÒNG ĐỜI

### 5.1 HIGH — Timer reconnect còn sống sau `disconnect()`, sinh port + WebSocket zombie

```js
// ws-client.js:160-174
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
```
Handle timeout **không bao giờ được lưu**, nên không thể huỷ:
```js
// ws-client.js:141-158
disconnect() {
  this._stopPing();
  if (this.port) {
    try {
      this.port.postMessage({ action: "DISCONNECT" });
      this.port.disconnect();
    } catch (e) {}
    this.port = null;
  }
  ...
  this.isConnected = false;
}
```
Không có cờ `isDisposed`/`shouldReconnect`. Trình tự: backend drop WS → `_emit("disconnected")` → `_scheduleReconnect()` arm timer với `delay` tới `maxReconnectDelay = 30000` (`:12`) → người dùng bấm Stop → `cleanup()` (`content-script.js:254-273`) đặt `wsClient = null` và gọi `disconnect()`, **không huỷ timer** → tối đa 30 s sau timer bắn, thấy `isConnected === false`, và gọi `this.connect()` trên object đã bị bỏ, mở **`runtime.connect` port mới và WebSocket mới tới backend** không có owner, không có gì người dùng chạm tới được để dừng ping interval, và không có tham chiếu từ content script. Vòng reconnect cũng retry tới 11 lần (`:161`), mỗi lần một timer không huỷ được.

### 5.2 MEDIUM — `SubtitleRenderer` không có `destroy()`; timer sống lâu hơn overlay

Không có `destroy` trong `lib/subtitle-renderer.js` (chỉ có `clear()`, `:671-689`), nhưng tồn tại hai chuỗi timer độc lập:
```js
// subtitle-renderer.js:105-141 (chuỗi lifecycle, 10 ms … 30 s)
this.lifecycleTimer = setTimeout(() => { this._onLifecycleTick(); }, delay);
```
```js
// subtitle-renderer.js:51-61 (chuỗi auto-clear 30 s, re-arm mỗi message preview)
this.autoClearTimer = setTimeout(() => { this.clear(); }, this.autoClearTimeoutMs);
```
```js
// subtitle-renderer.js:470-475 (timer 380 ms mỗi eviction, không được track)
setTimeout(() => { if (child.parentNode) { child.remove(); } }, 380);
```
Và destroy bỏ tham chiếu mà không clear chúng:
```js
// overlay-manager.js:217-226
if (this.host && this.host.parentNode) {
  this.host.parentNode.removeChild(this.host);
}
this.renderer = null;
```
Kết quả: sau Stop, renderer và các mảng của nó vẫn reachable từ timer đang chờ tới 10 s (lifecycle) / 30 s (auto-clear) và tiếp tục chạy `_renderAll()` trên node detached — công việc lãng phí và bộ nhớ bị giữ, chứ không phải bug nhìn thấy (node detached không tốn layout). Timer eviction bắn vào child detached và no-op. Từng cái mức thấp; nhóm lại vì đây là **thiếu contract `destroy()`**, và `clear()` thậm chí không được gọi khi destroy (dù nó tồn tại).

### 5.3 MEDIUM — `AudioContext` của video vẫn connected được cố ý không đóng

```js
// audio-capture.js:211-222
// Clean up AudioContext if it belongs to a disconnected video element (preventing memory leak)
if (this.videoElement && !this.videoElement.isConnected && this.videoElement.__bsAudioCtx) {
  try { this.videoElement.__bsAudioCtx.close(); } catch (e) {}
  delete this.videoElement.__bsAudioCtx;
  delete this.videoElement.__bsSourceNode;
}
// Do NOT close cached target.__bsAudioCtx if video is still active in DOM
if (this.audioContext && (!this.videoElement || this.audioContext !== this.videoElement.__bsAudioCtx)) {
  try { this.audioContext.close(); } catch (e) {}
}
```
Cache context trên `target.__bsAudioCtx` (`:36-38`) là **đúng và cần thiết** — `createMediaElementSource` chỉ được gọi một lần mỗi element — và comment cho thấy trade-off là cố ý. Khía cạnh không bound là context được key **theo video element**: vì extension có thể start/stop trên các video khác nhau (`findVideo` xếp hạng mọi ứng viên, `content-script.js:354-373`), mỗi video element mới tạo thêm một context 16kHz ở lại cho tới khi element rời DOM. Trên trang swap video element (SPA navigation), nhiều `AudioContext` sống tích luỹ, và mỗi cái vẫn route audio của video qua graph Web Audio 16kHz (§1.3) **kể cả sau khi người dùng bấm Stop**. Firefox giới hạn số `AudioContext` đồng thời, nên đây cũng là rủi ro chức năng, không chỉ bộ nhớ.

### 5.4 TỐT — Vệ sinh listener/lifecycle làm thực sự tốt

* Mọi listener scoped theo capture được đăng ký với abort signal và giải phóng bằng `abort()`:
  ```js
  // content-script.js:166-167
  captureAbortController = new AbortController();
  const { signal } = captureAbortController;
  ```
  dùng ở `:176` (seeked), `:234-236` (ba biến thể fullscreenchange), `:238-239` (play/playing), abort ở `:256-259`.
* Listener fullscreen của overlay dùng bound reference và được remove:
  ```js
  // overlay-manager.js:199-210
  _onFullscreenChange() { ... }
  document.removeEventListener("fullscreenchange", this._onFullscreenChange);
  document.removeEventListener("webkitfullscreenchange", this._onFullscreenChange);
  document.removeEventListener("mozfullscreenchange", this._onFullscreenChange);
  ```
  và `MutationObserver` được disconnect (`:212-215`).
* Audio graph được teardown hoàn toàn: `port.onmessage = null`, `onaudioprocess = null`, `disconnect()` cả hai node (`audio-capture.js:191-209`), track stream được stop (`:224-229`).
* Service worker null mọi handler `ws.on*` trước khi close (`service-worker.js:15-27`), ngăn vòng lặp reconnect khi teardown có chủ ý.
* Không tìm thấy collection client-side không bound: `completedSentences` (`:381-383`), `utteranceStore` ≤ 50 (`:262-265`), `playedIds` ≤ 300 (`tts-player.js:72-75`), `listeners` là `Set` keyed theo event với `off()` hoạt động (`ws-client.js:294-304`).
* `disconnect()` suppress `onclose` trước khi close để tránh arm reconnect (`:152`: `this.ws.onclose = null;`).

---

## 6. POLLING

Grep `setInterval|clearInterval` toàn extension trả **đúng một** interval:
```js
// ws-client.js:176-181
_startPing() {
  this._stopPing();
  this.pingInterval = setInterval(() => {
    this.sendJSON({ type: "ping", timestamp: performance.now() });
  }, 30000);
}
```
Keepalive 30 s, được clear đúng khi disconnect (`:183-188`, gọi ở `:142, :59, :79, :118`). **Đây là đúng** — không có polling cho trạng thái kết nối, không polling cho DOM attachment (thay bằng `MutationObserver`, `overlay-manager.js:67`), không polling trạng thái backend trong content script.

Các mục có dạng polling còn lại:
* **Vòng retry tìm video** `[LOW/MEDIUM]` — bốn lần poll 250 ms khi video lazily-created chưa có:
  ```js
  // content-script.js:156-162
  // Retry for up to ~1s if video element is lazily loaded upon play
  for (let i = 0; i < 4; i++) {
    await new Promise(r => setTimeout(r, 250));
    video = findVideo();
    if (video) break;
  }
  ```
  Bound ~1 s và chỉ khi start, nên tác động nhỏ — nhưng mỗi vòng chạy phép quét `findVideo` đầy đủ (§3.2). Extension đã có pattern observer để thay thế.
* **Sàn tick lifecycle 10 ms** `[LOW]` — `const delay = Math.max(10, Math.min(minWaitMs, 30000));` (`subtitle-renderer.js:137`). Bình thường wait tính ra ~500 ms (fade duration) hoặc thời gian hiển thị còn lại, nên **không phải busy loop**; sàn 10 ms chỉ chạm khi deadline đã trong 10 ms, khi đó tick kế giải quyết. Không cần hành động.
* **Timeout fetch của popup không được clear trên nhánh lỗi mạng** `[LOW]`:
  ```js
  // popup.js:241-244
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const res = await fetch(url, { ...fetchOpts, signal: controller.signal });
  clearTimeout(timer);
  ```
  Nếu `fetch` reject (đúng trường hợp code này tồn tại để xử lý — xem `catch` rỗng ở `:254-256`), `clearTimeout` không bao giờ chạy, nên timer và closure của nó sống tới khi bắn (tới 120 s cho engine switch, `:395`/`:475`), rồi gọi `abort()` trên request đã chết. Nhỏ, nhưng là timeout không được clear thật, và xảy ra hai lần mỗi lần gọi `fetchBackend` (vòng fallback scheme, `:235-257`). Debounce slider 150 ms của popup (`:685, :709`) là phù hợp.
* **Nút test popup** `setTimeout(..., 5000)` (`popup.js:677`) sống/chết cùng document popup — ổn.

---

## 7. LOGGING

Grep `console.(log|debug|warn|error|info)` trong `extension_firefox`: **38 kết quả** (37 trong code thật). Phân bố theo mức liên quan hot path:

| Call site | Tần suất | Đánh giá |
| --- | --- | --- |
| `audio-capture.js:159-161` | `idx === 0 \|\| idx % 200 === 0` → 1/200 chunk ≈ **1 mỗi 12.8 s** | **Throttle đúng** — và template literal chỉ được dựng bên trong `if`, nên không tốn string mỗi chunk. Đây là log hot-path quan trọng nhất của extension và nó được xử lý tốt. |
| `content-script.js:202-203` | mỗi `sentence_complete` + mỗi `translation` (~1–3/s) | `console.log("[BS] Translation:", p)` / `console.log("[BS] Sentence:", p)` log **cả object payload**. Trong Firefox, Web Console giữ tham chiếu live tới object đã log khi đang mở, nên trong phiên dài chúng tích luỹ (payload graph gồm cả text utterance đầy đủ). Không per-frame, nhưng là mục **khối lượng log lớn nhất** và là mục duy nhất có hệ quả bộ nhớ khả dĩ. |
| `content-script.js:64` | mỗi `tts_audio` | Chỉ log scalar (`utterance_id`, `text`, `duration_sec`) — **cố ý tránh base64**. Tốt. |
| `content-script.js:379` | 1 lần mỗi frame mỗi page load | Với `all_frames: true` + `match_origin_as_fallback: true` (`manifest.json:42-43`), một log mỗi frame — hàng chục dòng trên trang nhiều iframe khi load. Thuộc hình thức. |
| `audio-capture.js:51, 69, 79, 92` | 1 lần mỗi lần start capture | Ổn. |
| `content-script.js:143, :305` | mỗi lần đổi setting | Ổn (đã debounce 150 ms, `popup.js:709`). |
| `overlay-manager.js:81, 227` | 1 lần mỗi init/destroy | Ổn. |
| `service-worker.js:127` | 1 lần mỗi lần SW start | Ổn (lặp mỗi khi Firefox restart event page — xem §8.3). |
| `ws-client.js:248, 276, 289, 310`; `service-worker.js:62, 85, 107`; `tts-player.js:50, 63, 130, 135, 139`; `audio-capture.js:44, 94` | chỉ nhánh lỗi | Ổn. |

**Không tồn tại console call trong đường per-audio-frame hay per-render.** Wrapper `_emit` chỉ log khi listener ném exception (`ws-client.js:310`). Mục này phần lớn sạch; thay đổi duy nhất đáng làm là hạ cấp hai log whole-payload ở `content-script.js:202-203`.

---

## 8. KIẾN TRÚC / SCALE

### 8.1 Nhiều video trong một trang ⇒ nhiều capture session độc lập

`popup.js:503-538` broadcast start trên **mọi frame**:
```js
// popup.js:509-511
const results = await api.scripting.executeScript({
  target: { tabId: tab.id, allFrames: true },
  func: (act, p) => { ... window.__bsStartCapture(p) ... },
```
Mỗi frame có video chạy `startCapture` (`content-script.js:150-250`), tạo `WSClient` riêng (`:179`), port riêng (`ws-client.js:38`), `AudioCapture` và `AudioContext` riêng (`:211-217`). Hai video trong một tab ⇒ **hai backend session đồng thời, hai luồng audio, hai luồng TTS**. Mỗi session rồi broadcast phụ đề tới **mọi** frame (`content-script.js:189-196`), nên mỗi frame nhận N sự kiện mỗi utterance và mỗi frame có video render overlay cho **cả hai** session (utterance id giữa các session **không được namespace**, nên `completedSentences.find(s => s.id === ...)` ở `subtitle-renderer.js:268` có thể match chéo session). Với TTS, va chạm id giữa các session có thể làm dedup `playedIds` (`tts-player.js:70`) **chặn nhầm** một utterance hợp lệ, còn thiếu id thì gây trùng. `[SUSPECTED cho nhánh va chạm id]`

> **Liên hệ backend:** đây chính là kịch bản kích hoạt F-03 và F-04 trong báo cáo chính (share `_shared_session` + race `unload_shared_model`).

### 8.2 Fan-out all-frames làm chi phí nhân lên

Mỗi sự kiện subtitle được gửi tới mọi frame (`service-worker.js:119-123`), và mọi frame nhận gọi `getVideo()` → `findVideo()` (§3.2) **kể cả khi nó sẽ không render gì**. Vậy chi phí scale theo `frames × sự kiện/s × (kích thước DOM trang)`, không theo số video. Đây là rủi ro scale chính trong một tab; gửi nhắm theo `frameId` (SW có `sender.tab.id`, và có thể học frame sở hữu từ message START) sẽ đưa về O(1).

### 8.3 Service worker đơn lẻ làm phễu mọi tab

`background/service-worker.js:33-113` xử lý mọi port message của mọi tab trên một thread, và MV3 + `"background": { "scripts": [...] }` (`manifest.json:31-36`) nghĩa là Firefox chạy nó như **event page không persistent**. `[SUSPECTED]` Nếu Firefox suspend event page trong capture dài (không có hoạt động port), bridge WebSocket chết; capture audio dừng âm thầm và khả năng phục hồi phụ thuộc hoàn toàn vào vòng reconnect không huỷ được ở §5.1. Ping 30 s (§6) là thứ duy nhất sinh traffic định kỳ để giữ nó sống, và nó bị `_stopPing()` dừng khi disconnect.

### 8.4 Tăng trưởng phiên dài

Đã bound: `playedIds` (300), `utteranceStore` (50), `completedSentences` (maxLines), `listeners`. Không bound: **queue TTS** (§4.3) và **buffer gửi của WebSocket/port** (§2.3). `reconnectAttempts` cap 11 (`ws-client.js:161`). Vậy phiên dài an toàn *trừ khi* TTS bật và consumer tụt lại, hoặc socket nghẽn.

### 8.5 Kiến trúc latency tổng thể

Thiết kế đặt đường capture→send (main thread, ScriptProcessor, resample JS, framing không base64) **cùng thread** với đường receive→render (parse JSON, DOM subtitle, decode base64 WAV). Đường **binary** nhận về không bao giờ được dùng (backend chỉ gửi JSON theo giao thức hiện tại), nên có một cải thiện rẻ: gửi TTS dạng binary qua WS thay vì base64-trong-JSON, loại bỏ cùng lúc độ phình base64, ba structured clone (§4.4) và decode thủ công (§4.1).

---

## 9. CÔNG VIỆC TRÙNG LẶP

| Trùng lặp | Bằng chứng | Chi phí |
| --- | --- | --- |
| Cùng sự kiện render 2 lần ở frame gốc (render local + broadcast echo về mọi frame) | `content-script.js:184-198` + `service-worker.js:117-125` | 2× render + 2× `findVideo` mỗi sự kiện |
| `document.querySelectorAll("video")` gọi hai lần mỗi `findVideo` (quét trực tiếp `:317`, lại trong `collectVideosAndShadowRoots` `:325`) | `content-script.js:317, 325` | cấp phát NodeList trùng mỗi lần quét |
| `findVideo` chạy lại mỗi sự kiện vì kết quả `null` không được cache | `content-script.js:42-43` + `:311` | mục CPU chiếm ưu thế trên frame không có video |
| Payload unwrap hai lần cho translation (`overlay-manager.js:290-293` dựng `(sentenceId, text, status)`, rồi `subtitle-renderer.js:305-309` xử lý lại dạng object) | `overlay-manager.js:288-294`; `subtitle-renderer.js:300-309` | nhỏ, nhưng nhánh double-unwrap là dead weight |
| Header serialize lại mỗi frame dù chỉ 2 field đổi | `ws-client.js:214-224` (`JSON.stringify`) + `frame-builder.js:14-15` (`enc.encode`) + object literal mới mỗi frame | ~15.6 cấp phát string+Uint8Array/s; có thể là prefix tĩnh |
| `parseInt` / chuỗi ternary được tính hai lần trong cùng biểu thức | `content-script.js:20`: `cfg.minWordsToCommit !== undefined && !isNaN(parseInt(cfg.minWordsToCommit, 10)) ? ... parseInt(cfg.minWordsToCommit, 10) : 2` — và chuỗi `vadThreshold` y hệt ở `:17` và `:18` | không đáng kể (chỉ khi đổi setting) |
| `querySelector` mỗi item bên trong vòng lặp history thay vì lặp children đã cache | `subtitle-renderer.js:487-496` (children đã liệt kê ở `:464`) | bound bởi `maxLines` |
| **Cả hai** bản chuyển PCM (float→int16) được implement hai lần: main thread (`audio-capture.js:152-154`) và worklet (`audio-processor.js:18-21`) | — | dead duplication; chỉ một cái sống |
| Không trùng lặp, nhưng đáng lưu ý: đường bridge tính lại nhánh PCM mỗi lần gọi | `ws-client.js:196-207` | Thực tế `pcmData` luôn là `ArrayBuffer` từ `pcmData.buffer` (`audio-capture.js:162`), nên nhánh `slice()` ở `:201`/`:206` **không bao giờ chạy** — dead defensive code trên hot path |

---

## PHỤ LỤC A — Phát hiện SUSPECTED (không xác nhận được ở chế độ read-only)

1. **Cách Firefox xử lý `new AudioContext({ sampleRate: 16000 })`** (`audio-capture.js:38, 65, 76`). Nếu tôn trọng, buffer ScriptProcessor thành 256 ms và audio người dùng bị band-limit (§1.2, §1.3); nếu bỏ qua, resampler JS ở `:117-133` chạy. Chính code thể hiện sự không chắc chắn (`:105` đọc rate thực; `:92` log nó).
2. **Bội số buffering nội bộ của `ScriptProcessorNode` trong Firefox** — các con số ~2× ở §1.2 là điển hình, không phải đã kiểm chứng.
3. **`Port.postMessage` có hỗ trợ transferable trong extension messaging của Firefox hay không** (§2.2). Nếu không, khuyến nghị dự phòng (lắp packet trong content script) vẫn loại bỏ được một bản copy.
4. **Ngữ nghĩa delivery all-frames của `tabs.sendMessage` không có `frameId`** (§3.3, §8.2). Comment của tác giả ở `service-worker.js:116` giả định điều đó; nếu delivery chỉ tới frame đầu thì §3.3 và phần lớn §8.2 giảm nhẹ.
5. **MV3 event-page suspension đóng bridge giữa phiên** (§8.3).
6. **Va chạm `utterance_id` chéo session** gây chặn nhầm TTS dedup (§8.1).
7. **Khả năng nghe thấy của band-limit 8 kHz** — routing đã verified; kết quả cảm nhận phụ thuộc đường playback.

## PHỤ LỤC B — Những gì đã làm tốt (không được regress)

1. **Binary framing trên đường gửi** — 4-byte LE length + JSON header + PCM16 (`frame-builder.js:33-40`), khớp `backend/ws/protocol.py:30-41`. Không phình base64/JSON trên uplink audio.
2. **Zero-copy transfer trong worklet** (`audio-processor.js:23-31`) — cấp phát một lần mỗi 1024 sample, transfer buffer, không cấp phát mỗi `process()`. Đúng pattern; chỉ cần được nối vào.
3. **Log per-chunk được throttle** (`audio-capture.js:159-161`) — 1 log mỗi 12.8 s, và không có log nào khác trong hot path frame/render.
4. **Tính incremental của subtitle renderer** — fingerprint short-circuit (`subtitle-renderer.js:453-458`), cập nhật tại chỗ focus/live (`:526-566`, `:591-594`), eviction có bound kèm xoá DOM (`:470-484`).
5. **Cô lập Shadow DOM** cho overlay với một lần inject style (`overlay-manager.js:38, 52-54`) — không rò style recalc sang trang host.
6. **Vòng đời listener dựa trên `AbortController`** cho mọi listener scoped theo capture (`content-script.js:166-167, 176, 234-239`) cộng teardown audio graph đầy đủ (`audio-capture.js:191-209, 224-229`).
7. **State client-side có bound**: `utteranceStore` ≤ 50 (`subtitle-renderer.js:262-265`), `completedSentences` ≤ maxLines (`:381-383`), `playedIds` ≤ 300 (`tts-player.js:72-75`).
8. **Teardown element/URL TTS đúng** (`tts-player.js:111-126, 146-169`).
9. **Dùng `MutationObserver` thay vì interval polling** để re-attach overlay (`overlay-manager.js:67-78`) và keepalive WS 30 s là interval duy nhất (`ws-client.js:176-181`).
10. **Cache `AudioContext`/`MediaElementSource` theo video element** (`audio-capture.js:36-49`) — workaround cần thiết cho ràng buộc once-per-element, với logic close-only-when-detached có chủ ý.

## PHỤ LỤC C — Thay đổi giá trị cao nhất, theo thứ tự

1. Load và dùng `lib/audio-processor.js` qua `audioWorklet.addModule`, với bước resample 16kHz **bên trong worklet** (hoặc context 48kHz và worklet làm chuyển đổi), rồi xoá `_setupScriptProcessor` (`audio-capture.js:101-176`). Giữ pattern transfer của worklet (`audio-processor.js:25-28`).
2. Thay `atob`+loop+`new Audio()` bằng `decodeAudioData` trên **một** `AudioContext` sống lâu và `AudioBufferSourceNode` được schedule (`tts-player.js:83-137`); gửi TTS dạng **binary** thay vì base64 để triệt ba lần clone mỗi utterance (`service-worker.js:46-58`, `content-script.js:184-198`).
3. Thêm backpressure: đọc `ws.bufferedAmount` ở `service-worker.js:88-109`, drop hoặc halt theo ngưỡng, và công bố trạng thái cho content script; cap `tts-player.js:77` và drop utterance cũ theo timestamp.
4. Sửa §5.1 — lưu handle timeout reconnect và clear nó trong `disconnect()`.
5. Cache kết quả âm của `findVideo` (`content-script.js:42-43`) và ngừng gửi sự kiện subtitle tới frame không thể render (`service-worker.js:119-123`).
6. Sửa trôi timestamp (§1.5) bằng cách trừ `residualSamples.length / 16000` khỏi `chunkTime` (`audio-capture.js:157`).
7. Sửa `||` trong kiểm tra ownership TTS (`content-script.js:61`) để chỉ một frame phát.
8. Thu hẹp phạm vi `MutationObserver` (`overlay-manager.js:74-77`); thêm `destroy()` cho `SubtitleRenderer` và gọi từ `OverlayManager.destroy` (`overlay-manager.js:221`).
9. Chuyển animation eviction khỏi `max-height`/`margin`/`padding` (`overlay-manager.js:445, 463`) sang chỉ `transform`/`opacity`.
10. Hạ cấp log whole-payload ở `content-script.js:202-203`.

---

*Phụ lục read-only. Không file nào được tạo, sửa hay xoá trong `extension_firefox/`.*
