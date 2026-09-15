// Kiểm thử CHỨC NĂNG cho AudioWorklet của extension (P3.4) — chạy bằng Node, không cần browser.
//
// Vì sao: `lib/audio-processor.js` là đường capture chính, nhưng trước đây chỉ được kiểm tra
// ở mức "có chuỗi này trong file". Script này dựng một AudioWorkletGlobalScope giả
// (sampleRate/currentTime/AudioWorkletProcessor/registerProcessor) rồi CHẠY THẬT `process()`
// trên tín hiệu sine đã biết, để kiểm chứng:
//   1. Đăng ký đúng tên processor.
//   2. Resample 48 kHz -> 16 kHz đúng tỉ lệ (tần số sine giữ nguyên 440 Hz).
//   3. Giữ pha giữa các block (không mất mẫu, không vấp ở biên block) — đúng thiết kế "phase".
//   4. Gom đúng chunkSize mẫu/chunk, timestamp tăng đều 64 ms và không trôi.
//   5. Ping/pong vẫn hoạt động (đường watchdog của main thread).
//
// Dùng: node backend/tests/js/worklet_harness.js  (exit 0 = PASS)

const fs = require("fs");
const path = require("path");

// Tìm gốc dự án bằng cách đi lên cho tới khi thấy extension_firefox/lib.
function findWorklet() {
  let dir = __dirname;
  for (let i = 0; i < 6; i++) {
    const candidate = path.join(dir, "extension_firefox", "lib", "audio-processor.js");
    if (fs.existsSync(candidate)) return candidate;
    dir = path.dirname(dir);
  }
  throw new Error("không tìm thấy extension_firefox/lib/audio-processor.js");
}

const WORKLET = process.argv[2] || findWorklet();

const failures = [];
function check(cond, msg, extra) {
  if (cond) {
    console.log("  PASS " + msg);
  } else {
    console.log("  FAIL " + msg + (extra !== undefined ? `  (${extra})` : ""));
    failures.push(msg);
  }
}

// ---------------------------------------------------------------- scope giả
const SAMPLE_RATE = 48000;
const TARGET_RATE = 16000;
const CHUNK = 1024;
const QUANTUM = 128;

const scope = {
  sampleRate: SAMPLE_RATE,
  currentTime: 0,
  registered: {},
  messages: [],
};

class FakePort {
  constructor() {
    this.onmessage = null;
    this._owner = null;
  }
  postMessage(payload, transfer) {
    scope.messages.push({ payload, transfer: transfer || [] });
  }
}

class AudioWorkletProcessor {
  constructor() {
    this.port = new FakePort();
  }
}

scope.AudioWorkletProcessor = AudioWorkletProcessor;
scope.registerProcessor = (name, cls) => {
  scope.registered[name] = cls;
};

const src = fs.readFileSync(WORKLET, "utf8");
const vm = require("vm");
const context = vm.createContext(scope);
vm.runInContext(src, context, { filename: WORKLET });

console.log(`Worklet: ${path.relative(process.cwd(), WORKLET)}`);
check(!!scope.registered["audio-capture-processor"], "đăng ký processor 'audio-capture-processor'");

const Proc = scope.registered["audio-capture-processor"];
const proc = new Proc({ processorOptions: { targetSampleRate: TARGET_RATE, chunkSize: CHUNK } });

// ---------------------------------------------------------------- chạy 100 quantum sine 440 Hz
const freq = 440;
const quanta = 100;
let phase = 0;
for (let q = 0; q < quanta; q++) {
  const block = new Float32Array(QUANTUM);
  for (let i = 0; i < QUANTUM; i++) {
    block[i] = 0.8 * Math.sin(2 * Math.PI * freq * phase / SAMPLE_RATE);
    phase++;
  }
  scope.currentTime += QUANTUM / SAMPLE_RATE;
  proc.process([[block]], [[new Float32Array(QUANTUM)]], {});
}

const audioMsgs = scope.messages.filter((m) => m.payload.type === "audio_chunk");
const inSamples = quanta * QUANTUM;
const expectedOut = Math.floor(inSamples * TARGET_RATE / SAMPLE_RATE);

console.log(`Input: ${inSamples} mẫu @${SAMPLE_RATE}Hz -> kỳ vọng ~${expectedOut} mẫu @${TARGET_RATE}Hz`);
check(audioMsgs.length >= 3, `gửi được nhiều chunk (${audioMsgs.length})`);
check(
  audioMsgs.every((m) => m.payload.samples === CHUNK && m.payload.buffer.byteLength === CHUNK * 2),
  "mỗi chunk đúng chunkSize mẫu Int16"
);
check(
  audioMsgs.every((m) => Array.isArray(m.transfer) && m.transfer.length === 1),
  "buffer được transfer (zero-copy)"
);

// Ghép mẫu Int16 -> tín hiệu
const out = [];
for (const m of audioMsgs) {
  const i16 = new Int16Array(m.payload.buffer);
  for (let i = 0; i < i16.length; i++) out.push(i16[i] / 32767);
}
const outLen = out.length;
check(
  Math.abs(outLen - expectedOut) <= CHUNK,
  `tổng số mẫu ra khớp tỉ lệ resample (${outLen} vs ~${expectedOut})`
);

// Tần số: đếm điểm cắt 0 trong phần giữa (bỏ biên) -> phải ~440 Hz
let crossings = 0;
for (let i = 2; i < outLen - 2; i++) {
  if ((out[i - 1] <= 0 && out[i] > 0)) crossings++;
}
const seconds = outLen / TARGET_RATE;
const measuredFreq = crossings / seconds;
check(
  Math.abs(measuredFreq - freq) / freq < 0.05,
  `tần số sau resample giữ đúng ~${freq} Hz`,
  `đo được ${measuredFreq.toFixed(1)} Hz`
);

// Biên độ không bị suy giảm (nội suy tuyến tính ở 440 Hz/16 kHz gần như không mất)
let peak = 0;
for (const v of out) peak = Math.max(peak, Math.abs(v));
check(peak > 0.7 && peak <= 0.85, "biên độ giữ nguyên (không bị lọc mất)", `peak=${peak.toFixed(3)}`);

// Liên tục pha: bước nhảy giữa hai mẫu liền kề của sine 440 Hz @16 kHz phải nhỏ
const maxStep = 2 * Math.PI * freq / TARGET_RATE; // ~0.173 rad
let worst = 0;
for (let i = 1; i < outLen; i++) worst = Math.max(worst, Math.abs(out[i] - out[i - 1]));
// biên độ 0.8 -> bước tối đa ~0.8*sin(0.173) ~= 0.138; cho phép dư 2.2x
check(
  worst < 0.8 * Math.sin(maxStep) * 2.2,
  "không đứt gãy pha ở biên block (không mất mẫu)",
  `bước lớn nhất ${worst.toFixed(4)}`
);

// Timestamp: bắt đầu ~0, tăng đều 1024/16000 = 64 ms
const ts = audioMsgs.map((m) => m.payload.captureTimestamp);
let mono = true;
for (let i = 1; i < ts.length; i++) if (ts[i] <= ts[i - 1]) mono = false;
check(mono, "timestamp tăng đơn điệu");
const step = 1 / (TARGET_RATE / CHUNK);
let drift = 0;
for (let i = 1; i < ts.length; i++) drift = Math.max(drift, Math.abs((ts[i] - ts[i - 1]) - step));
check(drift < 1e-6, `timestamp không trôi (bước ~${(step * 1000).toFixed(1)} ms)`, `lệch max ${drift}`);

// ---------------------------------------------------------------- K11: độ trễ gom chunk
// Chunk đầu tiên chỉ được gửi khi gom đủ `chunkSize` mẫu @16 kHz ⇒ độ trễ thêm tối thiểu
// = chunkSize/targetRate. Đây là con số K11 (client capture latency) của đường worklet.
const chunkLatencyMs = (CHUNK / TARGET_RATE) * 1000;
console.log("");
console.log(`K11: độ trễ gom chunk của worklet = ${chunkLatencyMs.toFixed(1)} ms ` +
  `(${CHUNK} mẫu @ ${TARGET_RATE} Hz) — mục tiêu < 70 ms`);
check(chunkLatencyMs < 70, "K11: độ trễ gom chunk < 70 ms", `${chunkLatencyMs.toFixed(1)} ms`);

// ---------------------------------------------------------------- ping/pong
scope.messages.length = 0;
if (proc.port.onmessage) proc.port.onmessage({ data: { type: "ping" } });
const pong = scope.messages.find((m) => m.payload.type === "pong");
check(!!pong && typeof pong.payload.framesEmitted === "number", "ping -> pong (watchdog của main thread)");

// ---------------------------------------------------------------- kết luận
console.log("");
if (failures.length) {
  console.log(`KẾT QUẢ: FAIL (${failures.length} mục)`);
  process.exit(1);
}
console.log("KẾT QUẢ: PASS (audio worklet chạy đúng trong scope giả lập)");
process.exit(0);
