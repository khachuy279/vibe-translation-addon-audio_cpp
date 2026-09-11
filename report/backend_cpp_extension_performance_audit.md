# Deep Performance / Architecture Audit — `backend_cpp` + `extension_firefox`

**Audit type:** Static source audit + architecture/data-flow reconstruction  
**Scope:** `backend_cpp` + `extension_firefox`  
**Project constraint:** Windows desktop, RTX 5060 Ti 16GB, normally max 1 active session  
**Code changes:** **NONE**  
**Commit:** **NONE**  
**Runtime benchmark executed:** **NO** — report separates code facts from inference/hypothesis and marks runtime measurement as `NEEDS BENCHMARK`.

---

## Audit Method

The audit deliberately does **not** treat the project as a collection of unrelated files. The first pass reconstructed:

- application entry point and lifecycle;
- WebSocket/session execution model;
- audio data ownership and buffer lifecycle;
- VAD frame loop and callback model;
- ASR preview/commit execution;
- translation queue and singleton LLM;
- TTS queue and singleton PyTorch model;
- Firefox content-script / service-worker bridge;
- outbound WebSocket serialization;
- cleanup and model unload paths.

The production-source inventory found:

- **61 Python source files** under `backend_cpp` (plus compiled `__pycache__` artifacts in the ZIP);
- **10 JavaScript source files** under `extension_firefox`;
- model/config catalogs and popup HTML/CSS;
- multiple existing performance/leak/concurrency tests, which were inspected as design intent and benchmark scaffolding but not treated as runtime evidence.

Syntax-only validation was also performed:

- Python AST parse: **61/61 source files parse successfully**
- JavaScript `node --check`: **10/10 JS files pass syntax checking**

---

# 1. Executive Summary

## 1.1 Overall assessment

The project is already structured around a reasonable realtime pipeline:

```text
Firefox content script
    |
    | 16-bit PCM chunks
    v
WebSocket / background bridge
    |
    v
FastAPI / WS handler
    |
    v
VADProcessor
    |
    +--> ASR AudioBufferManager
              |
              +--> periodic ASR preview
              |
              +--> final ASR commit
                        |
                        v
                  Translation queue
                        |
                        v
                 Local GGUF translator
                        |
                        v
                    TTS queue
                        |
                        v
                  OmniVoice TTS
                        |
                        v
                  JSON + base64 WAV
                        |
                        v
                    Firefox player
```

The architecture is viable for a **single desktop session**, but two things dominate future performance:

1. **ASR preview is not truly incremental.** Every preview poll snapshots the entire accumulated utterance, converts the entire buffer again, normalizes the entire array again, and then re-runs inference over the growing audio. For non-streaming models this is naturally cumulative/repeated work; for “streaming” models the current implementation still creates a fresh stream context for each snapshot and finalizes it, so the code does not actually exploit persistent stream state.

2. **Model inference is globally serialized by design.** ASR uses a process-wide `_shared_infer_lock`; translation has a process-wide singleton with `_infer_lock`; TTS has a singleton with `_infer_lock`. This is acceptable for one session and beneficial for VRAM, but it becomes the hard scalability ceiling for multi-session operation.

The third major issue is correctness/lifecycle rather than raw compute:

3. **ASR model unload/swap is not synchronized with in-flight inference.** The unload path holds `_shared_lock` but does not coordinate with `_shared_infer_lock`, while inference releases `_shared_lock` before acquiring `_shared_infer_lock`. A model-switch/unload can therefore race with a running inference that is still using the model/session. This is a potential crash/data-corruption class defect.

A fourth issue is on the browser side:

4. **`popup.js` broadcasts START to `allFrames: true`.** Every matching content-script frame can attempt `startCapture()`. A product requirement of one session is therefore not actually enforced by the extension architecture. This can create multiple WebSocket connections, duplicate capture, extra CPU/memory load, and duplicated model contention.

---

## 1.2 Findings by priority

### P0 — Critical

| ID | Finding | Why P0 |
|---|---|---|
| P0-01 | ASR model/session unload/swap can race with inference | Potential crash, use-after-close, native-library instability, data corruption |
| P0-02 | Unbounded ASR token queue | Potential unbounded memory growth when outbound send is slower than preview production |

### P1 — High

| ID | Finding | Primary impact |
|---|---|---|
| P1-01 | Full-utterance ASR snapshot + repeated full inference on every preview | Latency, CPU/GPU utilization, repeated work |
| P1-02 | “Streaming” ASR path recreates/finalizes a fresh stream per poll | Missed streaming-state performance advantage |
| P1-03 | Large number of full-buffer copies/conversions before ASR | CPU/memory bandwidth/latency |
| P1-04 | Multi-frame START can create multiple sessions | Unexpected CPU/VRAM contention and duplicate processing |
| P1-05 | Shared ASR inference lock is hard serialization boundary | Multi-session throughput ceiling |
| P1-06 | Shared VAD engine lacks an explicit engine-level inference contract for concurrent sessions | Potential thread-safety issue for multi-session VAD |
| P1-07 | Translation timeout can leave underlying native inference thread running | Hidden lock occupancy, backlog, latency spikes |
| P1-08 | Default executor is shared across VAD/ASR/translation/TTS blocking work | Cross-stage thread starvation under contention |
| P1-09 | Audio dump executor is a single-thread executor with an unbounded work queue when enabled | I/O backlog + memory growth under slow disk |
| P1-10 | TTS path contains CPU-side postprocessing + base64 expansion | CPU, RAM, wire latency |
| P1-11 | TTS/translation are sequential workers with bounded queues and drop-on-full behavior | Throughput ceiling and data loss under load |

### P2 — Medium

| ID | Finding | Primary impact |
|---|---|---|
| P2-01 | 350 ms ASR polling granularity | Hidden preview latency |
| P2-02 | Stability logic adds deliberate extra commit latency | End-to-end latency |
| P2-03 | `AudioBufferManager` snapshots whole buffer even if only a small suffix is new | CPU/memory bandwidth |
| P2-04 | VAD per-frame float conversion + per-frame byte slicing | CPU/allocation pressure |
| P2-05 | `SpeechNormalizer.sanitize_input()` always copies + clamps | CPU/allocation pressure |
| P2-06 | Normalization runs on every preview inference over the full snapshot | Repeated CPU work |
| P2-07 | Firefox ScriptProcessor main-thread capture + JS resampling loop | Browser CPU/jank risk |
| P2-08 | Firefox bridge path structurally clones/transfers PCM before packet build | Copy/serialization overhead |
| P2-09 | JSON + base64 are used for TTS audio transport | Payload expansion and allocations |
| P2-10 | Global DEBUG logging configuration is aggressive for a realtime app | I/O/logging overhead |

### P3 — Low

| ID | Finding | Primary impact |
|---|---|---|
| P3-01 | Small object/dict construction in queue messages | Micro CPU |
| P3-02 | Some duplicated compatibility aliases and wrappers | Complexity/maintainability |
| P3-03 | Repeated catalog/config scans on UI/backend endpoints | Low-volume I/O |
| P3-04 | `AudioCapture` logs every 200 chunks | Small console overhead |

---

## 1.3 Biggest bottleneck

### Primary bottleneck: ASR preview architecture

The current preview loop is effectively:

```text
every 350 ms
    -> snapshot WHOLE utterance
    -> int16 -> float32 WHOLE utterance
    -> normalize WHOLE utterance
    -> run inference on WHOLE utterance
    -> compare text
```

The key point is that:

```python
snapshot = self._audio_buffer_mgr.get_snapshot_if_newer(...)
```

does not return only the delta. It returns the full buffer.

Then:

```python
await asyncio.to_thread(
    self._run_inference,
    pcm_snapshot,
    ...
)
```

passes the full utterance again.

This makes the amount of repeated work grow with utterance duration. With a bounded 8-second sentence, the absolute size is limited, but the architecture still performs repeated full-array materialization and repeated inference instead of incremental decoding.

For a true streaming model, the highest-ROI fix is **persistent model stream/session state per utterance**, not “optimize NumPy one line at a time.”

---

## 1.4 Three highest-ROI changes

### ROI #1 — Make ASR preview truly incremental

**Targets:** `asr/transcribe_engine.py`, `asr/audio_buffer.py`

For models whose native API supports streaming, maintain one stream/session state per active utterance and feed only the new PCM samples. Preserve a bounded overlap/replay mechanism only where the model requires it.

Expected benefit: major reduction in repeated ASR work.

**Risk:** medium/high because native `transcribe_cpp` stream semantics must be validated.

---

### ROI #2 — Fix ASR model lifecycle synchronization

**Targets:** `asr/model_manager.py`, `asr/transcribe_engine.py`, `main.py`

Introduce an explicit lifecycle barrier so that:

```text
model swap/unload
       |
       +--> waits for all in-flight inference to finish
       |
       +--> closes session
       |
       +--> closes model
       |
       +--> installs new model
```

This is correctness-first, not a micro-optimization.

---

### ROI #3 — Enforce exactly one browser capture owner

**Targets:** `extension_firefox/popup/popup.js`, `content/content-script.js`

Do not broadcast START indiscriminately to every frame. Select one capture owner:

```text
popup
  |
  +--> discover active video owner frame
          |
          +--> START exactly one capture session
```

Other frames should render/broadcast UI state, not capture audio or open independent backend sessions.

---

# 2. Architecture hiện tại

## 2.1 Backend component map

```text
backend_cpp/main.py
|
+-- FastAPI / Uvicorn
|     |
|     +-- /health
|     +-- /api/config
|     +-- /api/voices
|     +-- /api/tts/prewarm
|     +-- /api/perf/*
|     +-- /ws
|
+-- global config
|
+-- ModelRegistry ----------------------+
|                                       |
+-- ASRModelManager                     |
|     +-- shared transcribe_cpp.Model   |
|     +-- shared transcribe_cpp.Session |
|     +-- _shared_lock                  |
|     +-- _shared_infer_lock            |
|     +-- _commit_lock                  |
|                                       |
+-- Translation singleton                |
|     +-- llama_cpp / GGUF              |
|     +-- _load_lock                    |
|     +-- _infer_lock                   |
|                                       |
+-- VADEngineFactory                     |
|     +-- shared VAD engine(s)          |
|     +-- per-session stream state      |
|                                       |
+-- TTS singleton                        |
      +-- OmniVoice PyTorch             |
      +-- _init_lock                    |
      +-- _infer_lock                   |
```

## 2.2 Session component map

For each WebSocket session:

```text
SessionState
|
+-- TranscribeEngine
|     |
|     +-- AudioBufferManager
|     +-- SpeechNormalizer
|     +-- SentenceSegmenter
|     +-- CommitDeduplicator
|     +-- token_queue = asyncio.Queue()   <-- UNBOUNDED
|     +-- preview poller task
|
+-- VADProcessor
|     |
|     +-- shared engine singleton
|     +-- per-session VADStreamState
|     +-- raw_buffer <= 3 sec
|     +-- pre_speech_ring
|
+-- translation_queue = asyncio.Queue(maxsize=20)
|
+-- tts_queue = asyncio.Queue(maxsize=10)
|
+-- SafeWebSocketConnection
      +-- asyncio.Lock for outbound sends
```

---

# 3. Data Flow

## 3.1 End-to-end audio path

```text
Firefox HTMLVideoElement
        |
        v
MediaElementSource / captureStream
        |
        v
AudioContext
        |
        v
ScriptProcessor(4096)
        |
        +-- resample to 16 kHz if needed
        |
        +-- Float32 -> Int16 loop
        |
        +-- 1024 samples = ~64 ms
        |
        v
ArrayBuffer
        |
        v
WSClient.sendBinary()
        |
        +-- frame-builder: JSON header + PCM packet
        |
        v
[optional Firefox background bridge]
        |
        +-- SEND_BINARY
        +-- service worker builds packet
        |
        v
WebSocket binary message
        |
        v
parse_audio_frame()
        |
        +-- JSON header parse
        +-- PCM slice
        |
        v
VADProcessor.feed_chunk()
        |
        +-- raw_buffer.extend()
        +-- per-native-frame int16 -> float32
        +-- VAD inference
        +-- callback frame -> bytes(...)
        |
        v
TranscribeEngine.feed_audio()
        |
        +-- AudioBufferManager.feed_bytes()
        |
        v
AudioBufferManager._bytes_buffer
        |
        +-- periodic full snapshot
        |
        v
int16 -> float32 WHOLE utterance
        |
        v
SpeechNormalizer WHOLE utterance
        |
        v
transcribe_cpp inference
        |
        +-- preview OR commit
        |
        v
ASR token queue
        |
        v
WebSocket JSON
        |
        v
translation_queue
        |
        v
llama.cpp GGUF
        |
        v
translation JSON
        |
        v
tts_queue
        |
        v
OmniVoice PyTorch
        |
        +-- GPU inference
        +-- CPU tensor -> numpy
        +-- optional phase-vocoder
        +-- normalize
        +-- WAV encoding
        +-- base64
        |
        v
WebSocket JSON with base64 audio
        |
        v
Firefox JSON.parse()
        |
        v
atob()
        |
        v
Uint8Array
        |
        v
Blob
        |
        v
Object URL
        |
        v
HTML Audio playback
```

---

# 4. Critical Path

## 4.1 Subtitle critical path

```text
audio arrival
 -> VAD frame decision
 -> audio buffer
 -> next ASR poll
 -> ASR inference
 -> ASR token delivery
 -> translation queue wait
 -> translation inference
 -> outbound websocket send
 -> subtitle render
```

The **ASR-to-subtitle** path is dominated by:

1. polling delay;
2. full-buffer ASR preview/commit;
3. ASR global inference lock;
4. translation queue + translation inference.

The TTS branch is downstream/parallel to subtitle delivery and should not block subtitle rendering because it has a separate queue/worker.

## 4.2 TTS critical path

```text
translation complete
 -> TTS queue
 -> TTS global infer lock
 -> GPU generation
 -> CPU audio conversion
 -> optional time stretch
 -> normalization
 -> WAV encode
 -> base64
 -> JSON serialization
 -> WS send
 -> browser decode
 -> Blob URL
 -> playback
```

---

# 5. Bottleneck Table

| Priority | Component | Problem | Evidence | Impact | Recommendation |
|---|---|---|---|---|---|
| P0 | ASR model lifecycle | unload/swap not synchronized with in-flight inference | `asr/model_manager.py:66-84`, `asr/transcribe_engine.py:409-440` | crash/native instability | add lifecycle barrier; do not close model while infer is active |
| P0 | ASR token queue | unbounded `asyncio.Queue()` | `asr/transcribe_engine.py:296-299` | unbounded memory growth under blocked sender | bound queue and coalesce/drop stale preview messages |
| P1 | ASR preview | whole utterance re-snapshotted every poll | `asr/transcribe_engine.py:776-798`; `asr/audio_buffer.py:111-138` | repeated CPU/RAM bandwidth + inference work | incremental streaming or delta-aware pipeline |
| P1 | ASR streaming | fresh `session.stream()` context per preview and immediate finalize | `asr/transcribe_engine.py:447-470` | streaming API advantage largely lost | persist stream state for utterance |
| P1 | Audio copies | multiple full/slice copies across VAD -> ASR | `ws/frame_protocol.py:35`, `vad/vad_processor.py:167-228`, `asr/audio_buffer.py:56-94` | bandwidth/allocations | reduce ownership boundaries; pass views/deltas where APIs allow |
| P1 | Firefox session ownership | `allFrames: true` START | `extension_firefox/popup/popup.js:507-531` | duplicate capture/session | select exactly one capture frame |
| P1 | ASR serialization | global `_shared_infer_lock` | `asr/model_manager.py:34`, `asr/transcribe_engine.py:416-430` | multi-session throughput collapse | one-model/multi-stream strategy or model pool if scaling becomes required |
| P1 | VAD shared engine | session state isolated but underlying engine concurrency contract not explicit | `vad/engines.py:436-468`, no shared infer lock | possible multi-session race | document/verify thread safety; add engine lock if native engine is non-reentrant |
| P1 | Translation timeout | `wait_for` cannot stop native thread | `translation/local_translator.py:233-239`, `314-322` | hidden inference continues after timeout | cooperative cancellation or bounded dedicated worker |
| P1 | Default executor | multiple stages use `asyncio.to_thread` | ASR/VAD/translation/TTS call sites | thread pool starvation | dedicated executors for compute domains |
| P1 | Audio dump | single worker, unbounded executor queue | `utils/audio_dumper.py:22`, `71`, `96`, `132` | I/O backlog and RAM growth | disable in production; bounded queue/backpressure |
| P1 | TTS | sequential singleton worker | `ws/ws_handler.py:378-398`, `tts/omnivoice_engine.py:58-60` | throughput limited to one generation at a time | keep for 1 session; only change for multi-session |
| P2 | Polling | fixed 350 ms | `config.py:109`, `transcribe_engine.py:758-762` | latency floor | event-driven/new-audio-triggered scheduling |
| P2 | Stability | 0.8 s + minimum poll count | `config.py:178-180`, `sentence_segmenter.py:153-187` | deliberate commit latency | benchmark UX/accuracy before changing |
| P2 | VAD conversion | per frame `astype(float32)` | `vad/vad_processor.py:193-199` | CPU allocations | reuse frame buffer if supported |
| P2 | Normalizer | copy + clip + repeated full-array stats | `speech_normalizer.py:95-100`, `202-254` | CPU/allocations | normalize only new/required audio where model semantics allow |
| P2 | Browser capture | ScriptProcessor + JS sample loop | `lib/audio-capture.js:101-175` | browser CPU/jank | migrate actual path to AudioWorklet later |
| P2 | TTS wire format | base64 + JSON | `tts/audio_processor.py:124-132`, `ws/serializers.py` | memory and payload expansion | binary WS message for audio |
| P2 | Logging | process-global DEBUG + sentence-level INFO | `main.py:35-39`, `ws/ws_handler.py:250`, etc. | I/O under load | INFO production, DEBUG diagnostic only |
| P3 | compatibility wrappers | alias/indirection layers | multiple `__init__.py` and WS aliases | complexity | cleanup only after behavior freezes |

---

# 6. CPU Bottlenecks

## 6.1 ASR full-buffer transformation

### Evidence

`AudioBufferManager.get_snapshot_if_newer()`:

```text
bytes(_bytes_buffer)
 -> np.frombuffer(...)
 -> astype(np.float32)
```

The snapshot is the complete current utterance.

Then `SpeechNormalizer.process()` performs:

```text
sanitize copy
 -> finite check
 -> clip
 -> RMS frame matrix
 -> percentile calculations
 -> gain application
 -> peak check
 -> clip again
```

Then inference runs.

### Root cause

The preview cadence is incremental in **time**, but not incremental in **data**.

### Classification

**FACT:** full snapshots and full normalization are performed.

**INFERENCE:** repeated whole-utterance work is likely a major CPU/memory-bandwidth consumer.

**BENCHMARK REQUIRED:** percentage of CPU time spent in snapshot, normalization, and native inference.

---

## 6.2 VAD per-frame allocation

`vad/vad_processor.py` performs:

```python
samples_int16 = np.frombuffer(...)
samples_float32 = samples_int16.astype(np.float32) / 32768.0
```

for every native VAD frame.

Then speech frames often do:

```python
frame_bytes = bytes(raw_buf[offset:frame_end])
```

This creates another bytes object before ASR receives it.

For FireRed at 25 ms:

```text
16,000 / 400 = 40 frames/sec
```

That means up to ~40 VAD conversion iterations per realtime second.

For FSMN at 60 ms:

```text
~16.7 frames/sec
```

These frequencies are static consequences of the configured frame sizes, not benchmark results.

---

## 6.3 Audio normalization

`SpeechNormalizer.sanitize_input()` explicitly copies:

```python
x = np.asarray(pcm, dtype=np.float32).copy()
```

and then:

```python
return np.clip(x, -1.0, 1.0)
```

which may allocate another array because no `out=` buffer is supplied.

`process()` later does more elementwise full-array passes.

This is safe and clear, but not ideal on a repeated preview hot path.

### Recommendation

Do **not** remove sanitization blindly. Instead:

- establish the invariant that internal PCM is already float32 and bounded;
- validate that invariant at the ingress boundary;
- make hot-path processing conditional on whether sanitization is actually needed;
- keep a debug/strict mode for defensive checks.

This preserves quality/reliability while reducing unnecessary passes.

---

# 7. GPU / VRAM Bottlenecks

## 7.1 ASR + translation + TTS coexistence

The model catalog states approximately:

- Qwen3-ASR-1.7B Q8: **~2.1 GB VRAM estimate**
- Tencent Hunyuan-MT2 7B Q4: **~4.6 GB**
- Xiaomi translation: **~2.5 GB**
- GemmaX2: **~5.8 GB**
- Voxtral 4B: **~3.2 GB**
- SenseVoice: **~0.5 GB**
- Nemotron streaming: **~0.75 GB**
- Kotoba Whisper: **~1.5 GB**

These are **catalog estimates from `models.yaml` / `translation_models.yaml`, not measured allocator data**.

For the default ASR + default Tencent translation combination:

```text
2.1 GB + 4.6 GB = ~6.7 GB
```

before TTS weights, CUDA/driver memory, runtime workspaces, KV/cache/context allocations, and allocator reserve.

Therefore:

**Inference:** the 16 GB card is plausibly sufficient for a single-session configuration using the smaller ASR/translation combinations.

**NEEDS BENCHMARK:** actual peak `torch.cuda.memory_allocated()` / reserved memory plus native transcribe/llama allocations. The current perf profiler cannot fully represent non-PyTorch CUDA/native allocations.

---

## 7.2 ASR GPU transfer

The code sends `pcm_float32` into `transcribe_cpp`.

No explicit `.cuda()` appears on the ASR path, which is good: the Python code is not repeatedly creating a torch tensor and transferring it to GPU.

The main transfer concern therefore moves inside the native `transcribe_cpp` binding, which is not visible in this source archive.

**NEEDS BENCHMARK / native profiler:** inspect host-to-device transfer inside `transcribe_cpp`.

---

## 7.3 TTS CPU/GPU boundary

`OmniVoiceTTS.synthesize_sync()`:

1. performs generation under `_infer_lock`;
2. calls CUDA synchronize;
3. calls `AudioProcessor.convert_to_numpy()`.

For Tensor output:

```python
item.detach().cpu().numpy().astype(np.float32)
```

Potentially:

```text
GPU tensor
 -> CPU copy
 -> NumPy view
 -> float32 copy if dtype != float32
```

This is expected to some degree, because WAV encoding needs CPU-accessible PCM.

Do not optimize this until a profiler confirms TTS postprocessing is material compared with generation.

---

# 8. Memory Bottlenecks

## 8.1 P0 — Unbounded ASR token queue

`TranscribeEngine._get_queue()` creates:

```python
asyncio.Queue()
```

without `maxsize`.

Preview messages are emitted from `_partial_preview_poller()`.

Consumption is:

```python
msg = await q.get()
yield msg
```

followed by WebSocket send.

The send is protected by:

```python
SafeWebSocketConnection._send_lock
```

so outbound traffic can serialize.

If network/browser consumption stalls while preview production continues:

```text
preview producer
   -> token_queue grows
   -> each message retains text + metadata
   -> memory grows without hard bound
```

### Recommendation

Use a bounded queue and **coalesce preview messages**:

```text
final messages: lossless
preview messages: latest-only / replace-old
```

This preserves correctness while making memory bounded.

---

## 8.2 VAD buffer is bounded

Good design:

```python
max_buffer_bytes = sample_rate * 2 * 3.0
```

So VAD raw buffering is capped at approximately 3 seconds.

This is one of the stronger safeguards already present.

However, dropping oldest audio while speech is active means the system can lose audio when downstream processing falls behind.

Therefore:

**GOOD for safety:** prevents unbounded RAM.

**BAD for fidelity:** can drop speech.

The warning explicitly tells the operator that downstream processing is lagging.

---

## 8.3 ASR AudioBuffer is bounded but larger

`AudioBufferManager` defaults to 60 seconds, but the sentence config has an 8-second max-duration auto-commit.

This means the explicit storage bound and logical sentence bound are not aligned.

Recommended principle:

```text
buffer capacity = max_duration_sec + safety_margin
```

rather than a generic 60-second default, unless long utterances are an actual supported feature.

---

## 8.4 Audio dumper lifecycle

`_DUMP_EXECUTOR = ThreadPoolExecutor(max_workers=1)`

Every dump operation uses `.submit()`.

ThreadPoolExecutor's work queue is not bounded by this code.

When `dump_audio=True` and disk I/O falls behind:

```text
capture
 -> submit write task
 -> submit write task
 -> submit write task
 -> ...
 -> executor queue grows
```

The task closures retain references to the corresponding PCM data.

This is a real memory-growth mechanism under slow storage.

### Recommendation

Production mode:

```text
dump_audio = false
```

Diagnostic mode:

- bounded queue;
- explicit drop policy;
- periodic batch writes;
- one file handle per session already exists for ingress, which is good.

---

# 9. I/O Bottlenecks

## 9.1 Audio debug dumping

Ingress, VAD utterance, and ASR input are all written when enabled.

This causes:

```text
3 debug checkpoints
```

and ASR preview dumping can be especially expensive because every preview inference can generate a new WAV file.

`dump_asr_input(..., is_commit=False)` creates a new preview filename and submits a write.

This is **not acceptable as a default hot-path diagnostic feature** for long-running sessions.

Current config correctly defaults:

```python
dump_audio = False
```

### Decision

**DO NOT TOUCH** unless diagnostics are actively needed.

---

## 9.2 WebSocket JSON serialization

Every outbound payload is serialized with:

```python
json.dumps(payload)
```

then sent as text.

For subtitles this is fine.

For TTS audio it is expensive because the audio was already base64 encoded.

---

# 10. Lock / Thread Bottlenecks

## 10.1 ASR global inference lock

```python
_shared_infer_lock = threading.Lock()
```

This is intentionally acquired around the native inference.

### 1 session

This is basically a correctness boundary and should remain unless native session becomes truly concurrent-safe.

### 2+ sessions

All sessions contend on the same lock:

```text
Session A ASR
       |
       +---- lock ----+
                      |
Session B ASR --------+
                      |
Session C ASR --------+
```

Throughput cannot scale linearly.

---

## 10.2 ASR commit priority logic

The code has a good idea:

```text
preview = non-blocking
commit  = blocking
```

and:

```python
if _commit_waiting > 0:
    skip preview
```

This reduces commit starvation.

This part is a **DO NOT TOUCH** design unless the benchmark proves preview frequency is too high.

---

## 10.3 Translation lock

`LocalGGUFTranslator` uses:

```python
self._infer_lock = threading.RLock()
```

Because the translator is a singleton, all translation jobs in the process serialize.

For a single desktop session: reasonable.

For multiple sessions: immediate shared bottleneck.

---

## 10.4 TTS lock

Same pattern:

```python
self._infer_lock = threading.Lock()
```

Good for safety with one model instance.

Do not try to run concurrent generation on the same model until native/PyTorch model behavior and VRAM headroom are verified.

---

## 10.5 WebSocket send lock

`SafeWebSocketConnection` is correct in principle.

Multiple async workers can call:

```text
send translation
send TTS
send ASR
```

and the lock ensures serialized outbound WebSocket writes.

### Potential latency effect

If TTS sends a very large JSON payload while subtitle workers want to send short text messages, the subtitle path may wait for the TTS serialization/send operation.

Therefore the lock is necessary, but the **payload size** matters.

A binary TTS channel can reduce this blocking duration substantially.

---

# 11. Queue / Backpressure Problems

## 11.1 Translation queue

```python
asyncio.Queue(maxsize=20)
```

Good: bounded.

Full behavior:

```text
QueueFull -> sentence dropped
```

This is explicit data loss.

For one session this is probably rare if translation is faster than speech segmentation.

**NEEDS BENCHMARK:** sustained utterance frequency vs translation RTF.

---

## 11.2 TTS queue

```python
asyncio.Queue(maxsize=10)
```

Also bounded, but full behavior drops a translated sentence's TTS request.

This is a reasonable emergency policy for realtime playback, but the choice should be intentional:

```text
subtitle path = lossless priority
TTS path = lossy optional side branch
```

The code effectively behaves that way, which is good.

---

## 11.3 ASR token queue

Unlike downstream queues, this one is unbounded.

This is the most problematic queue policy.

Recommended:

```text
FINAL messages: never drop
PREVIEW messages: latest-wins
```

---

## 11.4 Producer-consumer balance

### Translation

```text
ASR commit producer
      |
      v
translation_queue
      |
      v
1 translator worker
```

If:

```text
translation_rate < sentence_commit_rate
```

backlog grows until 20, then sentences are dropped.

### TTS

```text
translation producer
      |
      v
tts_queue
      |
      v
1 TTS worker
```

If:

```text
TTS synthesis throughput < sentence output rate
```

backlog grows until 10.

This is predictable and bounded.

---

# 12. Audio Copy / Conversion Audit

## 12.1 Main path copy table

| From | To | Copy? | Frequency | Necessary? | Recommendation |
|---|---|---:|---|---|---|
| WebSocket frame `data` | `pcm = data[...]` | **YES** | every audio WS frame | technically simple, but avoidable with parser/view API | expose memoryview only if downstream can accept it |
| `pcm` bytes | VAD `raw_buffer` via `.extend()` | **YES** | every frame | currently yes | unavoidable if retaining bytes; consider ring buffer/view ownership |
| VAD raw buffer | per-frame `bytes(...)` | **YES** | every native VAD frame | mostly avoidable | make callback/API accept memoryview or process directly |
| per-frame bytes | ASR AudioBuffer `.extend()` | **YES** | speech frame | currently yes | preserve a single owned buffer where possible |
| ASR `_bytes_buffer` | snapshot `bytes(...)` | **YES** | every ASR preview | avoidable | maintain typed/ring buffer or incremental offset |
| int16 view | float32 `astype()` | **YES** | every preview | conversion may be required by model | convert once and reuse if model contract permits |
| normalized input | `SpeechNormalizer.sanitize_input()` | **YES** | every inference | some validation needed | avoid unconditional copy in trusted hot path |
| normalized PCM | ASR dump int16 | **YES** when debug | every dump | diagnostic only | leave disabled in production |
| Firefox Float32 -> Int16 | new Int16Array | **YES** | every 64ms chunk | required for protocol | AudioWorklet can move this off main thread |
| content frame -> background bridge | structured clone / message transfer | **LIKELY YES** | every chunk | bridge requires transfer | use transferable port payload where API permits |
| background bridge | final packet ArrayBuffer | **YES** | every chunk | current packet format requires concat | binary framing protocol could avoid an extra aggregate allocation |

### Copy count conclusion

The exact low-level physical copy count across browser/runtime boundaries depends on browser structured-clone/transfer semantics and ASGI internals, so **do not state an exact total as measured**.

The source clearly shows at least several explicit application-level copies.

A representative backend path is:

```text
WS data
 -> PCM bytes slice
 -> VAD raw buffer copy
 -> VAD frame bytes copy
 -> ASR raw buffer copy
 -> ASR snapshot copy
 -> int16 -> float32 copy
 -> normalizer copy
 -> normalization outputs
```

That is enough evidence to prioritize **buffer ownership redesign**, even before a native profiler is attached.

---

# 13. Latency Budget

Because the source archive does not contain actual benchmark samples or GPU timings, the compute cost is marked `UNKNOWN — benchmark required`.

| Stage | Estimated Cost | Blocking? | Critical Path? |
|---|---:|---|---|
| Browser chunk formation | ~64 ms chunk cadence | no | yes |
| WebSocket transport | UNKNOWN — benchmark | async I/O | yes |
| Binary header parse | UNKNOWN — likely small | CPU | yes |
| VAD frame processing | UNKNOWN | yes inside `to_thread` | yes |
| ASR poll quantization | **0–350 ms scheduling granularity** | no, but delay | yes |
| ASR preview inference | UNKNOWN | native compute in worker | yes |
| ASR final inference | UNKNOWN | native compute in worker | yes |
| Translation queue wait | UNKNOWN | queue wait | yes |
| Translation inference | UNKNOWN | native compute in worker | yes |
| Subtitle WS send | UNKNOWN | serialized by send lock | yes |
| TTS queue wait | UNKNOWN | queue wait | only TTS branch |
| TTS generation | UNKNOWN | GPU + `_infer_lock` | TTS critical path |
| TTS CPU postprocess | UNKNOWN | worker thread | TTS critical path |
| WAV/base64 serialization | UNKNOWN | CPU + allocation | TTS critical path |
| Browser base64 decode | UNKNOWN | main thread | TTS critical path |

### Hidden latency sources

1. **64 ms browser chunking**
2. **350 ms ASR poll interval**
3. **0.8 s stability duration**
4. **minimum stability poll count**
5. **global ASR inference lock**
6. **translation queue wait**
7. **global translation inference lock**
8. **outbound WebSocket send lock**
9. **TTS queue wait**
10. **CUDA synchronize in TTS**

The 350 ms polling interval is especially important: it quantizes preview responsiveness before model compute is even considered.

---

# 14. Scalability Analysis

## 14.1 1 session

This is the architecture's intended sweet spot.

Likely characteristics:

```text
ASR: serialized — acceptable
Translation: serialized — acceptable
TTS: serialized — acceptable
VAD: one active stream — acceptable
VRAM: one model instance per subsystem — desirable
```

The main performance problem is **repeated ASR work**, not concurrency.

---

## 14.2 2 sessions

Likely first new bottlenecks:

```text
Session A ASR ----\
Session B ASR -----+--> shared ASR infer lock
```

Translation and TTS also serialize through singletons.

VAD engine concurrency must be verified.

Firefox multi-frame session creation becomes even more harmful if more than one capture is accidentally created in the same tab.

---

## 14.3 4 sessions

Expected:

- ASR inference queueing becomes visible;
- translation lock wait increases;
- TTS lock wait increases;
- default executor becomes more relevant;
- shared VAD model concurrency becomes a correctness question;
- CPU-side audio copies multiply roughly with number of active streams.

The system will not scale linearly.

---

## 14.4 8 sessions

The architecture becomes dominated by shared singletons:

```text
N sessions
    |
    +--> one ASR model/session
    +--> one translation LLM
    +--> one TTS model
```

Throughput becomes queueing around three serial resources.

At this point a model pool or explicit admission-control architecture is required.

---

## 14.5 16 sessions

Do not attempt to scale this architecture by merely increasing worker counts.

The limiting resources become:

1. GPU VRAM;
2. GPU compute;
3. serialized model locks;
4. native runtime thread counts;
5. CPU-side copying;
6. default executor saturation;
7. network + JSON/base64 serialization.

At 16 sessions, the correct architecture would likely be a **resource scheduler** rather than one singleton per model.

---

# 15. Top 10 Optimizations

Ranking is based on:

```text
ROI = expected performance/reliability value / implementation risk
```

without inventing numerical speedups.

| Rank | Change | ROI | Risk | Metrics improved |
|---|---|---|---|---|
| 1 | True incremental ASR streaming | Very High | Medium/High | latency, throughput, CPU/GPU |
| 2 | Fix ASR model lifecycle barrier | Very High | Medium | reliability, latency spikes |
| 3 | Bound + coalesce ASR preview queue | Very High | Low | RAM, latency stability |
| 4 | Enforce one capture owner in extension | Very High | Low/Medium | CPU, RAM, session count |
| 5 | Reduce full-buffer copy chain | High | Medium | CPU, RAM bandwidth |
| 6 | Event-driven ASR preview scheduling | High | Medium | latency, CPU |
| 7 | Binary TTS WS messages | High | Medium | CPU, RAM, wire latency |
| 8 | Dedicated executors for blocking native calls | Medium/High | Medium | tail latency, thread contention |
| 9 | Verify/guard VAD engine thread safety | High for scale | Medium | reliability, scale |
| 10 | Production logging profile | Medium | Low | I/O, CPU |

---

# 16. Quick Wins

## QW-01 — Bound ASR token queue

**File:** `asr/transcribe_engine.py`

**Current:**
```python
asyncio.Queue()
```

**Change:**
Use a bounded queue.

Preview messages should be coalesced/dropped before final messages.

**Trade-off:** possible loss of intermediate previews, but no quality loss to final transcript.

---

## QW-02 — Production logging level

**File:** `main.py`

Current global level:

```python
logging.DEBUG
```

For normal realtime use:

```text
INFO
```

with explicit DEBUG opt-in.

This changes observability, not ASR/TTS quality.

---

## QW-03 — Keep `dump_audio=False`

Current default is already correct.

Do not enable it during normal performance testing.

---

## QW-04 — Prevent duplicate START across frames

**File:** `extension_firefox/popup/popup.js`

`broadcastToFrames("START_TRANSLATION")` with `allFrames: true` is inconsistent with the one-session requirement.

Select one frame for capture.

---

## QW-05 — Separate “preview” and “final” message policies

Final subtitle:

```text
lossless
```

Preview subtitle:

```text
latest-wins
```

This is an architecture-friendly form of backpressure.

---

# 17. Structural Improvements

## 17.1 Recommended target pipeline

```text
Firefox capture
    |
    v
Single Capture Owner
    |
    | 16-bit PCM
    v
Bounded ingress channel
    |
    v
VAD
    |
    +--> incremental ASR stream
    |       |
    |       +--> preview latest-wins queue
    |       |
    |       +--> final commit queue
    |
    v
translation
    |
    v
TTS optional side branch
```

---

## 17.2 ASR target architecture

For native streaming models:

```text
utterance start
    |
    v
create one persistent stream
    |
    +-- feed delta PCM
    +-- read partial
    +-- feed delta PCM
    +-- read partial
    +-- ...
    |
utterance end
    |
    v
finalize stream once
```

For non-streaming models:

```text
utterance buffer
    |
    +--> preview schedule on bounded window / controlled cadence
    |
    +--> final full inference
```

Do not pretend offline inference is streaming.

---

## 17.3 Buffer ownership target

Prefer:

```text
one canonical PCM representation
```

for each stage.

For example:

```text
Ingress PCM
   |
   +--> VAD uses view
   |
   +--> ASR owns bounded ring
```

rather than:

```text
bytes
 -> bytearray
 -> bytes frame
 -> bytearray
 -> bytes snapshot
 -> numpy
 -> numpy copy
```

The exact zero-copy design is constrained by native library APIs and must be benchmarked.

---

# 18. Benchmark Plan

## 18.1 Benchmark A — ASR preview cost

Measure for utterances:

```text
1 s
2 s
4 s
6 s
8 s
```

Metrics:

- preview infer ms;
- audio duration;
- RTF;
- snapshot conversion time;
- normalization time;
- CPU utilization;
- peak RSS;
- allocation count if profiler available.

Goal:

Determine whether inference cost scales approximately with total utterance duration.

---

## 18.2 Benchmark B — Incremental stream vs current preview

Run:

```text
Current implementation
vs
Persistent native stream
```

Measure:

- p50/p90/p99 preview latency;
- total GPU time;
- CPU time;
- number of samples processed by native ASR;
- transcript equivalence.

Do not accept the optimization until transcript quality is shown equivalent.

---

## 18.3 Benchmark C — Copy audit

Instrument:

```text
parse_audio_frame
VAD frame extraction
AudioBuffer snapshot
normalizer
transcribe_cpp input
```

Capture:

- bytes copied;
- NumPy array sizes;
- allocation count;
- per-stage memory bandwidth if possible.

---

## 18.4 Benchmark D — Queue/backpressure

Artificially slow:

```text
translation
TTS
WebSocket sender
```

Measure:

- queue size over time;
- message drop count;
- RSS;
- latency;
- recovery after downstream resumes.

Acceptance criteria:

```text
RSS remains bounded
final subtitles remain deliverable
preview messages may be dropped
TTS may drop under configured policy
```

---

## 18.5 Benchmark E — Memory leak cycles

Use existing scenario:

```text
connect
 -> stream
 -> disconnect
 -> repeat 20–50 cycles
```

Record:

- RSS after each cycle;
- PyTorch allocated/reserved VRAM;
- OS thread count;
- asyncio task count.

The existing benchmark only uses a small number of cycles; extend the measurement count during real validation.

---

## 18.6 Benchmark F — Model switch race

While ASR is active:

```text
start inference
 -> request model switch
 -> unload
 -> load next
```

Repeat many times.

Acceptance:

- no native exception;
- no stale session;
- no corrupted output;
- no VRAM growth across swaps.

This benchmark should run only after a safe synchronization fix is designed, because intentionally stress-triggering a suspected race is not a substitute for fixing it.

---

## 18.7 Benchmark G — 1/2/4/8 session scaling

Use synthetic clients.

Measure:

```text
per-session latency
aggregate throughput
ASR lock wait
translation lock wait
TTS lock wait
CPU %
VRAM
RSS
thread count
queue depth
```

The goal is to identify the first shared-resource saturation point.

---

# 19. Recommended Implementation Order

## Phase 1 — Correctness / containment

1. Fix ASR model unload/swap lifecycle race.
2. Bound ASR token queue.
3. Enforce one capture owner in Firefox.
4. Keep debug audio dump disabled for production.

## Phase 2 — Highest performance ROI

5. Measure full-buffer ASR preview cost.
6. Implement persistent streaming ASR for models that support it.
7. Reduce repeated buffer conversion/copying.
8. Replace polling with event/new-audio-driven preview scheduling where safe.

## Phase 3 — TTS/transport

9. Benchmark TTS postprocessing vs generation.
10. Introduce binary TTS WebSocket transport if wire serialization is material.

## Phase 4 — Scale preparation

11. Add dedicated executors/resource classes.
12. Verify VAD engine concurrency.
13. Build explicit per-stage admission/backpressure policy.
14. Only then consider multi-session model pools.

---

# 20. Files To Change

## P0 / P1 primary files

### `backend_cpp/asr/model_manager.py`

**Functions / areas:**

```text
ASRModelManager.unload_shared_model()
ASRModelManager.ensure_model()
ASRModelManager.ensure_session()
_shared_lock
_shared_infer_lock
```

**Required change:**

Create one authoritative lifecycle protocol for:

```text
load
swap
unload
inference
```

No close/unload may overlap native inference.

---

### `backend_cpp/asr/transcribe_engine.py`

**Functions / areas:**

```text
_get_queue()
_run_inference()
_partial_preview_poller()
_commit_async()
_commit_sync()
stream_tokens()
cleanup()
```

**Required change:**

1. Bound/coalesce preview queue.
2. Replace full snapshot/re-infer preview with persistent incremental streaming where model supports it.
3. Keep final commit priority.
4. Ensure cleanup cancels the correct native work lifecycle.

---

### `backend_cpp/asr/audio_buffer.py`

**Functions / areas:**

```text
get_snapshot_with_version()
get_snapshot_if_newer()
feed_bytes()
pop_all()
slice_after()
```

**Required change:**

Introduce delta-aware/ring-buffer semantics if native ASR input permits.

Do not remove versioning; it is useful.

---

### `backend_cpp/vad/vad_processor.py`

**Functions / areas:**

```text
feed_chunk()
```

**Required change:**

Potentially reduce explicit per-frame allocations.

First verify that VAD callback/engine latency actually matters relative to ASR.

---

### `backend_cpp/vad/engines.py`

**Functions / areas:**

```text
FireRedOfficialVADEngine.is_speech()
SileroOfficialVADEngine.is_speech()
FsmnOfficialVADEngine.is_speech()
VADEngineFactory.get_engine()
```

**Required change:**

Establish/verify native thread-safety of the shared engine object.

This is mainly a future multi-session concern.

---

### `backend_cpp/translation/local_translator.py`

**Functions / areas:**

```text
translate()
_infer()
reconfigure()
unload_model()
```

**Required change:**

Only after benchmark:

- decide whether timeout semantics should be revised;
- consider dedicated executor;
- avoid hidden background native inference after caller timeout.

---

### `backend_cpp/utils/audio_dumper.py`

**Functions / areas:**

```text
dump_ingress_chunk()
dump_vad_utterance()
dump_asr_input()
```

**Required change:**

Diagnostic-only bounded write queue if this feature is needed for long runs.

Otherwise **DO NOT TOUCH**.

---

## Firefox extension

### `extension_firefox/popup/popup.js`

**Functions / areas:**

```text
broadcastToFrames()
START_TRANSLATION handler
STOP_TRANSLATION handler
```

**Required change:**

Find one capture owner instead of `allFrames: true` for START.

---

### `extension_firefox/content/content-script.js`

**Functions / areas:**

```text
startCapture()
cleanup()
handleSubtitleEvent()
```

**Required change:**

Add explicit session ownership / duplicate-start guard.

---

### `extension_firefox/lib/audio-capture.js`

**Functions / areas:**

```text
_setupScriptProcessor()
onaudioprocess
```

**Required change only after benchmark:**

Move actual active capture path to the existing AudioWorklet design or another worker-side capture strategy.

The archive contains `lib/audio-processor.js`, but the manifest does not load it as a content-script and `AudioCapture` does not instantiate an `AudioWorkletNode`, so it is currently not the active path.

---

### `extension_firefox/background/service-worker.js`

**Functions / areas:**

```text
SEND_BINARY
```

**Required change only after measurement:**

Use transferable message payloads where Firefox API semantics allow, and avoid unnecessary bridge copies.

---

### `backend_cpp/tts/audio_processor.py`

**Functions / areas:**

```text
convert_to_numpy()
normalize_audio()
encode_wav_to_base64()
apply_time_stretch()
```

**Required change only after TTS profiling:**

Reduce redundant CPU copies; consider binary audio transport.

---

# 21. MUST FIX / SHOULD OPTIMIZE / OPTIONAL / DO NOT TOUCH

## MUST FIX

### 1. ASR model lifecycle race

Reason:

```text
native inference can overlap model close
```

This is correctness-critical.

### 2. Unbounded ASR token queue

Reason:

```text
potential unbounded memory growth
```

### 3. Single-session enforcement in extension

Reason:

```text
current `allFrames` broadcast can violate the application invariant
```

---

## SHOULD OPTIMIZE

1. true incremental ASR stream;
2. reduce whole-utterance snapshot/reconversion;
3. event-driven preview scheduling;
4. isolate blocking executors;
5. binary TTS transport;
6. VAD allocation reduction.

---

## OPTIONAL

1. minor wrapper cleanup;
2. small dict/object reductions;
3. catalog caching for UI endpoints;
4. console log frequency tweaks.

---

## DO NOT TOUCH

### `SafeWebSocketConnection._send_lock`

The lock exists for a valid reason: outbound WebSocket sends from multiple tasks must be serialized.

### ASR commit-priority logic

The current:

```text
commit waits
 -> preview skips
```

policy is sensible for realtime UX.

### Bounded VAD buffer

The bound protects against uncontrolled growth.

Do not make it unbounded just to “avoid dropped audio.”

### TTS singleton infer lock

For one session it is a safe/simple resource boundary.

Do not introduce concurrent TTS generation without proving VRAM and model safety.

### Audio debug dumping default

`dump_audio=False` is the correct production default.

---

# 22. Root Cause Matrix

| Root cause | Symptoms |
|---|---|
| Full-buffer preview architecture | repeated CPU work, repeated GPU work, increasing preview cost |
| Global ASR lock | serialization across sessions |
| Global translation singleton/lock | translation queueing under concurrency |
| Global TTS singleton/lock | TTS serialization |
| Unbounded ASR token queue | possible RAM growth |
| `allFrames: true` start broadcast | duplicate sessions/capture |
| Multiple representation boundaries | extra copies and allocations |
| Base64 JSON TTS | payload and memory expansion |
| Default executor shared by all blocking tasks | tail-latency spikes |
| Single-thread debug dumper | I/O backlog under slow disk |
| Polling model | deterministic hidden latency |
| Native cancellation limitation | timeout does not mean compute stopped |

---

# 23. Quantification Notes

The source allows some exact structural quantities, but not real performance percentages.

## Exact from source

- Browser capture chunk: **1024 samples**
- Browser target sample rate: **16 kHz**
- Browser chunk duration: **~64 ms**
- FireRed native frame: **400 samples / 25 ms**
- Silero native frame: **512 samples / 32 ms**
- FSMN native frame: **960 samples / 60 ms**
- ASR preview poll interval: **350 ms**
- Sentence stability duration: **800 ms**
- ASR max sentence duration: **8 s**
- VAD raw buffer cap: **3 s**
- Translation queue: **20**
- TTS queue: **10**
- ASR preview queue: **unbounded**
- TTS sample rate: **24 kHz**
- TTS default inference steps: **8**
- ASR worker native threads setting: **4**
- commit fallback executor: **2 threads**
- audio dumper executor: **1 thread**

These are source facts, not benchmark claims.

## Cannot be honestly quantified from source alone

- actual ASR ms;
- actual VAD ms;
- actual translation ms;
- actual TTS ms;
- GPU occupancy;
- GPU kernel utilization;
- VRAM peak including native allocator;
- exact browser structured-clone copy count;
- network latency;
- p99 end-to-end latency.

All require runtime measurement.

---

# 24. Final Architecture Verdict

## For the stated current use case — one Windows desktop session

**Verdict: workable, but the ASR preview implementation is over-processing audio and the ASR lifecycle has a correctness hole.**

The system is not primarily suffering from “not enough threads.”

It is primarily suffering from:

```text
repeated data processing
+
serialized model execution
+
buffer/message lifecycle duplication
```

The best optimization strategy is therefore:

```text
1. fix correctness boundaries
2. make ASR incrementally process new audio
3. bound/coalesce preview data
4. reduce representations/copies
5. keep optional branches (TTS/debug) isolated
```

---

# 25. Recommended Next Audit/Implementation Gate

Before any code change, collect these runtime numbers from a **single-session, real audio** run:

```text
ASR:
  preview infer p50/p90/p99
  commit infer p50/p90/p99
  preview count / minute
  commit count / minute
  ASR lock wait p50/p90/p99
  samples processed / inference

VAD:
  frame processing p50/p90/p99
  callback time
  overflow_count

Translation:
  queue_wait p50/p90/p99
  infer p50/p90/p99
  lock_wait p50/p90/p99

TTS:
  queue_wait p50/p90/p99
  synthesis p50/p90/p99
  RTF
  output bytes

System:
  RSS baseline/peak
  torch allocated/reserved VRAM
  OS threads
  asyncio tasks
  CPU %
  GPU utilization
```

The existing `tests/perf_benchmark.py` and `run_perf_test.py` already provide a useful starting framework, including conversational, continuous-speech, burst/lock-contention, and leak-detection scenarios. They should be extended rather than replaced wholesale.

---

# 26. Bottom Line

### Highest-risk correctness issue

**ASR model unload/swap can overlap an active native inference.**

### Highest-impact performance issue

**ASR repeatedly snapshots, converts, normalizes, and re-infers the entire growing utterance every preview cycle.**

### Highest-risk memory issue

**Unbounded ASR token queue; debug audio dumper is another unbounded queue when enabled.**

### Highest-impact scaling issue

**ASR + translation + TTS are all singleton/serialized model resources.**

### Highest-impact extension issue

**`allFrames: true` START can violate the one-session requirement.**

### Best practical optimization path

```text
P0 safety
  -> single-session enforcement
  -> bounded preview queue
  -> ASR model lifecycle barrier
        |
        v
P1 performance
  -> persistent ASR stream
  -> delta audio / fewer copies
  -> event-driven preview
        |
        v
P2 transport
  -> binary TTS audio
  -> dedicated executors
        |
        v
Scale only if needed
  -> resource scheduler / model pool
```

**Audit status: STOP — no source code was changed.**
