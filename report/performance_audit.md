# PERFORMANCE & ARCHITECTURE AUDIT REPORT
# Vibe Translation Addon — backend_cpp

**Date**: 2026-09-11  
**Auditor**: Senior Software Architect + Performance Engineer  
**Hardware Target**: AMD Ryzen 5600X (6C/12T), 64 GB RAM, NVIDIA RTX 5060 Ti 16 GB VRAM  
**Models**: FireRed-VAD (CPU) · qwen3-asr-1.7b (GPU via Vulkan/CUDA) · Hunyuan-MT2 7B GGUF (GPU) · OmniVoice TTS (GPU)  
**Max sessions**: 1 (by design, `WSConfig.max_sessions = 1`)

---

## 1. EXECUTIVE SUMMARY

| Category | Count |
|---|---|
| **P0 — Critical** | 3 |
| **P1 — High** | 9 |
| **P2 — Medium** | 8 |
| **P3 — Low** | 6 |
| **Total issues** | 26 |

### Largest bottleneck (measured)
**Translation stage: avg 478 ms / sentence** (Hunyuan-MT2 7B GGUF, ~64 tok/s).  
This is 85–95% of the full ASR→subtitle latency of ~480–625 ms.

### 3 changes with highest ROI

| Rank | Change | Expected gain | Risk |
|---|---|---|---|
| 1 | Enable `preview_min_growth_ratio = 0.5` (already coded, off by default) | Cut preview ASR inference −49%, reduce RTF spike frequency | Low |
| 2 | Switch default VAD from `firered-vad` → `fsmn-vad` (already coded, mislabelled) | VAD CPU −47% (108 ms/s → 58 ms/s per audio-second) | Low |
| 3 | Eliminate redundant `bytes(bytearray)` copy in `audio_buffer.py` `get_snapshot_*` | −1 large allocation per preview poll (~350 ms cadence) | Low |

> **Note**: The config file comments indicate these were benchmarked but the default `vad_engine` is set to `"firered-vad"` (line 94 config.py) despite the comments recommending `fsmn-vad`. This is a **confirmed config inconsistency**.

---

## 2. ARCHITECTURE HIỆN TẠI

```
PROCESS: backend_cpp (uvicorn / asyncio event loop)

  asyncio event loop (main thread)
  |
  +--[WebSocket receive loop]
  |       |
  |       v  asyncio.to_thread()
  |  _process_binary_chunk()          [worker thread]
  |       |
  |       v
  |  parse_audio_frame()             [struct.unpack + JSON decode]
  |       |
  |       v
  |  VADProcessor.feed_chunk()       [threading.Lock held for frame loop]
  |    [firered/fsmn/silero engine]
  |       |
  |       +--on_speech_start() --> TranscribeEngine._is_speech_active=True
  |       |
  |       +--on_speech_chunk(bytes) --> AudioBufferManager.feed_bytes()
  |       |                              [bytearray, threadsafe]
  |       |
  |       +--on_speech_end() --> asyncio.run_coroutine_threadsafe(_commit_async)
  |                               [_shared_infer_lock BLOCKING]
  |
  +--[_partial_preview_poller]       [asyncio.Task, every 350ms]
  |       |
  |       v  asyncio.to_thread()
  |  _run_inference()                [_shared_infer_lock NON-BLOCKING]
  |       |
  |       v
  |  _token_queue                    [asyncio.Queue maxsize=64, latest-wins preview]
  |       |
  +--[_stream_asr_tokens]            [asyncio.Task]
  |       |
  |       v
  |  translation_queue               [asyncio.Queue maxsize=20]
  |       |
  +--[_translation_worker]           [asyncio.Task]
  |       |
  |       v  asyncio.to_thread()
  |  LocalGGUFTranslator.translate() [threading.RLock]
  |       |
  |       v
  |  tts_queue                       [asyncio.Queue maxsize=10]
  |       |
  +--[_tts_worker]                   [asyncio.Task]
  |       |
  |       v  asyncio.to_thread()
  |  OmniVoiceTTS.synthesize_sync()  [threading.Lock]
  |       |
  |       v
  +-- WebSocket send (JSON + base64 audio)

SHARED SINGLETONS (global, cross-session):
  ASRModelManager._shared_model    (transcribe_cpp.Model in VRAM)
  ASRModelManager._shared_session  (transcribe_cpp.Session)
  LocalGGUFTranslator._instance    (llama_cpp.Llama in VRAM)
  OmniVoiceTTS._instance           (PyTorch OmniVoice in VRAM)
  VADEngineFactory._engines        (dict: name -> singleton engine)

THREAD POOLS:
  uvicorn asyncio executor:    ThreadPoolExecutor (default)
  _SYNC_COMMIT_EXECUTOR:       max_workers=2 (fallback commit path)
  _DUMP_EXECUTOR:              max_workers=1 (audio debug only)
```

---

## 3. DATA FLOW

```
Browser Extension (16kHz 16-bit mono PCM)
    |
    | WS binary frame: [4-byte header_len][JSON header][PCM bytes]
    v
parse_audio_frame()
    | returns: (pcm_bytes: bytes, capture_ts, chunk_idx)
    | COPY 1: data[4+N:] -- slice = zero-copy view
    v
VADProcessor.feed_chunk(pcm_bytes)
    | COPY 2: raw_buf.extend(pcm_bytes) -- bytearray extend = COPY
    |
    | [per frame, 400 samples / 25ms for firered/fsmn]
    | COPY 3: np.frombuffer(raw_buf, np.int16, offset=) -- ZERO-COPY view
    | COPY 4: samples_int16.astype(np.float32) / 32768.0 -- ALLOCATION
    |
    | [speech frame callback]
    | COPY 5: frame_bytes = bytes(raw_buf[a:b]) -- COPY
    |
    | [VAD engine inference, CPU]
    | firered: COPY 6: float32 -> clip -> *32767 -> astype(int16)
    | fsmn:    COPY 7: torch.from_numpy(float32) -- zero-copy shared
    v
on_speech_chunk(bytes) -> AudioBufferManager.feed_bytes()
    | COPY 8: _bytes_buffer.extend(pcm_bytes) -- COPY into bytearray
    v
[every 350ms: preview poller]
AudioBufferManager.get_snapshot_if_newer()
    | COPY 9:  bytes(self._bytes_buffer) -- UNNECESSARY: bytearray->bytes
    | COPY 10: np.frombuffer(raw_bytes, np.int16) -- zero-copy of COPY 9
    | COPY 11: .astype(np.float32) / 32768.0 -- ALLOCATION
    | COPY 12: bytearray(self._frame_states) -- copy of frame states
    | COPY 13: np.frombuffer(bytes(states[:N])) -- UNNECESSARY double-copy
    v
SpeechNormalizer.process(pcm_snapshot)
    | COPY 14: np.asarray(pcm, float32).copy() -- EXPLICIT defensive copy
    | x *= gain       [in-place]
    | COPY 15: np.clip(x, -1, 1) -- returns NEW array, not in-place
    v
transcribe.cpp session.stream()/run(pcm_float32)  [GPU inference]
    | returns: raw_text: str
    v
_token_queue (asyncio.Queue, maxsize=64)
    v
translation_queue (asyncio.Queue, maxsize=20)
    v
LocalGGUFTranslator.translate()  [GPU via llama_cpp]
    | returns: cleaned translated text
    v
WebSocket send_json (subtitle appears on browser)
    v
[tts_enabled] -> tts_queue (asyncio.Queue, maxsize=10)
    v
OmniVoiceTTS.synthesize_sync()  [GPU via PyTorch]
    | COPY 16: tensor.detach().cpu().numpy().astype(float32) -- GPU->CPU + cast
    | COPY 17: np.clip() in normalize_audio -- returns new array
    | COPY 18: sf.write(BytesIO) -- WAV encode
    | COPY 19: base64.b64encode() -- base64 encode
    v
WebSocket send_json (TTS audio as base64 WAV)
```

---

## 4. CRITICAL PATH

```
Speaker stops talking
    |
    v [silence_duration_ms = 250ms default]
VAD silence accumulation: 150ms - 480ms wait

    |
    v [VAD END detected]
on_speech_end() -> _commit_async() -> asyncio.to_thread(_run_inference)

    |
    +-- [wait _shared_infer_lock] -- if preview running: up to 141ms wait
    |
    v [avg 134ms, max 231ms]
transcribe.cpp GPU inference

    |
    v [~0ms queue]
translation_queue.put_nowait()

    |
    v [avg 478ms, max 624ms] <-- DOMINANT STAGE
LocalGGUFTranslator (Hunyuan-MT2 7B, 64 tok/s)

    |
    v
WebSocket send -> subtitle on browser

MEASURED E2E (ASR commit -> subtitle): avg=479ms, p90=608ms, max=625ms
FULL pipeline (silence+ASR+translation): ~730ms - 1100ms
```

---

## 5. BOTTLENECK TABLE

| Priority | Component | Problem | Evidence | Impact | Recommendation |
|---|---|---|---|---|---|
| P0 | config.py:94 | vad_engine default = "firered-vad" contradicts benchmarked comment recommending "fsmn-vad" | firered=108ms/s vs fsmn=58ms/s (code comments, measured) | +47% VAD CPU waste | Change default to "fsmn-vad" |
| P0 | audio_buffer.py:88,117,145 | bytes(self._bytes_buffer) unnecessary conversion on every snapshot | Code: raw_bytes = bytes(self._bytes_buffer) in 3 methods | 1 O(N) allocation per preview every 350ms | Use memoryview or bytearray directly with np.frombuffer |
| P0 | speech_normalizer.py:97 | np.asarray(pcm, float32).copy() explicit copy always, even when pcm already clean float32 | sanitize_input() line 97 | 1 large alloc per inference (preview+commit) | Add conditional: skip copy if already float32+finite+C_CONTIGUOUS |
| P1 | config.py:178 | preview_min_growth_ratio = 0.0 (gate off) | perf: 55,620ms preview vs 19,620ms commit = 2.84x amplification, gate implemented but disabled | 49% excess ASR inference; worsens lock contention | Enable ratio=0.5 after benchmark |
| P1 | vad_processor.py:202 | Per-frame astype(np.float32) allocation in hot VAD loop | Line 202-204: ~40 frames/sec during speech | ~40 allocs/sec x 1600 bytes = 64KB/sec GC pressure | Pre-allocate self._frame_f32_buf |
| P1 | vad_processor.py:234,250,261,274 | bytes(raw_buf[a:b]) copy per speech frame | Lines 234,250,261,274 in frame loop | ~40 copies/sec x 800 bytes during speech | Pass (offset, length) to callback |
| P1 | engines.py:188 | firered per-frame float32->int16 conversion: clip*32767.astype(int16) | FireRedOfficialVADEngine.is_speech():188 | Extra alloc per frame with firered engine | Switch to fsmn-vad (avoids entirely) |
| P1 | transcribe_engine.py:49 | _SYNC_COMMIT_EXECUTOR max_workers=2, global, never sized to workload | Global ThreadPoolExecutor definition | Arbitrary sizing; may contend | NEEDS BENCHMARK: profile under load |
| P1 | local_translator.py:116-117 | _load_lock + _infer_lock both RLock; reconfigure() holds both nested | reconfigure() lines 91-104 | Nested lock with potential reconfiguration during inference | Convert to single lock + condition |
| P1 | perf_profiler.py:152 | Every record_metric() acquires threading.Lock | Line 152: with self._lock in every call | Lock on hot path (per inference, per VAD frame) | Lock-free per-thread buffers |
| P2 | audio_buffer.py:101,130,160 | bytes(states[:N]) double-copy for frame_state array | np.frombuffer(bytes(states[:N])) pattern | Extra small alloc per snapshot | np.frombuffer(states, count=N) directly |
| P2 | speech_normalizer.py:243 | np.clip() returns new array instead of in-place | Line 243: x = np.clip(x, -1.0, 1.0) | 1 extra large alloc per inference | np.clip(x, -1.0, 1.0, out=x) |
| P2 | transcribe_engine.py:788-800 | SNR percentile computation (np.reshape, np.percentile) on every commit | Lines 788-800 in on_speech_end() | Non-trivial on short utterances, runs every commit | Make conditional on config flag |
| P2 | ws_handler.py:325 | logger.info() with full translated text on every translation | Line 325 | String formatting in hot path | Use logger.debug() |
| P2 | ws_handler.py:368 | logger.info() TTS queued message on every sentence | Line 368 | String format in hot path | Use logger.debug() |
| P2 | transcribe_engine.py:411 | Accesses asyncio.Queue._queue private attribute | getattr(q, "_queue", None) line 411 | CPython implementation detail; may break on Python version upgrade | Use explicit bounded deque for preview |
| P2 | engines.py:385-394 | FsmnVADEngine wraps tensor in list for model.generate() | Line 385-394: input=[tensor_chunk] | Potential list overhead per frame | NEEDS BENCHMARK: profile FunASR input |
| P2 | transcribe_engine.py:509 | normalize_speech() static creates new SpeechNormalizer() each call | Line 517: normalizer = SpeechNormalizer(...) | Allocates throwaway object; used only for compat | Mark deprecated; use instance normalizer |
| P3 | ws_handler.py:210-228 | Two-layer function wrapping: _process_binary_chunk + _handle_binary_message | Lines 210, 226 | Minor call overhead | Merge into single function |
| P3 | audio_buffer.py:63 | [vad_state]*new_frames creates temporary list every feed_bytes | Line 63 | Minor list alloc per chunk | bytes([vad_state]) * new_frames |
| P3 | vad_processor.py:170 | callbacks = [] allocated fresh every feed_chunk() call | Line 170 | Minor allocation | Pre-allocate or reuse cleared instance list |
| P3 | local_translator.py:246 | _async_load_lock created lazily via hasattr() check | Lines 246-247 | hasattr() called on every translate() even when model loaded | Initialize in __init__ |
| P3 | omnivoice_engine.py:57 | _voice_prompt_cache unbounded dict | Line 57 | Grows per unique voice+text pair; no TTL | Add LRU/maxsize limit |
| P3 | audio_processor.py:92-99 | Phase vocoder time-stretch: Python for-loop per STFT frame | Lines 92-99 | High CPU when speed != 1.0 for long audio | Replace with scipy.phase_vocoder or librosa |
| P3 | perf_profiler.py:413-415 | perf.record_metric() on every TTS dequeue | Lines 413-415 | Lock overhead; minor | Batch or reduce |
| P3 | ws_handler.py:413 | tts_queue_wait_ms recorded always even when not in debug mode | Line 413-415 | Lock per TTS item | Acceptable; document |

---

## 6. CPU BOTTLENECKS

**firered-vad CPU cost (measured)**: 108.7 ms per audio-second  
**fsmn-vad CPU cost (measured)**: 57.9 ms per audio-second (-47%)  
**silero-vad CPU cost (measured)**: 13.3 ms per audio-second (-88% vs firered)

Default is `firered-vad` (config.py:94) despite the comment saying fsmn is preferred. This is wasting ~50ms/audio-second of CPU.

**Per-frame allocations in VAD hot loop**:  
- 40 frames/sec during speech (25ms firered/fsmn frames)
- Each frame: 1 allocation of 1600 bytes (float32 array)
- = 64 KB/sec of allocations just for VAD frame conversion

**SpeechNormalizer double allocation per inference**:  
- sanitize_input(): `.copy()` → 1 allocation of size N
- process() end: `np.clip()` → 1 more allocation of size N
- N = utterance length in float32 samples (typically 16,000-48,000 samples = 64-192 KB)
- Frequency: every preview (every 350ms) + every commit

---

## 7. GPU / VRAM BOTTLENECKS

**Measured VRAM**: 2034 MB allocated, 2104 MB reserved out of 16384 MB total.

> WARNING: If ASR uses Vulkan backend (not CUDA), torch.cuda.memory_allocated() does NOT count ASR VRAM. True VRAM usage may be higher. Verify with `nvidia-smi` during active session.

**ASR + Translation serialization**:  
- ASR uses `_shared_infer_lock` (threading.Lock)
- Translation uses `_infer_lock` (threading.RLock)
- They do NOT share a lock → can overlap on GPU (scheduler handles)
- No explicit GPU serialization bug found

**TTS CUDA sync on every synthesis**:  
`torch.cuda.synchronize()` in synthesize_sync() adds a blocking GPU sync point. Minor for single-session; harmless as timing is already awaited.

**TTS tensor dtype chain**:  
Model outputs float16 → `.detach().cpu().numpy()` = CPU float16 → `.astype(float32)` = extra copy.  
Fix: `.to(dtype=torch.float32).cpu().numpy()` combines conversion steps.

---

## 8. MEMORY BOTTLENECKS

```
RAM at session start:  6,101 MB
RAM at session end:    7,755 MB
Delta:                +1,654 MB (ONE-TIME model load — EXPECTED)
Cross-session delta:  +0.03 MB -> NO MEMORY LEAK
```

**AudioBufferManager**: max 60s audio = 1.92 MB — bounded, no growth risk.  
**VAD raw_buffer**: max 3s audio = 96 KB — bounded.  
**_token_queue**: maxsize=64 — bounded.  
**translation_queue**: maxsize=20 — bounded.  
**tts_queue**: maxsize=10 — bounded.

**One unbounded structure**: `OmniVoiceTTS._voice_prompt_cache` — grows per unique (voice_path, text) pair. In practice bounded to 1-2 voices per session. Add LRU maxsize as precaution.

---

## 9. I/O BOTTLENECKS

**Audio debug dumper**: disabled by default (`dump_audio=False`). Zero cost in production. When enabled: offloaded to single worker thread. No hot-path impact.

**Performance report dump on disconnect**: synchronous JSON+markdown write. Not on realtime path (only on cleanup). Acceptable.

**WebSocket overhead**: ~200-500 bytes per message JSON. At 3 previews/sec: ~1.5 KB/sec. Negligible.

---

## 10. LOCK / THREAD BOTTLENECKS

**Lock hierarchy (documented, verified correct)**:
```
_shared_lock -> _shared_infer_lock  (correct ordering)
NEVER: _shared_infer_lock -> _shared_lock  (would deadlock)
```

**ASR inference lock contention (MEASURED)**:
```
asr.lock_wait_ms: avg=4.3ms, p50=0.01ms, p90=0.01ms, p99=100ms, max=141ms
ALERT FIRED: "High lock contention on asr.lock_wait_ms: 141.5ms"
```
Root cause: preview holds `_shared_infer_lock` for ~134ms avg. Commit waits up to full preview inference time.  
Mitigation already in place: `_commit_waiting` counter + preview skips. But current in-flight preview cannot be aborted.

**Translation lock**: avg=0.0ms wait. Sequential single-worker. Perfect.

**VADProcessor._lock**: Held for entire frame processing loop (includes engine inference ~2-5ms/frame). Callbacks correctly executed outside lock.

---

## 11. QUEUE / BACKPRESSURE

| Queue | Size | Producer rate | Consumer rate | Backlog risk? |
|---|---|---|---|---|
| _token_queue | 64 (latest-wins preview) | ~3/sec preview + 1-4/min final | immediate (async loop) | None |
| translation_queue | 20 | ~1-4/min | 1 per 478ms | Very low; 20 x 478ms = 9.6s buffer |
| tts_queue | 10 | ~1 per 478ms | 1 per 495ms | Balanced; minimal |

All queues have bounded backpressure. At normal speech cadence (<1 sentence/sec), no queue starvation or backlog expected.

---

## 12. AUDIO COPY / CONVERSION AUDIT

| # | From | To | Method | Copy? | Frequency | Necessary? | Recommendation |
|---|---|---|---|---|---|---|---|
| 1 | WS frame bytes | pcm slice | data[4+N:] | ZERO-COPY (slice) | Every chunk | N/A | OK |
| 2 | pcm bytes | VAD raw_buf | bytearray.extend() | COPY | Every chunk | YES | Unavoidable |
| 3 | raw_buf | int16 view | np.frombuffer(offset=) | ZERO-COPY | Per frame | N/A | OK |
| 4 | int16 view | float32 | .astype(float32) | COPY+ALLOC | Per frame ~40/sec | YES | Pre-allocate buffer |
| 5 | raw_buf slice | bytes | bytes(raw_buf[a:b]) | COPY | Per speech frame | PARTIAL | Pass offset+length |
| 6 | float32 | int16 | clip*32767.astype(int16) | COPY | Per frame (firered only) | YES for firered | Switch to fsmn-vad |
| 7 | float32 | torch CPU tensor | torch.from_numpy() | ZERO-COPY | Per frame | YES | OK |
| 8 | pcm bytes | AudioBuffer bytearray | bytearray.extend() | COPY | Per speech frame | YES | Unavoidable |
| 9 | bytearray | bytes | bytes(self._bytes_buffer) | UNNECESSARY COPY | Every 350ms snapshot | NO | Use memoryview |
| 10 | bytes (9) | int16 view | np.frombuffer() | ZERO-COPY of 9 | Every 350ms | N/A | Eliminated with fix 9 |
| 11 | int16 view | float32 | .astype(float32) | COPY+ALLOC | Every 350ms | YES | Acceptable |
| 12 | frame_states bytearray | bytearray | bytearray(states) | COPY | Every 350ms | PARTIAL | Use memoryview |
| 13 | states bytearray | bytes | bytes(states[:N]) | UNNECESSARY | Every 350ms | NO | np.frombuffer(states,count=N) |
| 14 | float32 snapshot | float32 copy | np.asarray().copy() | EXPLICIT COPY | Every inference | CONDITIONAL | Skip if already float32+finite |
| 15 | float32 gained | float32 clipped | np.clip() | COPY+ALLOC | Every inference | NO | np.clip(x,-1,1,out=x) |
| 16 | GPU float16 tensor | CPU float32 numpy | .detach().cpu().numpy().astype() | COPY (GPU->CPU) + CAST | Every TTS | YES | .to(float32).cpu().numpy() |
| 17 | float32 audio | normalized float32 | np.clip() in normalize_audio | COPY | Every TTS | NO | In-place clip |
| 18 | float32 audio | WAV bytes | sf.write(BytesIO) | COPY | Every TTS | YES | Unavoidable |
| 19 | WAV bytes | base64 string | base64.b64encode() | COPY | Every TTS | YES | Unavoidable |

**Unnecessary copies**: #9, #13, #15, #17 — all fixable with minimal risk.

---

## 13. LATENCY BUDGET

| Stage | Measured/Estimated Cost | Blocking? | Critical Path? |
|---|---|---|---|
| Browser audio capture + WS send | ~20-50ms (browser) | Yes | Upstream |
| WS receive + frame parse | ~1-2ms | No | No |
| asyncio.to_thread dispatch | ~0.1-1ms | No | No |
| VAD feed_chunk (per chunk) | ~2-5ms | Yes (thread) | No |
| Silence accumulation | **150-480ms (config=250ms)** | Yes | YES — Knob #1 |
| ASR infer lock wait | avg=4ms, max=141ms | Yes | Worst case |
| SpeechNormalizer | ~1-3ms | Yes (in thread) | Minimal |
| transcribe.cpp GPU inference | **avg=134ms, max=231ms** | Yes | YES |
| Token queue enqueue | <0.1ms | No | No |
| Translation queue enqueue | <0.1ms | No | No |
| Translation inference | **avg=478ms, max=624ms** | Yes | YES — DOMINANT |
| WS send (subtitle) | ~1-2ms | No | No |
| TTS queue enqueue | <0.1ms | No | Parallel |
| TTS inference | **avg=495ms, max=513ms** | Yes | Parallel to next utterance |
| TTS base64 + WS send | ~5-20ms | No | Parallel |
| **E2E ASR->Subtitle (measured)** | **avg=479ms, p90=608ms, max=625ms** | — | — |

---

## 14. SCALABILITY ANALYSIS

Design constraint: max_sessions=1. This analysis is for hypothetical future expansion.

**1 session (current)**: All resources dedicated. GPU handles 3 models sequentially. CPU idle between utterances. Optimal.

**2 sessions (hypothetical)**:
- ASR: _shared_infer_lock global -> serialized. Each session waits ~134ms avg per utterance. Latency doubles.
- Translation: _infer_lock singleton -> serialized. Latency doubles.
- TTS: _infer_lock singleton -> serialized. Latency doubles.
- VAD: OK — per-session state isolated, model shared read-only.
- VRAM: models shared, state per-session -> VRAM OK (~2034 MB unchanged).

**4 sessions**: 4x serialization. Translation alone = ~1900ms avg. Not viable.

**8+ sessions**: Not scalable with current single-instance architecture.

**Conclusion**: The single-session design is correct and optimal for the stated use case (personal desktop, single browser tab). Multi-session would require batched LLM inference or per-session model instances, both requiring major architectural changes.

---

## 15. TOP 10 OPTIMIZATIONS

| Rank | Optimization | File | Effort | Perf Gain | Risk |
|---|---|---|---|---|---|
| 1 | Fix VAD default: "firered-vad" -> "fsmn-vad" | config.py:94 | 1 line | -47% VAD CPU | Low |
| 2 | Enable preview growth gate ratio=0.5 | config.py:178 | 1 line | -49% preview inferences | Low |
| 3 | Eliminate bytearray->bytes copy in AudioBuffer snapshots | audio_buffer.py:88,117,145 | ~15 lines | -1 large alloc/350ms | Low |
| 4 | In-place np.clip in SpeechNormalizer | speech_normalizer.py:243 | 1 line | -1 alloc/inference | Low |
| 5 | Conditional skip .copy() in sanitize_input | speech_normalizer.py:97 | ~5 lines | -1 alloc/inference | Low |
| 6 | Eliminate frame_state bytes() intermediate | audio_buffer.py:101,130,160 | ~6 lines | -1 alloc/350ms | Low |
| 7 | Pre-allocate VAD float32 frame buffer | vad_processor.py:202 | ~10 lines | -40 allocs/sec during speech | Medium |
| 8 | TTS tensor dtype consolidation | audio_processor.py:39 | 1 line | -1 copy in TTS path | Low |
| 9 | Move _async_load_lock to __init__ | local_translator.py:246 | 2 lines | Eliminate hasattr() in hot path | Low |
| 10 | Add LRU maxsize to voice_prompt_cache | omnivoice_engine.py:57 | ~5 lines | Prevent unbounded growth | Low |

---

## 16. QUICK WINS (each < 5 min)

1. `config.py:94`: `"firered-vad"` → `"fsmn-vad"` — 1 line
2. `speech_normalizer.py:243`: `np.clip(x, -1.0, 1.0, out=x)` — 1 line  
3. `local_translator.py` `__init__`: Add `self._async_load_lock = asyncio.Lock()` — 2 lines
4. `audio_processor.py:39`: `.to(dtype=torch.float32).cpu().numpy()` — 1 line
5. `audio_processor.py:120`: `np.clip(normalized, ..., out=normalized)` — 1 line
6. `config.py:178`: `preview_min_growth_ratio: float = 0.5` (after benchmark) — 1 line

---

## 17. STRUCTURAL IMPROVEMENTS

### 17.1 AudioBuffer: Eliminate bytearray->bytes copy
BEFORE: `raw_bytes = bytes(self._bytes_buffer)` then `np.frombuffer(raw_bytes, np.int16)`  
AFTER: Snapshot into `bytearray` (one copy for thread safety, then release lock), then `np.frombuffer(snapshot, np.int16)` directly without the intermediate `bytes()`.

The key insight: `np.frombuffer` accepts `bytearray` directly. The `bytes()` call is purely redundant.

### 17.2 SpeechNormalizer: Conditional sanitize_input
Add fast path: if `pcm.dtype == np.float32 and pcm.flags['C_CONTIGUOUS'] and np.all(np.isfinite(pcm))`: return `np.clip(pcm, -1.0, 1.0)` with `out=` pre-allocated buffer to skip `.copy()`.

### 17.3 VADProcessor: Pre-allocated frame buffer
```python
# In __init__:
self._frame_f32_buf = np.empty(self._frame_samples, dtype=np.float32)

# In feed_chunk() frame loop replace lines 202-204:
int16_view = np.frombuffer(raw_buf, dtype=np.int16,
                           count=self._frame_samples, offset=offset)
np.copyto(self._frame_f32_buf, int16_view, casting='unsafe')
self._frame_f32_buf /= 32768.0
samples_float32 = self._frame_f32_buf  # no allocation
```
Must re-allocate buffer if `_frame_samples` changes (engine switch in `update_config()`).

---

## 18. BENCHMARK PLAN

**B1: VAD default verify**
```bash
python scratch/compare_vad_quality.py
# Expect: firered=108ms/s, fsmn=58ms/s, silero=13ms/s
```

**B2: Preview gate at ratio=0.5**
```bash
# Set preview_min_growth_ratio = 0.5 in config.py
python backend_cpp/run_perf_test.py
# Expect: asr.preview_audio_ms reduced ~49%, final transcripts bit-identical
```

**B3: AudioBuffer copy elimination**
```bash
python -c "
import time, numpy as np
from backend_cpp.asr.audio_buffer import AudioBufferManager
buf = AudioBufferManager()
buf.feed_bytes(bytes(32000), 1)  # 1s audio
t0 = time.perf_counter()
for _ in range(1000): buf.get_snapshot_if_newer(0)
print(f'Snapshot rate: {1000/(time.perf_counter()-t0):.0f}/sec')
"
```

**B4: SpeechNormalizer sanitize timing**
```bash
python -c "
import numpy as np, time
from backend_cpp.asr.speech_normalizer import SpeechNormalizer
n = SpeechNormalizer()
pcm = np.random.randn(48000).astype(np.float32)
t0 = time.perf_counter()
for _ in range(1000): n.sanitize_input(pcm)
print(f'sanitize_input: {(time.perf_counter()-t0):.3f}s for 1000 calls')
"
```

**B5: Full E2E with wav_test files**
```bash
python backend_cpp/run_perf_test.py
# Capture: VAD, ASR, translation, TTS latency breakdown
```

**B6: True VRAM usage verification**
```bash
# During active session (separate terminal):
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 1
```

---

## 19. RECOMMENDED IMPLEMENTATION ORDER

**Phase 0 — Config Fix (immediate, zero risk)**
1. Fix vad_engine default in config.py:94
2. Move _async_load_lock to __init__ in local_translator.py
3. In-place np.clip in speech_normalizer.py:243 and audio_processor.py:120

**Phase 1 — Memory Quick Wins (< 1 day, low risk)**
4. Eliminate bytes(states[:N]) in audio_buffer.py
5. TTS tensor dtype: .to(float32).cpu().numpy()
6. Conditional skip .copy() in sanitize_input

**Phase 2 — Preview Gate (1 hour + benchmark)**
7. Set preview_min_growth_ratio = 0.5 in config.py:178
8. Run B2 benchmark; verify transcript quality
9. Update config comment

**Phase 3 — VAD Frame Buffer (1-2 days, medium risk)**
10. Pre-allocate float32 frame buffer in VADProcessor
11. Test with all 3 VAD engines
12. Measure CPU impact

**Phase 4 — AudioBuffer Structural (2-3 days, medium risk)**
13. Eliminate bytearray->bytes intermediate in get_snapshot* and pop_all
14. Add LRU maxsize to voice_prompt_cache
15. Replace custom phase vocoder with scipy/librosa

---

## 20. FILES TO CHANGE

```
backend_cpp/config.py
  - VADConfig.vad_engine:              "firered-vad" -> "fsmn-vad"        (line 94)
  - ASRConfig.preview_min_growth_ratio: 0.0 -> 0.5                         (line 178, after B2 benchmark)

backend_cpp/asr/audio_buffer.py
  - get_snapshot_with_version():  bytes(states[:N]) -> np.frombuffer(states,count=N)  (line 101)
  - get_snapshot_if_newer():      bytes(states[:N]) -> np.frombuffer(states,count=N)  (line 130)
  - pop_all():                    bytes(states[:N]) -> np.frombuffer(states,count=N)  (line 160)
  - get_snapshot_*:               bytes(self._bytes_buffer) -> bytearray copy + frombuffer (line 88,117,145)

backend_cpp/asr/speech_normalizer.py
  - sanitize_input():   add conditional skip when pcm already float32+finite+contiguous  (line 97)
  - process():          np.clip(x, -1.0, 1.0) -> np.clip(x, -1.0, 1.0, out=x)           (line 243)

backend_cpp/vad/vad_processor.py
  - __init__():         add self._frame_f32_buf = np.empty(frame_samples, dtype=float32) (after line 101)
  - feed_chunk():       replace lines 202-204 with pre-allocated buffer approach
  - update_config():    re-allocate _frame_f32_buf when _frame_samples changes

backend_cpp/translation/local_translator.py
  - __init__():         add self._async_load_lock = asyncio.Lock()   (around line 115)
  - translate():        remove hasattr() guard, use self._async_load_lock directly (line 246-247)

backend_cpp/tts/audio_processor.py
  - convert_to_numpy():  item.detach().to(torch.float32).cpu().numpy()   (line 39)
  - normalize_audio():   np.clip(normalized, -0.99, 0.99, out=normalized).astype(float32)  (line 120)

backend_cpp/tts/omnivoice_engine.py
  - _voice_prompt_cache:  replace dict with LRU cache (collections.OrderedDict maxsize=8)  (line 57)
```

---

## APPENDIX: MEASURED DATA (perf_report.json, 2026-09-11)

```
Session stats:
  Connected sessions:    1
  Audio chunks received: 260
  ASR total inferences:  43  (35 preview + 8 commit)
  Preview audio total:   55,620 ms (2.84x amplification vs commit audio)
  Commit audio total:    19,620 ms
  Translations:          7
  TTS synthesized:       7
  Short commits filtered: 1

ASR inference timing:
  avg=134.7ms, p50=143.2ms, p90=169.2ms, p99=224.8ms, max=231.2ms
  RTF: avg=0.09, max=0.26 (all < 1.0: faster than realtime)
  Lock wait: avg=4.3ms, p99=100ms, max=141.5ms

Translation timing:
  avg=478ms, p50=438ms, p90=607ms, p99=623ms, max=625ms
  Speed: avg=64.2 tok/s, min=63.6, max=65.5

TTS timing:
  avg=495ms, p50=499ms, p90=509ms, max=513ms
  RTF: avg=0.19 (5x faster than realtime)

Pipeline E2E:
  ASR-commit->subtitle: avg=479ms, p90=608ms, max=625ms
  Subtitle->TTS-audio:  avg=502ms, p90=515ms, max=518ms (parallel)

Memory:
  RAM session start:  6,101 MB
  RAM session end:    7,755 MB (+1,654 MB model load, not leak)
  Cross-session RAM delta: +0.03 MB -> NO MEMORY LEAK
  GPU VRAM allocated: 2,034 MB / 16,384 MB
  GPU VRAM reserved:  2,104 MB

Alerts triggered:
  1x BOTTLENECK_LOCK_CONTENTION: asr.lock_wait_ms = 141.5ms
```

---

**AUDIT COMPLETE. Awaiting confirmation before any code changes.**

*All numeric claims sourced from perf_report.json (runtime) or config.py/source code comments (benchmarks).*  
*Items marked "NEEDS BENCHMARK" require runtime measurement before implementation.*
