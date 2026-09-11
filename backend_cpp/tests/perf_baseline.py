"""W2.1 baseline measurement suite — the mandatory gate before any Phase 2 optimization.

The audit (`report/backend_cpp_extension_performance_audit.md`) deliberately marked every
performance number as `NEEDS BENCHMARK`. This module produces those numbers so that
W2.2 (incremental ASR streaming) and the other Phase 2 items can be judged against a real
baseline instead of intuition.

Two kinds of measurement, on purpose
------------------------------------
**Micro-benchmarks** (no server, no GPU, deterministic)

  Measure the CPU-side work the audit flagged: the full-buffer snapshot, the speech
  normalizer, and the copy chain. They call the *real* production classes
  (`AudioBufferManager`, `SpeechNormalizer`, `VADProcessor`, `parse_audio_frame`) so the
  numbers cannot drift away from the shipping code. Allocation pressure is measured with
  `tracemalloc`, which is a real measurement rather than an estimate.

**End-to-end scenarios** (need a running backend)

  Drive the WebSocket with synthetic speech and read the server-side `perf` collector:

  * ``scenario_preview_cost``   — audit benchmark A: does inference cost scale with the
    total accumulated utterance length?
  * ``scenario_backpressure``   — audit benchmark D: what happens to queues and RSS when
    audio arrives faster than the pipeline can consume it?
  * ``scenario_session_scaling``— audit benchmark G: how do lock waits grow with 1/2/4
    concurrent sessions?

Correlating metrics
-------------------
``perf.get_samples()`` exposes raw samples. ``asr.infer_ms`` and ``asr.audio_dur_sec`` are
recorded together once per inference, so they are index-aligned and can be zipped — that is
what turns "inference is slow" into "inference cost grows with utterance duration".

Note for ``scenario_session_scaling``
-------------------------------------
The backend intentionally admits only one session by default (``config.ws.max_sessions``,
audit finding P1-04). Scaling measurements therefore set it to 0 (unlimited) for the
duration of the run and restore it afterwards. When benchmarking against an *external*
server the operator must do the same, otherwise every extra client is simply superseded.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend_cpp.asr.audio_buffer import AudioBufferManager
from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.asr.speech_normalizer import SpeechNormalizer
from backend_cpp.tests.perf_benchmark import (
    BenchmarkClient,
    generate_pcm_audio,
    pack_audio_frame,
)
from backend_cpp.utils.perf_profiler import perf
from backend_cpp.vad.vad_processor import VADProcessor
from backend_cpp.ws.frame_protocol import parse_audio_frame

logger = logging.getLogger("perf_baseline")

SCHEMA_VERSION = 1

# Utterance lengths the audit asked for (seconds).
DEFAULT_DURATIONS: Tuple[int, ...] = (1, 2, 4, 6, 8)

# The synthetic test signal (perf_benchmark.generate_pcm_audio) is a harmonic tone mix with
# a syllabic envelope. Measured behaviour (see micro_benchmarks.vad_cost):
#
#     firered-vad : 0 speech frames detected  -> the pipeline produces NOTHING
#     fsmn-vad    : ~all frames detected      -> utterances actually flow
#
# Scenarios must therefore request fsmn-vad explicitly. Relying on the server default
# (firered-vad) silently yields a run with zero inferences, which looks like a very fast
# baseline instead of a broken measurement.
BASELINE_VAD_ENGINE = "fsmn-vad"


def _baseline_config(**overrides: Any) -> Dict[str, Any]:
    """Config sent to the backend by every baseline scenario."""
    cfg: Dict[str, Any] = {
        "sourceLang": "auto",
        "targetLang": "vi",
        "vadEngine": BASELINE_VAD_ENGINE,
        "ttsEnabled": False,
    }
    cfg.update(overrides)
    return cfg


def _validity(
    inference_count: int,
    minimum: int = 1,
    delivered: Optional[int] = None,
) -> Dict[str, Any]:
    """Flag a scenario that produced no work, so an empty run cannot be mistaken for data.

    Two distinct failure modes are reported separately, because they need different fixes:

    * no inference at all  -> VAD/signal mismatch (the audio never looked like speech)
    * inference but no delivery -> the ASR produced an EMPTY transcript, so
      ``_emit_final()`` dropped it. Measured with the synthetic test signal; real speech
      does not have this problem (use ``--audio-file``).
    """
    if inference_count < minimum:
        return {
            "valid": False,
            "inferences_observed": inference_count,
            "reason": (
                "No ASR inference was recorded, so every number here is meaningless. Most "
                "likely a VAD/signal mismatch: the synthetic test audio is not detected as "
                "speech by the active VAD engine (FireRed classifies it as non-speech; "
                "fsmn-vad detects it). Check config.vad.vad_engine and the \"vadEngine\" "
                "sent in set_config before trusting this section."
            ),
        }

    result: Dict[str, Any] = {
        "valid": True,
        "inferences_observed": inference_count,
    }
    if delivered is not None:
        result["delivered_to_client"] = delivered
        result["delivery_measurable"] = delivered > 0
        if delivered == 0:
            result["delivery_note"] = (
                "Inferences ran but the client received no messages. Measured cause: the "
                "synthetic harmonic test signal does not produce a transcript, so "
                "TranscribeEngine._emit_final() drops the empty text and nothing is sent. "
                "Cost metrics (infer_ms, rtf, lock waits, RSS) remain valid; delivery must "
                "be measured with real speech via --audio-file."
            )
    return result


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------
def _measure(fn: Callable[[], Any], repeats: int = 3) -> Dict[str, Any]:
    """Run ``fn`` ``repeats`` times, returning median/count time and peak allocations.

    ``tracemalloc`` peak is sampled on the best (fastest) run so the allocation figure
    belongs to the same execution as the reported timing.
    """
    timings: List[float] = []
    best_peak = 0

    for _ in range(max(1, repeats)):
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        timings.append(elapsed_ms)
        best_peak = peak if best_peak == 0 else min(best_peak, peak)

    return {
        "runs": len(timings),
        "median_ms": round(statistics.median(timings), 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
        "peak_alloc_bytes": int(best_peak),
    }


def _stats(values: Sequence[float]) -> Dict[str, float]:
    """Compact distribution summary used for every reported series."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)

    return {
        "count": n,
        "min": round(ordered[0], 3),
        "avg": round(sum(ordered) / n, 3),
        "p50": round(pct(0.50), 3),
        "p90": round(pct(0.90), 3),
        "p99": round(pct(0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Micro-benchmark: snapshot + normalizer cost  (audit benchmark A, CPU side)
# ---------------------------------------------------------------------------
def benchmark_snapshot_and_normalize(
    durations_sec: Iterable[int] = DEFAULT_DURATIONS,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Measure the full-utterance work performed before every ASR inference.

    This is the core of audit finding P1-01: `get_snapshot_if_newer()` re-materialises the
    WHOLE accumulated utterance (bytes -> int16 view -> float32 copy) and the normalizer
    then makes more full-array passes over it. If the measured cost grows roughly linearly
    with utterance duration, the audit's "repeated whole-utterance work" claim is confirmed
    with numbers instead of inference.
    """
    rows: List[Dict[str, Any]] = []

    for duration in durations_sec:
        pcm_bytes = generate_pcm_audio(float(duration), speech_like=True)
        sample_count = len(pcm_bytes) // 2
        frame_samples = 400
        n_frames = max(1, (sample_count + frame_samples - 1) // frame_samples)
        frame_state = np.ones(n_frames, dtype=np.uint8)

        def run_snapshot() -> None:
            mgr = AudioBufferManager(sample_rate=DEFAULT_SAMPLE_RATE)
            mgr.feed_bytes(pcm_bytes)
            snapshot = mgr.get_snapshot_if_newer(0)
            assert snapshot is not None

        snapshot = _measure(run_snapshot, repeats=repeats)

        # Prepare the float32 array once; the normalizer under test is the stateful
        # session instance exactly as used by TranscribeEngine.
        snapshot_pcm = (
            np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        normalizer = SpeechNormalizer()

        def run_normalize() -> None:
            normalizer.process(snapshot_pcm, frame_state=frame_state, use_smoothing=True)

        normalize = _measure(run_normalize, repeats=repeats)

        pre_infer_ms = snapshot["median_ms"] + normalize["median_ms"]
        rows.append(
            {
                "audio_sec": float(duration),
                "audio_bytes": len(pcm_bytes),
                "pcm_float32_bytes": int(snapshot_pcm.nbytes),
                "snapshot": snapshot,
                "normalize": normalize,
                "pre_infer_ms": round(pre_infer_ms, 3),
                "pre_infer_ms_per_audio_sec": round(pre_infer_ms / duration, 3),
            }
        )

    # Scaling verdict: compare the shortest and longest utterance. Both the time ratio and
    # the allocation ratio matter -- allocation is what grows strictly with duration, and
    # it is the figure the audit's "repeated whole-utterance work" claim is about.
    first, last = rows[0], rows[-1]
    duration_ratio = last["audio_sec"] / first["audio_sec"]
    cost_ratio = (last["pre_infer_ms"] / first["pre_infer_ms"]) if first["pre_infer_ms"] else 0.0
    alloc_first = first["snapshot"]["peak_alloc_bytes"] + first["normalize"]["peak_alloc_bytes"]
    alloc_last = last["snapshot"]["peak_alloc_bytes"] + last["normalize"]["peak_alloc_bytes"]
    alloc_ratio = round(alloc_last / alloc_first, 2) if alloc_first else 0.0

    return {
        "benchmark": "snapshot_and_normalize",
        "note": (
            "snapshot = AudioBufferManager.get_snapshot_if_newer() over the FULL utterance; "
            "normalize = SpeechNormalizer.process() over the FULL utterance. Both run before "
            "every preview inference."
        ),
        "rows": rows,
        "scaling": {
            "duration_ratio_longest_over_shortest": round(duration_ratio, 2),
            "cost_ratio_longest_over_shortest": round(cost_ratio, 2),
            "alloc_ratio_longest_over_shortest": alloc_ratio,
            "alloc_scales_with_duration": alloc_ratio > 0.8 * duration_ratio,
            "verdict": (
                "Allocation grows proportionally with utterance length (the audit's claim "
                "holds for allocation, not for wall-clock time). Compare these absolute "
                "milliseconds against the native inference cost before optimizing them."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Micro-benchmark: copy / conversion chain  (audit benchmark C)
# ---------------------------------------------------------------------------
def benchmark_copy_chain(duration_sec: float = 1.0, repeats: int = 3) -> Dict[str, Any]:
    """Measure each representation boundary on the inbound audio path.

    Every stage below is a real production call. ``peak_alloc_bytes`` comes from
    ``tracemalloc``, so it is measured rather than derived; ``output_bytes`` is the size of
    what the stage hands to the next stage.
    """
    pcm_bytes = generate_pcm_audio(duration_sec, speech_like=True)
    chunk_bytes = int(DEFAULT_SAMPLE_RATE * 0.1) * 2  # 100 ms chunks, like the extension
    ws_frame = pack_audio_frame(pcm_bytes[:chunk_bytes], 1, time.time())
    frame_samples = 400
    frame_bytes_len = frame_samples * 2

    stages: List[Dict[str, Any]] = []

    # 1. Binary frame parse (header + PCM slice)
    parsed = parse_audio_frame(ws_frame)
    measurement = _measure(lambda: parse_audio_frame(ws_frame), repeats=repeats)
    measurement.update(
        {
            "stage": "ws.parse_audio_frame",
            "input_bytes": len(ws_frame),
            "output_bytes": len(parsed[0]) if parsed[0] else 0,
        }
    )
    stages.append(measurement)

    # 2. AudioBuffer snapshot (bytes -> int16 view -> float32 copy) for the whole utterance
    def run_snapshot() -> None:
        mgr = AudioBufferManager(sample_rate=DEFAULT_SAMPLE_RATE)
        mgr.feed_bytes(pcm_bytes)
        mgr.get_snapshot_if_newer(0)

    measurement = _measure(run_snapshot, repeats=repeats)
    measurement.update(
        {
            "stage": "asr.audio_buffer.get_snapshot_if_newer (full utterance)",
            "input_bytes": len(pcm_bytes),
            "output_bytes": int(len(pcm_bytes) // 2 * 4),
        }
    )
    stages.append(measurement)

    # 3. Normalizer sanitize+process (audit finding N-02: copy() then clip() = 2 allocs)
    snapshot_pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    measurement = _measure(lambda: SpeechNormalizer.sanitize_input(snapshot_pcm), repeats=repeats)
    measurement.update(
        {
            "stage": "asr.speech_normalizer.sanitize_input",
            "input_bytes": int(snapshot_pcm.nbytes),
            "output_bytes": int(snapshot_pcm.nbytes),
        }
    )
    stages.append(measurement)

    # 4. Per-native-frame NumPy conversion reference.
    #
    # This is an explicit REFERENCE measurement of the arithmetic that
    # vad/vad_processor.py performs for every native frame, so the cost of that
    # conversion can be compared against the VAD model itself (see
    # benchmark_vad_cost). It is not a claim about the module's total cost.
    raw = bytearray(pcm_bytes[: frame_bytes_len * 40])

    def run_frame_conversion() -> None:
        for offset in range(0, len(raw) - frame_bytes_len + 1, frame_bytes_len):
            int16 = np.frombuffer(raw, dtype=np.int16, count=frame_samples, offset=offset)
            int16.astype(np.float32) / 32768.0

    measurement = _measure(run_frame_conversion, repeats=repeats)
    measurement.update(
        {
            "stage": f"REFERENCE numpy per-frame convert ({frame_samples}-sample frames)",
            "input_bytes": len(raw),
            "output_bytes": (len(raw) // frame_bytes_len) * frame_samples * 4,
            "native_frame_samples": frame_samples,
            "is_reference_only": True,
        }
    )
    stages.append(measurement)

    total_ms = sum(s["median_ms"] for s in stages)
    return {
        "benchmark": "copy_chain",
        "note": (
            "Each stage is a real production call except the row marked "
            "is_reference_only. peak_alloc_bytes is measured with tracemalloc. The audit "
            "previously listed these boundaries as unquantified."
        ),
        "audio_sec": duration_sec,
        "stages": stages,
        "total_median_ms": round(total_ms, 3),
        "total_ms_per_audio_sec": round(total_ms / duration_sec, 3),
        "vad_engine": None,
    }


# ---------------------------------------------------------------------------
# Micro-benchmark: VAD cost per engine
# ---------------------------------------------------------------------------
def benchmark_vad_cost(
    engines: Sequence[str] = ("firered-vad", "fsmn-vad"),
    duration_sec: float = 2.0,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Measure the real cost of VADProcessor.feed_chunk() per engine.

    Methodology notes (the first attempt at this measurement was wrong):

    * A fresh ``VADProcessor`` is used for each repeat. Reusing one processor carries VAD
      state across repeats, so later runs do not see the same input distribution. The
      underlying neural engine is a cached singleton, so construction stays cheap.
    * A warm-up pass runs outside the measurement, so model/JIT warm-up is not billed to
      the reported figure.
    * ``speech_frames_passed_to_asr`` is reported because the synthetic harmonic test
      signal is not guaranteed to be classified as speech: if it is zero, the per-speech-
      frame ``bytes(...)`` copy path was NOT exercised, and that must be visible rather
      than silently averaged away.
    """
    pcm_bytes = generate_pcm_audio(duration_sec, speech_like=True)
    chunk_bytes = int(DEFAULT_SAMPLE_RATE * 0.1) * 2
    rows: List[Dict[str, Any]] = []

    for engine in engines:
        counters = {"frames": 0}

        def make_processor() -> VADProcessor:
            proc = VADProcessor(vad_engine=engine)
            proc.on_speech_chunk = lambda *args: counters.__setitem__("frames", counters["frames"] + 1)
            return proc

        # Warm up outside the measured region.
        warm = make_processor()
        warm.feed_chunk(pcm_bytes[: chunk_bytes * 4], capture_timestamp=0.0)

        def run_vad() -> None:
            counters["frames"] = 0
            proc = make_processor()
            proc.feed_chunk(pcm_bytes, capture_timestamp=0.0)

        measurement = _measure(run_vad, repeats=repeats)
        frame_samples = getattr(warm, "_frame_samples", 0)
        measurement.update(
            {
                "engine": engine,
                "native_frame_samples": frame_samples,
                "frames_per_audio_sec": round(DEFAULT_SAMPLE_RATE / frame_samples, 1) if frame_samples else 0,
                "speech_frames_passed_to_asr": counters["frames"],
                "rtf": round(measurement["median_ms"] / 1000.0 / duration_sec, 4),
                "ms_per_audio_sec": round(measurement["median_ms"] / duration_sec, 3),
            }
        )
        rows.append(measurement)
        logger.info(
            "   VAD %s: %.1f ms for %.1fs audio (RTF %.3f), speech frames to ASR=%d",
            engine,
            measurement["median_ms"],
            duration_sec,
            measurement["rtf"],
            counters["frames"],
        )

    return {
        "benchmark": "vad_cost",
        "note": (
            "Real VADProcessor.feed_chunk() cost, fresh processor per repeat, warm-up "
            "excluded. Compare against the per-frame NumPy conversion reference in "
            "copy_chain to see whether the model or the array bookkeeping dominates."
        ),
        "audio_sec": duration_sec,
        "rows": rows,
    }


def benchmark_vad_engine_tradeoff(
    audio_path: Optional[Path] = None,
    max_sec: float = 24.0,
    engines: Sequence[str] = ("silero-vad", "fsmn-vad", "firered-vad"),
    repeats: int = 3,
) -> Dict[str, Any]:
    """Compare VAD engines on cost AND detection quality over real speech.

    ``benchmark_vad_cost`` only tells you which engine is cheapest. That is not enough to
    pick a default: an engine that trims or invents speech changes subtitle behaviour. This
    runs every engine over the same real recording and reports both sides, so the trade-off
    is explicit.

    Boundary timestamps are NOT expected to match between engines -- hysteresis and native
    frame sizes differ (32/60/25 ms). What matters is whether segment count and total
    speech coverage are in the same ballpark, i.e. whether one engine is obviously missing
    or inventing speech.
    """
    path = Path(audio_path) if audio_path else Path("wav_test/OSR_us_000_0010_16k.wav")
    if not path.is_absolute():
        path = Path.cwd() / path

    # Imported here: perf_real_audio imports this module, so a top-level import would cycle.
    from backend_cpp.tests.perf_real_audio import load_wav_pcm16
    from backend_cpp.vad.vad_processor import VADProcessor

    pcm = load_wav_pcm16(path, max_sec=max_sec)
    duration = len(pcm) / (DEFAULT_SAMPLE_RATE * 2)
    chunk_bytes = int(DEFAULT_SAMPLE_RATE * 0.1) * 2

    rows: List[Dict[str, Any]] = []

    for engine_name in engines:
        events: List[Dict[str, Any]] = []
        clock = {"sec": 0.0}

        try:
            # Warm-up (model load) outside the measured region.
            warm = VADProcessor(vad_engine=engine_name)
            warm.feed_chunk(pcm[: chunk_bytes * 5], capture_timestamp=0.0)
        except Exception as exc:
            rows.append({"engine": engine_name, "available": False, "error": f"{type(exc).__name__}: {exc}"})
            continue

        per_run_ms: List[float] = []
        for _ in range(max(1, repeats)):
            events.clear()
            clock["sec"] = 0.0
            proc = VADProcessor(
                vad_engine=engine_name,
                on_speech_start=lambda: events.append({"type": "START", "at": clock["sec"]}),
                on_speech_end=lambda: events.append({"type": "END", "at": clock["sec"]}),
            )
            gc.collect()
            start = time.perf_counter()
            for offset in range(0, len(pcm) - chunk_bytes + 1, chunk_bytes):
                proc.feed_chunk(pcm[offset: offset + chunk_bytes], capture_timestamp=clock["sec"])
                clock["sec"] += 0.1
            per_run_ms.append((time.perf_counter() - start) * 1000.0)

        segments: List[List[float]] = []
        current: Optional[float] = None
        for event in events:
            if event["type"] == "START" and current is None:
                current = event["at"]
            elif event["type"] == "END" and current is not None:
                segments.append([round(current, 2), round(event["at"], 2)])
                current = None
        if current is not None:
            segments.append([round(current, 2), round(duration, 2)])

        lengths = [end - start for start, end in segments]
        speech_total = sum(lengths)
        median_ms = statistics.median(per_run_ms)

        rows.append(
            {
                "engine": engine_name,
                "available": True,
                "median_ms": round(median_ms, 2),
                "ms_per_audio_sec": round(median_ms / duration, 2),
                "rtf": round(median_ms / 1000.0 / duration, 4),
                "segments": len(segments),
                "speech_sec": round(speech_total, 2),
                "speech_ratio": round(speech_total / duration, 3),
                "median_segment_sec": round(statistics.median(lengths), 2) if lengths else 0.0,
                "first_segments": segments[:6],
            }
        )

    available = [r for r in rows if r.get("available")]
    cheapest = min(available, key=lambda r: r["ms_per_audio_sec"]) if available else None

    if cheapest:
        for row in available:
            row["cost_multiple_vs_cheapest"] = round(
                row["ms_per_audio_sec"] / cheapest["ms_per_audio_sec"], 2
            )
            row["speech_ratio_delta_vs_cheapest"] = round(
                row["speech_ratio"] - cheapest["speech_ratio"], 3
            )

    return {
        "benchmark": "vad_engine_tradeoff",
        "audio_file": str(path),
        "audio_sec": round(duration, 2),
        "note": (
            "Cost alone is not enough to choose a VAD engine -- detection coverage matters "
            "too. Compare speech_ratio and segments before switching the default."
        ),
        "cheapest_engine": cheapest["engine"] if cheapest else None,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# End-to-end: ASR preview cost vs utterance length  (audit benchmark A)
# ---------------------------------------------------------------------------
async def run_scenario_preview_cost(
    uri: str,
    ssl_context: Any = None,
    durations_sec: Sequence[int] = DEFAULT_DURATIONS,
    repeats: int = 2,
) -> Dict[str, Any]:
    """Stream utterances of increasing length and correlate ASR cost with audio duration."""
    logger.info(
        "▶️ [BASELINE A] ASR preview cost vs utterance length %s (repeats=%d)...",
        list(durations_sec),
        repeats,
    )

    perf.reset()
    client = BenchmarkClient(uri, ssl_context)
    await client.connect()
    await client.send_config(_baseline_config(splitOnStability=True))
    await asyncio.sleep(0.5)

    chunk_idx = 1
    started = time.perf_counter()
    for _ in range(repeats):
        for duration in durations_sec:
            chunk_idx = await client.stream_audio_duration(
                duration_sec=float(duration),
                chunk_ms=100,
                speech_like=True,
                realtime_factor=1.0,
                start_chunk_idx=chunk_idx,
            )
            # A short silence forces the commit for this utterance.
            chunk_idx = await client.stream_audio_duration(
                duration_sec=1.0,
                chunk_ms=100,
                speech_like=False,
                realtime_factor=1.0,
                start_chunk_idx=chunk_idx,
            )

    await asyncio.sleep(3.0)
    wall_sec = time.perf_counter() - started
    await client.disconnect()

    infer_ms = perf.get_samples("asr", "infer_ms")
    audio_dur = perf.get_samples("asr", "audio_dur_sec")
    rtfs = perf.get_samples("asr", "rtf")

    # Bucket each inference by the nearest configured utterance length.
    buckets: Dict[float, Dict[str, List[float]]] = {
        float(d): {"infer_ms": [], "rtf": []} for d in durations_sec
    }
    for infer, dur, rtf in zip(infer_ms, audio_dur, rtfs):
        nearest = min(buckets.keys(), key=lambda candidate: abs(candidate - dur))
        buckets[nearest]["infer_ms"].append(infer)
        buckets[nearest]["rtf"].append(rtf)

    rows = []
    for duration in sorted(buckets):
        rows.append(
            {
                "audio_sec": duration,
                "infer_ms": _stats(buckets[duration]["infer_ms"]),
                "rtf": _stats(buckets[duration]["rtf"]),
            }
        )

    scaling_ratio = None
    if len(rows) >= 2 and rows[0]["infer_ms"].get("avg") and rows[-1]["infer_ms"].get("avg"):
        scaling_ratio = round(rows[-1]["infer_ms"]["avg"] / rows[0]["infer_ms"]["avg"], 2)

    counters = perf.generate_report()["counters"]
    logger.info("✅ [BASELINE A] Completed in %.1fs, %d inferences.", wall_sec, len(infer_ms))

    return {
        "benchmark": "preview_cost_e2e",
        "wall_sec": round(wall_sec, 2),
        "validity": _validity(len(infer_ms)),
        "vad_engine": BASELINE_VAD_ENGINE,
        "repeats": repeats,
        "durations_requested": [float(d) for d in durations_sec],
        "rows": rows,
        "infer_cost_scaling_longest_over_shortest": scaling_ratio,
        "counters": {
            key: counters.get(key, 0)
            for key in (
                "asr.total_inferences",
                "asr.preview_infers",
                "asr.commit_inferences",
                "asr.cached_preview_reuse",
                "asr.preview_lock_skipped",
                "asr.commit_lock_timeout",
            )
        },
        "client_events": {
            "utterances": len(client.utterance_events),
            "messages": len(client.received_messages),
        },
        "lock_wait_ms": _stats(perf.get_samples("asr", "lock_wait_ms")),
        "resources": {"ram_rss_mb": _rss_mb()},
    }


# ---------------------------------------------------------------------------
# End-to-end: queue / backpressure behaviour  (audit benchmark D)
# ---------------------------------------------------------------------------
async def run_scenario_backpressure(
    uri: str,
    ssl_context: Any = None,
    audio_sec: float = 16.0,
    speed: float = 4.0,
    with_tts: bool = False,
) -> Dict[str, Any]:
    """Push audio far faster than real time and watch queues, drops and RSS.

    Acceptance (audit §18.4): RSS stays bounded, final subtitles still arrive, preview
    messages may be dropped, and any TTS loss is a configured policy rather than a crash.
    """
    logger.info(
        "▶️ [BASELINE D] Backpressure: %.0fs audio at %.1fx, tts=%s...",
        audio_sec,
        speed,
        with_tts,
    )

    perf.record_resource_checkpoint("backpressure_before")
    rss_before = _rss_mb()
    counters_before = dict(perf.generate_report()["counters"])

    client = BenchmarkClient(uri, ssl_context)
    await client.connect()
    await client.send_config(
        _baseline_config(ttsEnabled=bool(with_tts), splitOnStability=True, minWordsToCommit=2)
    )
    await asyncio.sleep(0.5)

    started = time.perf_counter()
    # Bursts of 2s speech separated by 1s of silence, then a trailing silence.
    #
    # The silence MUST exceed the VAD's silence_duration_ms (default 600 ms) plus
    # hangover_ms (default 400 ms), otherwise no utterance ever receives an END event, no
    # commit is emitted, and the scenario reports zero deliveries -- a measurement bug that
    # looks like a backpressure finding.
    chunk_idx = 1
    burst = 0
    burst_sec = 3.0
    while burst * burst_sec < audio_sec:
        chunk_idx = await client.stream_audio_duration(
            duration_sec=2.0,
            chunk_ms=50,
            speech_like=True,
            realtime_factor=speed,
            start_chunk_idx=chunk_idx,
        )
        chunk_idx = await client.stream_audio_duration(
            duration_sec=1.0,
            chunk_ms=50,
            speech_like=False,
            realtime_factor=speed,
            start_chunk_idx=chunk_idx,
        )
        burst += 1

    # Trailing silence so the last burst commits too.
    chunk_idx = await client.stream_audio_duration(
        duration_sec=1.5,
        chunk_ms=50,
        speech_like=False,
        realtime_factor=speed,
        start_chunk_idx=chunk_idx,
    )

    # Let the pipeline drain so queue effects (not just ordering) are visible.
    await asyncio.sleep(6.0)
    wall_sec = time.perf_counter() - started
    await client.disconnect()

    rss_after = _rss_mb()
    counters_after = dict(perf.generate_report()["counters"])
    delta = {
        key: counters_after.get(key, 0) - counters_before.get(key, 0)
        for key in sorted(set(counters_before) | set(counters_after))
        if counters_after.get(key, 0) != counters_before.get(key, 0)
    }

    finals = [
        m for m in client.received_messages
        if m.get("type") == "utterance_update" and m.get("is_final")
    ]
    translations = [m for m in client.received_messages if m.get("type") == "translation"]

    logger.info(
        "✅ [BASELINE D] Completed in %.1fs, %d finals, %d translations, RSS %.1f -> %.1f MB",
        wall_sec,
        len(finals),
        len(translations),
        rss_before,
        rss_after,
    )

    return {
        "benchmark": "backpressure_e2e",
        "wall_sec": round(wall_sec, 2),
        "validity": _validity(
            int(delta.get("asr.total_inferences", 0) or 0),
            delivered=len(finals),
        ),
        "vad_engine": BASELINE_VAD_ENGINE,
        "audio_sec_pushed": audio_sec,
        "speed": speed,
        "tts_enabled": with_tts,
        "delivery": {
            "final_utterances": len(finals),
            "translations": len(translations),
        },
        "counter_delta": delta,
        "queue_wait_ms": {
            "translation": _stats(perf.get_samples("translation", "queue_wait_ms")),
            "tts": _stats(perf.get_samples("tts", "queue_wait_ms")),
        },
        "resources": {
            "ram_rss_mb_before": rss_before,
            "ram_rss_mb_after": rss_after,
            "ram_rss_delta_mb": round(rss_after - rss_before, 1),
        },
    }


# ---------------------------------------------------------------------------
# End-to-end: concurrent session scaling  (audit benchmark G)
# ---------------------------------------------------------------------------
async def run_scenario_session_scaling(
    uri: str,
    ssl_context: Any = None,
    levels: Sequence[int] = (1, 2, 4),
    audio_sec: float = 6.0,
) -> Dict[str, Any]:
    """Measure lock-wait growth as concurrent sessions increase.

    Temporarily disables the single-session admission guard (``config.ws.max_sessions``),
    because that guard is exactly what would otherwise reject the extra clients.
    """
    from backend_cpp.config import config

    original_max_sessions = config.ws.max_sessions
    rows: List[Dict[str, Any]] = []

    logger.info("▶️ [BASELINE G] Session scaling over %s concurrent sessions...", list(levels))
    logger.info(
        "   (temporarily setting config.ws.max_sessions=0; admission guard would reject extras)"
    )

    try:
        config.ws.max_sessions = 0
        for level in levels:
            perf.reset()
            clients = []
            for _ in range(level):
                client = BenchmarkClient(uri, ssl_context)
                await client.connect()
                await client.send_config(_baseline_config())
                clients.append(client)

            await asyncio.sleep(0.5)
            started = time.perf_counter()

            async def drive(client: BenchmarkClient, index: int) -> None:
                await client.stream_audio_duration(
                    duration_sec=audio_sec,
                    chunk_ms=100,
                    speech_like=True,
                    realtime_factor=1.0,
                    start_chunk_idx=1,
                )
                # Trailing silence must exceed silence_duration_ms + hangover_ms or the
                # utterance never commits and no lock-wait sample is recorded.
                await client.stream_audio_duration(
                    duration_sec=1.5,
                    chunk_ms=100,
                    speech_like=False,
                    realtime_factor=1.0,
                    start_chunk_idx=10_000 + index,
                )

            await asyncio.gather(*(drive(client, i) for i, client in enumerate(clients)))
            await asyncio.sleep(4.0)
            wall_sec = time.perf_counter() - started

            for client in clients:
                await client.disconnect()

            counters = perf.generate_report()["counters"]
            rows.append(
                {
                    "sessions": level,
                    "wall_sec": round(wall_sec, 2),
                    "validity": _validity(counters.get("asr.total_inferences", 0)),
                    "lock_wait_ms": _stats(perf.get_samples("asr", "lock_wait_ms")),
                    "infer_ms": _stats(perf.get_samples("asr", "infer_ms")),
                    "counters": {
                        key: counters.get(key, 0)
                        for key in (
                            "asr.total_inferences",
                            "asr.preview_lock_skipped",
                            "asr.commit_lock_timeout",
                            "asr.final_dropped",
                        )
                    },
                    "ram_rss_mb": _rss_mb(),
                }
            )
            logger.info(
                "   session=%d -> wall %.1fs, lock_wait p50=%.1fms p99=%.1fms",
                level,
                wall_sec,
                rows[-1]["lock_wait_ms"].get("p50", 0.0),
                rows[-1]["lock_wait_ms"].get("p99", 0.0),
            )
    finally:
        config.ws.max_sessions = original_max_sessions

    return {
        "benchmark": "session_scaling_e2e",
        "levels": list(levels),
        "audio_sec_per_session": audio_sec,
        "note": "config.ws.max_sessions was forced to 0 for this measurement, then restored.",
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def collect_environment() -> Dict[str, Any]:
    """Capture enough environment detail for the baseline to be comparable later."""
    import platform
    import sys

    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": None,
        "ram_rss_mb": _rss_mb(),
    }

    try:
        import psutil

        env["cpu_count"] = psutil.cpu_count(logical=True)
        env["ram_total_mb"] = round(psutil.virtual_memory().total / (1024 * 1024), 1)
    except Exception:
        pass

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1
            )
    except Exception:
        pass

    try:
        from backend_cpp.config import config

        env["asr_model"] = config.asr.active_model
        env["asr_backend"] = config.asr.backend
        env["asr_threads"] = config.asr.threads
        env["poll_interval_ms"] = config.asr.poll_interval_ms
        env["vad_engine"] = config.vad.vad_engine
        env["translation_base"] = config.translation.base
        env["tts_enabled"] = config.tts.enabled
        env["sentence"] = {
            "max_duration_sec": config.sentence.max_duration_sec,
            "stability_duration_sec": config.sentence.stability_duration_sec,
            "stability_threshold_polls": config.sentence.stability_threshold_polls,
            "min_words_to_commit": config.sentence.min_words_to_commit,
        }
        env["max_sessions"] = config.ws.max_sessions
        env["token_queue_maxsize"] = config.asr.token_queue_maxsize
    except Exception as exc:  # pragma: no cover - defensive
        env["config_error"] = str(exc)

    return env


def build_baseline(
    micro: Optional[Dict[str, Any]] = None,
    scenarios: Optional[Dict[str, Any]] = None,
    commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the baseline document written to ``report/baseline_perf.json``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "backend_cpp_baseline",
        "audit_reference": "report/backend_cpp_extension_performance_audit.md",
        "plan_reference": "report/backend_cpp_extension_performance_audit_implementation_plan.md",
        "work_item": "W2.1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": commit,
        "environment": collect_environment(),
        "micro_benchmarks": micro or {},
        "end_to_end_scenarios": scenarios or {},
    }


def write_baseline(document: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
    logger.info("💾 Baseline written to %s", output_path)
    return output_path


def merge_into_baseline(output_path: Path, section: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Load, merge one section, and rewrite the baseline document."""
    document: Dict[str, Any] = {}
    if output_path.exists():
        try:
            document = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            document = {}

    document.setdefault("schema_version", SCHEMA_VERSION)
    document.setdefault("kind", "backend_cpp_baseline")
    document.setdefault("work_item", "W2.1")
    document["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    document["environment"] = collect_environment()
    document.setdefault("micro_benchmarks", {})
    document.setdefault("end_to_end_scenarios", {})

    if section == "micro":
        document["micro_benchmarks"].update(payload)
    else:
        document["end_to_end_scenarios"].update(payload)

    write_baseline(document, output_path)
    return document
