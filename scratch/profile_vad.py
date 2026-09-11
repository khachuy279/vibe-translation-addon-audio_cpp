"""Profile where VAD time actually goes, per engine.

W2.1 measured `VADProcessor.feed_chunk()` at 81 ms (fsmn-vad) and 154 ms (firered-vad)
per second of audio -- the largest CPU stage in the pipeline, and one the audit never
costed. This script answers the next question: WHICH part is slow?

It separates:
  1. the engine model call itself (`is_speech` per native frame),
  2. the surrounding `feed_chunk` bookkeeping (locking, state, callbacks, copies),
  3. the internal call tree of the model call (cProfile), to see whether the cost is
     real neural-network work or per-call framework overhead.

Usage:
    python scratch/profile_vad.py
    python scratch/profile_vad.py --engine fsmn-vad --seconds 3
"""

import argparse
import cProfile
import gc
import io
import pstats
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE  # noqa: E402
from backend_cpp.tests.perf_benchmark import generate_pcm_audio  # noqa: E402
from backend_cpp.vad.engines import VADEngineFactory  # noqa: E402
from backend_cpp.vad.vad_processor import VADProcessor  # noqa: E402


def _pcm_f32(duration_sec: float) -> np.ndarray:
    raw = generate_pcm_audio(duration_sec, speech_like=True)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def time_engine_call(engine, pcm: np.ndarray, repeats: int = 5) -> Dict[str, Any]:
    """Cost of the raw engine call, one native frame at a time."""
    frame = engine.native_frame_samples
    n_frames = len(pcm) // frame
    per_run_ms: List[float] = []

    for _ in range(repeats):
        state = engine.create_initial_state(threshold=0.4)
        gc.collect()
        start = time.perf_counter()
        for index in range(n_frames):
            chunk = pcm[index * frame: (index + 1) * frame]
            engine.is_speech(chunk, state, 0.4)
        per_run_ms.append((time.perf_counter() - start) * 1000.0)

    total_audio = n_frames * frame / DEFAULT_SAMPLE_RATE
    median_ms = statistics.median(per_run_ms)
    return {
        "frames": n_frames,
        "frame_samples": frame,
        "audio_sec": round(total_audio, 3),
        "median_ms": round(median_ms, 2),
        "ms_per_frame": round(median_ms / max(1, n_frames), 3),
        "ms_per_audio_sec": round(median_ms / total_audio, 2),
        "frames_per_sec": round(n_frames / total_audio, 1),
    }


def time_feed_chunk(engine_name: str, pcm_bytes: bytes, duration_sec: float, repeats: int = 5) -> Dict[str, Any]:
    """Cost of the full production entry point (locking, frame split, callbacks)."""
    per_run_ms: List[float] = []

    # Warm-up outside measurement: model/JIT load must not be billed.
    warm = VADProcessor(vad_engine=engine_name)
    warm.feed_chunk(pcm_bytes[:16000], capture_timestamp=0.0)

    for _ in range(repeats):
        proc = VADProcessor(vad_engine=engine_name)
        gc.collect()
        start = time.perf_counter()
        proc.feed_chunk(pcm_bytes, capture_timestamp=0.0)
        per_run_ms.append((time.perf_counter() - start) * 1000.0)

    median_ms = statistics.median(per_run_ms)
    return {
        "median_ms": round(median_ms, 2),
        "ms_per_audio_sec": round(median_ms / duration_sec, 2),
        "rtf": round(median_ms / 1000.0 / duration_sec, 4),
    }


def profile_feed_chunk(engine_name: str, pcm_bytes: bytes, top: int = 18) -> str:
    """cProfile the production entry point and return the hottest frames."""
    proc = VADProcessor(vad_engine=engine_name)
    proc.feed_chunk(pcm_bytes[:16000], capture_timestamp=0.0)  # warm-up

    proc = VADProcessor(vad_engine=engine_name)
    profiler = cProfile.Profile()
    profiler.enable()
    proc.feed_chunk(pcm_bytes, capture_timestamp=0.0)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("tottime")
    stats.print_stats(top)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=None, help="Engine to profile (default: all)")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-profile", action="store_true", help="Skip the cProfile section")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("vad_profile_report.txt")),
        help="Also write the full report to this file (console output gets truncated by shells)",
    )
    args = parser.parse_args()

    engines = [args.engine] if args.engine else ["fsmn-vad", "firered-vad", "silero-vad"]
    duration = args.seconds
    pcm_f32 = _pcm_f32(duration)
    pcm_bytes = generate_pcm_audio(duration, speech_like=True)

    lines: List[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit(f"audio: {duration}s @ {DEFAULT_SAMPLE_RATE}Hz")
    emit()
    emit(f"{'engine':<14} {'frames':>7} {'ms/frame':>9} {'engine ms/s':>12} "
         f"{'feed_chunk ms/s':>17} {'bookkeeping':>12}")
    emit("-" * 78)

    for name in engines:
        try:
            engine = VADEngineFactory.get_engine(name)
        except Exception as exc:
            emit(f"{name:<14} unavailable: {type(exc).__name__}: {exc}")
            continue

        call = time_engine_call(engine, pcm_f32, repeats=args.repeats)
        feed = time_feed_chunk(name, pcm_bytes, duration, repeats=args.repeats)

        # How much of feed_chunk is NOT the raw engine call? That is the bookkeeping
        # (locking, per-frame slicing/astype, callbacks) we could plausibly remove.
        bookkeeping = feed["ms_per_audio_sec"] - call["ms_per_audio_sec"]
        share = bookkeeping / feed["ms_per_audio_sec"] * 100 if feed["ms_per_audio_sec"] else 0.0

        emit(
            f"{name:<14} {call['frames']:>7} {call['ms_per_frame']:>9.3f} "
            f"{call['ms_per_audio_sec']:>12.2f} {feed['ms_per_audio_sec']:>17.2f} "
            f"{bookkeeping:>9.2f} ms ({share:.0f}%)"
        )

        if not args.no_profile:
            emit()
            emit("=" * 78)
            emit(f"cProfile: VADProcessor.feed_chunk() [{name}]  (sorted by tottime)")
            emit("=" * 78)
            emit(profile_feed_chunk(name, pcm_bytes))
            emit()

    report = "\n".join(lines)
    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
