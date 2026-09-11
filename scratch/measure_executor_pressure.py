"""Is the shared default executor actually starving stages? (evidence for W2.5 / P1-08)

W2.5 claims "cross-starvation between VAD/ASR/translation/TTS" and proposes dedicated
executors. That is an audit *hypothesis*, never measured. Two facts make it doubtful:

1. `asyncio.to_thread` uses the loop's default executor, whose max_workers is
   `min(32, cpu_count + 4)` = 16 on this 12-CPU machine. The hot path submits at most a handful
   of concurrent calls (VAD per 100ms chunk, ASR preview per ~350ms, ASR commit, translation).
2. The heavy VAD engine (fsmn) does a lot of *Python-level* work (`copy.deepcopy` x6567 per 2s
   of audio, from the cProfile in the plan). That holds the GIL, which is process-wide, so
   separate executors CANNOT fix it.

So measure the thing that would prove starvation: the dispatch delay from `to_thread` submit to
the worker actually starting. If it is ~0, the pool is not saturated and W2.5 is not needed.

Usage:  python scratch/measure_executor_pressure.py
"""

import asyncio
import collections
import ssl
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HEALTH = "https://127.0.0.1:8765/health"
URI = "wss://127.0.0.1:8765/ws"

SAMPLE = collections.defaultdict(lambda: {"dispatch_ms": [], "exec_ms": []})
IN_FLIGHT = {"now": 0, "max": 0}
_REAL_TO_THREAD = None


def _label(func) -> str:
    module = getattr(func, "__module__", "?")
    name = getattr(func, "__qualname__", getattr(func, "__name__", str(func)))
    short = module.split(".")[-1]
    return f"{short}.{name}"


def patch_to_thread() -> None:
    """Wrap asyncio.to_thread to record submit->start delay and in-flight concurrency."""
    global _REAL_TO_THREAD
    _REAL_TO_THREAD = asyncio.to_thread

    async def patched(func, /, *args, **kwargs):
        key = _label(func)
        t_submit = time.perf_counter()

        def wrapper():
            t_start = time.perf_counter()
            IN_FLIGHT["now"] += 1
            IN_FLIGHT["max"] = max(IN_FLIGHT["max"], IN_FLIGHT["now"])
            try:
                return func(*args, **kwargs)
            finally:
                IN_FLIGHT["now"] -= 1
                SAMPLE[key]["dispatch_ms"].append((t_start - t_submit) * 1000.0)
                SAMPLE[key]["exec_ms"].append((time.perf_counter() - t_start) * 1000.0)

        return await _REAL_TO_THREAD(wrapper)

    asyncio.to_thread = patched


def _stats(values):
    if not values:
        return "n/a"
    ordered = sorted(values)
    return (
        f"n={len(values):<4} p50={statistics.median(ordered):>7.2f} "
        f"p99={ordered[int(len(ordered) * 0.99) - 1 if len(ordered) > 1 else 0]:>7.2f} "
        f"max={ordered[-1]:>7.2f}"
    )


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def main() -> int:
    from backend_cpp.config import config

    config.perf.enabled = True

    from backend_cpp.run_perf_test import is_server_running, start_in_process_server
    from backend_cpp.tests.perf_real_audio import run_scenario_real_audio
    from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths

    setup_cuda_dll_paths()

    patch_to_thread()

    if not is_server_running(HEALTH):
        start_in_process_server(8765)
        for _ in range(40):
            await asyncio.sleep(0.5)
            if is_server_running(HEALTH):
                break

    result = await run_scenario_real_audio(URI, _ssl_ctx(), max_sec=24.0, silence_tail_sec=2.0)
    print(f"\nutterances={result['delivery']['final_utterances']} "
          f"translations={result['delivery']['translations']}\n")

    print("=" * 78)
    print("asyncio.to_thread pressure (all call sites in the hot path)")
    print("=" * 78)
    header = f"{'call site':40s} {'dispatch (submit->start)':>32s} {'occupancy (exec)':>32s}"
    print(header)
    print("-" * len(header))
    for key in sorted(SAMPLE, key=lambda k: -len(SAMPLE[k]["dispatch_ms"])):
        print(
            f"{key[:40]:40s} {_stats(SAMPLE[key]['dispatch_ms']):>32s} "
            f"{_stats(SAMPLE[key]['exec_ms']):>32s}"
        )
    print()
    print(f"max concurrent to_thread calls in flight: {IN_FLIGHT['max']}")
    print("  default executor max_workers = min(32, cpu_count + 4) = 16 on this 12-CPU machine")
    print()
    print("Interpretation: a dispatch delay near 0 ms means the pool is NOT saturated, so")
    print("dedicated executors (W2.5) would not remove any waiting.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
