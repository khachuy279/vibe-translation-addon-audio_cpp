"""Is the +660MB RESOURCE_LEAK_WARNING a real leak or a lazy-model-load artifact?

`perf.reset()` re-baselines the resource snapshot, so each run below reports the RSS growth
attributable to THAT run alone:

* if every run grows by ~660MB  -> real leak (or per-run accumulation)
* if only run 1 grows, then flat -> one-time lazy model load, i.e. a FALSE POSITIVE

Run from the project root:  python scratch/leak_double_run.py
"""

import asyncio
import ssl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend_cpp.config import config  # noqa: E402

config.perf.enabled = True

from backend_cpp.run_perf_test import is_server_running, start_in_process_server  # noqa: E402
from backend_cpp.tests.perf_baseline import _rss_mb  # noqa: E402
from backend_cpp.tests.perf_real_audio import run_scenario_real_audio  # noqa: E402
from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths  # noqa: E402
from backend_cpp.utils.perf_profiler import perf  # noqa: E402

HEALTH = "https://127.0.0.1:8765/health"
URI = "wss://127.0.0.1:8765/ws"


def _ssl_ctx():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _alert_types():
    report = perf.generate_report()
    alerts = report.get("alerts") or []
    return sorted({str(a.get("type", "")) for a in alerts})


async def main():
    setup_cuda_dll_paths()

    if not is_server_running(HEALTH):
        start_in_process_server(8765)
        for _ in range(40):
            await asyncio.sleep(0.5)
            if is_server_running(HEALTH):
                break
    print(f"[setup] baseline RSS {_rss_mb():.1f} MB (models not loaded yet)\n", flush=True)

    for run in (1, 2, 3):
        perf.reset()
        before = _rss_mb()
        result = await run_scenario_real_audio(
            URI, _ssl_ctx(), max_sec=8.0, silence_tail_sec=1.5
        )
        after = _rss_mb()
        # The session_end checkpoint runs in the server handler, which may land after the
        # client disconnects; let it settle so a timing race cannot hide the alert.
        await asyncio.sleep(1.5)
        print(
            f"[run {run}] RSS {before:.1f} -> {after:.1f} MB "
            f"(delta {after - before:+.1f}) | "
            f"utterances {result['delivery']['final_utterances']} | "
            f"alerts {_alert_types()}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
