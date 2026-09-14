"""1-Click Performance Benchmark & Bottleneck Diagnostic Tool for backend_cpp.

Usage:
    python run_perf_test.py
    python run_perf_test.py --mode=conversational
    python run_perf_test.py --mode=burst
    python run_perf_test.py --mode=leak
    python run_perf_test.py --server-url=wss://localhost:8765/ws

If a backend server is already running, this tool connects to it directly.
If not, it automatically boots up an in-process backend server with pre-warmed models,
runs the automated benchmark suite, prints diagnostic metrics, and saves the report.
"""

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure backend_cpp and project root in sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from backend_cpp.config import config
from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths
from backend_cpp.utils.perf_profiler import perf
from backend_cpp.tests.perf_benchmark import (
    run_scenario_conversational_stream,
    run_scenario_continuous_speech,
    run_scenario_rapid_burst_contention,
    run_scenario_leak_detection,
)
from backend_cpp.tests.perf_baseline import (
    benchmark_copy_chain,
    benchmark_snapshot_and_normalize,
    benchmark_vad_cost,
    benchmark_vad_engine_tradeoff,
    merge_into_baseline,
    run_scenario_backpressure,
    run_scenario_preview_cost,
    run_scenario_session_scaling,
)
from backend_cpp.tests.perf_real_audio import (
    DEFAULT_REFERENCE_AUDIO,
    run_scenario_real_audio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_perf_test")


def is_server_running(health_url: str) -> bool:
    """Check if backend is already running and reachable."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "PerfTest"})
        with urllib.request.urlopen(req, context=ctx, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_in_process_server(port: int = 8765) -> threading.Thread:
    """Start local uvicorn server in a background daemon thread."""
    import uvicorn
    from backend_cpp.main import app
    from backend_cpp.utils.ssl_utils import ensure_ssl_certificates

    cert_path, key_path = ensure_ssl_certificates()
    ssl_kwargs = {
        "ssl_certfile": cert_path,
        "ssl_keyfile": key_path,
    }

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        ws_ping_interval=20.0,
        ws_ping_timeout=30.0,
        **ssl_kwargs,
    )
    server = uvicorn.Server(server_config)

    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn_perf_server")
    thread.start()
    return thread


async def main_async(args: argparse.Namespace) -> None:
    setup_cuda_dll_paths()

    # The perf collector is DISABLED by default (config.perf.enabled = False) and every
    # record_metric / increment_counter call is a no-op while it is off. Without this the
    # server silently discards every measurement and the baseline comes back empty even
    # though the pipeline really did run -- which is exactly how a benchmark lies.
    #
    # Only effective for the in-process server; an externally started backend must be
    # launched with perf.enabled = True for these scenarios to report anything.
    from backend_cpp.config import config as _app_config

    _app_config.perf.enabled = True

    mode = args.mode.lower()
    baseline_path = Path(args.baseline_out)
    if not baseline_path.is_absolute():
        baseline_path = _PROJECT_ROOT / baseline_path

    # ------------------------------------------------------------------
    # W2.1 micro-benchmarks: no server and no GPU required, so run them first and
    # allow `--mode micro` to exit without paying the model pre-warm cost.
    # ------------------------------------------------------------------
    if mode in ("micro", "baseline"):
        print("\n" + "=" * 80)
        print("   W2.1 MICRO-BENCHMARKS (in-process, no server)")
        print("=" * 80 + "\n")
        micro = {
            "snapshot_and_normalize": benchmark_snapshot_and_normalize(),
            "copy_chain": benchmark_copy_chain(duration_sec=1.0),
            "vad_cost": benchmark_vad_cost(),
            "vad_engine_tradeoff": benchmark_vad_engine_tradeoff(),
        }
        merge_into_baseline(baseline_path, "micro", micro)
        print(f"\n💾 Micro-benchmarks written to {baseline_path}\n")
        if mode == "micro":
            return

    host = args.host
    port = args.port
    ws_scheme = "wss" if args.ssl else "ws"
    http_scheme = "https" if args.ssl else "http"

    health_url = f"{http_scheme}://{host}:{port}/health"
    ws_url = args.server_url or f"{ws_scheme}://{host}:{port}/ws"

    logger.info(f"🔍 Checking if backend is running at {health_url}...")
    server_thread = None

    if not is_server_running(health_url):
        logger.info("⚡ Backend is not running. Starting automated in-process test server...")
        server_thread = start_in_process_server(port=port)
        # Wait up to 20s for server to start and prewarm
        started = False
        for _ in range(40):
            await asyncio.sleep(0.5)
            if is_server_running(health_url):
                started = True
                break
        if not started:
            logger.error("❌ Failed to start in-process backend server. Aborting.")
            return
        logger.info("✅ In-process backend server is live and ready!")
    else:
        logger.info("✅ Connected to already-running backend server!")
        if mode in ("baseline", "preview-cost", "backpressure", "scaling"):
            logger.warning(
                "⚠️  Using an EXTERNAL server. The baseline scenarios read the server-side "
                "perf collector, which is off unless the server was started with "
                "config.perf.enabled = True (and, for the scaling scenario, "
                "config.ws.max_sessions = 0). Expect an empty baseline otherwise."
            )

    # SSL context for benchmark client
    ssl_ctx = None
    if args.ssl:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    # Reset metrics collector
    perf.reset()
    perf.record_resource_checkpoint("benchmark_start")

    scenario_results = {}

    print("\n" + "=" * 80)
    print("         STARTING BACKEND_CPP AUTOMATED PERFORMANCE TEST SUITE")
    print("=" * 80 + "\n")

    try:
        # Scenario 1: Conversational stream
        if mode in ("all", "conversational"):
            res1 = await run_scenario_conversational_stream(
                ws_url,
                ssl_context=ssl_ctx,
                num_turns=args.turns,
                speech_sec=3.0,
                silence_sec=1.0,
            )
            scenario_results["conversational"] = res1

        # Scenario 2: Continuous long speech
        if mode in ("all", "continuous"):
            res2 = await run_scenario_continuous_speech(
                ws_url,
                ssl_context=ssl_ctx,
                duration_sec=12.0,
            )
            scenario_results["continuous"] = res2

        # Scenario 3: Rapid burst & lock contention
        if mode in ("all", "burst"):
            res3 = await run_scenario_rapid_burst_contention(
                ws_url,
                ssl_context=ssl_ctx,
                num_bursts=4,
            )
            scenario_results["burst"] = res3

        # Scenario 4: Leak detection
        if mode in ("all", "leak"):
            res4 = await run_scenario_leak_detection(
                ws_url,
                ssl_context=ssl_ctx,
                cycles=3,
            )
            scenario_results["leak"] = res4

        # --- W2.1 baseline scenarios (audit benchmarks A, D, G) ---
        if mode in ("baseline", "preview-cost"):
            scenario_results["preview_cost"] = await run_scenario_preview_cost(
                ws_url, ssl_context=ssl_ctx
            )

        if mode in ("baseline", "backpressure"):
            scenario_results["backpressure"] = await run_scenario_backpressure(
                ws_url,
                ssl_context=ssl_ctx,
                with_tts=args.with_tts,
            )

        if mode in ("baseline", "scaling"):
            scenario_results["session_scaling"] = await run_scenario_session_scaling(
                ws_url,
                ssl_context=ssl_ctx,
                levels=tuple(args.scale_levels),
            )

        # Real speech: the only scenario that can measure subtitle/translation DELIVERY
        # (the synthetic signal commits but produces an empty transcript).
        if mode in ("baseline", "real-audio"):
            scenario_results["real_audio"] = await run_scenario_real_audio(
                ws_url,
                ssl_context=ssl_ctx,
                audio_path=args.audio_file,
                max_sec=args.audio_max_sec,
                with_tts=args.with_tts,
                vad_engine=args.vad_engine,
                vad_silence_ms=args.vad_silence_ms,
            )

    except Exception as e:
        logger.error(f"Error during benchmark execution: {e}", exc_info=True)

    # Record final resource checkpoint
    perf.record_resource_checkpoint("benchmark_finished")

    # Generate and print summary
    summary_text = perf.generate_markdown_summary()
    print("\n" + summary_text + "\n")

    # Dump output files
    output_json = _HERE / "perf_test_summary.json"
    output_txt = _HERE / "perf_test_summary.txt"

    perf.dump_report_file(str(output_json))
    logger.info(f"💾 Benchmark summary saved to:\n   - {output_json}\n   - {output_txt}")

    # Persist the W2.1 baseline sections (audit benchmarks A, D, G)
    if mode in ("baseline", "preview-cost", "backpressure", "scaling", "real-audio"):
        relevant = {k: v for k, v in scenario_results.items() if k in (
            "preview_cost", "backpressure", "session_scaling", "real_audio"
        )}
        if relevant:
            merge_into_baseline(baseline_path, "e2e", relevant)
            logger.info(f"💾 W2.1 baseline updated: {baseline_path}")

    # Print action advice for user
    print("\n" + "=" * 80)
    print("💡 NEXT STEPS FOR THE USER:")
    print(f"   1. Đã hoàn thành đo hiệu năng toàn diện!")
    print(f"   2. Hãy gửi lại file `{output_json.name}` hoặc nội dung bảng trên cho tôi.")
    print("   3. Tôi sẽ phân tích chính xác từng bottleneck, lock contention và đề xuất tối ưu.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Automated Performance Benchmark for backend_cpp")
    parser.add_argument(
        "--mode",
        default="all",
        choices=[
            "all", "conversational", "continuous", "burst", "leak",
            # W2.1 baseline modes (audit benchmarks A/C/D/G)
            "micro", "baseline", "preview-cost", "backpressure", "scaling", "real-audio",
        ],
        help="Benchmark scenario ('micro' needs no server; 'baseline' runs everything)",
    )
    parser.add_argument(
        "--audio-file",
        default=DEFAULT_REFERENCE_AUDIO,
        help=f"16kHz mono 16-bit WAV for the real-audio scenario (default: {DEFAULT_REFERENCE_AUDIO})",
    )
    parser.add_argument(
        "--audio-max-sec",
        type=float,
        default=24.0,
        help="Only use the first N seconds of the reference audio",
    )
    parser.add_argument(
        "--baseline-out",
        default="report/baseline_perf.json",
        help="Where to write the W2.1 baseline JSON (relative to the project root)",
    )
    parser.add_argument(
        "--vad-engine",
        default="fsmn-vad",
        help=(
            "VAD engine for the real-audio scenario (default: fsmn-vad, the engine that "
            "reliably segments real speech in this harness). Compare engines with "
            "scratch/compare_vad_quality.py."
        ),
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=None,
        help=(
            "Override the VAD trailing-silence limit for the real-audio scenario. The safety "
            "net commits an utterance only after this much silence (default 450ms, measured to "
            "cost 480ms of the ~1.1s perceived subtitle latency). Lower it to trade robustness "
            "for latency; compare with scratch/analyze_latency_budget.py."
        ),
    )
    parser.add_argument(
        "--with-tts",
        dest="with_tts",
        action="store_true",
        help="Enable TTS during the backpressure scenario (loads the TTS model)",
    )
    parser.add_argument(
        "--scale-levels",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Concurrent session counts for the scaling scenario",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--server-url", default=None, help="Custom WebSocket URL (e.g. wss://localhost:8765/ws)")
    parser.add_argument("--turns", type=int, default=3, help="Number of turns for conversational stream")
    parser.add_argument("--no-ssl", dest="ssl", action="store_false", help="Disable SSL/TLS")
    parser.set_defaults(ssl=True)

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
