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

    # SSL context for benchmark client
    ssl_ctx = None
    if args.ssl:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    # Reset metrics collector
    perf.reset()
    perf.record_resource_checkpoint("benchmark_start")

    mode = args.mode.lower()
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

    # Print action advice for user
    print("\n" + "=" * 80)
    print("💡 NEXT STEPS FOR THE USER:")
    print(f"   1. Đã hoàn thành đo hiệu năng toàn diện!")
    print(f"   2. Hãy gửi lại file `{output_json.name}` hoặc nội dung bảng trên cho tôi.")
    print("   3. Tôi sẽ phân tích chính xác từng bottleneck, lock contention và đề xuất tối ưu.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Automated Performance Benchmark for backend_cpp")
    parser.add_argument("--mode", default="all", choices=["all", "conversational", "continuous", "burst", "leak"], help="Benchmark scenario")
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
