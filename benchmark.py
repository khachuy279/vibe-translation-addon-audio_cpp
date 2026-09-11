"""Streaming Audio/ASR Benchmark CLI & Regression Tool for Firefox Extension Pipeline.

Usage:
    python benchmark.py
    python benchmark.py --scenarios=baseline,low_latency,balanced
    python benchmark.py --speed=2.0
    python benchmark.py --chunk-ms=64
    python benchmark.py --with-stress
    python benchmark.py --server-url=wss://localhost:8765/ws
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root in sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths
setup_cuda_dll_paths()

from benchmarks.dataset import discover_dataset
from benchmarks.runner import (
    TestCaseResult,
    is_server_alive,
    run_multi_session_stress,
    run_single_test_case,
    start_local_server,
)
from benchmarks.scenarios import get_standard_scenarios, ScenarioConfig
from benchmarks.reporter import generate_markdown_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


async def main_async(args: argparse.Namespace) -> None:
    # 1. Discover dataset
    dataset_dir = Path(args.dataset_dir)
    dataset = discover_dataset(dataset_dir)
    if args.filter:
        dataset = [p for p in dataset if args.filter.lower() in p.pair_id.lower()]
        logger.info(f"🔍 Filter applied ('{args.filter}'): {len(dataset)} pairs matched")
    else:
        logger.info(f"📂 Discovered {len(dataset)} audio/reference pairs in {dataset_dir}")

    # 2. Check or start backend server
    health_url = f"https://127.0.0.1:{args.port}/health"
    if not is_server_alive(health_url):
        logger.info(f"🚀 No running backend detected at {health_url}. Starting in-process server...")
        start_local_server(port=args.port)
    else:
        logger.info(f"⚡ Connected to existing backend server at {health_url}")

    ws_url = args.server_url or f"wss://127.0.0.1:{args.port}/ws"

    # 3. Resolve scenarios
    all_scenarios = get_standard_scenarios()
    if args.scenarios == "all":
        selected_keys = list(all_scenarios.keys())
    else:
        selected_keys = [k.strip() for k in args.scenarios.split(",") if k.strip() in all_scenarios]

    if not selected_keys:
        selected_keys = ["baseline"]

    output_jsonl_path = Path(args.output_jsonl)
    output_report_path = Path(args.output_report)

    # Clean old report if starting a fresh comprehensive run
    logger.info(f"💾 Persistent test results will be appended to: {output_jsonl_path}")

    all_scenario_results: Dict[str, List[TestCaseResult]] = {}
    baseline_results: List[TestCaseResult] = []

    # 4. Execute Scenarios
    for sc_key in selected_keys:
        scenario = all_scenarios[sc_key]
        if args.speed != 1.0:
            scenario.speed = args.speed
        if args.chunk_ms != 64:
            scenario.chunk_ms = args.chunk_ms

        logger.info(f"\n========================================================")
        logger.info(f"▶️ EXECUTING SCENARIO: {scenario.name} (chunk={scenario.chunk_ms}ms, speed={scenario.speed}x)")
        logger.info(f"========================================================")

        results_for_sc: List[TestCaseResult] = []
        for pair in dataset:
            res = await run_single_test_case(
                pair=pair,
                scenario=scenario,
                server_ws_url=ws_url,
                output_jsonl_path=output_jsonl_path,
            )
            results_for_sc.append(res)
            # Brief pause between test cases for server resource cleanup
            await asyncio.sleep(0.5)

        all_scenario_results[sc_key] = results_for_sc
        if sc_key == "baseline":
            baseline_results = results_for_sc

    # If baseline was not run in selection, use first scenario as baseline
    if not baseline_results and all_scenario_results:
        baseline_results = list(all_scenario_results.values())[0]

    # 5. Multi-session stress test if requested
    stress_results = None
    if args.with_stress and dataset:
        # Use first representative speech file
        stress_pair = next((p for p in dataset if "noise" not in p.pair_id.lower()), dataset[0])
        logger.info(f"\n========================================================")
        logger.info(f"⚡ RUNNING MULTI-SESSION STRESS TEST ON {stress_pair.pair_id}")
        logger.info(f"========================================================")
        stress_results = await run_multi_session_stress(
            pair=stress_pair,
            scenario=all_scenarios.get("baseline", list(all_scenarios.values())[0]),
            server_ws_url=ws_url,
            concurrency_levels=[1, 2, 4],
            output_jsonl_path=output_jsonl_path,
        )

    # 6. Generate Markdown report conforming to Section 20
    report_text = generate_markdown_report(
        dataset=dataset,
        baseline_results=baseline_results,
        comparison_results=all_scenario_results,
        stress_results=stress_results,
        output_path=output_report_path,
    )

    logger.info(f"\n📊 Benchmark Report generated successfully at: {output_report_path}")
    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETED — REPORT SAVED: {output_report_path}")
    print(f"RESULTS JSONL SAVED: {output_jsonl_path}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Audio/ASR Benchmark Suite for Firefox Extension")
    parser.add_argument("--dataset-dir", default="wav_test", help="Path to wav_test directory")
    parser.add_argument("--filter", default=None, help="Substring filter for audio filename (e.g. Japanese_5s)")
    parser.add_argument(
        "--scenarios",
        default="baseline,low_latency,balanced",
        help="Comma-separated scenarios: baseline, low_latency, high_accuracy, balanced, all",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Pacing factor (1.0 = realtime, 2.0 = fast, inf = unthrottled)")
    parser.add_argument("--chunk-ms", type=int, default=64, help="Chunk duration in ms (10, 20, 25, 40, 50, 64, 100, 200)")
    parser.add_argument("--port", type=int, default=8765, help="Port of backend server")
    parser.add_argument("--server-url", default=None, help="WebSocket URL e.g. wss://127.0.0.1:8765/ws")
    parser.add_argument("--with-stress", action="store_true", help="Run multi-session concurrency stress test")
    parser.add_argument("--output-jsonl", default="benchmark_results.jsonl", help="Output JSONL filepath")
    parser.add_argument("--output-report", default="report/streaming_benchmark_report.md", help="Output Markdown report path")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
