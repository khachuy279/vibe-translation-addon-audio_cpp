"""Silence Duration Benchmark Matrix Runner (100 / 120 / 150 / 180 / 200 ms).

Executes all 5 silence durations across the dataset and compares:
- TTFS
- Final subtitle latency
- WER
- CER
- Missing speech (deletions)
- Utterance fragmentation (number of committed utterances)
- Subtitle stability
- Number of ASR commits
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List

from benchmarks.dataset import discover_dataset
from benchmarks.runner import (
    TestCaseResult,
    is_server_alive,
    run_single_test_case,
    start_local_server,
)
from benchmarks.scenarios import get_standard_scenarios

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("silence_matrix")

SILENCE_LEVELS = [100, 120, 150, 180, 200]


async def run_silence_matrix(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    dataset = discover_dataset(dataset_dir)
    if args.filter:
        dataset = [p for p in dataset if args.filter.lower() in p.pair_id.lower()]
    logger.info(f"📂 Running Silence Matrix on {len(dataset)} dataset pairs...")

    health_url = f"https://127.0.0.1:{args.port}/health"
    if not is_server_alive(health_url):
        logger.info(f"🚀 Starting local backend server on port {args.port}...")
        start_local_server(port=args.port)
    ws_url = args.server_url or f"wss://127.0.0.1:{args.port}/ws"

    all_scenarios = get_standard_scenarios()
    matrix_keys = [f"silence_{ms}ms" for ms in SILENCE_LEVELS]

    results_by_silence: Dict[int, List[TestCaseResult]] = {}

    for ms in SILENCE_LEVELS:
        sc_key = f"silence_{ms}ms"
        scenario = all_scenarios[sc_key]
        logger.info(f"\n========================================================")
        logger.info(f"▶️ RUNNING SILENCE DURATION: {ms}ms")
        logger.info(f"========================================================")

        res_list: List[TestCaseResult] = []
        for pair in dataset:
            res = await run_single_test_case(
                pair=pair,
                scenario=scenario,
                server_ws_url=ws_url,
                output_jsonl_path=Path(args.output_jsonl),
            )
            res_list.append(res)
            await asyncio.sleep(0.3)

        results_by_silence[ms] = res_list

    # Generate Markdown Silence Comparison Table
    lines: List[str] = []
    lines.append("# VAD Silence Duration Benchmark Matrix (100ms – 200ms)")
    lines.append("")
    lines.append("| Silence Setting | Avg TTFS (ms) | Final Latency (ms) | WER (%) | CER (%) | Missing Chars | Total Commits | Total Partials | Rollbacks | Fragmentation (Utterances/File) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for ms in SILENCE_LEVELS:
        res_list = results_by_silence[ms]
        valid_ttfs = [r.latency.ttfs_ms for r in res_list if r.latency.ttfs_ms is not None]
        avg_ttfs = sum(valid_ttfs) / len(valid_ttfs) if valid_ttfs else 0.0

        valid_finals = [r.latency.final_latency_ms for r in res_list if r.latency.final_latency_ms is not None]
        avg_final = sum(valid_finals) / len(valid_finals) if valid_finals else 0.0

        ann = [r for r in res_list if r.accuracy.is_annotated]
        wers = [r.accuracy.wer for r in ann if r.accuracy.wer is not None]
        avg_wer = (sum(wers) / len(wers) * 100.0) if wers else 0.0

        cers = [r.accuracy.cer for r in ann if r.accuracy.cer is not None]
        avg_cer = (sum(cers) / len(cers) * 100.0) if cers else 0.0

        total_dels = sum(r.accuracy.deletions for r in ann)
        total_commits = sum(r.quality.unique_finals for r in res_list)
        total_partials = sum(r.quality.total_partials for r in res_list)
        total_rollbacks = sum(r.quality.rollback_count for r in res_list)
        avg_frag = float(total_commits) / max(1, len(res_list))

        lines.append(
            f"| **{ms} ms** | {avg_ttfs:.1f} | {avg_final:.1f} | {avg_wer:.1f}% | {avg_cer:.1f}% | {total_dels} | {total_commits} | {total_partials} | {total_rollbacks} | {avg_frag:.1f} |"
        )

    matrix_report = "\n".join(lines)
    report_file = Path(args.output_report)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(matrix_report)

    print("\n" + "=" * 60)
    print(matrix_report)
    print("=" * 60 + "\n")
    logger.info(f"Silence matrix report saved to: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VAD Silence Duration Benchmark Matrix")
    parser.add_argument("--dataset-dir", default="wav_test", help="Dataset directory")
    parser.add_argument("--filter", default=None, help="Filter audio file substring")
    parser.add_argument("--port", type=int, default=8765, help="Backend port")
    parser.add_argument("--server-url", default=None, help="WebSocket URL")
    parser.add_argument("--output-jsonl", default="benchmark_results.jsonl", help="Output JSONL")
    parser.add_argument("--output-report", default="report/silence_duration_matrix_report.md", help="Output Markdown report")
    args = parser.parse_args()
    asyncio.run(run_silence_matrix(args))


if __name__ == "__main__":
    main()
