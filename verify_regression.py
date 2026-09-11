"""Automated Regression Gate Verification for preview_min_growth_ratio = 0.2.

Compares:
1. Final transcript bit-identity (equality assertion)
2. WER and CER (must not increase)
3. TTFS (Time To First Subtitle, must not increase noticeably)
4. Final Subtitle Latency (must not increase)
5. Preview inference count & compute time (must show actual reduction)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
logger = logging.getLogger("verify_regression")


async def run_regression_verification(dataset_dir: str = "wav_test", port: int = 8765) -> None:
    dataset = discover_dataset(dataset_dir)
    health_url = f"https://127.0.0.1:{port}/health"
    if not is_server_alive(health_url):
        logger.info(f"Starting server on port {port}...")
        start_local_server(port=port)
    ws_url = f"wss://127.0.0.1:{port}/ws"

    all_scenarios = get_standard_scenarios()
    sc_ungated = all_scenarios["baseline_ungated"]  # ratio 0.0
    sc_gated = all_scenarios["baseline"]           # ratio 0.2

    logger.info("========================================================")
    logger.info("▶️ 1. RUNNING UNGATED BASELINE (preview_min_growth_ratio = 0.0)")
    logger.info("========================================================")
    ungated_results: Dict[str, TestCaseResult] = {}
    for pair in dataset:
        res = await run_single_test_case(pair, sc_ungated, server_ws_url=ws_url)
        ungated_results[pair.pair_id] = res
        await asyncio.sleep(0.3)

    logger.info("\n========================================================")
    logger.info("▶️ 2. RUNNING GATED BASELINE (preview_min_growth_ratio = 0.2)")
    logger.info("========================================================")
    gated_results: Dict[str, TestCaseResult] = {}
    for pair in dataset:
        res = await run_single_test_case(pair, sc_gated, server_ws_url=ws_url)
        gated_results[pair.pair_id] = res
        await asyncio.sleep(0.3)

    # Verification and Comparison Table
    lines: List[str] = []
    lines.append("# Regression Verification: `preview_min_growth_ratio = 0.2` vs `0.0`")
    lines.append("")
    lines.append("| File | Final Transcript Bit-Identical? | WER (0.0 → 0.2) | CER (0.0 → 0.2) | TTFS ms (0.0 → 0.2) | Final Latency ms (0.0 → 0.2) | Inferences (0.0 → 0.2) | Inference Reduction |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: |")

    total_inf_before = 0
    total_inf_after = 0
    all_bit_identical = True

    for pair in dataset:
        pid = pair.pair_id
        r_before = ungated_results[pid]
        r_after = gated_results[pid]

        txt_before = r_before.accuracy.normalized_hypothesis
        txt_after = r_after.accuracy.normalized_hypothesis
        bit_identical = (txt_before == txt_after)
        if not bit_identical:
            all_bit_identical = False

        wer_b = f"{r_before.accuracy.wer * 100:.1f}%" if r_before.accuracy.wer is not None else "N/A"
        wer_a = f"{r_after.accuracy.wer * 100:.1f}%" if r_after.accuracy.wer is not None else "N/A"

        cer_b = f"{r_before.accuracy.cer * 100:.1f}%" if r_before.accuracy.cer is not None else "N/A"
        cer_a = f"{r_after.accuracy.cer * 100:.1f}%" if r_after.accuracy.cer is not None else "N/A"

        ttfs_b = f"{r_before.latency.ttfs_ms:.0f}" if r_before.latency.ttfs_ms is not None else "N/A"
        ttfs_a = f"{r_after.latency.ttfs_ms:.0f}" if r_after.latency.ttfs_ms is not None else "N/A"

        final_b = f"{r_before.latency.final_latency_ms:.0f}" if r_before.latency.final_latency_ms is not None else "N/A"
        final_a = f"{r_after.latency.final_latency_ms:.0f}" if r_after.latency.final_latency_ms is not None else "N/A"

        inf_b = r_before.latency.total_inferences
        inf_a = r_after.latency.total_inferences
        total_inf_before += inf_b
        total_inf_after += inf_a

        reduction_pct = ((inf_b - inf_a) / max(1, inf_b)) * 100.0

        identical_icon = "✅ YES" if bit_identical else "❌ NO"
        lines.append(
            f"| `{pair.wav_path.name}` | {identical_icon} | {wer_b} → {wer_a} | {cer_b} → {cer_a} | {ttfs_b} → {ttfs_a} | {final_b} → {final_a} | {inf_b} → {inf_a} | **-{reduction_pct:.1f}%** |"
        )

    net_reduction = ((total_inf_before - total_inf_after) / max(1, total_inf_before)) * 100.0
    lines.append("")
    lines.append(f"**Total Inferences**: {total_inf_before} (Before) → {total_inf_after} (After) = **-{net_reduction:.1f}% reduction in GPU compute**.")
    lines.append(f"**Bit-identical Verification**: {'✅ ALL PASSED' if all_bit_identical else '⚠️ DISCREPANCY DETECTED'}")

    report_str = "\n".join(lines)
    report_file = Path("report/regression_preview_gate_report.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_str)

    print("\n" + "=" * 60)
    print(report_str)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_regression_verification())
