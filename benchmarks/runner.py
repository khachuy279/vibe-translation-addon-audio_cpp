"""Benchmark Runner: executes test suite, outputs benchmark_results.jsonl, and renders markdown report."""

import asyncio
import json
import logging
import os
import platform
import ssl
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch

from benchmarks.accuracy import evaluate_accuracy, AccuracyResult
from benchmarks.client import BenchmarkSession
from benchmarks.dataset import discover_dataset, DatasetPair
from benchmarks.metrics import TimelineCollector, LatencyReport, SubtitleQualityReport
from benchmarks.resources import ResourceMonitor, ResourceSnapshot
from benchmarks.scenarios import ScenarioConfig, get_standard_scenarios
from benchmarks.simulator import StreamingAudioSimulator

logger = logging.getLogger("benchmarks.runner")


@dataclass
class TestCaseResult:
    """Individual test case execution record."""
    pair_id: str
    file_name: str
    scenario_id: str
    scenario_name: str
    language: str
    condition: str
    speed_mode: str
    audio_duration_sec: float
    wall_duration_sec: float
    accuracy: AccuracyResult
    latency: LatencyReport
    quality: SubtitleQualityReport
    resources: ResourceSnapshot
    raw_hypothesis: str
    server_telemetry: Optional[Dict[str, Any]]
    config_snapshot: Dict[str, Any]

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Format test result for persistent benchmark_results.jsonl."""
        return {
            "timestamp": time.time(),
            "test_id": f"{self.scenario_id}_{self.pair_id}",
            "scenario": self.scenario_id,
            "scenario_name": self.scenario_name,
            "file_name": self.file_name,
            "language": self.language,
            "condition": self.condition,
            "speed_mode": self.speed_mode,
            "audio_duration_sec": self.audio_duration_sec,
            "wall_duration_sec": round(self.wall_duration_sec, 2),
            "config": self.config_snapshot,
            "ground_truth": {
                "raw_reference": self.accuracy.raw_reference,
                "normalized_reference": self.accuracy.normalized_reference,
                "is_annotated": self.accuracy.is_annotated,
            },
            "hypothesis": {
                "raw_hypothesis": self.raw_hypothesis,
                "normalized_hypothesis": self.accuracy.normalized_hypothesis,
            },
            "accuracy": self.accuracy.to_dict(),
            "latency": self.latency.to_dict(),
            "quality": self.quality.to_dict(),
            "resources": self.resources.to_dict(),
        }


def is_server_alive(health_url: str) -> bool:
    """Check if server is already reachable."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "HealthCheck"})
        with urllib.request.urlopen(req, context=ctx, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_local_server(port: int = 8765) -> threading.Thread:
    """Start local uvicorn backend server in background daemon thread."""
    import uvicorn
    from backend_cpp.main import app
    from backend_cpp.utils.ssl_utils import ensure_ssl_certificates
    from backend_cpp.config import config as app_cfg

    # Enable perf profiler so /api/perf/summary records metrics
    app_cfg.perf.enabled = True
    # Allow concurrent sessions for multi-session benchmarking & stress test
    app_cfg.ws.max_sessions = 16

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
        ws_ping_interval=None,
        **ssl_kwargs,
    )
    server = uvicorn.Server(server_config)

    t = threading.Thread(target=server.run, daemon=True, name="in_process_benchmark_server")
    t.start()

    # Wait for server to boot
    health_url = f"https://127.0.0.1:{port}/health"
    for _ in range(60):
        if is_server_alive(health_url):
            logger.info(f"Local benchmark server up on port {port}")
            return t
        time.sleep(0.5)

    raise RuntimeError(f"Server failed to start on port {port} within 30s")


async def run_single_test_case(
    pair: DatasetPair,
    scenario: ScenarioConfig,
    server_ws_url: str,
    output_jsonl_path: Optional[Path] = None,
) -> TestCaseResult:
    """Execute a single test case: stream audio, collect metrics, evaluate accuracy."""
    logger.info(
        f"▶️ [{scenario.scenario_id.upper()}] Running {pair.pair_id} "
        f"({pair.actual_duration_sec:.1f}s, lang={pair.inferred_language}, cond={pair.inferred_condition})..."
    )

    session_cfg = scenario.to_ws_config()
    # Explicitly guide language if known from dataset
    if pair.inferred_language not in ("UNKNOWN", "multi"):
        session_cfg["sourceLang"] = pair.inferred_language

    client = BenchmarkSession(server_ws_url, session_config=session_cfg)
    client.reset_server_perf()

    # Setup simulator
    sim = StreamingAudioSimulator(
        audio_source=pair.wav_path,
        chunk_ms=scenario.chunk_ms,
        speed=scenario.speed,
        trailing_silence_sec=1.5,
    )

    timeline = TimelineCollector(audio_duration_sec=pair.actual_duration_sec)
    res_monitor = ResourceMonitor(interval_sec=0.05)

    res_monitor.start()
    t_start = time.perf_counter()

    try:
        await client.connect()
        await client.stream_audio(sim, timeline, drain_timeout_sec=4.0)
    finally:
        await client.disconnect()
        t_end = time.perf_counter()
        res_snapshot = res_monitor.stop()

    wall_duration = t_end - t_start

    # Query server-side performance summary
    telemetry = client.get_server_perf_summary()

    # Assemble delivered hypothesis text
    # Group unique final commits by utterance_id preserving arrival order
    finals_by_utt: Dict[str, str] = {}
    for f in timeline.finals:
        uid = f.get("utterance_id")
        txt = (f.get("text") or "").strip()
        if uid and txt and uid not in finals_by_utt:
            finals_by_utt[uid] = txt

    raw_hypothesis = " ".join(finals_by_utt.values()).strip()
    if not raw_hypothesis and timeline.partials:
        # If utterance didn't commit for some reason, use latest partial
        raw_hypothesis = timeline.partials[-1].get("text", "").strip()

    # Accuracy calculation
    acc_result = evaluate_accuracy(
        raw_reference=pair.raw_reference,
        raw_hypothesis=raw_hypothesis,
        language=pair.inferred_language,
    )

    # Latency and quality
    lat_result = timeline.compute_latency(server_telemetry=telemetry)
    quality_result = timeline.analyze_quality()

    test_result = TestCaseResult(
        pair_id=pair.pair_id,
        file_name=pair.wav_path.name,
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        language=pair.inferred_language,
        condition=pair.inferred_condition,
        speed_mode=f"{scenario.speed:.1f}x",
        audio_duration_sec=round(pair.actual_duration_sec, 2),
        wall_duration_sec=wall_duration,
        accuracy=acc_result,
        latency=lat_result,
        quality=quality_result,
        resources=res_snapshot,
        raw_hypothesis=raw_hypothesis,
        server_telemetry=telemetry,
        config_snapshot=session_cfg,
    )

    # Append to benchmark_results.jsonl if requested
    if output_jsonl_path:
        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(test_result.to_jsonl_dict(), ensure_ascii=False) + "\n")

    logger.info(
        f"✅ [{scenario.scenario_id.upper()}] {pair.pair_id}: "
        f"TTFS={lat_result.ttfs_ms if lat_result.ttfs_ms is not None else 'N/A'}ms, "
        f"FinalLat={lat_result.final_latency_ms if lat_result.final_latency_ms is not None else 'N/A'}ms, "
        f"RTF_asr={lat_result.rtf_asr:.3f}, "
        f"WER={f'{acc_result.wer*100:.1f}%' if acc_result.wer is not None else 'N/A'}, "
        f"CER={f'{acc_result.cer*100:.1f}%' if acc_result.cer is not None else 'N/A'}"
    )

    return test_result


async def run_multi_session_stress(
    pair: DatasetPair,
    scenario: ScenarioConfig,
    server_ws_url: str,
    concurrency_levels: List[int] = [1, 2, 4],
    output_jsonl_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    """Run concurrent streams to detect serialization, lock contention, and queue depths."""
    stress_results: Dict[int, Dict[str, Any]] = {}

    for N in concurrency_levels:
        logger.info(f"⚡ [STRESS TEST] Running {N} concurrent session(s) on {pair.pair_id}...")
        res_monitor = ResourceMonitor(interval_sec=0.05)
        res_monitor.start()
        t0 = time.perf_counter()

        tasks = []
        for i in range(N):
            tasks.append(
                run_single_test_case(
                    pair=pair,
                    scenario=scenario,
                    server_ws_url=server_ws_url,
                    output_jsonl_path=None,  # Do not duplicate in jsonl
                )
            )

        results: List[TestCaseResult] = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.perf_counter() - t0
        res = res_monitor.stop()

        valid_results = [r for r in results if isinstance(r, TestCaseResult)]
        failures = len(results) - len(valid_results)

        avg_ttfs = (
            sum(r.latency.ttfs_ms for r in valid_results if r.latency.ttfs_ms is not None)
            / max(1, sum(1 for r in valid_results if r.latency.ttfs_ms is not None))
            if valid_results
            else 0.0
        )
        avg_rtf = (
            sum(r.latency.rtf_asr for r in valid_results) / max(1, len(valid_results))
            if valid_results
            else 0.0
        )

        stress_record = {
            "concurrency": N,
            "wall_time_sec": round(wall_time, 2),
            "failures": failures,
            "avg_ttfs_ms": round(avg_ttfs, 1),
            "avg_rtf_asr": round(avg_rtf, 3),
            "cpu_peak_percent": res.cpu_peak_percent,
            "ram_peak_mb": res.ram_peak_mb,
            "vram_peak_mb": res.vram_peak_mb,
        }
        stress_results[N] = stress_record

        if output_jsonl_path:
            with open(output_jsonl_path, "a", encoding="utf-8") as f:
                rec = {
                    "timestamp": time.time(),
                    "test_id": f"stress_N{N}_{pair.pair_id}",
                    "scenario": "stress",
                    "stress_record": stress_record,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stress_results
