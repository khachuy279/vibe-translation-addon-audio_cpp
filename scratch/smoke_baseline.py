"""Smoke-test the W2.1 baseline micro-benchmarks (no server or GPU required)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend_cpp.tests.perf_baseline import (  # noqa: E402
    benchmark_copy_chain,
    benchmark_snapshot_and_normalize,
    benchmark_vad_cost,
    collect_environment,
)

print("=== environment ===")
env = collect_environment()
print(json.dumps({k: env[k] for k in ("python", "cpu_count", "asr_model", "vad_engine", "poll_interval_ms") if k in env}, indent=2))

print("\n=== A (CPU side): snapshot + normalizer cost vs utterance length ===")
result = benchmark_snapshot_and_normalize(durations_sec=(1, 2, 4, 6, 8), repeats=3)
for row in result["rows"]:
    print(
        f"  {row['audio_sec']:>4.0f}s | snapshot {row['snapshot']['median_ms']:>6.2f} ms "
        f"({row['snapshot']['peak_alloc_bytes']/1024:>8.1f} KiB) | "
        f"normalize {row['normalize']['median_ms']:>6.2f} ms "
        f"({row['normalize']['peak_alloc_bytes']/1024:>8.1f} KiB) | "
        f"pre-infer {row['pre_infer_ms']:>6.2f} ms "
        f"({row['pre_infer_ms_per_audio_sec']:.2f} ms per audio-sec)"
    )
print("  scaling:", json.dumps(result["scaling"]))

print("\n=== C: copy / conversion chain (1s of audio) ===")
chain = benchmark_copy_chain(duration_sec=1.0, repeats=2)
for stage in chain["stages"]:
    tag = " [REFERENCE]" if stage.get("is_reference_only") else ""
    print(
        f"  {stage['stage']:<55} {stage['median_ms']:>7.2f} ms | "
        f"peak alloc {stage['peak_alloc_bytes']/1024:>9.1f} KiB | out {stage['output_bytes']:>8} B{tag}"
    )
print(f"  total: {chain['total_median_ms']:.2f} ms  ({chain['total_ms_per_audio_sec']:.2f} ms/audio-sec)")

print("\n=== VAD: real feed_chunk cost per engine (2s audio) ===")
vad = benchmark_vad_cost(duration_sec=2.0, repeats=3)
for row in vad["rows"]:
    print(
        f"  {row['engine']:<14} frame={row['native_frame_samples']:>4} samples "
        f"({row['frames_per_audio_sec']:.1f} fps) | {row['median_ms']:>8.2f} ms | "
        f"RTF {row['rtf']:.3f} | {row['ms_per_audio_sec']:.1f} ms/audio-sec | "
        f"speech frames -> ASR: {row['speech_frames_passed_to_asr']}"
    )
