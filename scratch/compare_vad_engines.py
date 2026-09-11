"""Thin CLI wrapper around the maintained VAD engine trade-off benchmark.

The real implementation lives in ``backend_cpp/tests/perf_baseline.py``
(``benchmark_vad_engine_tradeoff``) so the measurement is covered by the baseline guard
tests and lands in ``report/baseline_perf.json``. This script only renders it.

Usage:
    python scratch/compare_vad_engines.py
    python scratch/compare_vad_engines.py --audio wav_test/OSR_us_000_0010_16k.wav --seconds 24
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend_cpp.tests.perf_baseline import benchmark_vad_engine_tradeoff  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default="wav_test/OSR_us_000_0010_16k.wav")
    parser.add_argument("--seconds", type=float, default=24.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out", default=str(Path(__file__).with_name("vad_engine_comparison.json"))
    )
    args = parser.parse_args()

    result = benchmark_vad_engine_tradeoff(
        audio_path=Path(args.audio),
        max_sec=args.seconds,
        repeats=args.repeats,
    )

    print(f"audio: {Path(result['audio_file']).name}  ({result['audio_sec']}s of REAL speech)\n")
    print(f"{'engine':<14} {'ms/audio-sec':>12} {'RTF':>7} {'segments':>9} "
          f"{'speech s':>9} {'ratio':>7} {'x cheap':>8} {'ratio delta':>12}")
    print("-" * 86)

    for row in result["rows"]:
        if not row.get("available"):
            print(f"{row['engine']:<14} unavailable: {row.get('error')}")
            continue
        print(
            f"{row['engine']:<14} {row['ms_per_audio_sec']:>12.2f} {row['rtf']:>7.4f} "
            f"{row['segments']:>9} {row['speech_sec']:>9.2f} {row['speech_ratio']:>7.3f} "
            f"{row.get('cost_multiple_vs_cheapest', 0):>8.2f} "
            f"{row.get('speech_ratio_delta_vs_cheapest', 0):>+12.3f}"
        )

    print("\nfirst detected segments (start, end) per engine:")
    for row in result["rows"]:
        if row.get("available"):
            print(f"  {row['engine']:<14} {row['first_segments']}")

    print(f"\ncheapest: {result['cheapest_engine']}")
    print(
        "\nReminder: a lower speech_ratio means the engine trims boundaries more (or misses\n"
        "speech). Check the delta against subtitle quality before switching the default."
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
