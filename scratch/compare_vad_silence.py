"""Compare VAD silence settings: trailing-silence latency vs transcript safety.

The VAD safety net commits an utterance only after `silence_duration_ms` of silence (quantized to
the engine frame, 60 ms for fsmn). That is the single largest stage of perceived subtitle latency
and costs nothing in compute, so the tradeoff is purely "latency vs splitting on natural pauses".

Usage:  python scratch/compare_vad_silence.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ("450", "250", "150", "100")


def trailing_silence(tag: str) -> list:
    log = ROOT / "scratch" / f"sil{tag}.log"
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"after (\d+)ms silence", line)
        if match:
            out.append(int(match.group(1)))
    return out


def scenario(tag: str) -> dict:
    path = ROOT / "scratch" / f"sil{tag}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["end_to_end_scenarios"]["real_audio"]


def main() -> int:
    reference = scenario(SETTINGS[0])
    ref_texts = [t["text"] for t in reference.get("transcripts", [])]
    if not ref_texts:
        print("run scratch/sil450.* first")
        return 1

    header = f"{'silence_ms':>10} {'trailing_silence_ms':>20} {'utt':>4} {'subs':>5} {'transcripts':>12}"
    print(header)
    print("-" * len(header))

    for tag in SETTINGS:
        silence = trailing_silence(tag)
        data = scenario(tag)
        if not data:
            print(f"{tag:>10} {'(no run)':>20}")
            continue
        texts = [t["text"] for t in data["transcripts"]]
        verdict = "IDENTICAL" if texts == ref_texts else f"DIFF {len(texts)}/{len(ref_texts)}"
        trailing = ",".join(str(v) for v in sorted(set(silence))) or "n/a"
        print(
            f"{tag:>10} {trailing:>20} {data['delivery']['final_utterances']:>4} "
            f"{data['delivery']['translations']:>5} {verdict:>12}"
        )

    print("\nPerceived latency = trailing_silence + (VAD END -> COMMIT) + (COMMIT -> subtitle)")
    print("  (VAD END -> COMMIT measures ~155-165 ms, COMMIT -> subtitle ~437 ms from the log)")
    for tag in SETTINGS:
        silence = trailing_silence(tag)
        if silence:
            print(f"  silence={tag:>4} -> approx total {silence[0] + 160 + 437:>5} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
