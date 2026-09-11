"""Compare two real-audio baseline runs: preview growth gate OFF vs ON.

W2.2's mandatory gate is transcript equivalence: the gate only skips PREVIEWS, so the final
transcripts must be identical. Any difference means the optimisation changed ASR behaviour and
must be rejected.

Usage:  python scratch/compare_preview_gate.py <off.json> <on.json>
"""

import json
import sys


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["end_to_end_scenarios"]["real_audio"]


def summarise(name, data):
    amp = data["preview_amplification"]
    counters = data["counters"]
    infer = data["asr"]["infer_ms"]
    return {
        "name": name,
        "previews": counters["asr.preview_infers"],
        "commits": counters["asr.commit_inferences"],
        "skipped": counters["asr.preview_skipped_growth"],
        "preview_audio_ms": amp["preview_audio_ms"],
        "commit_audio_ms": amp["commit_audio_ms"],
        "total_audio_ms": amp["preview_audio_ms"] + amp["commit_audio_ms"],
        "amplification": amp["factor_vs_speech"],
        "infer_total_ms": round(infer["count"] * infer["avg"], 1),
        "infer_avg_ms": infer["avg"],
        "infer_p90_ms": infer["p90"],
        "utterances": data["delivery"]["final_utterances"],
    }


def main():
    off = summarise("gate OFF", load(sys.argv[1]))
    on = summarise("gate ON ", load(sys.argv[2]))

    keys = [
        "previews",
        "commits",
        "skipped",
        "preview_audio_ms",
        "commit_audio_ms",
        "total_audio_ms",
        "amplification",
        "infer_total_ms",
        "infer_avg_ms",
        "infer_p90_ms",
        "utterances",
    ]
    width = max(len(k) for k in keys)
    print(f"{'metric':<{width}}  {'gate OFF':>14}  {'gate ON':>14}  {'change':>10}")
    print("-" * (width + 44))
    for key in keys:
        a, b = off[key], on[key]
        if isinstance(a, str):
            change = "-"
        elif a:
            change = f"{(b - a) / a * 100:+.1f}%"
        else:
            change = "n/a"
        print(f"{key:<{width}}  {str(a):>14}  {str(b):>14}  {change:>10}")

    texts_off = [t["text"] for t in load(sys.argv[1])["transcripts"]]
    texts_on = [t["text"] for t in load(sys.argv[2])["transcripts"]]
    identical = texts_off == texts_on
    print("\ntranscript equivalence:", "IDENTICAL" if identical else "DIFFERENT")
    if not identical:
        print("  gate OFF:", texts_off)
        print("  gate ON :", texts_on)
    print(f"  {len(texts_off)} utterances compared")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
