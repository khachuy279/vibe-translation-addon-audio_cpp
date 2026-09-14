"""Speaking-rate plausibility: is the reference or the hypothesis the right length?

Japanese broadcast speech runs ~6-10 characters/second. If the reference implies a
rate far below that while a model's output implies a normal rate, the reference is
under-labelled (caption-style) and the model is transcribing the audio correctly.

This is independent of any ASR model's opinion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "ja_eval"

rows = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "utts.jsonl", encoding="utf-8") if l.strip()}
data = json.loads((REPO / "report" / "ja_offline_bench.json").read_text(encoding="utf-8"))
models = {r["model"]: {u["id"]: u for u in r["per_utt"]} for r in data["results"] if "error" not in r}

# Japanese punctuation is stripped for a fair char count.
import re  # noqa: E402

PUNCT = re.compile(r"[\s、。，．・！？!?「」『』（）()\[\]{}〜～ー―—–\-…‥：:；;]")


def chars(s: str) -> int:
    return len(PUNCT.sub("", s))


ids = [i for i in rows if i in models["qwen3-asr-1.7b"]]

print(f"{'model':<24}{'chars/s median':>16}{'p10':>8}{'p90':>8}   (audio-second normalised)")
rates = {}
for name, utts in models.items():
    vals = []
    for uid in ids:
        dur = rows[uid]["duration_sec"]
        vals.append(chars(utts[uid]["hyp"]) / max(0.3, dur))
    vals = np.array(vals)
    rates[name] = vals
    print(f"{name:<24}{np.median(vals):>16.2f}{np.percentile(vals,10):>8.2f}{np.percentile(vals,90):>8.2f}")

ref_vals = np.array([chars(rows[uid]["text"]) / max(0.3, rows[uid]["duration_sec"]) for uid in ids])
print(f"{'REFERENCE':<24}{np.median(ref_vals):>16.2f}{np.percentile(ref_vals,10):>8.2f}{np.percentile(ref_vals,90):>8.2f}")

print("\nPer-utterance comparison on the 10 worst qwen cases:")
q = models["qwen3-asr-1.7b"]
c = models["cohere-transcribe"]
worst = sorted(ids, key=lambda uid: -(q[uid]["cer"] or 0))[:10]
print(f"{'id':<10}{'dur':>6}{'ref c/s':>9}{'qwen c/s':>10}{'cohe c/s':>10}   ref / qwen")
for uid in worst:
    dur = rows[uid]["duration_sec"]
    print(
        f"{uid:<10}{dur:>6.2f}{chars(rows[uid]['text'])/dur:>9.2f}"
        f"{chars(q[uid]['hyp'])/dur:>10.2f}{chars(c[uid]['hyp'])/dur:>10.2f}"
        f"   {rows[uid]['text'][:22]} / {q[uid]['hyp'][:34]}"
    )

frac_ref_slow = float(np.mean(ref_vals < 3.0))
frac_ref_normal = float(np.mean((ref_vals >= 4.0) & (ref_vals <= 11.0)))
print(f"\nReference rate below 3.0 chars/s (implausibly sparse): {100*frac_ref_slow:.1f}% of utterances")
print(f"Reference rate in normal spoken band 4-11 chars/s      : {100*frac_ref_normal:.1f}%")
print(f"qwen rate in normal spoken band 4-11 chars/s           : {100*np.mean((rates['qwen3-asr-1.7b']>=4)&(rates['qwen3-asr-1.7b']<=11)):.1f}%")
print(f"cohere rate in normal spoken band 4-11 chars/s         : {100*np.mean((rates['cohere-transcribe']>=4)&(rates['cohere-transcribe']<=11)):.1f}%")
