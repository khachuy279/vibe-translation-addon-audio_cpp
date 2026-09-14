"""Why did raising silence_duration_ms make CER worse, despite near-perfect segmentation?

Compares the committed hypotheses at 150 ms vs 600 ms against the reference to see whether
merging introduced cross-sentence contamination.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.ja_text import normalize_ja, score_ja  # noqa: E402

runs = {}
for tag, f in [("s150", "report/jacv_stream_qwen.json"), ("s600", "report/jacv_stream_sil600.json"),
               ("s900", "report/jacv_stream_sil900.json")]:
    p = REPO / f
    if p.exists():
        runs[tag] = {r["pair_id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["results"]}

for pid in sorted(runs["s150"]):
    ref = runs["s150"][pid]["raw_reference"]
    print(f"\n{'='*100}\n{pid}  REF: {ref[:120]}")
    for tag in ("s150", "s600", "s900"):
        if tag not in runs or pid not in runs[tag]:
            continue
        r = runs[tag][pid]
        print(f"\n  [{tag}] CER={r['cer']*100:5.2f}%  commits={r['commit_count']}")
        print(f"        HYP: {r['raw_hypothesis'][:170]}")

# Aggregate: does the 600 ms run put MORE text in one commit than the reference sentence?
print(f"\n\n{'='*100}\nCOMMIT LENGTH PROFILE")
print(f"{'run':<6}{'commits':>9}{'mean chars/commit':>20}{'max':>7}")
for tag, by in runs.items():
    lens = [len(c["text"]) for r in by.values() for c in r["commits"] if c["text"]]
    print(f"{tag:<6}{len(lens):>9}{sum(lens)/max(1,len(lens)):>20.1f}{max(lens) if lens else 0:>7}")
