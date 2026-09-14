"""Compare the VAD segmentation sweep points: gap to ceiling, error profile, segment lengths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.ja_text import score_ja  # noqa: E402

STREAM = REPO / "data" / "ja_long3"
meta = json.loads((STREAM / "streams.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
REF = (STREAM / "streams" / f"{meta['id']}.txt").read_text(encoding="utf-8")

FILES = [
    ("silence=150 hangover=0 (base)", "report/sweep_base.json"),
    ("silence=250 hangover=0", "report/sweep_s250_h0.json"),
    ("silence=400 hangover=0", "report/sweep_s400_h0.json"),
    ("silence=600 hangover=0", "report/sweep_s600_h0.json"),
    ("silence=300 hangover=400", "report/sweep_s300_h400.json"),
]

rows = []
for label, path in FILES:
    p = REPO / path
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    sc = score_ja(REF, d["pipeline_text"])
    rows.append({
        "label": label,
        "cer": d["pipeline_cer_pct"],
        "gap": d["gap_pp"],
        "commits": d["commits"],
        "max_seg": d["segment_max_sec"],
        "p95": d["segment_p95_sec"],
        "over": d["segments_over_limit"],
        "S": sc.substitutions, "D": sc.deletions, "I": sc.insertions,
        "file": path,
    })

ceiling = None
base = REPO / "report" / "refs" / f"{meta['id']}.json"
if base.exists():
    ceiling = json.loads(base.read_text(encoding="utf-8"))["ceiling_cer_pct"]

print(f"stream {meta['id']}: {meta['duration_sec']:.0f}s, {len(meta['true_sentences_sec'])} sentences, "
      f"max sentence {max(b - a for a, b in meta['true_sentences_sec']):.1f}s")
print(f"(1) ceiling CER = {ceiling}%\n")
print(f"{'configuration':<32}{'CER':>8}{'gap':>8}{'segs':>6}{'max':>7}{'p95':>7}{'>8s':>5}{'S':>6}{'D':>5}{'I':>5}")
for r in rows:
    print(f"{r['label']:<32}{r['cer']:>7.2f}%{r['gap']:>+7.2f}{r['commits']:>6}"
          f"{r['max_seg']:>7.2f}{r['p95']:>7.2f}{r['over']:>5}{r['S']:>6}{r['D']:>5}{r['I']:>5}")

if len(rows) >= 2:
    best = min(rows, key=lambda r: r["gap"])
    print(f"\nbest: {best['label']}  gap={best['gap']:+.2f}pp  ({best['file']})")

    worst = max(rows, key=lambda r: r["gap"])
    if best["file"] != worst["file"]:
        b = json.loads((REPO / best["file"]).read_text(encoding="utf-8"))
        w = json.loads((REPO / worst["file"]).read_text(encoding="utf-8"))
        print(f"\n{'='*100}\nBEST vs WORST commit-by-commit\n{'='*100}")
        print(f"\n--- {best['label']} (CER {b['pipeline_cer_pct']}%) ---")
        for c in b["commit_detail"]:
            print(f"   [{c['audio_sec']:>5.2f}s {c['reason'][:14]:<14}] {c['text']}")
        print(f"\n--- {worst['label']} (CER {w['pipeline_cer_pct']}%) ---")
        for c in w["commit_detail"]:
            print(f"   [{c['audio_sec']:>5.2f}s {c['reason'][:14]:<14}] {c['text']}")
