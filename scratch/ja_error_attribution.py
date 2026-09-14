"""Attribute Japanese errors: compare model outputs on the SAME utterances.

Reads report/ja_offline_bench.json and prints side-by-side hypotheses for the
utterances where qwen3-asr-1.7b diverges most from cohere-transcribe, plus the
insertion-heavy cases that look like repetition loops.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "report" / "ja_offline_bench.json"

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

by_model = {r["model"]: r for r in data["results"] if "error" not in r}
qwen = by_model.get("qwen3-asr-1.7b")
cohere = by_model.get("cohere-transcribe")
kotoba = by_model.get("kotoba-whisper-v2.2")
if not qwen or not cohere:
    raise SystemExit("need both qwen3-asr-1.7b and cohere-transcribe in the report")

q = {u["id"]: u for u in qwen["per_utt"]}
c = {u["id"]: u for u in cohere["per_utt"]}
k = {u["id"]: u for u in kotoba["per_utt"]} if kotoba else {}

# ---- 1. Insertion-heavy qwen utterances (loop / hallucination signature) ----
print("=" * 100)
print("QWEN3-ASR-1.7B: utterances with the most INSERTED characters")
print("=" * 100)
ins_sorted = sorted(qwen["per_utt"], key=lambda u: -u["insertions"])
for u in ins_sorted[:12]:
    ref = u["ref"]
    hyp = u["hyp"]
    print(f"\n[{u['id']}] ref_len={u['ref_len']} S={u['substitutions']} D={u['deletions']} I={u['insertions']} CER={u['cer']*100:.1f}%")
    print(f"  REF : {ref}")
    print(f"  QWEN: {hyp}")
    if u["id"] in c:
        print(f"  COHE: {c[u['id']]['hyp']}   (CER={c[u['id']]['cer']*100:.1f}%)")

# ---- 2. Where cohere wins biggest ----
print("\n\n" + "=" * 100)
print("BIGGEST COHERE WINS (qwen CER - cohere CER)")
print("=" * 100)
common = [uid for uid in q if uid in c]
ranked = sorted(common, key=lambda uid: -((q[uid]["cer"] or 0) - (c[uid]["cer"] or 0)))
for uid in ranked[:10]:
    a, b = q[uid], c[uid]
    print(f"\n[{uid}] qwenCER={a['cer']*100:.0f}%  cohereCER={b['cer']*100:.0f}%  qwenI={a['insertions']}")
    print(f"  REF : {a['ref']}")
    print(f"  QWEN: {a['hyp']}")
    print(f"  COHE: {b['hyp']}")

# ---- 3. Where qwen wins ----
print("\n\n" + "=" * 100)
print("BIGGEST QWEN WINS")
print("=" * 100)
ranked2 = sorted(common, key=lambda uid: -((c[uid]["cer"] or 0) - (q[uid]["cer"] or 0)))
for uid in ranked2[:6]:
    a, b = q[uid], c[uid]
    print(f"\n[{uid}] qwenCER={a['cer']*100:.0f}%  cohereCER={b['cer']*100:.0f}%")
    print(f"  REF : {a['ref']}")
    print(f"  QWEN: {a['hyp']}")
    print(f"  COHE: {b['hyp']}")

# ---- 4. Aggregate error profile ----
print("\n\n" + "=" * 100)
print("ERROR PROFILE (share of total edits)")
print("=" * 100)
print(f"{'model':<24}{'S':>8}{'D':>8}{'I':>8}{'CER':>9}{'ref':>8}")
for name, m in by_model.items():
    tot = m["total_substitutions"] + m["total_deletions"] + m["total_insertions"]
    ref = sum(u["ref_len"] for u in m["per_utt"])
    print(
        f"{name:<24}"
        f"{100*m['total_substitutions']/tot:>7.1f}%{100*m['total_deletions']/tot:>7.1f}%{100*m['total_insertions']/tot:>7.1f}%"
        f"{m['corpus_cer_pct']:>8.2f}%{ref:>8}"
    )

# ---- 5. Empty / truncated outputs ----
print("\n\n" + "=" * 100)
print("EMPTY OR DEGENERATE HYPOTHESES")
print("=" * 100)
for name, m in by_model.items():
    empties = [u for u in m["per_utt"] if len(u["hyp"].strip()) < len(u["ref"]) * 0.25]
    print(f"{name:<24} {len(empties):>3} / {len(m['per_utt'])} utterances with <25% of reference length")
