"""Check whether the verbatim filter actually removed the under-labelled clips.

If the qwen3-asr-1.7b "extra text" signature survives the coverage filter, the
filter did not work and the ReazonSpeech set cannot be salvaged by that heuristic.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

for name in ["ja_offline_bench.json", "ja_offline_verbatim.json"]:
    path = REPO / "report" / name
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{'='*90}\n{name}  (n={data['n_utts']})\n{'='*90}")
    for r in data["results"]:
        if "error" in r:
            continue
        ref_chars = sum(u["ref_len"] for u in r["per_utt"])
        print(
            f"  {r['model']:<24} CER={r['corpus_cer_pct']:6.2f}%  "
            f"ref_chars={ref_chars:>6}  S/D/I={r['total_substitutions']}/{r['total_deletions']}/{r['total_insertions']}"
        )
    # How many utterances does qwen over-produce on?
    q = next((r for r in data["results"] if r["model"] == "qwen3-asr-1.7b" and "error" not in r), None)
    if q:
        over = [u for u in q["per_utt"] if u["insertions"] >= 5]
        print(f"  qwen utterances with >=5 inserted chars: {len(over)} / {len(q['per_utt'])} "
              f"({100*len(over)/len(q['per_utt']):.1f}%)")
        for u in over[:5]:
            print(f"    [{u['id']}] REF: {u['ref']}")
            print(f"           HYP: {u['hyp']}")
