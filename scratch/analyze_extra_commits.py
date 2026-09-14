"""What are the extra commits when min_words_to_emit_final is lowered to 1?

Determines whether the recovered short finals are genuine short turns (worth emitting)
or truncated VAD fragments that decode into garbage (better dropped).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

old = json.loads((REPO / "report" / "jacv_stream_qwen.json").read_text(encoding="utf-8"))
new = json.loads((REPO / "report" / "jacv_stream_new_emit.json").read_text(encoding="utf-8"))

o = {r["pair_id"]: r for r in old["results"]}
n = {r["pair_id"]: r for r in new["results"]}

print(f"{'stream':<10}{'old commits':>12}{'new commits':>12}{'extra':>8}")
tot = 0
for pid in o:
    e = n[pid]["commit_count"] - o[pid]["commit_count"]
    tot += e
    print(f"{pid:<10}{o[pid]['commit_count']:>12}{n[pid]['commit_count']:>12}{e:>8}")
print(f"{'TOTAL':<10}{'':>12}{'':>12}{tot:>8}")

print("\nShort commits (< 4 content tokens) now emitted, with surrounding reference:")
from benchmarks.ja_text import normalize_ja  # noqa: E402
from backend_cpp.asr.sentence_segmenter import count_content_tokens  # noqa: E402

shown = 0
for pid in n:
    ref = n[pid]["raw_reference"]
    for c in n[pid]["commits"]:
        t = c["text"]
        if t and count_content_tokens(t) < 4:
            idx = ref.find(normalize_ja(t)[:3]) if len(normalize_ja(t)) >= 3 else -1
            ctx = ref[max(0, idx - 12): idx + 18] if idx >= 0 else "(not found in reference)"
            print(f"  [{pid}] tokens={count_content_tokens(t)} {t!r:<20} ref-context: …{ctx}…")
            shown += 1
            if shown >= 30:
                break
    if shown >= 30:
        break
print(f"\n(shown {shown} short commits)")
