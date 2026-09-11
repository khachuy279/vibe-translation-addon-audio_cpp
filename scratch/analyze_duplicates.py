"""Analyze the real-audio run log to find out why transcripts appeared twice.

Hypotheses to distinguish:
  A) the ASR really emits the same sentence twice (stability split + commit),
  B) the WAV itself contains each sentence twice,
  C) the client-side collection records the same message twice.

The server log settles it: count `[ASR COMMIT]` lines per utterance id and their `reason`,
and look for `[ASR DEDUP]` lines that should have suppressed a repeat.
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "scratch" / "dup_investigation.log"

text = log_path.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

commit_re = re.compile(r"\[ASR COMMIT\]\s*\[utt=(?P<utt>[^\]]+)\]\s*\[(?P<model>[^\]]+)\]\s*\[(?P<reason>[^\]]+)\]\s*\((?P<lang>[^)]*)\):\s*'(?P<text>.*)'")
dedup_re = re.compile(r"\[ASR DEDUP\]\s*\[utt=(?P<utt>[^\]]+)\]\s*Skipped duplicate commit \[(?P<reason>[^\]]+)\]:\s*'(?P<text>.*)'")
filter_re = re.compile(r"\[ASR FILTER\]")
short_re = re.compile(r"\[TRANSLATE FILTER\]")
final_sent_re = re.compile(r"\[BS\] Sentence")

commits = []
dedups = []
for line in lines:
    m = commit_re.search(line)
    if m:
        commits.append(m.groupdict())
        continue
    m = dedup_re.search(line)
    if m:
        dedups.append(m.groupdict())

print("=" * 78)
print("DUPLICATE TRANSCRIPT ANALYSIS")
print("=" * 78)
print(f"log: {log_path.name}  ({len(lines)} lines)")
print(f"[ASR COMMIT] lines: {len(commits)}")
print(f"[ASR DEDUP]  lines: {len(dedups)}")

print("\n--- commits by reason ---")
for reason, count in Counter(c["reason"] for c in commits).most_common():
    print(f"  {reason:<16} {count}")

print("\n--- commit sequence (utt | reason | text) ---")
seen = Counter()
for c in commits:
    seen[c["text"]] += 1
    print(f"  {c['utt']:<10} {c['reason']:<16} {c['text']!r}")

print("\n--- texts committed more than once ---")
dupes = {t: n for t, n in seen.items() if n > 1}
if not dupes:
    print("  (none: every committed text is unique in the log)")
else:
    for t, n in sorted(dupes.items(), key=lambda kv: -kv[1]):
        print(f"  x{n}  {t!r}")

print("\n--- dedup suppressions (texts the deduplicator blocked) ---")
if not dedups:
    print("  (none)")
else:
    for d in dedups:
        print(f"  {d['utt']:<10} reason={d['reason']:<16} {d['text']!r}")

print("\n--- other filters ---")
print(f"  [ASR FILTER]     : {sum(1 for line in lines if filter_re.search(line))}")
print(f"  [TRANSLATE FILTER]: {sum(1 for line in lines if short_re.search(line))}")

print("\n--- interpretation ---")
by_utt = defaultdict(list)
for c in commits:
    by_utt[c["utt"]].append(c["reason"])
multi_reason = {u: r for u, r in by_utt.items() if len(r) > 1}
if dupes and multi_reason:
    print("  Same text committed more than once -> hypothesis A (pipeline emits twice).")
elif dupes and not multi_reason:
    print("  Duplicate texts appear under DIFFERENT utterance ids.")
else:
    print("  No duplicate commits server-side -> hypothesis B/C (audio or client-side).")

if not dedups and dupes:
    print("  NOTE: the deduplicator never fired even though texts repeated.")
print("=" * 78)
