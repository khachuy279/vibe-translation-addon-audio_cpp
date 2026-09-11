"""Dump which log lines exist in the real-audio run, to see what actually got logged."""

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log_path = PROJECT_ROOT / "scratch" / "dup_investigation.log"
text = log_path.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

print(f"total lines: {len(lines)}\n")

markers = ["ASR", "COMMIT", "DEDUP", "TRANSLATE", "VAD", "subtitle", "SUBTITLE", "pipeline", "PERF"]
print("marker counts:")
for marker in markers:
    print(f"  {marker:<12} {sum(1 for line in lines if marker in line)}")

print("\nall lines mentioning ASR/COMMIT/TRANSLATE/subtitle (first 40):")
shown = 0
for line in lines:
    if any(m in line for m in ("ASR", "COMMIT", "TRANSLATE", "subtitle", "SUBTITLE")):
        print(f"  {line.strip()[:160]}")
        shown += 1
        if shown >= 40:
            break
if shown == 0:
    print("  (none)")

print("\nlogger names seen (top 15):")
loggers = Counter()
for line in lines:
    if "] " in line and " [INFO] " in line or " [WARNING] " in line:
        try:
            after = line.split("[INFO] ", 1)[-1] if "[INFO] " in line else line.split("[WARNING] ", 1)[-1]
            loggers[after.split(":", 1)[0].strip()] += 1
        except Exception:
            pass
for name, count in loggers.most_common(15):
    print(f"  {count:>5}  {name}")
