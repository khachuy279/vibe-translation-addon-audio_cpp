"""Measure the REAL end-to-end subtitle latency budget from a benchmark log.

Why this matters: the audit's Phase 2 focused on compute cost (ms per audio-second), and the
translation investigation measured 452 ms/sentence. But the metric the user actually perceives is
"how long after I stop talking does the subtitle appear". That budget is dominated by stages that
are NOT compute: the VAD trailing-silence wait before a commit is even attempted.

Markers in the log:
    [VAD END]     speech end detected by the engine
    [ASR COMMIT]  the utterance was committed (text finalized)
    [TRANSLATE]   the translated subtitle was delivered

Usage:  python scratch/analyze_latency_budget.py [log ...]
"""

import re
import sys
from pathlib import Path
from statistics import mean, median

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")

MARKERS = (
    ("vad_end", re.compile(r"\[VAD END\]")),
    ("commit", re.compile(r"\[ASR COMMIT\]")),
    ("translate", re.compile(r"\[TRANSLATE\]")),
)


def _to_ms(stamp: str) -> float:
    hh, mm, rest = stamp.split(" ")[1].split(":")
    ss, msec = rest.split(",")
    return ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * 1000.0 + int(msec)


def parse(path: Path):
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TS.match(line)
        if not match:
            continue
        for kind, pattern in MARKERS:
            if pattern.search(line):
                events.append((kind, _to_ms(match.group(1)), line.strip()))
                break
    return events


def report(path: Path) -> int:
    events = parse(path)
    if not events:
        print(f"{path.name}: no markers found")
        return 1

    vad_ends = [e for e in events if e[0] == "vad_end"]
    commits = [e for e in events if e[0] == "commit"]
    translations = [e for e in events if e[0] == "translate"]

    print(f"\n{'=' * 78}\n{path.name}\n{'=' * 78}")
    print(f"markers: vad_end={len(vad_ends)} commit={len(commits)} translate={len(translations)}")

    # Pair each commit with the nearest preceding speech-end, then the first translation after it.
    pa_tr = []
    for _kind, t_commit, line in commits:
        prev_end = [t for _k, t, _l in vad_ends if t <= t_commit]
        next_tr = [t for _k, t, _l in translations if t >= t_commit]
        pa_tr.append(
            (
                (t_commit - prev_end[-1]) if prev_end else None,
                (next_tr[0] - t_commit) if next_tr else None,
                line,
            )
        )

    print(f"\n{'#':>2} {'silence_wait_ms':>15} {'commit->sub_ms':>14} {'total_ms':>9}")
    print("-" * 46)
    totals = []
    for idx, (silence, to_sub, _line) in enumerate(pa_tr, 1):
        total = (silence + to_sub) if (silence is not None and to_sub is not None) else None
        print(
            f"{idx:>2} {('%.0f' % silence) if silence is not None else 'n/a':>15} "
            f"{('%.0f' % to_sub) if to_sub is not None else 'n/a':>14} "
            f"{('%.0f' % total) if total is not None else 'n/a':>9}"
        )
        if total is not None:
            totals.append(total)

    if totals:
        print("-" * 46)
        print(f"   median total = {median(totals):.0f} ms   mean = {mean(totals):.0f} ms")
        silences = [s for s, _t, _l in pa_tr if s is not None]
        to_subs = [t for _s, t, _l in pa_tr if t is not None]
        if silences:
            print(f"   median silence_wait = {median(silences):.0f} ms  <-- VAD trailing silence")
        if to_subs:
            print(f"   median commit->subtitle = {median(to_subs):.0f} ms  <-- ASR commit + translate")
    return 0


def main() -> int:
    logs = sys.argv[1:] or ["scratch/preview_amp.log", "scratch/dup_verify.log"]
    root = Path(__file__).resolve().parents[1]
    status = 0
    for rel in logs:
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            print(f"missing: {path}")
            continue
        status |= report(path)
    return status


if __name__ == "__main__":
    sys.exit(main())
