"""Fetch the awesome-dsh-plugin list and extract plugins relevant to this project.

Project profile: Python 3.13 + FastAPI/asyncio backend, native C++ (transcribe.cpp) audio
pipeline, a Firefox MV3 extension, plus pytest suites and benchmark harnesses.
"""
import re
import urllib.request

URL = "https://raw.githubusercontent.com/awesome-dsh-plugin/awesome-dsh-plugin/main/README.md"
req = urllib.request.Request(URL, headers={"User-Agent": "probe"})
text = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
print(f"README length: {len(text)} chars\n")

# Plugin entries look like: - [**name**](url) — description   or   - [name](url) - desc
entries = re.findall(r"^\s*[-*]\s*\[([^\]]{2,80})\]\(([^)]+)\)\s*[—–:-]?\s*(.{0,220})", text, re.M)
print(f"parsed {len(entries)} linked entries\n")

CATEGORIES = {
    "audio/speech/asr": ["audio", "speech", "asr", "transcri", "subtitle", "whisper", "tts"],
    "python": ["python", "pytest", "ruff", "uv ", "pyright", "mypy"],
    "testing/benchmark": ["test", "bench", "coverage", "profil", "perf"],
    "code search/index": ["search", "index", "grep", "semantic", "codebase", "repo map"],
    "memory/context": ["memory", "context", "recall", "knowledge", "note"],
    "git/review": ["git", "diff", "review", "commit", "branch", "pr "],
    "lsp/diagnostics": ["lsp", "language server", "diagnostic", "typecheck"],
    "docs/planning": ["doc", "plan", "spec", "task", "todo"],
}

hits = {k: [] for k in CATEGORIES}
for name, url, desc in entries:
    blob = f"{name} {desc}".lower()
    for cat, keys in CATEGORIES.items():
        if any(k in blob for k in keys):
            hits[cat].append((name.strip(), url.strip(), re.sub(r"\s+", " ", desc).strip()))

for cat, items in hits.items():
    print(f"=== {cat}  ({len(items)}) ===")
    seen = set()
    for name, url, desc in items[:14]:
        if name in seen:
            continue
        seen.add(name)
        print(f"  {name}")
        print(f"     {url}")
        if desc:
            print(f"     {desc[:170]}")
    print()
