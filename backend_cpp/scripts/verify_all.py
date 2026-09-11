"""Static verification harness for backend_cpp + extension_firefox.

Runs syntax-only checks across the whole project so that regressions are caught
immediately after every change:

  * Python: ``py_compile`` on every ``.py`` file under ``backend_cpp``
  * JavaScript: ``node --check`` on every ``.js`` file under ``extension_firefox``

Usage (from the project root)::

    python backend_cpp/scripts/verify_all.py
    python backend_cpp/scripts/verify_all.py --quiet

Exit code is 0 when every file passes, 1 otherwise.

Notes
-----
This is a *syntax* gate only. It does not import the modules (which would require
CUDA / native ``transcribe_cpp`` bindings) and it does not execute tests. It exists
so that a broken file like the historical ``model_manager.py`` IndentationError can
never silently land in the tree again.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Directories that never contain hand-written source worth compiling.
_EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "models",
    "voices",
    "node_modules",
    ".git",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = _PROJECT_ROOT / "backend_cpp"
_EXTENSION_DIR = _PROJECT_ROOT / "extension_firefox"


def _iter_source_files(root: Path, suffix: str) -> List[Path]:
    """Collect source files under *root*, skipping generated/vendored directories."""
    if not root.is_dir():
        return []
    found: List[Path] = []
    for path in root.rglob(f"*{suffix}"):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        found.append(path)
    return sorted(found)


def _check_python(files: List[Path]) -> List[Tuple[Path, str]]:
    """Return ``(path, reason)`` for every Python file that fails to compile."""
    failures: List[Tuple[Path, str]] = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            # Keep the message on a single line for readable CI output.
            reason = " ".join(str(exc).split())
            failures.append((path, reason))
        except Exception as exc:  # pragma: no cover - defensive
            failures.append((path, f"{type(exc).__name__}: {exc}"))
    return failures


def _check_javascript(files: List[Path]) -> List[Tuple[Path, str]]:
    """Return ``(path, reason)`` for every JS file that fails ``node --check``."""
    if not files:
        return []

    node = shutil.which("node")
    if node is None:
        return [(files[0], "node executable not found on PATH - skipped JS checks")]

    failures: List[Tuple[Path, str]] = []
    for path in files:
        try:
            proc = subprocess.run(
                [node, "--check", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - defensive
            failures.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        if proc.returncode != 0:
            reason = (proc.stderr or proc.stdout or "node --check failed").strip()
            failures.append((path, " ".join(reason.split())))

    return failures


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary and any failures.",
    )
    args = parser.parse_args(argv)

    py_files = _iter_source_files(_BACKEND_DIR, ".py")
    js_files = _iter_source_files(_EXTENSION_DIR, ".js")

    if not args.quiet:
        print(f"[verify_all] python files: {len(py_files)}")
        print(f"[verify_all] javascript files: {len(js_files)}")

    py_failures = _check_python(py_files)
    js_failures = _check_javascript(js_files)

    def _report(label: str, failures: List[Tuple[Path, str]], total: int) -> None:
        if failures:
            print(f"[FAIL] {label}: {len(failures)}/{total} file(s) did not pass")
            for path, reason in failures:
                print(f"       {_relative(path)}: {reason}")
        else:
            print(f"[ OK ] {label}: {total}/{total} file(s) passed")

    _report("python", py_failures, len(py_files))
    _report("javascript", js_failures, len(js_files))

    if py_failures or js_failures:
        return 1
    print("[verify_all] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
