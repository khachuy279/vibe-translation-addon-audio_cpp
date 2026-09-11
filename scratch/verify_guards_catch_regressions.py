"""Prove that the extension invariant guards actually catch regressions.

A test suite that only ever passes proves nothing. This script deliberately breaks one
invariant at a time, runs the guard tests, asserts they FAIL, then restores the original
file content -- in a try/finally so the working tree is always restored, even on error.

Usage:
    python scratch/verify_guards_catch_regressions.py
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "extension_firefox" / "manifest.json"
POPUP = PROJECT_ROOT / "extension_firefox" / "popup" / "popup.js"
CONTENT = PROJECT_ROOT / "extension_firefox" / "content" / "content-script.js"

GUARD_TEST = "backend_cpp/tests/test_extension_invariants.py"
BASELINE_GUARD_TEST = "backend_cpp/tests/test_perf_baseline_gate.py"
BASELINE_MODULE = PROJECT_ROOT / "backend_cpp" / "tests" / "perf_baseline.py"
PREVIEW_GATE_GUARD_TEST = "backend_cpp/tests/test_preview_growth_gate.py"
TRANSCRIBE_ENGINE = PROJECT_ROOT / "backend_cpp" / "asr" / "transcribe_engine.py"
APP_CONFIG = PROJECT_ROOT / "backend_cpp" / "config.py"
VAD_ENGINES = PROJECT_ROOT / "backend_cpp" / "vad" / "engines.py"
VAD_GUARD_TEST = "backend_cpp/tests/test_vad_concurrency.py"
CUDA_UTILS = PROJECT_ROOT / "backend_cpp" / "utils" / "cuda_utils.py"
CUDA_GUARD_TEST = "backend_cpp/tests/test_cuda_dll_discovery.py"
VAD_PROCESSOR = PROJECT_ROOT / "backend_cpp" / "vad" / "vad_processor.py"
VAD_SILENCE_GUARD_TEST = "backend_cpp/tests/test_vad_silence_latency.py"


def run_guards(test_path: str = GUARD_TEST) -> bool:
    """Return True when the guard tests pass."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-q", "-p", "no:cacheprovider"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        # Decode explicitly: pytest output contains non-ASCII text and the Windows default
        # (cp1252) would raise UnicodeDecodeError inside subprocess's reader thread.
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode == 0


def check(
    name: str,
    path: Path,
    mutate,
    expected_to_fail: bool = True,
    test_path: str = GUARD_TEST,
) -> bool:
    """Break an invariant, confirm the guards notice, then restore."""
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(mutate(original), encoding="utf-8")
        guards_passed = run_guards(test_path)
        result = (not guards_passed) if expected_to_fail else guards_passed
        status = "OK  " if result else "FAIL"
        detail = "guards detected the break" if not guards_passed else "guards did NOT notice!"
        print(f"  [{status}] {name}: {detail}")
        return result
    finally:
        path.write_text(original, encoding="utf-8")


# ---------------------------------------------------------------------------
# Mutations: each reintroduces a real bug that was fixed previously
# ---------------------------------------------------------------------------
def break_inv1_duplicate_host_permission(source: str) -> str:
    """INV-1: declare <all_urls> in BOTH host_permissions and optional_host_permissions."""
    manifest = json.loads(source)
    manifest["host_permissions"] = ["<all_urls>"] + manifest.get("host_permissions", [])
    return json.dumps(manifest, indent=2) + "\n"


def break_inv1_remove_optional(source: str) -> str:
    """INV-1: drop <all_urls> from optional_host_permissions (the original bug)."""
    manifest = json.loads(source)
    manifest["optional_host_permissions"] = []
    return json.dumps(manifest, indent=2) + "\n"


def break_inv3_all_frames_false(source: str) -> str:
    """INV-3: stop declaring the content script for every frame."""
    manifest = json.loads(source)
    manifest["content_scripts"][0]["all_frames"] = False
    return json.dumps(manifest, indent=2) + "\n"


def break_inv6_drop_script_from_popup(source: str) -> str:
    """INV-6: let CONTENT_SCRIPT_FILES drift from the manifest list."""
    return source.replace(
        '    "content/content-script.js"\n  ];',
        '    "content/overlay-manager.js"\n  ];',
        1,
    )


def break_inv7_remove_silent_skip(source: str) -> str:
    """INV-7: let a video-less frame answer a broadcast START (masks the real frame)."""
    return source.replace(
        "if (!captureOwnerToken && !findVideo()) {\n          return false;\n        }",
        "if (false) {\n          return false;\n        }",
        1,
    )


def break_inv4_request_after_await(source: str) -> str:
    """INV-4: move the permission request after another await (kills the user gesture)."""
    return source.replace(
        'btnStart.addEventListener("click", async () => {\n    btnStart.disabled = true;',
        'btnStart.addEventListener("click", async () => {\n    btnStart.disabled = true;\n    await fetchBackendEngineConfig();',
        1,
    )


# --- W2.1 baseline gate mutations -------------------------------------------------
def break_baseline_vad_engine(source: str) -> str:
    """Use the VAD engine that classifies the synthetic test signal as non-speech."""
    return source.replace(
        'BASELINE_VAD_ENGINE = "fsmn-vad"',
        'BASELINE_VAD_ENGINE = "firered-vad"',
        1,
    )


def break_baseline_reference_marking(source: str) -> str:
    """Stop labelling the NumPy row as a reference measurement."""
    return source.replace('"is_reference_only": True,', '', 1).replace(
        'f"REFERENCE numpy per-frame convert ({frame_samples}-sample frames)"',
        'f"numpy per-frame convert ({frame_samples}-sample frames)"',
        1,
    )


def break_baseline_validity_flag(source: str) -> str:
    """Pretend an all-zero run is valid data."""
    return source.replace(
        '            "valid": False,\n            "inferences_observed": inference_count,',
        '            "valid": True,\n            "inferences_observed": inference_count,',
        1,
    )


# --- W2.2 preview growth gate mutations ---------------------------------------
def break_preview_gate_reference_advance(source: str) -> str:
    """Reintroduce the bug that silently collapsed the gate to one preview per utterance.

    Advancing the gate's reference point on a SKIPPED poll makes the threshold grow faster
    than the audio arrives, so the gate never reopens. It looked plausible and measured
    0.26x amplification; the real (correct) figure is ~0.70x at the same ratio.
    """
    return source.replace(
        "        self._last_polled_samples = snapshot_samples\n\n    def update_sentence_config",
        "        self._last_polled_samples = snapshot_samples\n"
        "        self._last_preview_duration_sec = snapshot_samples / 16000.0\n\n"
        "    def update_sentence_config",
        1,
    )


def break_preview_gate_disabled(source: str) -> str:
    """Make the gate always allow a preview, i.e. silently restore the 2.27x amplification."""
    return source.replace(
        "        required = self._preview_required_growth_sec(duration_sec)\n"
        "        if required <= 0.0:",
        "        required = self._preview_required_growth_sec(duration_sec)\n"
        "        if True:",
        1,
    )


# --- VAD default engine mutations ---------------------------------------------
def break_vad_default_engine(source: str) -> str:
    """Revert the VAD default to the most expensive engine (~8x silero, ~1.9x fsmn)."""
    return source.replace(
        'vad_engine: str = "fsmn-vad"  # fsmn-vad (default), firered-vad, silero-vad',
        'vad_engine: str = "firered-vad"  # firered-vad, silero-vad, fsmn-vad',
        1,
    )


def break_vad_fallback_engine(source: str) -> str:
    """Make an unknown engine name fall back to the priciest engine again."""
    return source.replace(
        '                ", ".join(sorted(DEFAULT_THRESHOLDS)),\n'
        "            )\n"
        '            engine = "fsmn-vad"',
        '                ", ".join(sorted(DEFAULT_THRESHOLDS)),\n'
        "            )\n"
        '            engine = "firered-vad"',
        1,
    )


# --- CUDA DLL discovery mutations ----------------------------------------------
def break_cuda_discovery_last_entry_only(source: str) -> str:
    """Reproduce the discovery bug: only the LAST site-packages entry is considered.

    The original code had the torch/lib + nvidia/ derivation OUTSIDE its `for sp in
    site_packages_dirs` loop, so only the final entry was used -- and that entry is
    site.USER_SITE, which normally does not exist. The result was an empty set, so no DLL
    directory was registered and `import llama_cpp` failed in a fresh process.
    """
    return source.replace(
        "        for sp in site_packages_dirs:\n",
        "        for sp in site_packages_dirs[-1:]:  # MUTATION\n",
        1,
    )


# --- VAD silence latency mutations --------------------------------------------
def break_vad_silence_default(source: str) -> str:
    """Give back the 300ms of latency the 150ms default was chosen to remove."""
    return source.replace(
        "    silence_duration_ms: int = 150\n",
        "    silence_duration_ms: int = 450\n",
        1,
    )


def break_vad_hangover_adds_latency(source: str) -> str:
    """The semantic trap: make hangover_ms add to the trailing silence.

    hangover_ms is only a grace period for attaching silent frames to the committed audio. If it
    is added to the end threshold, the configured 400ms silently adds another 400ms of latency --
    which is exactly the mistake made while investigating (assuming 450 + 400 = 850ms).
    """
    return source.replace(
        "                        total_silence_limit_ms = float(self.silence_duration_ms)\n",
        "                        total_silence_limit_ms = float(\n"
        "                            self.silence_duration_ms + self.hangover_ms)\n",
        1,
    )


def main() -> int:
    print("Confirming the invariant guards are not vacuous:\n")

    results = [
        check("INV-1  <all_urls> duplicated in host_permissions", MANIFEST, break_inv1_duplicate_host_permission),
        check("INV-1  <all_urls> removed from optional_host_permissions", MANIFEST, break_inv1_remove_optional),
        check("INV-3  content_scripts all_frames=false", MANIFEST, break_inv3_all_frames_false),
        check("INV-6  CONTENT_SCRIPT_FILES drifts from manifest", POPUP, break_inv6_drop_script_from_popup),
        check("INV-7  video-less frame answers the broadcast", CONTENT, break_inv7_remove_silent_skip),
        check("INV-4  permission request moved after an await", POPUP, break_inv4_request_after_await),
    ]

    print("\nConfirming the W2.1 baseline gate is not vacuous:\n")
    results += [
        check(
            "W2.1  baseline VAD engine set to the one that ignores the test signal",
            BASELINE_MODULE,
            break_baseline_vad_engine,
            test_path=BASELINE_GUARD_TEST,
        ),
        check(
            "W2.1  reference row no longer marked as a reference measurement",
            BASELINE_MODULE,
            break_baseline_reference_marking,
            test_path=BASELINE_GUARD_TEST,
        ),
        check(
            "W2.1  empty run no longer flagged as invalid",
            BASELINE_MODULE,
            break_baseline_validity_flag,
            test_path=BASELINE_GUARD_TEST,
        ),
    ]

    print("\nConfirming the W2.2 preview growth gate is not vacuous:\n")
    results += [
        check(
            "W2.2  gate reference advances on a skipped poll (collapses to 1 preview)",
            TRANSCRIBE_ENGINE,
            break_preview_gate_reference_advance,
            test_path=PREVIEW_GATE_GUARD_TEST,
        ),
        check(
            "W2.2  gate disabled (restores the 2.27x preview amplification)",
            TRANSCRIBE_ENGINE,
            break_preview_gate_disabled,
            test_path=PREVIEW_GATE_GUARD_TEST,
        ),
    ]

    print("\nConfirming the VAD default-engine guard is not vacuous:\n")
    results += [
        check(
            "VAD  default engine reverted to the priciest one",
            APP_CONFIG,
            break_vad_default_engine,
            test_path=VAD_GUARD_TEST,
        ),
        check(
            "VAD  unknown engine name falls back to the priciest one again",
            VAD_ENGINES,
            break_vad_fallback_engine,
            test_path=VAD_GUARD_TEST,
        ),
    ]

    print("\nConfirming the CUDA DLL discovery guard is not vacuous:\n")
    results += [
        check(
            "CUDA discovery only scans the last site-packages entry (empty set)",
            CUDA_UTILS,
            break_cuda_discovery_last_entry_only,
            test_path=CUDA_GUARD_TEST,
        ),
    ]

    print("\nConfirming the VAD silence-latency guards are not vacuous:\n")
    results += [
        check(
            "VAD  silence_duration_ms default reverted to 450 (gives back 300ms)",
            APP_CONFIG,
            break_vad_silence_default,
            test_path=VAD_SILENCE_GUARD_TEST,
        ),
        check(
            "VAD  hangover_ms added to the end threshold (doubles trailing silence)",
            VAD_PROCESSOR,
            break_vad_hangover_adds_latency,
            test_path=VAD_SILENCE_GUARD_TEST,
        ),
    ]

    print()
    if all(results):
        print(f"All {len(results)} mutations were caught, and the tree is restored.")
    else:
        print("Some mutations were NOT caught -- the guards are incomplete.")

    # Final sanity check: everything restored, guards green again.
    restored = run_guards() and run_guards(BASELINE_GUARD_TEST)
    print(f"Guards after restore: {'PASS' if restored else 'FAIL'}")
    return 0 if (all(results) and restored) else 1


if __name__ == "__main__":
    sys.exit(main())
