"""Guard the CUDA DLL directory discovery.

`setup_cuda_dll_paths()` reported ``Configured CUDA DLL directories: []`` -- it found nothing --
because the per-site-package block sat OUTSIDE its own loop and therefore used only the last
entry (``site.USER_SITE``, which does not exist). The consequence was not cosmetic:
``llama_cpp/lib/ggml.dll`` statically imports ``ggml-cuda.dll``, which needs cudart64_12.dll and
cublas64_12.dll from the ``nvidia-*-cu12`` runtime wheels, so with nothing registered
``import llama_cpp`` fails in a fresh process. Translation only worked because ASR imports
``transcribe_cpp`` first, which loads a ``ggml.dll`` that llama.dll then binds to by name.

These tests fail if the discovery silently returns nothing again, or if it stops covering the
directories that exist on this machine.
"""

from __future__ import annotations

import os
import site
import sys

import pytest

from backend_cpp.utils.cuda_utils import discover_cuda_dll_dirs, setup_cuda_dll_paths


def _expected_dirs() -> set:
    """Every directory the discovery is supposed to find, computed independently."""
    expected = set()
    dirs = [p for p in (site.getsitepackages() + [site.USER_SITE]) if p]
    for sp in dirs:
        if not os.path.exists(sp):
            continue
        for rel in ("torch/lib", "llama_cpp/lib"):
            candidate = os.path.join(sp, *rel.split("/"))
            if os.path.isdir(candidate):
                expected.add(os.path.abspath(candidate))
        nvidia = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia):
            for item in os.listdir(nvidia):
                if item.lower() == "cudnn":
                    continue
                for sub in ("bin", "lib"):
                    candidate = os.path.join(nvidia, item, sub)
                    if os.path.isdir(candidate):
                        expected.add(os.path.abspath(candidate))
    return expected


def test_discovery_is_not_empty():
    """An empty result is exactly the regression: nothing gets registered at all."""
    dirs = discover_cuda_dll_dirs()
    assert dirs, (
        "discover_cuda_dll_dirs() returned nothing, so setup_cuda_dll_paths() would register "
        "no directory and 'import llama_cpp' would fail in a fresh process"
    )


def test_discovery_covers_every_existing_candidate():
    """Not just the last site-packages entry -- the bug that made the set empty."""
    expected = _expected_dirs()
    if not expected:
        pytest.skip("no torch/llama_cpp/nvidia directories installed in this interpreter")

    missing = expected - discover_cuda_dll_dirs()
    assert not missing, f"discovery missed existing directories: {sorted(missing)}"


def test_discovery_includes_llama_cpp_lib_when_present():
    """llama_cpp/lib holds ggml-cuda.dll and must be registered for the CUDA build to load."""
    sp = [p for p in site.getsitepackages() if os.path.isdir(p)]
    llama_lib = [os.path.join(p, "llama_cpp", "lib") for p in sp]
    llama_lib = [p for p in llama_lib if os.path.isdir(p)]
    if not llama_lib:
        pytest.skip("llama_cpp not installed")

    dirs = discover_cuda_dll_dirs()
    assert any(os.path.abspath(p) in dirs for p in llama_lib)


def test_discovery_excludes_cudnn_to_avoid_version_mismatch():
    """nvidia/cudnn is deliberately skipped: pytorch ships its own cuDNN 9 build.

    Registering both makes the loader pick one arbitrarily and can raise
    CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
    """
    for path in discover_cuda_dll_dirs():
        components = {part.lower() for part in path.replace("\\", "/").split("/")}
        assert "cudnn" not in components, f"cuDNN directory must not be registered: {path}"


@pytest.mark.skipif(sys.platform != "win32", reason="DLL search path is Windows-only")
def test_setup_registers_the_discovered_directories_on_path():
    """The registration step must actually apply what discovery found.

    Only asserted when setup has not run yet in this process, because it is intentionally
    one-shot (guarded by _cuda_paths_initialized).
    """
    from backend_cpp.utils import cuda_utils

    if cuda_utils._cuda_paths_initialized:
        pytest.skip("setup_cuda_dll_paths() already ran in this process")

    expected = discover_cuda_dll_dirs()
    if not expected:
        pytest.skip("nothing to register on this machine")

    setup_cuda_dll_paths()

    path_entries = set(os.environ.get("PATH", "").split(os.path.pathsep))
    missing = {d for d in expected if d not in path_entries}
    assert not missing, f"discovered directories were not added to PATH: {sorted(missing)}"
