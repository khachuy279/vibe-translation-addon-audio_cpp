"""Helper module to synchronize converter scripts and libraries from transcribe.cpp upstream.

Supports both online synchronization (fetching from GitHub) and offline operation
using already cached/downloaded scripts in backend_cpp/convert/scripts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

logger = logging.getLogger("convert_sync")

CONVERT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CONVERT_DIR / "scripts"
LIB_DIR = SCRIPTS_DIR / "lib"

RAW_BASE_URL = "https://raw.githubusercontent.com/handy-computer/transcribe.cpp/main"

# Shared lib files required by converters
SHARED_LIB_FILES = [
    "__init__.py",
    "gguf_common.py",
    "hf_source.py",
    "quant_policy.py",
    "ref_dump.py",
]

# Supported model family mapping: family_id -> metadata
FAMILY_CATALOG: Dict[str, Dict[str, Any]] = {
    "qwen3_asr": {
        "name": "Qwen3-ASR (Alibaba)",
        "script": "convert-qwen3_asr.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",  # [model, out_path, --repo-id, ...]
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 2100,
        "status": "stable",
        "description": "Audio-LLM model, high accuracy, multilingual (0.6B, 1.7B, fine-tunes)",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0", "librosa"],
    },
    "sensevoice": {
        "name": "SenseVoice (Alibaba FunASR)",
        "script": "convert-sensevoice.py",
        "architecture_type": "non_autoregressive",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": ["zh", "en", "ja", "ko", "yue"],
        "vram_mb": 500,
        "status": "stable",
        "description": "Ultra-fast non-autoregressive speech model, latency ~20ms",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "whisper": {
        "name": "OpenAI Whisper",
        "script": "convert-whisper.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 1500,
        "status": "stable",
        "description": "Classic encoder-decoder Whisper models (tiny to large-v3)",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "parakeet": {
        "name": "NVIDIA Parakeet / Nemotron",
        "script": "convert-parakeet.py",
        "architecture_type": "streaming",
        "source_format": "nemo_model",
        "cmd_style": "standard",
        "supports_revision": False,
        "variant_flag": None,
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 750,
        "status": "stable",
        "description": "FastConformer CTC / RNNT / streaming ASR models (requires nemo_toolkit)",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "moonshine": {
        "name": "Useful Sensors Moonshine",
        "script": "convert-moonshine.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": ["en"],
        "vram_mb": 400,
        "status": "stable",
        "description": "Resource-efficient speech recognition for edge & desktop",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "moonshine_streaming": {
        "name": "Useful Sensors Moonshine Streaming",
        "script": "convert-moonshine_streaming.py",
        "architecture_type": "streaming",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": ["en"],
        "vram_mb": 450,
        "status": "stable",
        "description": "Low-latency streaming Moonshine ASR",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "canary": {
        "name": "NVIDIA Canary",
        "script": "convert-canary.py",
        "architecture_type": "offline_llm",
        "source_format": "nemo_model",
        "cmd_style": "standard",
        "supports_revision": False,
        "variant_flag": None,
        "sample_rate": 16000,
        "default_languages": ["en", "de", "es", "fr"],
        "vram_mb": 1800,
        "status": "supported",
        "description": "NVIDIA 1B parameter multilingual ASR & AST model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "canary_qwen": {
        "name": "NVIDIA Canary-Qwen",
        "script": "convert-canary-qwen.py",
        "architecture_type": "offline_llm",
        "source_format": "nemo_model",
        "cmd_style": "standard",
        "supports_revision": False,
        "variant_flag": None,
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 2500,
        "status": "supported",
        "description": "Canary speech encoder coupled with Qwen LLM decoder",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "voxtral": {
        "name": "Voxtral",
        "script": "convert-voxtral.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 2000,
        "status": "supported",
        "description": "Mistral-based speech recognition model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "voxtral_realtime": {
        "name": "Voxtral Realtime",
        "script": "convert-voxtral_realtime.py",
        "architecture_type": "streaming",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 2000,
        "status": "supported",
        "description": "Streaming real-time Voxtral speech model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "granite": {
        "name": "IBM Granite Speech",
        "script": "convert-granite.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": ["en"],
        "vram_mb": 2200,
        "status": "supported",
        "description": "IBM Granite Speech autoregressive model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "granite_nar": {
        "name": "IBM Granite Non-Autoregressive",
        "script": "convert-granite_nar.py",
        "architecture_type": "non_autoregressive",
        "source_format": "hf_snapshot",
        "cmd_style": "outdir",  # expects --outdir <dir>
        "supports_revision": True,
        "variant_flag": None,
        "sample_rate": 16000,
        "default_languages": ["en"],
        "vram_mb": 800,
        "status": "supported",
        "description": "IBM Granite Non-autoregressive fast speech model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "gigaam": {
        "name": "Sber GigaAM",
        "script": "convert-gigaam.py",
        "architecture_type": "non_autoregressive",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": False,
        "variant_flag": "--variant-key",  # uses --variant-key instead of --variant
        "sample_rate": 16000,
        "default_languages": ["ru"],
        "vram_mb": 600,
        "status": "supported",
        "description": "Russian speech recognition model by SberDevices",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "cohere": {
        "name": "Cohere Transcribe",
        "script": "convert-cohere.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": False,
        "variant_flag": None,
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 2200,
        "status": "supported",
        "description": "Cohere speech recognition model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "funasr_nano": {
        "name": "FunASR Nano",
        "script": "convert-funasr_nano.py",
        "architecture_type": "non_autoregressive",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": ["zh", "en"],
        "vram_mb": 300,
        "status": "supported",
        "description": "Compact lightweight Chinese/English ASR model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "medasr": {
        "name": "MedASR",
        "script": "convert-medasr.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "outdir",  # expects --outdir <dir>
        "supports_revision": True,
        "variant_flag": None,
        "sample_rate": 16000,
        "default_languages": ["en"],
        "vram_mb": 1200,
        "status": "supported",
        "description": "Specialized clinical and medical speech recognition model",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "moss": {
        "name": "OpenBMB MOSS-Speech",
        "script": "convert-moss.py",
        "architecture_type": "offline_llm",
        "source_format": "hf_snapshot",
        "cmd_style": "standard",
        "supports_revision": True,
        "variant_flag": "--variant",
        "sample_rate": 16000,
        "default_languages": ["zh", "en"],
        "vram_mb": 1600,
        "status": "experimental",
        "description": "Conversational speech model from OpenBMB (experimental upstream)",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
    "sortformer": {
        "name": "Sortformer Diarization",
        "script": "convert-sortformer.py",
        "architecture_type": "streaming",
        "source_format": "nemo_model",
        "cmd_style": "out_flag",  # expects --out <file>
        "supports_revision": False,
        "variant_flag": None,
        "sample_rate": 16000,
        "default_languages": "auto",
        "vram_mb": 400,
        "status": "experimental",
        "description": "Multi-speaker diarization model (experimental upstream)",
        "required_pkgs": ["torch>=2.0.0", "safetensors", "gguf>=0.10.0"],
    },
}


def download_file(url: str, dest_path: Path, timeout: float = 30.0) -> bool:
    """Download single file from URL with user-agent header."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "transcribe-convert-sync/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
        dest_path.write_bytes(content)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url} -> {dest_path}: {e}")
        return False


def sync_lib(force: bool = False) -> bool:
    """Ensure scripts/lib/* shared helper modules are present."""
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for f in SHARED_LIB_FILES:
        target = LIB_DIR / f
        if target.exists() and not force:
            continue
        url = f"{RAW_BASE_URL}/scripts/lib/{f}"
        print(f"  [Sync] Fetching shared lib: {f} ...")
        if not download_file(url, target):
            all_ok = False
    return all_ok


def sync_family_script(family_id: str, force: bool = False) -> Optional[Path]:
    """Ensure specific family converter script is available locally."""
    info = FAMILY_CATALOG.get(family_id)
    if not info:
        logger.error(f"Unknown family: {family_id}")
        return None

    script_name = info["script"]
    target = SCRIPTS_DIR / script_name
    if target.exists() and not force:
        return target

    # Make sure lib is also synced
    sync_lib(force=force)

    url = f"{RAW_BASE_URL}/scripts/{script_name}"
    print(f"  [Sync] Fetching converter script for {family_id} ({script_name}) ...")
    if download_file(url, target):
        return target
    return target if target.exists() else None


def is_family_available(family_id: str) -> bool:
    """Check if the family script and lib files exist locally."""
    info = FAMILY_CATALOG.get(family_id)
    if not info:
        return False
    script_path = SCRIPTS_DIR / info["script"]
    if not script_path.exists():
        return False
    for f in SHARED_LIB_FILES:
        if not (LIB_DIR / f).exists():
            return False
    return True


def _parse_version_tuple(v_str: str) -> Tuple[int, ...]:
    """Extract numeric components for version comparison."""
    parts = re.findall(r"\d+", v_str)
    return tuple(int(p) for p in parts) if parts else (0,)


def check_family_dependencies(family_id: str) -> Tuple[bool, List[str]]:
    """Check if Python dependencies required by this family converter are installed with required versions."""
    import importlib.util
    import importlib.metadata

    info = FAMILY_CATALOG.get(family_id)
    if not info:
        return False, ["unknown family"]

    missing: List[str] = []

    # Map package/import names to distribution names in metadata
    dist_map = {
        "torch": "torch",
        "safetensors": "safetensors",
        "gguf": "gguf",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "numpy": "numpy",
    }

    for req in info.get("required_pkgs", []):
        # Check if version specifier exists (e.g. gguf>=0.10.0)
        m = re.match(r"^([a-zA-Z0-9_\-]+)(>=|==|>|<)?(.*)$", req.strip())
        if not m:
            pkg_name, op, req_ver = req, None, ""
        else:
            pkg_name, op, req_ver = m.group(1), m.group(2), m.group(3)

        # 1. Check module import spec
        spec = importlib.util.find_spec(pkg_name)
        if spec is None:
            missing.append(req)
            continue

        # 2. Check version constraint if specified
        if op and req_ver:
            dist_name = dist_map.get(pkg_name, pkg_name)
            try:
                installed_ver = importlib.metadata.version(dist_name)
                inst_tuple = _parse_version_tuple(installed_ver)
                req_tuple = _parse_version_tuple(req_ver)
                if op == ">=" and inst_tuple < req_tuple:
                    missing.append(f"{pkg_name} (installed {installed_ver} < required {req_ver})")
                elif op == "==" and inst_tuple != req_tuple:
                    missing.append(f"{pkg_name} (installed {installed_ver} != required {req_ver})")
            except Exception:
                # If metadata not found but spec is present, assume ok
                pass

    if info.get("source_format") == "nemo_model":
        if importlib.util.find_spec("nemo") is None:
            missing.append("nemo_toolkit[asr]")

    return (len(missing) == 0), missing


def build_converter_command(
    family_id: str,
    script_path: Path,
    model_source: str | Path,
    out_path: Path,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[str]:
    """Adapter layer to construct the exact CLI command according to the family's upstream argument schema."""
    info = FAMILY_CATALOG.get(family_id, {})
    cmd_style = info.get("cmd_style", "standard")
    supports_rev = info.get("supports_revision", True)
    variant_flag = info.get("variant_flag", "--variant")

    cmd: List[str] = [
        sys.executable,
        str(script_path),
        str(model_source),
    ]

    # Destination argument handling based on style
    if cmd_style == "standard":
        cmd.append(str(out_path))
    elif cmd_style == "outdir":
        # Scripts like convert-granite_nar.py and convert-medasr.py take --outdir <dir>
        cmd.extend(["--outdir", str(out_path.parent)])
    elif cmd_style == "out_flag":
        # Scripts like convert-sortformer.py take --out <file>
        cmd.extend(["--out", str(out_path)])

    # Repo ID flag
    if repo_id:
        cmd.extend(["--repo-id", repo_id])

    # Revision flag
    if revision and supports_rev:
        cmd.extend(["--revision", revision])

    # Variant flag
    if variant and variant_flag:
        cmd.extend([variant_flag, variant])

    return cmd


def sync_all(force: bool = False) -> Dict[str, bool]:
    """Sync all shared libraries and converter scripts."""
    print("Synchronizing shared libraries...")
    sync_lib(force=force)
    results = {}
    print(f"Synchronizing {len(FAMILY_CATALOG)} converter scripts...")
    for fam, info in FAMILY_CATALOG.items():
        res = sync_family_script(fam, force=force)
        results[fam] = (res is not None and res.exists())
    return results


if __name__ == "__main__":
    force_sync = "--force" in sys.argv
    fam_arg = next((a for a in sys.argv[1:] if not a.startswith("--")), None)
    if fam_arg:
        res = sync_family_script(fam_arg, force=force_sync)
        if res:
            print(f"Successfully synced {fam_arg} -> {res}")
        else:
            print(f"Failed to sync {fam_arg}")
            sys.exit(1)
    else:
        results = sync_all(force=force_sync)
        synced = sum(1 for v in results.values() if v)
        print(f"\nCompleted: {synced}/{len(results)} family scripts ready in {SCRIPTS_DIR}")
