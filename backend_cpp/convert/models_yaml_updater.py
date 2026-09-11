"""Safe updater for backend_cpp/models.yaml.

Features:
- Automatic key & name generation from HF repo and quantization preset.
- Preserves existing models and creates automatic backup (.bak) before modifying.
- Note: yaml.safe_dump serializes dictionary data; comments in models.yaml may be normalized.
- Atomic file write to prevent corruption.
- Checks for duplicate model keys.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger("models_yaml_updater")

BACKEND_CPP_DIR = Path(__file__).resolve().parent.parent
MODELS_YAML_PATH = BACKEND_CPP_DIR / "models.yaml"


def slugify_model_key(hf_repo: str, quant: Optional[str] = None) -> str:
    """Derive clean, lowercase model registry key from HF repo id and quant."""
    slug = hf_repo.strip().strip("/").split("/")[-1]
    # Remove common repetitive extensions/words
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "-", slug).lower()
    cleaned = re.sub(r"[-_]+", "-", cleaned).strip("-")
    # If key already ends with quant tag like -q8-0 or -bf16, strip it first
    known_quants = ["q8-0", "q8_0", "q4-k-m", "q4_k_m", "q5-k-m", "q5_k_m", "bf16", "f16", "f32"]
    for kq in known_quants:
        if cleaned.endswith(f"-{kq}"):
            cleaned = cleaned[:-len(kq)-1]
            break
    if quant:
        q_suffix = quant.lower().replace("_", "-")
        return f"{cleaned}-{q_suffix}"
    return cleaned


def generate_display_name(hf_repo: str, family: str, quant: str) -> str:
    """Create friendly display name for UI dropdown."""
    slug = hf_repo.strip().strip("/").split("/")[-1]
    # Keep uppercase letters if helpful
    clean_name = slug.replace("-", " ").replace("_", " ")
    return f"{clean_name} ({quant.upper()})"


def register_model(
    model_key: str,
    name: str,
    family: str,
    architecture_type: str,
    hf_repo: str,
    default_quant: str,
    file_name: str,
    sample_rate: int = 16000,
    languages: Any = "auto",
    vram_estimate_mb: int = 2000,
    description: str = "",
    yaml_path: Optional[Path] = None,
    overwrite: bool = False,
) -> bool:
    """Register or update a model definition in models.yaml with backup and atomic write."""
    target_yaml = Path(yaml_path or MODELS_YAML_PATH).resolve()
    if not target_yaml.exists():
        raise FileNotFoundError(f"models.yaml not found at: {target_yaml}")

    # 1. Read existing yaml content
    try:
        raw_text = target_yaml.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse existing models.yaml: {e}")

    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}

    existing_models = data["models"]

    if model_key in existing_models and not overwrite:
        raise ValueError(
            f"Model key '{model_key}' already exists in models.yaml. "
            f"Use overwrite=True or choose a different key."
        )

    # 2. Build model entry dict
    model_entry: Dict[str, Any] = {
        "name": name,
        "family": family,
        "architecture_type": architecture_type,
        "hf_repo": hf_repo,
        "default_quant": default_quant.upper(),
        "file": file_name,
        "sample_rate": sample_rate,
        "languages": languages,
        "vram_estimate_mb": vram_estimate_mb,
    }
    if architecture_type == "offline_llm":
        model_entry["context_window_sec"] = 30
    if description:
        model_entry["description"] = description
    else:
        model_entry["description"] = f"{name} model ported to transcribe.cpp"

    existing_models[model_key] = model_entry

    # 3. Create backup file (models.yaml.bak)
    backup_path = target_yaml.with_suffix(".yaml.bak")
    try:
        shutil.copy2(target_yaml, backup_path)
    except Exception as e:
        logger.warning(f"Could not create backup of models.yaml: {e}")

    # 4. Atomic write using temporary file
    temp_dir = target_yaml.parent
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        temp_name = tf.name
        yaml.safe_dump(
            data,
            tf,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            indent=2,
        )

    try:
        # Atomic replace
        Path(temp_name).replace(target_yaml)
        print(f"  ✅ [models.yaml] Successfully registered model '{model_key}' into {target_yaml.name}")
        return True
    except Exception as e:
        if Path(temp_name).exists():
            Path(temp_name).unlink(missing_ok=True)
        raise RuntimeError(f"Atomic update of models.yaml failed: {e}")


if __name__ == "__main__":
    import sys
    print(f"models.yaml path: {MODELS_YAML_PATH}")
    if MODELS_YAML_PATH.exists():
        with open(MODELS_YAML_PATH, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        print("Registered models:", list((content.get("models") or {}).keys()))
