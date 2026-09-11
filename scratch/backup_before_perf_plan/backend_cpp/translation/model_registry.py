"""Translation Model Registry for local GGUF translation models in backend_cpp."""

import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import yaml

BACKEND_CPP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_CPP_DIR / "models"
TRANSLATION_MODELS_YAML_PATH = BACKEND_CPP_DIR / "translation_models.yaml"

logger = logging.getLogger(__name__)

# Built-in fallback definitions in case YAML file is missing or corrupted
DEFAULT_TRANSLATION_MODELS: Dict[str, Dict[str, Any]] = {
    "tencent-1.8b": {
        "name": "Hunyuan-MT2 1.8B (Tencent)",
        "model": "unsloth/Hy-MT2-1.8B-GGUF",
        "gguf_file": "Hy-MT2-1.8B-UD-Q8_K_XL.gguf",
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "prompt_style": "tencent",
        "aliases": ["hy-mt2-1.8b", "tencent-1.8"],
        "description": "Siêu nhanh, nhẹ (~2.0GB, Q8_K_XL)",
    },
    "tencent": {
        "name": "Hunyuan-MT2 7B (Tencent)",
        "model": "unsloth/Hy-MT2-7B-GGUF",
        "gguf_file": "Hy-MT2-7B-UD-Q4_K_XL.gguf",
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "prompt_style": "tencent",
        "aliases": ["hy-mt2-7b", "tencent-7b"],
        "description": "Chất lượng cao (~4.6GB, Q4_K_XL)",
    },
    "xiaomi": {
        "name": "MiLM-MT 4.6B (Xiaomi)",
        "model": "mradermacher/MiLMMT-46-4B-v1.0-GGUF",
        "gguf_file": "MiLMMT-46-4B-v1.0.Q4_K_M.gguf",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "prompt_style": "milmmt",
        "aliases": ["milmmt"],
        "description": "Nhanh & Nhẹ (~2.5GB)",
    },
    "gemmax": {
        "name": "GemmaX2 28-9B v0.2 (Xiaomi)",
        "model": "mradermacher/GemmaX2-28-9B-v0.2-i1-GGUF",
        "gguf_file": "GemmaX2-28-9B-v0.2.i1-Q4_K_M.gguf",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "prompt_style": "gemmax",
        "aliases": ["gemma", "gemmax2", "gemmax-9b"],
        "description": "Chất lượng cao 28 ngôn ngữ (~5.8GB)",
    },
}


class TranslationModelRegistry:
    """Manages available translation GGUF models catalog, aliases, and specs."""

    _instance: Optional["TranslationModelRegistry"] = None
    _instance_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls, yaml_path: Optional[Path] = None) -> "TranslationModelRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(yaml_path=yaml_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = yaml_path or TRANSLATION_MODELS_YAML_PATH
        self.models: Dict[str, Dict[str, Any]] = {}
        self.alias_map: Dict[str, str] = {}
        self.default_model_key: str = "tencent"
        self.load_registry()

    def load_registry(self) -> None:
        """Load or reload model catalog from translation_models.yaml."""
        self.models.clear()
        self.alias_map.clear()

        if not self.yaml_path.exists():
            logger.warning(
                f"translation_models.yaml not found at {self.yaml_path}, using default catalog"
            )
            self.models = dict(DEFAULT_TRANSLATION_MODELS)
        else:
            try:
                with open(self.yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self.default_model_key = data.get("default_model", "tencent")
                self.models = data.get("models", {})
                logger.info(
                    f"Loaded {len(self.models)} translation model definition(s) from {self.yaml_path}"
                )
            except Exception as e:
                logger.error(
                    f"Error parsing {self.yaml_path}: {e}. Falling back to default catalog."
                )
                self.models = dict(DEFAULT_TRANSLATION_MODELS)

        # Build alias lookup map (lowercase for case-insensitive lookup)
        for key, info in self.models.items():
            k_lower = key.lower().strip()
            self.alias_map[k_lower] = key
            for alias in info.get("aliases", []):
                self.alias_map[alias.lower().strip()] = key

    def resolve_key(self, key_or_alias: str) -> str:
        """Resolve model key or alias to canonical model key, falling back to default."""
        if not key_or_alias:
            return self.default_model_key
        normalized = key_or_alias.lower().strip()
        return self.alias_map.get(normalized, key_or_alias)

    def get_model(self, key_or_alias: str) -> Optional[Dict[str, Any]]:
        """Get model specification dictionary by key or alias."""
        canonical = self.resolve_key(key_or_alias)
        return self.models.get(canonical)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models with downloaded status for API and UI clients."""
        result = []
        for key, info in self.models.items():
            gguf_file = info.get("gguf_file", "")
            is_downloaded = False
            if gguf_file:
                target_path = MODELS_DIR / gguf_file
                if target_path.exists():
                    is_downloaded = True
                else:
                    # Check common alternative naming variations
                    alt_names = []
                    if ".i1-" in gguf_file:
                        alt_names.append(gguf_file.replace(".i1-", "."))
                        alt_names.append(gguf_file.replace(".i1-", "-"))
                    elif ".Q4_K_M" in gguf_file:
                        alt_names.append(gguf_file.replace(".Q4_K_M", ".i1-Q4_K_M"))
                        alt_names.append(gguf_file.replace("-Q4_K_M", ".i1-Q4_K_M"))
                    for alt in alt_names:
                        if (MODELS_DIR / alt).exists():
                            is_downloaded = True
                            break

            result.append(
                {
                    "id": key,
                    "name": info.get("name", key),
                    "desc": info.get("description", ""),
                    "gguf_file": gguf_file,
                    "model": info.get("model", ""),
                    "temperature": info.get("temperature", 0.0),
                    "top_p": info.get("top_p", 1.0),
                    "top_k": info.get("top_k", 1),
                    "repetition_penalty": info.get("repetition_penalty", 1.0),
                    "prompt_style": info.get("prompt_style", "tencent"),
                    "is_downloaded": is_downloaded,
                }
            )
        return result
