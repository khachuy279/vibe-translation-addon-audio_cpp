"""Translation Model Registry for backend_audio_cpp.

Manages dynamic registration and resolution of GGUF translation models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import logging

logger = logging.getLogger("backend_audio_cpp.translation.registry")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_FILE = _PROJECT_ROOT / "backend_audio_cpp" / "translation_models.yaml"
_MODELS_DIR = _PROJECT_ROOT / "backend_audio_cpp" / "models"


class TranslationModelRegistry:
    """Registry managing available translation models and aliases."""

    _instance: Optional["TranslationModelRegistry"] = None

    @classmethod
    def get_instance(cls) -> "TranslationModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or _CONFIG_FILE
        self.models_dir = _MODELS_DIR
        self.models: Dict[str, Dict[str, Any]] = {}
        self.aliases: Dict[str, str] = {}
        self.default_model_key: str = "tencent"
        self.active_model_key: str = "tencent"
        self.load()

    def load(self) -> None:
        """Load translation models configuration from YAML."""
        if not self.config_path.exists():
            logger.warning(f"Translation config not found at {self.config_path}")
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.default_model_key = data.get("default_model", "tencent")
        self.active_model_key = self.default_model_key
        raw_models = data.get("models", {})

        self.models.clear()
        self.aliases.clear()

        for key, m in raw_models.items():
            self.models[key] = m
            self.aliases[key.lower()] = key
            for alias in m.get("aliases", []):
                self.aliases[str(alias).lower()] = key

        logger.info(f"Loaded {len(self.models)} translation model(s) from {self.config_path}")

    def resolve_key(self, key_or_alias: Optional[str]) -> str:
        """Resolve model alias or key to canonical translation model key."""
        if not key_or_alias:
            return self.active_model_key
        return self.aliases.get(key_or_alias.lower().strip(), key_or_alias)

    def get_model(self, key_or_alias: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata by key or alias."""
        key = self.resolve_key(key_or_alias)
        return self.models.get(key)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models for frontend options."""
        result = []
        for key, m in self.models.items():
            gguf_name = m.get("gguf_file", "")
            is_present = (_MODELS_DIR / gguf_name).exists()
            result.append({
                "id": key,
                "name": m.get("name", key),
                "model": m.get("model", ""),
                "gguf_file": gguf_name,
                "is_present": is_present,
                "is_active": (key == self.active_model_key),
                "description": m.get("description", ""),
            })
        return result

    def set_active_model_key(self, key_or_alias: str) -> str:
        """Set active translation model key."""
        resolved = self.resolve_key(key_or_alias)
        if resolved not in self.models:
            raise KeyError(f"Unknown translation model '{key_or_alias}'")
        self.active_model_key = resolved
        return resolved
