"""Model Registry for transcribe.cpp models in backend_cpp."""

import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import yaml

from backend_cpp.config import config, MODELS_DIR, MODELS_YAML_PATH, LEGACY_MODELS_DIR

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages available transcribe.cpp GGUF models, resolution, and downloads."""

    _instance: Optional["ModelRegistry"] = None
    _instance_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = yaml_path or MODELS_YAML_PATH
        self.models: Dict[str, Dict[str, Any]] = {}
        cfg_default = getattr(config.asr, "active_model", "qwen3-asr-1.7b")
        self.default_model_key: str = cfg_default
        self._active_model_key: str = cfg_default
        self._resolved_path_cache: Dict[str, Optional[str]] = {}
        self.load_registry()

    def load_registry(self) -> None:
        """Load or reload model catalog from models.yaml."""
        self._resolved_path_cache.clear()
        if not self.yaml_path.exists():
            logger.warning(f"models.yaml not found at {self.yaml_path}, using defaults")
            return

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            cfg_model = getattr(config.asr, "active_model", None)
            self.default_model_key = cfg_model or data.get("default_model", self.default_model_key)
            self.models = data.get("models", {})
            self._active_model_key = self.default_model_key
            logger.info(f"Loaded {len(self.models)} model definition(s) from {self.yaml_path}")
        except Exception as e:
            logger.error(f"Error parsing {self.yaml_path}: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models with downloaded status."""
        result = []
        for key, info in self.models.items():
            model_file = info.get("file", "")
            is_downloaded = self.resolve_model_path(key) is not None
            result.append({
                "id": key,
                "name": info.get("name", key),
                "family": info.get("family", ""),
                "architecture_type": info.get("architecture_type", "offline_llm"),
                "is_downloaded": is_downloaded,
                "is_active": (key == self._active_model_key),
                "vram_mb": info.get("vram_estimate_mb", 0),
                "description": info.get("description", ""),
            })
        return result

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get specification for a given model key."""
        return self.models.get(model_key)

    def is_streaming_model(self, model_key: str) -> bool:
        """Check if model key supports native streaming from metadata or known streaming families."""
        info = self.models.get(model_key)
        if not info:
            return False
        if info.get("architecture_type") == "streaming":
            return True
        if info.get("family") in ("nemotron", "parakeet", "voxtral", "moonshine"):
            return True
        return False

    def resolve_model_path(self, model_key: str) -> Optional[str]:
        """Find absolute path to GGUF model file on disk, with caching."""
        if model_key in self._resolved_path_cache:
            cached = self._resolved_path_cache[model_key]
            if cached is None or Path(cached).exists():
                return cached

        info = self.models.get(model_key)
        if not info:
            self._resolved_path_cache[model_key] = None
            return None

        filename = info.get("file", "")
        if not filename:
            self._resolved_path_cache[model_key] = None
            return None

        # 1. Check in backend_cpp/models/
        p1 = MODELS_DIR / filename
        if p1.exists():
            resolved = str(p1)
            self._resolved_path_cache[model_key] = resolved
            return resolved

        # 2. Check in legacy backend/models/
        p2 = LEGACY_MODELS_DIR / filename
        if p2.exists():
            resolved = str(p2)
            self._resolved_path_cache[model_key] = resolved
            return resolved

        # 3. Check in legacy backend/models/huggingface/
        p3 = LEGACY_MODELS_DIR / "huggingface" / filename
        if p3.exists():
            resolved = str(p3)
            self._resolved_path_cache[model_key] = resolved
            return resolved

        self._resolved_path_cache[model_key] = None
        return None

    def invalidate_cache(self, model_key: Optional[str] = None) -> None:
        """Clear resolved model path cache."""
        if model_key:
            self._resolved_path_cache.pop(model_key, None)
        else:
            self._resolved_path_cache.clear()

    def ensure_model(self, model_key: str) -> str:
        """Ensure model is present on disk, downloading if missing."""
        path = self.resolve_model_path(model_key)
        if path:
            return path

        info = self.models.get(model_key)
        if not info:
            raise ValueError(f"Unknown model '{model_key}' in registry")

        repo_id = info.get("hf_repo")
        filename = info.get("file")

        if not repo_id or not filename:
            raise ValueError(f"Model '{model_key}' missing 'hf_repo' or 'file' for auto-download")

        logger.info(f"Downloading model '{model_key}' ({filename}) from HuggingFace '{repo_id}'...")
        from huggingface_hub import hf_hub_download

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
        )
        self._resolved_path_cache[model_key] = downloaded
        logger.info(f"Model '{model_key}' downloaded successfully to {downloaded}")
        return downloaded

    def get_active_model_key(self) -> str:
        return self._active_model_key

    def set_active_model_key(self, key: str) -> None:
        if key not in self.models:
            raise ValueError(f"Model key '{key}' not found in registry. Available: {list(self.models.keys())}")
        self._active_model_key = key
        config.asr.active_model = key
