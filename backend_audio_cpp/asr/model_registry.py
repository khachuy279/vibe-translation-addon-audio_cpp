"""ASR Model Registry and Live Dynamic Switching Manager for backend_audio_cpp.

Parses models.yaml, resolves model aliases, tracks active model key for live switching,
and ensures GGUF model files are present or downloaded from Hugging Face.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from huggingface_hub import hf_hub_download

logger = logging.getLogger("backend_audio_cpp.asr.registry")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend_audio_cpp"
_MODELS_DIR = _BACKEND_DIR / "models"
_YAML_PATH = _BACKEND_DIR / "models.yaml"
_SERVER_CONFIG_PATH = _BACKEND_DIR / "config_server.json"


class ASRModelRegistry:
    """Singleton registry managing ASR model catalog and live dynamic switching."""

    _instance: Optional["ASRModelRegistry"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ASRModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ASRModelRegistry()
            return cls._instance

    def __init__(self):
        self._models_dir = _MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._models_catalog: Dict[str, Dict[str, Any]] = {}
        self._alias_map: Dict[str, str] = {}
        self._default_model_key: str = "qwen3-asr-1.7b"
        self._active_model_key: str = "qwen3-asr-1.7b"
        self._state_lock = threading.Lock()
        self.reload_catalog()

    def reload_catalog(self) -> None:
        """Parse models.yaml and build alias map."""
        if not _YAML_PATH.exists():
            raise FileNotFoundError(f"models.yaml not found at: {_YAML_PATH}")

        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._default_model_key = data.get("default_model", "qwen3-asr-1.7b")
        self._active_model_key = self._default_model_key
        raw_models = data.get("models", {})

        self._models_catalog.clear()
        self._alias_map.clear()

        for key, m_cfg in raw_models.items():
            self._models_catalog[key] = m_cfg
            self._alias_map[key.lower()] = key
            for alias in m_cfg.get("aliases", []):
                self._alias_map[alias.lower()] = key

        logger.info(f"Loaded {len(self._models_catalog)} ASR models from models.yaml")

    def resolve_key(self, query: str) -> str:
        """Resolve model alias or key to canonical key."""
        if not query:
            return self._default_model_key
        q = query.strip().lower()
        return self._alias_map.get(q, self._default_model_key)

    def get_active_model_key(self) -> str:
        with self._state_lock:
            return self._active_model_key

    def set_active_model_key(self, key: str) -> str:
        """Switch active model live without restarting the server."""
        canonical = self.resolve_key(key)
        self.ensure_model_downloaded(canonical)
        with self._state_lock:
            old_key = self._active_model_key
            self._active_model_key = canonical
        logger.info(f"[REGISTRY] Switched active ASR model: '{old_key}' -> '{canonical}'")
        return canonical

    def get_model(self, key: str) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_key(key)
        return self._models_catalog.get(canonical)

    def list_models(self) -> List[Dict[str, Any]]:
        """Return list of models for extension dropdown UI."""
        res = []
        for k, v in self._models_catalog.items():
            res.append({
                "id": k,
                "name": v.get("name", k),
                "family": v.get("family", ""),
                "description": v.get("description", ""),
                "languages": v.get("languages", "auto"),
                "is_active": (k == self._active_model_key),
            })
        return res

    def get_model_file_path(self, key: str) -> Path:
        """Get path to the local model file."""
        canonical = self.resolve_key(key)
        cfg = self._models_catalog.get(canonical, {})
        filename = cfg.get("file", f"{canonical}.gguf")
        return self._models_dir / filename

    def is_model_downloaded(self, key: str) -> bool:
        """Check if local GGUF file exists."""
        return self.get_model_file_path(key).exists()

    def ensure_model_downloaded(self, key: str) -> Path:
        """Ensure model file exists locally; if not, download from HF Hub."""
        canonical = self.resolve_key(key)
        target_path = self.get_model_file_path(canonical)
        if target_path.exists():
            return target_path

        cfg = self._models_catalog.get(canonical, {})
        repo_id = cfg.get("hf_repo", "audio-cpp/audio.cpp-gguf")
        hf_filename = cfg.get("hf_filename")
        if not hf_filename:
            raise ValueError(f"No hf_filename specified for model '{canonical}' in models.yaml")

        logger.info(f"[REGISTRY] Downloading {canonical} from {repo_id}:{hf_filename}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            local_dir=str(self._models_dir),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != target_path:
            downloaded_path.rename(target_path)

        logger.info(f"[REGISTRY] Downloaded and ready: {target_path}")
        return target_path

    def generate_server_config(self) -> Dict[str, Any]:
        """Generate audio.cpp config_server.json declaring all models with lazy loading."""
        models_list = []
        for key, cfg in self._models_catalog.items():
            file_path = self.get_model_file_path(key)
            models_list.append({
                "id": key,
                "family": cfg.get("family", "qwen3_asr"),
                "path": str(file_path).replace("\\", "/"),
                "task": cfg.get("task", "asr"),
                "mode": cfg.get("mode", "offline"),
            })

        # Ensure omnivoice TTS is always registered in audio.cpp server
        omnivoice_path = self._models_dir / "omnivoice-q8_0.gguf"
        if omnivoice_path.exists():
            models_list.append({
                "id": "omnivoice",
                "family": "omnivoice",
                "path": str(omnivoice_path).replace("\\", "/"),
                "task": "tts",
            })

        server_cfg = {
            "host": "127.0.0.1",
            "port": 8089,
            "backend": "cuda",
            "device": 0,
            "threads": 4,
            "lazy_load": True,
            "max_loaded_models": 1,
            "models": models_list,
        }

        with open(_SERVER_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(server_cfg, f, indent=2)

        logger.info(f"[REGISTRY] Generated server config with {len(models_list)} models at {_SERVER_CONFIG_PATH}")
        return server_cfg


# Alias for backward compatibility and clean imports
ModelRegistry = ASRModelRegistry
