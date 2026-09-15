"""ModelRegistry quản lý danh mục cấu hình và đường dẫn các mô hình ASR."""

import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import yaml

from backend.config import config, MODELS_DIR, MODELS_YAML_PATH
from backend.utils.logger import logger


class ModelRegistry:
    """Singleton quản lý catalog mô hình ASR từ models.yaml."""

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ModelRegistry()
            return cls._instance

    def __init__(self, yaml_path: Optional[str] = None):
        self._yaml_path = Path(yaml_path) if yaml_path else MODELS_YAML_PATH
        self._models: Dict[str, Dict[str, Any]] = {}
        self._default_model_key: str = "qwen3-asr-1.7b"
        self._active_model_key: str = config.asr.active_model
        self._load_yaml()

    def _load_yaml(self) -> None:
        """Đọc file models.yaml."""
        if not self._yaml_path.exists():
            logger.warning(f"models.yaml không tồn tại tại: {self._yaml_path}")
            return

        with open(self._yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._default_model_key = data.get("default_model", "qwen3-asr-1.7b")
        self._models = data.get("models", {})
        if not self._active_model_key or self._active_model_key not in self._models:
            self._active_model_key = self._default_model_key

    def get_active_model_key(self) -> str:
        return self._active_model_key

    def set_active_model_key(self, model_key: str) -> None:
        clean = (model_key or "").strip().lower()
        if clean in self._models:
            self._active_model_key = clean
            config.asr.active_model = clean
        else:
            raise ValueError(f"Mô hình '{model_key}' không có trong catalog models.yaml")

    def get_model_info(self, model_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        key = model_key or self._active_model_key
        return self._models.get(key)

    def list_models(self) -> List[Dict[str, Any]]:
        """Trả về danh sách tất cả các mô hình ASR khả dụng."""
        result = []
        for k, v in self._models.items():
            item = dict(v)
            item["id"] = k
            item["is_active"] = (k == self._active_model_key)
            result.append(item)
        return result

    def is_streaming_model(self, model_key: Optional[str] = None) -> bool:
        """Kiểm tra xem model có phải là kiến trúc streaming (session.stream) không."""
        info = self.get_model_info(model_key)
        if not info:
            return False
        return info.get("architecture_type") == "streaming"

    def resolve_model_path(self, model_key: Optional[str] = None) -> str:
        """Tìm đường dẫn file GGUF cục bộ của model."""
        info = self.get_model_info(model_key)
        if not info:
            raise ValueError(f"Không tìm thấy thông tin cho model key: {model_key}")

        filename = info.get("file", "")
        if not filename:
            raise ValueError(f"Không có trường 'file' trong cấu hình của model {model_key}")

        return str(MODELS_DIR / filename)
