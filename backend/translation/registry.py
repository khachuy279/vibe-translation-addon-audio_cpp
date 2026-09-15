"""TranslationModelRegistry: Quản lý catalog các mô hình dịch từ translation_models.yaml."""

import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import yaml

from backend.config import (
    config,
    MODELS_DIR,
    TRANSLATION_MODELS_YAML_PATH,
)
from backend.utils.logger import logger


class TranslationModelRegistry:
    """Singleton quản lý catalog Translation Model."""

    _instance: Optional["TranslationModelRegistry"] = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "TranslationModelRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = TranslationModelRegistry()
            return cls._instance

    def __init__(self, yaml_path: Optional[str] = None):
        self._yaml_path = Path(yaml_path) if yaml_path else TRANSLATION_MODELS_YAML_PATH
        self._models: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}
        self.default_model_key: str = "tencent"
        self._load_yaml()

    def _load_yaml(self) -> None:
        """Đọc và phân tích file translation_models.yaml."""
        if not self._yaml_path.exists():
            logger.warning(f"Không tìm thấy translation_models.yaml tại: {self._yaml_path}", extra={"module_tag": "TRANSLATE"})
            return

        with open(self._yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self.default_model_key = data.get("default_model", "tencent")
        self._models = data.get("models", {})

        # Xây dựng bảng alias
        self._aliases.clear()
        for key, info in self._models.items():
            self._aliases[key.lower()] = key
            for alias in info.get("aliases", []):
                self._aliases[alias.lower()] = key

    def resolve_key(self, key_or_alias: str) -> str:
        """Chuẩn hóa alias về canonical model key."""
        clean = (key_or_alias or "").strip().lower()
        return self._aliases.get(clean, self.default_model_key)

    def is_known(self, key_or_alias: str) -> bool:
        """True nếu key/alias có thật trong catalog (khác với `resolve_key` luôn fallback)."""
        return (key_or_alias or "").strip().lower() in self._aliases

    def get_model(self, key_or_alias: str) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_key(key_or_alias)
        return self._models.get(canonical)

    def list_models(self) -> List[Dict[str, Any]]:
        """Trả về danh sách các model dịch khả dụng (kèm cờ đã có file cục bộ hay chưa)."""
        result = []
        active_key = self.resolve_key(config.translation.base)
        for k, v in self._models.items():
            item = dict(v)
            item["id"] = k
            item["is_active"] = (k == active_key)
            # Popup dùng `is_downloaded` để đánh dấu ⚡ (đã có sẵn) hay "chưa tải".
            item["is_downloaded"] = self.is_downloaded(k)
            result.append(item)
        return result

    def is_downloaded(self, key_or_alias: str) -> bool:
        """True nếu file GGUF của model đã có sẵn trong `backend/models` (không chạm mạng)."""
        try:
            return os.path.isfile(self.resolve_gguf_path(key_or_alias))
        except Exception:  # noqa: BLE001
            return False

    def repo_id(self, key_or_alias: str) -> str:
        """Repo HuggingFace của model (trường `model` trong YAML)."""
        return str((self.get_model(key_or_alias) or {}).get("model", "") or "")

    def resolve_gguf_path(self, key_or_alias: str) -> str:
        """Tìm đường dẫn file GGUF của model dịch trên đĩa."""
        info = self.get_model(key_or_alias)
        if not info:
            raise ValueError(f"Không tìm thấy model dịch cho key: {key_or_alias}")

        filename = info.get("gguf_file", "")
        if not filename:
            raise ValueError(f"Không có trường 'gguf_file' trong model {key_or_alias}")

        return str(MODELS_DIR / filename)
