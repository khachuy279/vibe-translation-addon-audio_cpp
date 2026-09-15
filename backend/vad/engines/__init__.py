"""Factory quản lý và cache các Singleton VAD Engine."""

import threading
from typing import Dict, Tuple, Any, Optional

from backend.config import config
from backend.vad.base import BaseVADEngine
from backend.vad.engines.firered import FireRedVADEngine
from backend.vad.engines.fsmn import FsmnVADEngine
from backend.vad.engines.silero import SileroVADEngine

SUPPORTED_VAD_ENGINES: Tuple[str, ...] = ("firered-vad", "fsmn-vad", "silero-vad")


class VADEngineFactory:
    """Thread-safe Factory & Pool cho các VAD Engine."""

    _engines: Dict[str, BaseVADEngine] = {}
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def get_engine(cls, engine_name: str) -> BaseVADEngine:
        """Lấy hoặc khởi tạo instance Engine VAD dùng chung."""
        engine = (engine_name or "firered-vad").lower().strip()
        if engine not in SUPPORTED_VAD_ENGINES:
            engine = "firered-vad"

        with cls._lock:
            if engine in cls._engines:
                return cls._engines[engine]

            if engine == "firered-vad":
                instance = FireRedVADEngine()
            elif engine == "silero-vad":
                instance = SileroVADEngine()
            else:  # fsmn-vad
                instance = FsmnVADEngine()

            cls._engines[engine] = instance
            return instance

    @classmethod
    def reset_pool(cls) -> None:
        """Giải phóng các instance engine đang cache."""
        with cls._lock:
            cls._engines.clear()


__all__ = [
    "BaseVADEngine",
    "FireRedVADEngine",
    "SileroVADEngine",
    "FsmnVADEngine",
    "VADEngineFactory",
    "SUPPORTED_VAD_ENGINES",
]
