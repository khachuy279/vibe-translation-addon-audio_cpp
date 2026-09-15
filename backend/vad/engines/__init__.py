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
    def is_cached(cls, engine_name: str) -> bool:
        """Engine đã được nạp sẵn trong pool chưa (không kích hoạt nạp/tải)."""
        engine = (engine_name or "").lower().strip()
        with cls._lock:
            return engine in cls._engines

    @classmethod
    def peek_engine(cls, engine_name: str) -> Optional[BaseVADEngine]:
        """Lấy engine CHỈ KHI đã có trong pool; trả None nếu chưa (không nạp, không tải)."""
        engine = (engine_name or "").lower().strip()
        with cls._lock:
            return cls._engines.get(engine)

    @classmethod
    def prewarm_engines(cls, names, threshold: Optional[float] = None) -> Dict[str, str]:
        """P1.9: nạp trước nhiều engine để chuyển nóng không phải tải/nạp model.

        Trả về dict {engine_name: "ok" | "error: ..."} để caller log rõ.
        Hàm này BLOCKING — gọi từ thread nền (asyncio.to_thread) khi khởi động.
        """
        results: Dict[str, str] = {}
        for name in names:
            key = (name or "").lower().strip()
            if key not in SUPPORTED_VAD_ENGINES:
                continue
            try:
                engine = cls.get_engine(key)
                # create_initial_state cũng tốn thời gian (Silero load JIT model) nên
                # warm luôn một lần rồi bỏ state — state thật là per-session.
                engine.create_initial_state(threshold=threshold)
                results[key] = "ok"
            except Exception as exc:  # noqa: BLE001
                results[key] = f"error: {type(exc).__name__}: {exc}"
        return results

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
