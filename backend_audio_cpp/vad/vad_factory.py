"""Factory and registry manager for VAD engines (FSMN-VAD & Silero-VAD)."""

import logging
import threading
from typing import Any, Dict, List, Optional

from backend_audio_cpp.vad.base_vad import BaseVADEngine, VADConfig
from backend_audio_cpp.vad.fsmn_vad_engine import FsmnVADEngine
from backend_audio_cpp.vad.vad_engine import SileroVADEngine

logger = logging.getLogger("backend_audio_cpp.vad.factory")


class VADFactory:
    """Manages creation and caching of VAD engines."""

    _instances: Dict[str, BaseVADEngine] = {}
    _lock = threading.Lock()

    @classmethod
    def get_engine(cls, engine_id: Optional[str] = None, config: Optional[VADConfig] = None) -> BaseVADEngine:
        """Get or create singleton VAD engine instance.

        Supported IDs:
            - 'fsmn-vad' / 'fsmn' / 'funasr' (Default, recommended for ASMR & Whisper)
            - 'silero-vad' / 'silero' (Standard Silero v5)
        """
        key = (engine_id or "fsmn-vad").strip().lower()
        if "silero" in key:
            canonical = "silero-vad"
        else:
            canonical = "fsmn-vad"

        with cls._lock:
            if canonical not in cls._instances:
                cfg = config or VADConfig()
                if canonical == "silero-vad":
                    cfg.model_path = "backend_audio_cpp/models/silero_vad.onnx"
                    cls._instances[canonical] = SileroVADEngine(cfg)
                else:
                    cfg.model_path = "backend_audio_cpp/models/fsmn_vad.onnx"
                    cls._instances[canonical] = FsmnVADEngine(cfg)

            return cls._instances[canonical]

    @classmethod
    def list_engines(cls) -> List[Dict[str, Any]]:
        """List catalog of available VAD engines for Extension UI."""
        return [
            {
                "id": "fsmn-vad",
                "name": "FSMN-VAD (ASMR / Thì thầm 🎙️)",
                "description": "FunASR FSMN deep neural VAD, cực nhạy với giọng nhỏ, thì thầm & đa ngôn ngữ",
                "is_default": True,
            },
            {
                "id": "silero-vad",
                "name": "Silero VAD v5 (Tiêu chuẩn ⚡)",
                "description": "Silero VAD v5 siêu nhẹ cho giọng nói hội thoại thông thường",
                "is_default": False,
            },
        ]
