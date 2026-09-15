"""Abstract Base Class cho các ASR Engine."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional
import numpy as np


class BaseASREngine(ABC):
    """Interface tiêu chuẩn cho engine nhận dạng giọng nói ASR."""

    @abstractmethod
    def feed_audio(self, audio_data: np.ndarray, timestamp: float = 0.0) -> None:
        """Nạp chunk âm thanh float32 vào bộ đệm của ASR engine."""
        pass

    @abstractmethod
    def on_speech_start(self) -> None:
        """Callback khi VAD phát hiện bắt đầu nói."""
        pass

    @abstractmethod
    def on_speech_end(self, reason: str = "VAD_SILENCE") -> None:
        """Callback khi VAD phát hiện kết thúc nói (kích hoạt chốt câu)."""
        pass

    @abstractmethod
    async def stream_tokens(self) -> AsyncIterator[Dict[str, Any]]:
        """Async generator phát trực tiếp các bản tin utterance_update về client."""
        pass

    @abstractmethod
    def set_language(self, language: str) -> None:
        """Cập nhật ngôn ngữ nhận dạng."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Giải phóng tài nguyên session khi client ngắt kết nối."""
        pass
