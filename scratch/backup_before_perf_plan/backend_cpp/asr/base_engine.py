"""Abstract Base Class for ASR Engines in backend_cpp."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional


class BaseASREngine(ABC):
    """Abstract interface that all ASR engines must implement."""

    @abstractmethod
    def feed_audio(self, pcm_bytes: bytes, timestamp: float = 0.0) -> None:
        """Feed incoming audio PCM (16kHz 16-bit mono) from VAD/audio capture."""
        pass

    @abstractmethod
    def on_speech_start(self) -> None:
        """Callback when VAD detects speech start."""
        pass

    @abstractmethod
    def on_speech_end(self, reason: str = "VAD_SILENCE") -> None:
        """Callback when VAD detects speech end (trigger final commit)."""
        pass

    @abstractmethod
    async def stream_tokens(self) -> AsyncIterator[Dict[str, Any]]:
        """Async generator yielding utterance_update messages to broadcast to clients."""
        pass

    @abstractmethod
    def set_language(self, language: str) -> None:
        """Update active language or language hint."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup session resources on client disconnect."""
        pass
