"""Deduplication utilities for streaming translation and TTS in backend_cpp."""

from dataclasses import dataclass
import time
from typing import Optional
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter


class SlidingWindowDedup:
    """Generic sliding window deduplicator for streaming texts."""

    def __init__(self, window_sec: float = 3.5):
        self.window_sec: float = window_sec
        self.last_norm: str = ""
        self.last_time: float = 0.0

    def is_duplicate(self, text: str, now: Optional[float] = None) -> bool:
        """Check if incoming text is duplicate within the configured sliding time window.

        If not duplicate, updates state with the normalized text and timestamp.
        """
        if not text:
            return False

        norm = SentenceSegmenter.normalize_for_comparison(text)
        current_time = time.monotonic() if now is None else now

        if norm and norm == self.last_norm and (current_time - self.last_time < self.window_sec):
            return True

        self.last_norm = norm
        self.last_time = current_time
        return False

    def reset(self) -> None:
        """Clear state."""
        self.last_norm = ""
        self.last_time = 0.0


class TranslationDedupState(SlidingWindowDedup):
    """Encapsulates state for translation deduplication with backward-compatible property aliases."""

    @property
    def last_trans_norm(self) -> str:
        return self.last_norm

    @last_trans_norm.setter
    def last_trans_norm(self, val: str) -> None:
        self.last_norm = val

    @property
    def last_trans_time(self) -> float:
        return self.last_time

    @last_trans_time.setter
    def last_trans_time(self, val: float) -> None:
        self.last_time = val


class TTSDedupState(SlidingWindowDedup):
    """Encapsulates state for TTS synthesis deduplication with backward-compatible property aliases."""

    @property
    def last_tts_norm(self) -> str:
        return self.last_norm

    @last_tts_norm.setter
    def last_tts_norm(self, val: str) -> None:
        self.last_norm = val

    @property
    def last_tts_time(self) -> float:
        return self.last_time

    @last_tts_time.setter
    def last_tts_time(self, val: float) -> None:
        self.last_time = val
