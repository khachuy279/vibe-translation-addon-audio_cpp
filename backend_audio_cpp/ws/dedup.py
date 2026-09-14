"""Deduplication trackers for Translation and TTS tasks."""

import time
from typing import Dict


class TranslationDedupState:
    """Prevents re-translating identical sentences within short time windows."""

    def __init__(self, window_sec: float = 3.0):
        self.window_sec = window_sec
        self._history: Dict[str, float] = {}

    def is_duplicate(self, text: str) -> bool:
        clean = text.strip()
        if not clean:
            return True
        now = time.time()
        # Clean up old records
        self._history = {k: v for k, v in self._history.items() if now - v < self.window_sec}
        if clean in self._history:
            return True
        self._history[clean] = now
        return False


class TTSDedupState:
    """Prevents generating duplicate audio for identical sentences within short windows."""

    def __init__(self, window_sec: float = 3.0):
        self.window_sec = window_sec
        self._history: Dict[str, float] = {}

    def is_duplicate(self, text: str) -> bool:
        clean = text.strip()
        if not clean:
            return True
        now = time.time()
        self._history = {k: v for k, v in self._history.items() if now - v < self.window_sec}
        if clean in self._history:
            return True
        self._history[clean] = now
        return False
