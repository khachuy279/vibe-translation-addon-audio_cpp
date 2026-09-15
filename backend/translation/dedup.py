"""Bộ lọc chống dịch trùng lặp cho luồng Translation Worker."""

from collections import deque
import time
from typing import Deque, Optional, Tuple


class TranslationDedupState:
    """Quản lý các câu vừa dịch để ngăn chặn dịch lặp lại nhiều lần."""

    def __init__(self, cache_ttl_sec: float = 10.0, max_items: int = 20):
        self.cache_ttl_sec = cache_ttl_sec
        self._history: Deque[Tuple[float, str]] = deque(maxlen=max_items)

    def is_duplicate(self, text: str) -> bool:
        if not text or not text.strip():
            return True

        clean = text.strip().lower()
        now = time.time()

        # Xóa các mục hết hạn
        while self._history and (now - self._history[0][0] > self.cache_ttl_sec):
            self._history.popleft()

        for ts, prev in reversed(self._history):
            if clean == prev:
                return True

        self._history.append((now, clean))
        return False

    def clear(self) -> None:
        self._history.clear()


# Alias tương thích
TranslationDeduplicator = TranslationDedupState

