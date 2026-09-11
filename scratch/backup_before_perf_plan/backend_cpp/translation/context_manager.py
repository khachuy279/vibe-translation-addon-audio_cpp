"""Context Manager for keeping recent conversation history for translation."""

from collections import deque
import threading
from typing import List


class ContextManager:
    """Maintains a rolling window of recent sentences for translation context.

    Thread-safe implementation allowing concurrent add, clear, and get_context_str.
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add(self, source_text: str, translated_text: str) -> None:
        """Add a completed sentence pair to context."""
        if source_text and source_text.strip():
            with self._lock:
                self._history.append((source_text.strip(), translated_text.strip()))

    def get_context_str(self) -> str:
        """Format recent context as background text for the translation model."""
        with self._lock:
            if not self._history:
                return ""
            items = []
            for src, tgt in self._history:
                if tgt:
                    items.append(f"{src} -> {tgt}")
                else:
                    items.append(src)
            return "\n".join(items)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()

