"""ContextManager: Quản lý bộ nhớ ngữ cảnh dịch (Sliding Window FIFO)."""

from collections import deque
from typing import Deque, List, Tuple


class ContextManager:
    """Quản lý cửa sổ trượt N câu dịch gần nhất để bổ trợ ngữ cảnh."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._history: Deque[Tuple[str, str]] = deque(maxlen=window_size)

    def add(self, source_text: str, translated_text: str) -> None:
        """Thêm 1 cặp câu (gốc, dịch) vào lịch sử."""
        if source_text and translated_text:
            self._history.append((source_text.strip(), translated_text.strip()))

    def get_context_str(self) -> str:
        """Tạo chuỗi ngữ cảnh kết hợp các câu gần nhất."""
        if not self._history:
            return ""
        lines = []
        for src, tgt in self._history:
            lines.append(f"{src} -> {tgt}")
        return " | ".join(lines)

    def clear(self) -> None:
        """Xóa sạch ngữ cảnh."""
        self._history.clear()


# Alias tương thích
TranslationContextTracker = ContextManager

