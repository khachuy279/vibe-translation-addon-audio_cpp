"""Module Lọc Trùng Lặp 3 Lớp (Commit Deduplication & Anti-Repeat).

Bảo vệ hệ thống khỏi:
1. Trùng lặp chính xác (Exact Hash / Normalized Match).
2. Trùng lặp tiền tố / tập con (Substring & Prefix Overlap >= 85%).
3. Trùng lặp từ vựng mật độ cao (Jaccard Word Overlap >= 80%).
4. Cửa sổ thời gian bảo vệ (Timestamp Window Guard).
"""

from collections import deque
import re
import time
from typing import Deque, FrozenSet, Optional, Tuple

_RE_PUNCT = re.compile(r'[.,!?;:，。！？；：…\'"“”‘’\-—()\[\]{}<>]')
_RE_SPACES = re.compile(r'\s+')


def normalize_for_dedup(text: str) -> str:
    """Chuẩn hóa văn bản để so khớp dedup: Bỏ dấu câu, chuyển chữ thường, gộp khoảng trắng."""
    if not text:
        return ""
    clean = _RE_PUNCT.sub(" ", text).lower()
    return _RE_SPACES.sub(" ", clean).strip()


class CommitDeduplicator:
    """Bộ lọc chống trùng lặp câu commit đa tầng."""

    def __init__(
        self,
        cache_ttl_sec: float = 30.0,
        window_sec: float = 3.5,
        substring_ratio: float = 0.85,
        jaccard_threshold: float = 0.80,
    ):
        self.cache_ttl_sec = cache_ttl_sec
        self.window_sec = window_sec
        self.substring_ratio = substring_ratio
        self.jaccard_threshold = jaccard_threshold
        # Lưu trữ: (timestamp, raw_text, norm_text, word_set)
        self._history: Deque[Tuple[float, str, str, FrozenSet[str]]] = deque()

    def _prune_expired(self, current_time: float) -> None:
        """Loại bỏ các bản ghi đã quá thời hạn cache_ttl_sec."""
        while self._history and (current_time - self._history[0][0] > self.cache_ttl_sec):
            self._history.popleft()

    def is_duplicate(self, text: str, timestamp: Optional[float] = None) -> bool:
        """Kiểm tra xem text có bị trùng lặp với các câu vừa commit không.
        
        Returns:
            True nếu bị trùng lặp (cần drop), False nếu là câu mới hợp lệ.
        """
        if not text or not text.strip():
            return True

        norm_curr = normalize_for_dedup(text)
        if not norm_curr:
            return True

        now = timestamp if timestamp is not None else time.time()
        self._prune_expired(now)

        curr_words = frozenset(norm_curr.split())
        len_c = len(norm_curr)

        for ts, prev_raw, prev_norm, prev_words in reversed(self._history):
            # Nếu vượt quá cửa sổ so khớp gần nhất -> Dừng kiểm tra
            if now - ts > self.window_sec:
                break

            # Lớp 1: So khớp chính xác 100% (Exact Match)
            if norm_curr == prev_norm:
                return True

            # Lớp 2: So khớp tập con / chuỗi con (Substring Containment)
            len_p = len(prev_norm)
            if len_c > 4 and len_p > 4:
                if norm_curr in prev_norm or prev_norm in norm_curr:
                    ratio = min(len_c, len_p) / max(len_c, len_p)
                    if ratio >= self.substring_ratio:
                        return True

            # Lớp 3: Độ tương đồng từ vựng Jaccard Overlap
            if curr_words and prev_words:
                intersection = len(curr_words & prev_words)
                union = len(curr_words | prev_words)
                if union > 0 and (intersection / union) >= self.jaccard_threshold:
                    return True

        return False

    def record_commit(self, raw_text: str, timestamp: Optional[float] = None) -> None:
        """Ghi nhận một câu đã commit thành công vào bộ đệm dedup."""
        now = timestamp if timestamp is not None else time.time()
        norm_text = normalize_for_dedup(raw_text)
        words = frozenset(norm_text.split()) if norm_text else frozenset()
        self._history.append((now, raw_text, norm_text, words))

    def clear(self) -> None:
        """Xóa sạch lịch sử dedup."""
        self._history.clear()
