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

    # ------------------------------------------------------------------ P2.7
    def trim_boundary_overlap(self, text: str) -> str:
        """Cắt phần ĐẦU câu bị lặp lại do chồng lấn ở ranh giới cắt (P2.7).

        Khi CommitManager cắt câu giữa lúc đang nói, đoạn audio của câu kế tiếp được
        lùi lại `boundary_overlap_ms` để không mất từ ở ranh giới. Hệ quả là vài từ đầu
        của câu mới trùng với đuôi câu trước. Hàm này TRIM phần trùng đó thay vì
        DROP cả câu (khác hẳn `is_duplicate`, vốn dùng để lọc rác hallucination).

        Hỗ trợ cả chữ Latin (so khớp theo từ) và CJK (so khớp theo ký tự).
        """
        if not text or not self._history:
            return text

        prev_raw = self._history[-1][1]
        if not prev_raw:
            return text

        prev_tokens = prev_raw.split()
        cur_tokens = text.split()
        if len(prev_tokens) > 1 and len(cur_tokens) > 1:
            return self._trim_by_tokens(text, prev_tokens, cur_tokens)
        return self._trim_by_chars(text, prev_raw)

    @staticmethod
    def _norm_token(tok: str) -> str:
        return normalize_for_dedup(tok)

    def _trim_by_tokens(self, text: str, prev_tokens, cur_tokens) -> str:
        prev_norm = [self._norm_token(t) for t in prev_tokens]
        cur_norm = [self._norm_token(t) for t in cur_tokens]
        max_k = min(len(prev_norm), len(cur_norm))
        best_k = 0
        for k in range(max_k, 0, -1):
            if prev_norm[-k:] == cur_norm[:k] and any(cur_norm[:k]):
                best_k = k
                break
        if best_k <= 0:
            return text
        remainder = " ".join(cur_tokens[best_k:]).strip()
        return remainder if remainder else text

    def _trim_by_chars(self, text: str, prev_raw: str) -> str:
        prev_norm = normalize_for_dedup(prev_raw).replace(" ", "")
        cur_norm = normalize_for_dedup(text).replace(" ", "")
        if not prev_norm or not cur_norm:
            return text
        max_k = min(len(prev_norm), len(cur_norm))
        best_k = 0
        for k in range(max_k, 0, -1):
            if prev_norm[-k:] == cur_norm[:k]:
                best_k = k
                break
        if best_k <= 0:
            return text
        # CJK: cắt theo số ký tự (bỏ qua dấu câu/khoảng trắng khi đếm)
        stripped = text.lstrip()
        count = 0
        idx = 0
        while idx < len(stripped) and count < best_k:
            if normalize_for_dedup(stripped[idx]) and not stripped[idx].isspace():
                count += 1
            idx += 1
        remainder = stripped[idx:].strip()
        return remainder if remainder else text

    def clear(self) -> None:
        """Xóa sạch lịch sử dedup."""
        self._history.clear()
