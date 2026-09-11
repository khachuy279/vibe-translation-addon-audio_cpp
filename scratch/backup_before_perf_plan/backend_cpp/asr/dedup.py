"""Deduplication and repeat filtering for ASR committed utterances."""

from collections import deque
import time
from typing import Deque, Optional, Tuple, FrozenSet

from backend_cpp.asr.constants import (
    DEDUP_JACCARD_THRESHOLD,
    DEDUP_SUBSTRING_RATIO,
    DEDUP_WINDOW_SEC,
    RECENT_COMMITS_CACHE_SEC,
)
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter


class CommitDeduplicator:
    """Tracks recently committed utterances to prevent duplicate commits."""

    def __init__(self, cache_ttl_sec: float = RECENT_COMMITS_CACHE_SEC):
        self.cache_ttl_sec = cache_ttl_sec
        self._recent_commits: Deque[Tuple[float, str, str, FrozenSet[str]]] = deque()

    def prune_expired(self, now: Optional[float] = None) -> None:
        """Prune commits older than cache_ttl_sec from the front in O(k) time."""
        current_time = now if now is not None else time.time()
        while self._recent_commits and (current_time - self._recent_commits[0][0] > self.cache_ttl_sec):
            self._recent_commits.popleft()

    def is_duplicate(self, text: str, window_sec: float = DEDUP_WINDOW_SEC) -> bool:
        """Check if text is an exact or near duplicate of a recently committed sentence."""
        if not text:
            return True

        norm_curr = SentenceSegmenter.normalize_for_comparison(text)
        if not norm_curr:
            return True

        now = time.time()
        self.prune_expired(now)

        curr_words = set(norm_curr.split())

        for entry in reversed(self._recent_commits):
            ts = entry[0]
            if now - ts > window_sec:
                # Since commits are ordered chronologically, all earlier commits also exceed window_sec
                break

            prev_norm = entry[2]

            # 1. Exact match
            if norm_curr == prev_norm:
                return True

            # 2. Substring containment if lengths are comparable
            len_c = len(norm_curr)
            len_p = len(prev_norm)
            if len_c > 3 and len_p > 3:
                if norm_curr in prev_norm or prev_norm in norm_curr:
                    ratio = min(len_c, len_p) / max(len_c, len_p)
                    if ratio >= DEDUP_SUBSTRING_RATIO:
                        return True

            # 3. High word token overlap (Word containment / overlap ratio >= 0.80)
            prev_words = entry[3] if len(entry) > 3 else set(prev_norm.split())
            if curr_words and prev_words:
                overlap = len(curr_words & prev_words) / max(len(curr_words), len(prev_words))
                if overlap >= DEDUP_JACCARD_THRESHOLD:
                    return True

        return False

    def record_commit(self, raw_text: str, norm_text: str, timestamp: Optional[float] = None) -> None:
        """Register a committed sentence into the deduplication cache."""
        ts = timestamp if timestamp is not None else time.time()
        words = frozenset(norm_text.split()) if norm_text else frozenset()
        self._recent_commits.append((ts, raw_text, norm_text, words))

    def clear(self) -> None:
        """Clear all cached commits."""
        self._recent_commits.clear()

    @property
    def recent_commits(self) -> Deque[Tuple[float, str, str]]:
        return self._recent_commits
