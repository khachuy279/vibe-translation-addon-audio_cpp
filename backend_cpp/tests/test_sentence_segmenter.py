"""Unit tests for SentenceSegmenter in backend_cpp."""

import time
import pytest

from backend_cpp.asr.sentence_segmenter import SentenceSegmenter, count_content_tokens


def test_count_content_tokens():
    assert count_content_tokens("Hello world") == 2
    assert count_content_tokens("Guess who? A cover for me.") == 6
    assert count_content_tokens("") == 0
    assert count_content_tokens("   ") == 0


def test_count_content_tokens_cjk():
    # Chinese characters
    assert count_content_tokens("你好世界") == 4
    # Japanese characters
    assert count_content_tokens("こんにちは世界") == 7
    # Mixed English and CJK
    assert count_content_tokens("Hello 世界") == 3


def test_normalize_for_comparison():
    assert SentenceSegmenter.normalize_for_comparison("What's up?") == "whats up"
    assert SentenceSegmenter.normalize_for_comparison("Hello, world!") == "hello world"
    assert SentenceSegmenter.normalize_for_comparison("  Xin chào ... ") == "xin chào"


def test_remove_prefix_overlap():
    p = "What's up?"
    t = "What's up? Can you show us"
    assert SentenceSegmenter.remove_prefix_overlap(p, t) == "Can you show us"


def test_check_stability():
    seg = SentenceSegmenter(
        split_on_stability=True,
        stability_duration_sec=0.1,  # short duration for testing
        stability_threshold_polls=2,
    )

    # First poll with text
    assert seg.check_stability("Hello") is False

    # Text changes
    assert seg.check_stability("Hello world") is False

    # Text stable poll 1
    time.sleep(0.06)
    assert seg.check_stability("Hello world") is False

    # Text stable poll 2 after duration elapsed
    time.sleep(0.06)
    assert seg.check_stability("Hello world") is True


def test_stability_disabled():
    seg = SentenceSegmenter(split_on_stability=False)
    assert seg.check_stability("Hello world") is False


def test_check_stability_with_trailing_punctuation():
    seg = SentenceSegmenter(
        split_on_stability=True,
        stability_duration_sec=0.1,
        stability_threshold_polls=2,
    )

    # Initial text without punctuation
    assert seg.check_stability("Hello world") is False

    # Second poll with trailing punctuation - should still accumulate stability
    time.sleep(0.06)
    assert seg.check_stability("Hello world.") is False

    # Third poll after elapsed duration - should be recognized as stable
    time.sleep(0.06)
    assert seg.check_stability("Hello world.") is True

