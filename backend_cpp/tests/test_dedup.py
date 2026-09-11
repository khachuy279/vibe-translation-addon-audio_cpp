"""Unit test for ASR deduplication and prefix stripping."""

from backend_cpp.asr.dedup import CommitDeduplicator
from backend_cpp.asr.sentence_segmenter import SentenceSegmenter


def test_remove_prefix_overlap():
    p = "What's up?"
    t = "What's up? Can you show us the dating app?"
    rem = SentenceSegmenter.remove_prefix_overlap(p, t)
    assert rem == "Can you show us the dating app?"

    # Exact match leaves empty
    assert SentenceSegmenter.remove_prefix_overlap("Hello world", "Hello world") == ""


def test_duplicate_commit_filtering():
    dedup = CommitDeduplicator()

    # First occurrence passes
    assert not dedup.is_duplicate("What's up?")

    # Register commit
    dedup.record_commit("What's up?", "whats up")

    # Duplicates are blocked
    assert dedup.is_duplicate("What's up?")
    assert dedup.is_duplicate("what's up")
    assert dedup.is_duplicate("What's up!")

    # Substring duplicate ('Probably don't need to mention' vs 'I probably don't need to mention')
    dedup.record_commit("I probably don't need to mention.", "i probably dont need to mention")
    assert dedup.is_duplicate("Probably don't need to mention.")

    # Distinct sentence passes
    assert not dedup.is_duplicate("Can you show us the dating app?")
