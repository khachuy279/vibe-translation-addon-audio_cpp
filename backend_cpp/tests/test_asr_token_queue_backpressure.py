"""Regression tests for ASR outbound token-queue backpressure (audit finding P0-02).

The audit found ``TranscribeEngine._get_queue()`` creating an *unbounded*
``asyncio.Queue``. If the WebSocket sender could not keep up with the preview poller,
the queue grew without limit while retaining every preview's text and metadata.

Policy under test
-----------------
* The queue is bounded by ``config.asr.token_queue_maxsize``.
* FINAL messages are lossless: they are never dropped while any preview can be evicted.
* PREVIEW messages are latest-wins: at most one queued preview per utterance.
"""

import asyncio

import pytest

from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.config import config


def _preview(utterance_id: str, text: str) -> dict:
    return {
        "type": "utterance_update",
        "utterance_id": utterance_id,
        "text": text,
        "is_final": False,
    }


def _final(utterance_id: str, text: str) -> dict:
    return {
        "type": "utterance_update",
        "utterance_id": utterance_id,
        "text": text,
        "is_final": True,
    }


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(config.asr, "token_queue_maxsize", 4)
    eng = TranscribeEngine()
    eng._running = True
    eng._token_queue = asyncio.Queue(maxsize=4)
    yield eng
    eng._running = False


# ---------------------------------------------------------------------------
# Boundedness
# ---------------------------------------------------------------------------
def test_queue_is_bounded(engine):
    """The queue must not be unbounded any more."""
    assert engine._token_queue.maxsize == 4
    assert engine._get_queue() is engine._token_queue


def test_queue_never_exceeds_maxsize(engine):
    for i in range(50):
        engine._enqueue_token_message(_final(f"u{i}", f"final-{i}"))
        engine._enqueue_token_message(_preview(f"u{i}", f"preview-{i}"))

    assert engine._token_queue.qsize() <= 4


# ---------------------------------------------------------------------------
# Preview: latest-wins
# ---------------------------------------------------------------------------
def test_preview_is_latest_wins(engine):
    """Repeated previews for one utterance must collapse to a single newest message."""
    for i in range(10):
        engine._enqueue_token_message(_preview("u1", f"text-{i}"))

    assert engine._token_queue.qsize() == 1
    assert engine._token_queue._queue[0]["text"] == "text-9"


def test_distinct_utterances_keep_one_preview_each(engine):
    engine._enqueue_token_message(_preview("u1", "a"))
    engine._enqueue_token_message(_preview("u2", "b"))

    assert engine._token_queue.qsize() == 2
    texts = {m["utterance_id"]: m["text"] for m in engine._token_queue._queue}
    assert texts == {"u1": "a", "u2": "b"}


def test_preview_is_dropped_when_queue_full_of_finals(engine):
    """Under pressure an optional preview is sacrificed; finals survive."""
    for i in range(4):
        engine._enqueue_token_message(_final(f"u{i}", f"final-{i}"))

    engine._enqueue_token_message(_preview("uP", "optional preview"))

    assert engine._token_queue.qsize() == 4
    assert all(m.get("is_final") for m in engine._token_queue._queue)


# ---------------------------------------------------------------------------
# Final: lossless
# ---------------------------------------------------------------------------
def test_final_is_never_dropped_when_previews_fill_queue(engine):
    for i in range(4):
        engine._enqueue_token_message(_preview(f"u{i}", f"preview-{i}"))
    assert engine._token_queue.qsize() == 4

    engine._enqueue_token_message(_final("uF", "FINAL"))

    items = list(engine._token_queue._queue)
    finals = [m for m in items if m.get("is_final")]
    assert len(finals) == 1
    assert finals[0]["text"] == "FINAL"
    assert engine._token_queue.qsize() == 4  # still bounded


def test_final_survives_repeated_preview_flood(engine):
    """A long sentence flooding previews must not starve the commit message."""
    for i in range(200):
        engine._enqueue_token_message(_preview("uLong", f"partial-{i}"))
    engine._enqueue_token_message(_final("uLong", "the committed sentence"))

    finals = [m for m in engine._token_queue._queue if m.get("is_final")]
    assert len(finals) == 1
    assert finals[0]["text"] == "the committed sentence"


# ---------------------------------------------------------------------------
# Shutdown safety
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cleanup_with_full_queue_unblocks_consumer(monkeypatch):
    """A bounded queue must not make teardown raise when it is completely full."""
    monkeypatch.setattr(config.asr, "token_queue_maxsize", 2)

    engine = TranscribeEngine()
    engine._running = True

    consumed = []

    async def consumer():
        async for token in engine.stream_tokens():
            consumed.append(token)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    for i in range(20):
        engine._push_message(_preview(f"u{i}", f"preview-{i}"))
        engine._push_message(_final(f"u{i}", f"final-{i}"))
    await asyncio.sleep(0.05)

    # Must not raise even though the queue is full.
    await engine.cleanup()
    await asyncio.wait_for(task, timeout=2.0)

    assert task.done()


def test_clear_token_queue_keeps_task_accounting_balanced(engine):
    """Draining must not leave unbalanced task counters behind."""
    for i in range(4):
        engine._enqueue_token_message(_final(f"u{i}", f"final-{i}"))

    engine._clear_token_queue()

    assert engine._token_queue.qsize() == 0
    assert engine._token_queue._unfinished_tasks == 0
