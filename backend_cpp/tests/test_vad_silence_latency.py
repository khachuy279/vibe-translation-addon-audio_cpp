"""Guards for the VAD trailing-silence settings.

Trailing silence is the single largest stage of perceived subtitle latency (measured: 480 ms of a
~1092 ms budget at the old default). Two things must not silently regress:

1. the default value, which was chosen deliberately as a latency/fragmentation tradeoff;
2. the SEMANTICS -- `silence_duration_ms` drives the end time, `hangover_ms` does not. I got this
   wrong while investigating (I assumed the two added up to 850 ms), so it is pinned here.
"""

from __future__ import annotations

import pytest

from backend_cpp.config import config
from backend_cpp.vad.vad_processor import VADProcessor

FRAME_MS_FSMN = 60


def _feed_then_silence(vad_engine: str, silence_ms: int, hangover_ms: int,
                       speech_sec: float = 1.5, silence_sec: float = 2.0) -> float:
    """Feed synthetic speech then silence; return the audio position of `on_speech_end`, in ms."""
    from backend_cpp.tests.perf_benchmark import generate_pcm_audio

    ended_at = {"ms": None}
    pos = {"ms": 0.0}

    def on_end():
        ended_at["ms"] = pos["ms"]

    vad = VADProcessor(
        sample_rate=16000,
        vad_engine=vad_engine,
        threshold=None,
        silence_duration_ms=silence_ms,
        hangover_ms=hangover_ms,
        pre_speech_buffer_ms=550,
        enabled=True,
        on_speech_start=lambda: None,
        on_speech_end=on_end,
    )

    pcm = generate_pcm_audio(speech_sec, speech_like=True) + b"\x00\x00" * int(
        16000 * silence_sec
    )
    chunk_bytes = 16000 * 30 // 1000 * 2  # 30 ms
    for offset in range(0, len(pcm), chunk_bytes):
        piece = pcm[offset:offset + chunk_bytes]
        pos["ms"] = (offset + len(piece)) / 2.0 / 16000 * 1000.0
        vad.feed_chunk(piece, capture_timestamp=pos["ms"] / 1000.0)
        if ended_at["ms"] is not None:
            return ended_at["ms"] - speech_sec * 1000.0

    if ended_at["ms"] is None:
        pytest.fail("VAD never reported speech end on the synthetic signal")
    return ended_at["ms"] - speech_sec * 1000.0


# ---------------------------------------------------------------------------
# The semantic trap: only silence_duration_ms gates the end
# ---------------------------------------------------------------------------
def test_hangover_does_not_delay_speech_end():
    """`hangover_ms` is a grace period for attaching silent frames, not added latency.

    If this test fails, someone started adding hangover to the end time -- which would mean the
    configured 400 ms silently doubles the trailing silence.
    """
    short = _feed_then_silence("fsmn-vad", silence_ms=150, hangover_ms=100)
    long = _feed_then_silence("fsmn-vad", silence_ms=150, hangover_ms=400)

    assert short == pytest.approx(long, abs=FRAME_MS_FSMN), (
        f"changing hangover_ms moved the end time: {short:.0f}ms vs {long:.0f}ms"
    )


def test_silence_duration_does_drive_speech_end():
    """Lowering silence_duration_ms must make the end arrive sooner."""
    at_450 = _feed_then_silence("fsmn-vad", silence_ms=450, hangover_ms=400)
    at_150 = _feed_then_silence("fsmn-vad", silence_ms=150, hangover_ms=400)

    assert at_450 > at_150, "the 450ms setting must wait longer than the 150ms one"
    # Quantized to the 60ms engine frame, so allow a frame of slack on each side.
    assert at_450 == pytest.approx(480, abs=2 * FRAME_MS_FSMN)
    assert at_150 == pytest.approx(180, abs=2 * FRAME_MS_FSMN)


# ---------------------------------------------------------------------------
# The chosen default
# ---------------------------------------------------------------------------
def test_default_silence_duration_is_the_deliberate_tradeoff():
    """150 ms was chosen on 2026-09-11 for ~300 ms lower latency.

    Safe range check: too high gives back the latency this change bought; too low starts cutting
    sentences at normal inter-word pauses.
    """
    assert config.vad.silence_duration_ms == 150, (
        "default changed; re-measure with scratch/compare_vad_silence.py and update the budget "
        "table in backend_cpp/config.py before accepting a new value"
    )
    assert 100 <= config.vad.silence_duration_ms <= 450
    assert config.vad.hangover_ms >= 0, "hangover is a grace period; keep it non-negative"


def test_session_vad_uses_the_configured_silence_duration():
    """The session must build its VADProcessor from config, not from the constructor default."""
    from backend_cpp.ws.session_state import SessionState
    from unittest.mock import MagicMock

    session = SessionState(MagicMock())
    session.init_components()

    assert session.vad_processor.silence_duration_ms == config.vad.silence_duration_ms
    assert session.vad_processor.hangover_ms == config.vad.hangover_ms
