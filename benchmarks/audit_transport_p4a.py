"""Stage P4-A Transport, Jitter, Clock Drift & Reset Protocol Audit Test Suite.

Automated verification covering:
1. Format A frame unpacking and backward compatibility.
2. Pre-VAD rejection of duplicate and out-of-order chunks.
3. Dropped chunk sequence gap detection.
4. Runtime-driven deterministic jitter calculation (| arrival_delta - expected_cadence |).
5. Pause-resilient active clock drift measurement.
6. Generation barrier stream_reset queue drainage and task accounting.
7. Telemetry schema deliverable completeness.
8. In-flight ASR commit dropped when stream_reset occurs before emission.
9. Non-monotonic/duplicate epoch reset rejection (only epoch > current_epoch allowed).
10. Paired Client Render Acknowledgement and latency percentiles.
"""

import asyncio
import os
from pathlib import Path
import struct
import sys
import time
from unittest.mock import patch, MagicMock

import numpy as np
import json
import pytest

from backend_cpp.ws.frame_protocol import parse_audio_frame, ParsedFrame
from backend_cpp.ws.session_state import TransportTelemetry, SessionState
from backend_cpp.ws.serializers import make_utterance_update_msg, make_translation_msg


def test_format_a_parsing_and_backward_compatibility():
    """Verify Format A unpacking exposes media timings and is backward compatible."""
    print("▶️ Running Test 1: Format A Frame Protocol Unpacking...")
    pcm_bytes = b"\x00\x01" * 1024  # 2048 bytes
    ts = 12345.678
    chunk_idx = 42
    start_media = 10.5
    end_media = 10.564
    epoch = 3
    rate = 1.0

    # Format A: 4-byte uint32 header length + JSON header + 16-bit PCM data
    header_dict = {
        "type": "audio_chunk",
        "captureTimestamp": ts,
        "chunkIndex": chunk_idx,
        "chunkStartMediaTime": start_media,
        "chunkEndMediaTime": end_media,
        "epoch": epoch,
        "playbackRate": rate,
        "chunkDurationMs": 64.0,
    }
    header_json = json.dumps(header_dict).encode("utf-8")
    packet = struct.pack("<I", len(header_json)) + header_json + pcm_bytes

    frame = parse_audio_frame(packet)

    assert isinstance(frame, tuple), "ParsedFrame must be a tuple for backward compatibility"
    unpacked_pcm, unpacked_ts, unpacked_idx = frame
    assert unpacked_pcm == pcm_bytes
    assert abs(unpacked_ts - ts) < 1e-3
    assert unpacked_idx == chunk_idx

    assert abs(frame.media_start_time - start_media) < 1e-4
    assert abs(frame.media_end_time - end_media) < 1e-4
    assert frame.epoch == epoch
    assert abs(frame.playback_rate - 1.0) < 1e-3
    assert frame.chunk_duration_ms == 64.0

    # Test Format B backward compatibility (8-byte float64 capture_ts + PCM)
    format_b_packet = struct.pack("<d", ts) + pcm_bytes
    frame_b = parse_audio_frame(format_b_packet)
    assert isinstance(frame_b, tuple)
    b_pcm, b_ts, b_idx = frame_b
    assert b_pcm == pcm_bytes
    assert abs(b_ts - ts) < 1e-3
    assert b_idx is None

    print("  ✅ PASS: Format A and Format B parsed successfully with full backward compatibility")



def test_pre_vad_duplicate_and_out_of_order_rejection():
    """Verify duplicate and out-of-order chunks are rejected before reaching VAD."""
    print("▶️ Running Test 2: Pre-VAD Filtering of Corrupted Frames...")
    telemetry = TransportTelemetry()
    pcm_len = 2048

    # Chunk 1: accepted
    assert telemetry.record_chunk(chunk_idx=1, pcm_len=pcm_len) is True
    assert telemetry.accepted_chunks == 1

    # Chunk 1 again: rejected as duplicate
    assert telemetry.record_chunk(chunk_idx=1, pcm_len=pcm_len) is False
    assert telemetry.duplicate_chunks == 1
    assert telemetry.accepted_chunks == 1

    # Chunk 3: accepted (chunk 2 dropped)
    assert telemetry.record_chunk(chunk_idx=3, pcm_len=pcm_len) is True
    assert telemetry.dropped_chunks == 1
    assert telemetry.accepted_chunks == 2

    # Chunk 2 (arriving late): rejected as out-of-order
    assert telemetry.record_chunk(chunk_idx=2, pcm_len=pcm_len) is False
    assert telemetry.out_of_order_chunks == 1
    assert telemetry.accepted_chunks == 2
    print("  ✅ PASS: Duplicates and out-of-order frames rejected cleanly")


def test_dropped_chunk_gap_detection():
    """Verify missing sequence gaps increment dropped_chunks."""
    print("▶️ Running Test 3: Dropped Chunk Gap Detection...")
    telemetry = TransportTelemetry()
    pcm_len = 2048

    telemetry.record_chunk(chunk_idx=1, pcm_len=pcm_len)
    telemetry.record_chunk(chunk_idx=2, pcm_len=pcm_len)
    # Gap: skip 3 and 4 -> jump to 5
    telemetry.record_chunk(chunk_idx=5, pcm_len=pcm_len)

    assert telemetry.dropped_chunks == 2, f"Expected 2 dropped chunks (3 and 4), got {telemetry.dropped_chunks}"
    print("  ✅ PASS: Gap detection accurately identified 2 dropped chunks")


def test_runtime_jitter_calculation():
    """Verify jitter calculation matches | inter_arrival - expected_cadence | using real record_chunk calls."""
    print("▶️ Running Test 4: Runtime-Driven Jitter Calculation...")
    telemetry = TransportTelemetry()
    cadence = 64.0  # ms
    pcm_len = 2048

    # Sequence of arrival timestamps:
    # 0ms (init), +64ms (diff=64ms, jitter=0ms), +84ms (diff=84ms, jitter=20ms), +54ms (diff=54ms, jitter=10ms)
    t0 = 100.0
    clock_steps = [t0, t0 + 0.064, t0 + 0.064 + 0.084, t0 + 0.064 + 0.084 + 0.054]

    with patch("time.perf_counter", side_effect=clock_steps):
        assert telemetry.record_chunk(chunk_idx=1, pcm_len=pcm_len, expected_cadence_ms=cadence) is True
        assert telemetry.record_chunk(chunk_idx=2, pcm_len=pcm_len, expected_cadence_ms=cadence) is True
        assert telemetry.record_chunk(chunk_idx=3, pcm_len=pcm_len, expected_cadence_ms=cadence) is True
        assert telemetry.record_chunk(chunk_idx=4, pcm_len=pcm_len, expected_cadence_ms=cadence) is True

    jitters = list(telemetry.jitter_history)
    assert len(jitters) == 3, f"Expected 3 jitter measurements, got {len(jitters)}"
    assert round(jitters[0], 1) == 0.0, f"Expected 0.0ms jitter, got {jitters[0]}"
    assert round(jitters[1], 1) == 20.0, f"Expected 20.0ms jitter, got {jitters[1]}"
    assert round(jitters[2], 1) == 10.0, f"Expected 10.0ms jitter, got {jitters[2]}"

    summary = telemetry.get_summary()
    assert summary["p50_jitter_ms"] == 10.0, f"P50 jitter should be 10.0, got {summary['p50_jitter_ms']}"
    print("  ✅ PASS: Runtime jitter calculation validated via real record_chunk calls")


def test_pause_resilient_clock_drift():
    """Verify clock drift measures active playback windows and ignores user video pauses."""
    print("▶️ Running Test 5: Pause-Resilient Clock Drift Validation...")
    telemetry = TransportTelemetry()
    # 5 seconds of 16kHz 16-bit PCM = 160,000 bytes
    pcm_5s = 160000

    # Simulate Window 1:
    # Starts at 100.0, feeds 5.0s PCM, paused at 105.0
    with patch("time.perf_counter", return_value=100.0):
        telemetry.start_window()
        telemetry.record_chunk(chunk_idx=1, pcm_len=pcm_5s)

    with patch("time.perf_counter", return_value=105.0):
        telemetry.pause_window()

    # User pauses for 60 seconds (simulated clock at 165.0)
    # Window 2 resumes at 165.0, feeds another 5.0s PCM, ends at 170.0
    with patch("time.perf_counter", return_value=165.0):
        telemetry.start_window()
        telemetry.record_chunk(chunk_idx=2, pcm_len=pcm_5s)

    with patch("time.perf_counter", return_value=170.0):
        telemetry.pause_window()

    summary = telemetry.get_summary()
    # Total PCM = 320,000 bytes / 32,000 bytes/sec = 10.0s
    assert summary["accepted_pcm_duration_sec"] == 10.0, f"Expected 10.0s PCM, got {summary['accepted_pcm_duration_sec']}"
    # Total Active Wall = (105.0 - 100.0) + (170.0 - 165.0) = 10.0s (60s pause completely excluded)
    assert summary["active_wall_duration_sec"] == 10.0, f"Expected 10.0s active wall, got {summary['active_wall_duration_sec']}"
    assert summary["clock_drift_pct"] == 0.0, f"Expected 0.0% drift, got {summary['clock_drift_pct']}%"
    print("  ✅ PASS: Clock drift isolates active playback windows without pause distortion")


class MockWebSocket:
    def __init__(self):
        self.sent_messages = []

    async def send_json(self, payload):
        self.sent_messages.append(payload)
        return True

    async def send_text(self, text):
        self.sent_messages.append(text)
        return True


def test_generation_barrier_stream_reset():
    """Verify stream_reset increments epoch, clears buffers, and drains queues with task_done()."""
    print("▶️ Running Test 6: Generation Barrier stream_reset Execution...")
    asyncio.run(_async_test_generation_barrier_stream_reset())


async def _async_test_generation_barrier_stream_reset():
    mock_ws = MockWebSocket()
    session = SessionState(mock_ws)
    session.init_components()

    # Initial epoch is 0
    assert session.current_epoch == 0

    # Put items into translation and TTS queues
    await session.translation_queue.put({"epoch": 0, "text": "Stale subtitle from epoch 0"})
    await session.tts_queue.put({"epoch": 0, "text": "Stale audio from epoch 0"})
    assert session.translation_queue.qsize() == 1
    assert session.tts_queue.qsize() == 1

    # Trigger stream_reset with epoch 1
    session.handle_stream_reset(epoch=1, reason="seek", media_time=45.2)

    assert session.current_epoch == 1, "Epoch must advance to 1"
    assert session.transport_telemetry.resets_count == 1, "Reset count must increment"
    assert session.translation_queue.empty(), "Translation queue must be drained on reset"
    assert session.tts_queue.empty(), "TTS queue must be drained on reset"

    # Verify task accounting: joining empty queues must not hang because task_done was called
    await session.translation_queue.join()
    await session.tts_queue.join()

    session.close()
    print("  ✅ PASS: Generation barrier advanced epoch and drained queues with task_done()")


def test_telemetry_schema_deliverable():
    """Verify get_summary() outputs all schema fields required for P4-A gate evaluation."""
    print("▶️ Running Test 7: Telemetry Schema Deliverable...")
    telemetry = TransportTelemetry()
    telemetry.record_chunk(chunk_idx=1, pcm_len=2048)
    summary = telemetry.get_summary()

    required_keys = [
        "total_chunks_received",
        "accepted_chunks",
        "dropped_chunks",
        "duplicate_chunks",
        "out_of_order_chunks",
        "p50_jitter_ms",
        "p95_jitter_ms",
        "mean_jitter_ms",
        "accepted_pcm_duration_sec",
        "active_wall_duration_sec",
        "clock_drift_pct",
        "resets_count",
        "paired_render_acks_count",
        "speech_offset_to_visible_lag",
        "client_render_cost_ms",
        "commit_to_ack_delay_ms",
    ]

    for k in required_keys:
        assert k in summary, f"Missing required telemetry key: {k}"

    print("  ✅ PASS: Telemetry summary contains all required schema fields")


def test_inflight_commit_dropped_on_reset():
    """Verify an in-flight ASR commit scheduled before reset is dropped at the boundary."""
    print("▶️ Running Test 8: In-flight ASR Commit Dropped on Reset...")
    asyncio.run(_async_test_inflight_commit_dropped())


async def _async_test_inflight_commit_dropped():
    from backend_cpp.ws.ws_handler import _stream_asr_tokens

    mock_ws = MockWebSocket()
    session = SessionState(mock_ws)
    session.init_components()

    # Session starts at epoch 0
    assert session.current_epoch == 0

    # Simulate an ASR engine commit that was scheduled during epoch 0
    # The message is tagged with epoch 0
    stale_commit_msg = {
        "type": "utterance_update",
        "utterance_id": "utt-old-123",
        "text": "Stale speech from previous epoch",
        "ui_text": "Stale speech from previous epoch",
        "stable_text": "Stale speech from previous epoch",
        "unstable_text": "",
        "is_final": True,
        "language": "en",
        "model": "qwen3-asr-1.7b",
        "commit_method": "VAD_SILENCE",
        "epoch": 0,
        "media_start_time": 10.0,
        "media_end_time": 12.5,
        "asr_commit_wall_time": time.time(),
    }

    # 1. Direct Engine Level Test:
    # Verify on_speech_end snapshots epoch at schedule time, so mutating engine._current_epoch does not leak
    engine = session.asr_engine
    engine._current_epoch = 0
    engine._current_media_start_time = 10.0
    engine._current_media_end_time = 12.5
    engine._audio_buffer_mgr.feed_bytes(b"\x00" * 3200)

    captured_commit_kwargs = {}
    async def mock_commit_async(*args, **kwargs):
        captured_commit_kwargs.update(kwargs)

    engine._commit_async = mock_commit_async
    engine._loop = asyncio.get_running_loop()

    # on_speech_end is scheduled at epoch 0
    engine.on_speech_end(reason="VAD_SILENCE")

    # In-flight reset advances engine epoch to 1
    engine.reset_stream(epoch=1)
    await asyncio.sleep(0.02)

    assert captured_commit_kwargs.get("epoch") == 0, (
        f"In-flight commit must retain snapshot epoch 0, got {captured_commit_kwargs.get('epoch')}"
    )

    # 2. WebSocket Worker Generation Barrier Test:
    # Now a stream_reset arrives, advancing session epoch to 1
    session.handle_stream_reset(epoch=1, reason="seek", media_time=5.0)
    assert session.current_epoch == 1

    # Mock engine stream_tokens to yield the stale message
    async def mock_stream_tokens():
        yield stale_commit_msg

    session.asr_engine.stream_tokens = mock_stream_tokens

    # Run the worker briefly to consume the token stream
    task = asyncio.create_task(_stream_asr_tokens(session))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # The stale commit message with epoch=0 MUST NOT have been queued to translation_queue or sent
    assert session.translation_queue.empty(), "Stale commit must NOT enter translation queue"
    assert len(mock_ws.sent_messages) == 0, "Stale commit must NOT be sent to client WebSocket"

    session.close()
    print("  ✅ PASS: In-flight commit with old epoch was strictly dropped by generation barrier")


def test_non_monotonic_epoch_reset_rejected():
    """Verify reset with an older or duplicate epoch is rejected, protecting epoch monotonicity."""
    print("▶️ Running Test 9: Non-Monotonic Epoch Reset Rejection...")
    mock_ws = MockWebSocket()
    session = SessionState(mock_ws)
    session.init_components()

    # Set epoch to 2
    session.current_epoch = 2

    # Attempt reset with duplicate epoch (2)
    session.handle_stream_reset(epoch=2, reason="dup")
    assert session.current_epoch == 2, "Duplicate epoch must not advance epoch"
    assert session.transport_telemetry.resets_count == 0, "Reset count must not increment for duplicate"

    # Attempt reset with older epoch (1)
    session.handle_stream_reset(epoch=1, reason="old")
    assert session.current_epoch == 2, "Older epoch must not decrease epoch"
    assert session.transport_telemetry.resets_count == 0, "Reset count must not increment for older epoch"

    # Valid strictly greater epoch (3)
    session.handle_stream_reset(epoch=3, reason="valid")
    assert session.current_epoch == 3, "Newer epoch must advance epoch"
    assert session.transport_telemetry.resets_count == 1, "Reset count must increment for valid reset"

    session.close()
    print("  ✅ PASS: Non-monotonic and duplicate reset epochs rejected cleanly")


def test_client_render_ack_and_latency_distribution():
    """Verify client render acknowledgement records paired metrics and rejects duplicates."""
    print("▶️ Running Test 10: Client Render Acknowledgement Tracking...")
    telemetry = TransportTelemetry()

    # Record first ACK for utterance 1
    t_commit = time.time() - 0.200  # committed 200ms ago
    res1 = telemetry.record_render_ack(
        epoch=1,
        utterance_id="utt-001",
        render_revision=1,
        media_end_time=45.0,
        video_current_time=45.45,
        speech_offset_to_visible_lag_sec=0.450,
        client_render_cost_ms=12.5,
        asr_commit_wall_time=t_commit,
    )
    assert res1 is True, "First render ack must be accepted"
    assert telemetry.get_summary()["paired_render_acks_count"] == 1

    # Duplicate ACK for utterance 1 (same epoch, utt_id, revision)
    res_dup = telemetry.record_render_ack(
        epoch=1,
        utterance_id="utt-001",
        render_revision=1,
        media_end_time=45.0,
        video_current_time=45.46,
        speech_offset_to_visible_lag_sec=0.460,
        client_render_cost_ms=13.0,
        asr_commit_wall_time=t_commit,
    )
    assert res_dup is False, "Duplicate render ack must be rejected"
    assert telemetry.get_summary()["paired_render_acks_count"] == 1

    # Record second ACK for utterance 2
    res2 = telemetry.record_render_ack(
        epoch=1,
        utterance_id="utt-002",
        render_revision=1,
        media_end_time=52.0,
        video_current_time=52.55,
        speech_offset_to_visible_lag_sec=0.550,
        client_render_cost_ms=14.5,
        asr_commit_wall_time=time.time() - 0.250,
    )
    assert res2 is True
    summary = telemetry.get_summary()
    assert summary["paired_render_acks_count"] == 2
    assert summary["speech_offset_to_visible_lag"]["p50_sec"] == 0.500
    assert summary["client_render_cost_ms"]["p50"] == 13.5

    print("  ✅ PASS: Render ACK correctly paired, deduplicated, and aggregated")


def run_all_tests():
    print("================================================================================")
    print("      STAGE P4-A TRANSPORT & PROTOCOL AUDIT TEST SUITE (10/10 TESTS)           ")
    print("================================================================================")
    test_format_a_parsing_and_backward_compatibility()
    test_pre_vad_duplicate_and_out_of_order_rejection()
    test_dropped_chunk_gap_detection()
    test_runtime_jitter_calculation()
    test_pause_resilient_clock_drift()
    test_generation_barrier_stream_reset()
    test_telemetry_schema_deliverable()
    test_inflight_commit_dropped_on_reset()
    test_non_monotonic_epoch_reset_rejected()
    test_client_render_ack_and_latency_distribution()
    print("================================================================================")
    print("      🎉 ALL 10/10 STAGE P4-A AUDIT TESTS PASSED WITH ZERO ERRORS!             ")
    print("================================================================================")


if __name__ == "__main__":
    run_all_tests()
