"""Unit tests for ws_handler refactorings and bug fixes."""

import asyncio
import os
import struct
import sys
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend_cpp.ws.session_state import SessionState
from backend_cpp.ws.ws_handler import (
    TranslationDedupState,
    TTSDedupState,
    _handle_binary_message,
    _process_binary_chunk,
    _translation_worker,
    _tts_worker,
    _process_translation_item,
    _process_tts_item,
)
from backend_cpp.translation.context_manager import ContextManager
from backend_cpp.translation.local_translator import LocalGGUFTranslator
from backend_cpp.config import config


class TestWSHandlerRefactor(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_ws = AsyncMock()
        self.session = SessionState(self.mock_ws)
        self.session.init_components()

    async def test_handle_binary_message_uses_thread(self):
        """Verify _handle_binary_message delegates CPU decoding to asyncio.to_thread."""
        dummy_pcm = b"\x00\x00" * 160
        # Format B: 8-byte float timestamp + PCM
        data = struct.pack("<d", 12345.67) + dummy_pcm

        self.session.vad_processor = MagicMock()

        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy_to_thread:
            await _handle_binary_message(self.session, data)
            spy_to_thread.assert_called_once_with(_process_binary_chunk, self.session, data)

        self.session.vad_processor.feed_chunk.assert_called_once_with(
            dummy_pcm,
            capture_timestamp=12345.67,
        )

    def test_dedup_dataclasses(self):
        """Verify TranslationDedupState and TTSDedupState dataclasses."""
        trans_dedup = TranslationDedupState()
        self.assertEqual(trans_dedup.last_trans_norm, "")
        self.assertEqual(trans_dedup.last_trans_time, 0.0)

        trans_dedup.last_trans_norm = "hello"
        trans_dedup.last_trans_time = 100.0
        self.assertEqual(trans_dedup.last_trans_norm, "hello")
        self.assertEqual(trans_dedup.last_trans_time, 100.0)

        tts_dedup = TTSDedupState()
        self.assertEqual(tts_dedup.last_tts_norm, "")
        self.assertEqual(tts_dedup.last_tts_time, 0.0)

    async def test_translation_worker_task_done_on_early_return(self):
        """Verify task_done is called even when translation item is skipped/empty."""
        ctx = ContextManager()
        dedup_state = TranslationDedupState()

        # 1. Empty item should return early but caller try/finally calls task_done
        empty_item = {"text": "", "utterance_id": "u1"}
        await self.session.translation_queue.put(empty_item)

        item = await self.session.translation_queue.get()
        try:
            await _process_translation_item(self.session, ctx, item, dedup_state)
        finally:
            self.session.translation_queue.task_done()

        self.assertEqual(self.session.translation_queue._unfinished_tasks, 0)

        # 2. Duplicate item should return early but caller try/finally calls task_done
        valid_item = {"text": "Hello world", "utterance_id": "u2"}
        dedup_state.last_trans_norm = "hello world"
        import time
        dedup_state.last_trans_time = time.monotonic()

        await self.session.translation_queue.put(valid_item)
        item = await self.session.translation_queue.get()
        try:
            await _process_translation_item(self.session, ctx, item, dedup_state)
        finally:
            self.session.translation_queue.task_done()

        self.assertEqual(self.session.translation_queue._unfinished_tasks, 0)

    async def test_tts_worker_lazy_init(self):
        """Verify get_tts_engine is NOT called at startup when tts_enabled is False."""
        self.session.config["tts_enabled"] = False

        with patch("backend_cpp.ws.ws_handler.get_tts_engine") as mock_get_tts:
            worker_task = asyncio.create_task(_tts_worker(self.session))
            await asyncio.sleep(0.02)
            mock_get_tts.assert_not_called()

            # Now put an item in queue while tts_enabled is still False: should skip and not call get_tts_engine
            await self.session.tts_queue.put({"text": "test", "utterance_id": "u1"})
            await asyncio.sleep(0.02)
            mock_get_tts.assert_not_called()
            self.assertEqual(self.session.tts_queue._unfinished_tasks, 0)

            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)

    async def test_tts_worker_calls_get_tts_when_enabled(self):
        """Verify get_tts_engine is lazily called once an item arrives and tts_enabled is True."""
        self.session.config["tts_enabled"] = True
        mock_engine = MagicMock()
        mock_engine.synthesize_clone = AsyncMock(return_value=("dGVzdA==", 1.0))
        mock_engine.sample_rate = 24000

        with patch("backend_cpp.ws.ws_handler.get_tts_engine", return_value=mock_engine) as mock_get_tts:
            worker_task = asyncio.create_task(_tts_worker(self.session))
            await asyncio.sleep(0.01)
            # Not called yet until item is in queue
            mock_get_tts.assert_not_called()

            await self.session.tts_queue.put({"text": "test audio", "utterance_id": "u2"})
            await asyncio.sleep(0.05)

            mock_get_tts.assert_called_once()
            self.assertEqual(self.session.tts_queue._unfinished_tasks, 0)

            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)

    async def test_local_gguf_translator_cancelled_error_propagation(self):
        """Verify LocalGGUFTranslator.translate does not swallow asyncio.CancelledError."""
        translator = LocalGGUFTranslator()
        translator._llm = MagicMock()

        with patch("asyncio.to_thread", side_effect=asyncio.CancelledError()):
            with self.assertRaises(asyncio.CancelledError):
                await translator.translate("test cancellation")

    async def test_safe_websocket_concurrent_writes(self):
        """Verify SafeWebSocketConnection serializes concurrent writes without error."""
        from backend_cpp.ws.connection import SafeWebSocketConnection

        mock_raw_ws = AsyncMock()
        # Simulate slight delay in raw send_text to expose any lack of locking
        async def slow_send_text(text):
            await asyncio.sleep(0.005)
        mock_raw_ws.send_text.side_effect = slow_send_text

        safe_ws = SafeWebSocketConnection(mock_raw_ws)

        # Run 20 concurrent sends
        async def do_send(i):
            return await safe_ws.send_json({"idx": i})

        results = await asyncio.gather(*(do_send(i) for i in range(20)))
        self.assertTrue(all(results))
        self.assertEqual(mock_raw_ws.send_text.call_count, 20)

    def test_parse_audio_frame_format_a_and_b(self):
        """Verify parse_audio_frame parses valid Format A & B and rejects corrupted data."""
        import json
        from backend_cpp.ws.frame_protocol import parse_audio_frame

        dummy_pcm = b"\x01\x02\x03\x04"

        # 1. Format A valid
        hdr = json.dumps({"type": "audio_chunk", "captureTimestamp": 10.5, "chunkIndex": 3}).encode("utf-8")
        data_a = struct.pack("<I", len(hdr)) + hdr + dummy_pcm
        pcm, ts, idx = parse_audio_frame(data_a)
        self.assertEqual(pcm, dummy_pcm)
        self.assertEqual(ts, 10.5)
        self.assertEqual(idx, 3)

        # 2. Format B valid
        data_b = struct.pack("<d", 99.125) + dummy_pcm
        pcm_b, ts_b, idx_b = parse_audio_frame(data_b)
        self.assertEqual(pcm_b, dummy_pcm)
        self.assertEqual(ts_b, 99.125)
        self.assertIsNone(idx_b)

        # 3. Format A with odd byte length PCM (invalid for 16-bit) -> reject
        odd_pcm = b"\x01\x02\x03"
        data_odd = struct.pack("<I", len(hdr)) + hdr + odd_pcm
        pcm_odd, _, _ = parse_audio_frame(data_odd)
        self.assertIsNone(pcm_odd)

        # 4. Corrupted Format A (header len claims 500, but data is short) -> reject, no blind fallback
        corrupt_a = struct.pack("<I", 500) + b"too short"
        pcm_c, _, _ = parse_audio_frame(corrupt_a)
        self.assertIsNone(pcm_c)

    def test_tts_audio_msg_no_duplicate_base64(self):
        """Verify make_tts_audio_msg emits 'audio' and removes 'audio_base64'."""
        from backend_cpp.ws.serializers import make_tts_audio_msg
        msg = make_tts_audio_msg(
            utt_id="utt_1",
            text="Hello",
            audio_b64="AA==",
            duration_sec=1.5,
            sample_rate=24000,
        )
        self.assertIn("audio", msg)
        self.assertEqual(msg["audio"], "AA==")
        self.assertNotIn("audio_base64", msg)
        self.assertEqual(msg["type"], "tts_audio")

    def test_sliding_window_dedup(self):
        """Verify SlidingWindowDedup suppresses duplicates within sliding window."""
        from backend_cpp.ws.dedup import SlidingWindowDedup
        dedup = SlidingWindowDedup(window_sec=2.0)

        # First time: not duplicate
        self.assertFalse(dedup.is_duplicate("Xin chào các bạn", now=10.0))

        # Within window (1.0s later) with identical normalized text: duplicate!
        self.assertTrue(dedup.is_duplicate("xin chào  các bạn", now=11.0))

        # Outside window (2.5s later): not duplicate!
        self.assertFalse(dedup.is_duplicate("Xin chào các bạn", now=13.5))

    def test_record_chunk_gap_detection_counter(self):
        """Verify record_chunk detects gaps in chunk index sequence and increments counter."""
        from unittest.mock import MagicMock, patch
        from backend_cpp.utils.perf_profiler import perf
        mock_ws = MagicMock()
        session = SessionState(mock_ws)

        with patch("backend_cpp.config.config.perf.enabled", True):
            # Baseline count
            initial_dropped = perf.generate_report()["counters"].get("ws.audio_chunks_dropped", 0)

            # First chunk
            session.record_chunk(1)
            self.assertEqual(session.chunk_index, 1)

            # Gap from 1 to 5 (dropped chunks = 5 - (1 + 1) = 3)
            session.record_chunk(5)
            self.assertEqual(session.chunk_index, 5)

            new_dropped = perf.generate_report()["counters"].get("ws.audio_chunks_dropped", 0)
            self.assertEqual(new_dropped - initial_dropped, 3)


class TestWSAdmissionControl(unittest.IsolatedAsyncioTestCase):
    """Audit finding P1-04: at most ``config.ws.max_sessions`` sessions at a time.

    Policy is NEWEST-WINS. Rejecting a newcomer would let one stale connection (a
    forgotten tab, or the extension's background bridge holding a socket open) lock the
    user out of START forever -- exactly the field failure that motivated this test:
    the extension saw close code 1013 on every start attempt after an earlier session
    had been left behind.
    """

    async def asyncSetUp(self):
        import backend_cpp.ws.ws_handler as ws_handler

        self.ws_handler = ws_handler
        self._orig_max_sessions = config.ws.max_sessions
        self._orig_dump_report = config.perf.dump_report_on_disconnect
        config.ws.max_sessions = 1
        config.perf.dump_report_on_disconnect = False
        with ws_handler._active_sessions_lock:
            ws_handler._active_sessions.clear()

    async def asyncTearDown(self):
        config.ws.max_sessions = self._orig_max_sessions
        config.perf.dump_report_on_disconnect = self._orig_dump_report
        with self.ws_handler._active_sessions_lock:
            self.ws_handler._active_sessions.clear()

    @staticmethod
    def _make_ws(receive=None):
        ws = AsyncMock()
        if receive is None:
            receive = AsyncMock(return_value={"type": "websocket.disconnect"})
        ws.receive = receive
        return ws

    async def _wait_for_active_sessions(self, expected, timeout=5.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self.ws_handler.get_active_session_count() == expected:
                return True
            await asyncio.sleep(0.01)
        return self.ws_handler.get_active_session_count() == expected

    async def test_session_slot_is_released_after_disconnect(self):
        ws = self._make_ws()
        await self.ws_handler.handle_ws(ws)

        ws.accept.assert_awaited()
        # The slot must be given back once the session ends, otherwise a single
        # failure would permanently occupy the only slot.
        self.assertEqual(self.ws_handler.get_active_session_count(), 0)

    async def test_new_session_supersedes_the_older_one(self):
        """A new START must be admitted; the stale session is the one that closes."""
        release_first = asyncio.Event()

        async def first_receive():
            await release_first.wait()
            return {"type": "websocket.disconnect"}

        first_ws = self._make_ws(receive=first_receive)
        first_task = asyncio.create_task(self.ws_handler.handle_ws(first_ws))
        self.assertTrue(await self._wait_for_active_sessions(1))

        release_second = asyncio.Event()

        async def second_receive():
            await release_second.wait()
            return {"type": "websocket.disconnect"}

        second_ws = self._make_ws(receive=second_receive)
        second_task = asyncio.create_task(self.ws_handler.handle_ws(second_ws))

        # Let the newcomer's handler get past accept() and the supersede step.
        await asyncio.sleep(0.2)

        # The newcomer is admitted, not rejected.
        second_ws.accept.assert_awaited()
        self.assertEqual(self.ws_handler.get_active_session_count(), 1)

        # ...and the OLDER session is the one asked to close (code 1013).
        first_ws.close.assert_awaited()
        _, kwargs = first_ws.close.call_args
        self.assertEqual(kwargs.get("code"), 1013)
        second_ws.close.assert_not_awaited()

        # The superseded session's teardown must not evict the newcomer.
        release_first.set()
        await asyncio.wait_for(first_task, timeout=5.0)
        self.assertEqual(self.ws_handler.get_active_session_count(), 1)

        release_second.set()
        await asyncio.wait_for(second_task, timeout=5.0)
        self.assertEqual(self.ws_handler.get_active_session_count(), 0)

    async def test_concurrency_stays_bounded(self):
        """Under a burst of connects the live count never exceeds the limit."""
        config.ws.max_sessions = 2

        release = asyncio.Event()

        async def blocking_receive():
            await release.wait()
            return {"type": "websocket.disconnect"}

        tasks = [
            asyncio.create_task(self.ws_handler.handle_ws(self._make_ws(receive=blocking_receive)))
            for _ in range(3)
        ]
        await asyncio.sleep(0.5)

        self.assertLessEqual(self.ws_handler.get_active_session_count(), 2)

        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        self.assertEqual(self.ws_handler.get_active_session_count(), 0)

    async def test_limit_can_be_disabled(self):
        """max_sessions=0 means unlimited (opt-out for deliberate multi-session use)."""
        config.ws.max_sessions = 0

        release = asyncio.Event()

        async def blocking_receive():
            await release.wait()
            return {"type": "websocket.disconnect"}

        tasks = [
            asyncio.create_task(self.ws_handler.handle_ws(self._make_ws(receive=blocking_receive)))
            for _ in range(3)
        ]
        self.assertTrue(await self._wait_for_active_sessions(3))

        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        self.assertEqual(self.ws_handler.get_active_session_count(), 0)


if __name__ == "__main__":
    unittest.main()
