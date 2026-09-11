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


if __name__ == "__main__":
    unittest.main()
