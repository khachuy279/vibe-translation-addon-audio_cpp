"""Unit tests for Phase 3C.3 VAD-Paced Soft Boundary Controller.

Tests:
1. test_normal_vad_silence_commit: Utterance < 15s commits naturally via VAD_SILENCE.
2. test_max_duration_safe_boundary: Utterance >= 15s enters FORCED_PENDING, commits via MAX_DURATION_SAFE on >=80ms silence.
3. test_segment_handoff_no_leakage_or_drop: Verifies clean audio separation between Segment A and Segment B without dropped onsets.
4. test_synthetic_runaway_emergency_cut: Pathological continuous speech >17s triggers exactly 1 MAX_DURATION_EMERGENCY.
"""

import asyncio
import time
import unittest
import numpy as np

from backend_cpp.asr.audio_buffer import VAD_STATE_SPEECH, VAD_STATE_NON_SPEECH
from backend_cpp.asr.constants import DEFAULT_SAMPLE_RATE
from backend_cpp.asr.transcribe_engine import TranscribeEngine, BoundaryState, _is_max_duration_reason
from backend_cpp.config import SentenceConfig


from unittest.mock import patch, MagicMock

class TestVADPacedBoundary(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        super().setUp()
        self.patch_sync = patch.object(TranscribeEngine, "_commit_sync", return_value=None)
        self.patch_async = patch.object(TranscribeEngine, "_commit_async", return_value=None)
        self.patch_sync.start()
        self.patch_async.start()

    def tearDown(self):
        self.patch_sync.stop()
        self.patch_async.stop()
        super().tearDown()

    def _create_sine_pcm(self, duration_sec: float, freq: float = 440.0) -> bytes:
        """Generate 16kHz 16-bit mono sine wave PCM bytes."""
        samples = int(duration_sec * DEFAULT_SAMPLE_RATE)
        t = np.linspace(0, duration_sec, samples, endpoint=False, dtype=np.float32)
        wave = (0.5 * np.sin(2 * np.pi * freq * t) * 32767.0).astype(np.int16)
        return wave.tobytes()

    def _create_silence_pcm(self, duration_sec: float) -> bytes:
        """Generate 16kHz 16-bit mono silence PCM bytes."""
        samples = int(duration_sec * DEFAULT_SAMPLE_RATE)
        return bytes(samples * 2)

    async def test_normal_vad_silence_commit(self):
        """Utterances < 15s commit naturally via VAD_SILENCE."""
        engine = TranscribeEngine(session_id="test_normal")
        engine.sentence_config.max_duration_sec = 15.0
        engine.sentence_config.max_duration_grace_sec = 2.0
        engine.sentence_config.boundary_candidate_silence_ms = 80

        commits = []
        engine.send_json_fn = lambda d: commits.append(d) if d.get("type") == "final" else None

        # Feed 4s of speech
        engine.on_speech_start()
        speech_pcm = self._create_sine_pcm(4.0)
        engine.feed_audio(speech_pcm, vad_state=VAD_STATE_SPEECH)

        self.assertEqual(engine._boundary_state, BoundaryState.NORMAL)

        # End of speech triggered by VAD
        engine.on_speech_end(reason="VAD_SILENCE")

        self.assertFalse(engine._is_speech_active)
        self.assertEqual(engine._boundary_state, BoundaryState.NORMAL)
        await engine.cleanup()

    async def test_max_duration_safe_boundary(self):
        """Utterance >= 15s enters FORCED_PENDING, commits via MAX_DURATION_SAFE on >=80ms silence candidate."""
        engine = TranscribeEngine(session_id="test_safe")
        engine.sentence_config.max_duration_sec = 15.0
        engine.sentence_config.max_duration_grace_sec = 2.0
        engine.sentence_config.boundary_candidate_silence_ms = 80

        commit_reasons = []
        original_on_speech_end = engine.on_speech_end

        def mock_on_speech_end(reason="VAD_SILENCE"):
            commit_reasons.append(reason)
            return original_on_speech_end(reason=reason)

        engine.on_speech_end = mock_on_speech_end

        engine.on_speech_start()

        # Feed 15.0s of continuous speech in 250ms chunks
        chunk_250ms = self._create_sine_pcm(0.25)
        for _ in range(60):  # 60 * 0.25s = 15.0s
            engine.feed_audio(chunk_250ms, vad_state=VAD_STATE_SPEECH)

        # Should now be in FORCED_PENDING
        self.assertEqual(engine._boundary_state, BoundaryState.FORCED_PENDING)
        self.assertEqual(len(commit_reasons), 0)  # No premature blind cut!

        # Feed 50ms silence (under 80ms threshold) -> still pending
        silence_50ms = self._create_silence_pcm(0.050)
        engine.feed_audio(silence_50ms, vad_state=VAD_STATE_NON_SPEECH)
        self.assertEqual(engine._boundary_state, BoundaryState.FORCED_PENDING)
        self.assertEqual(len(commit_reasons), 0)

        # Feed another 50ms silence (total 100ms >= 80ms) -> triggers MAX_DURATION_SAFE!
        engine.feed_audio(silence_50ms, vad_state=VAD_STATE_NON_SPEECH)
        self.assertIn("MAX_DURATION_SAFE", commit_reasons)
        self.assertEqual(engine._boundary_state, BoundaryState.NORMAL)

        # Verify speech remains active for continuous flow
        self.assertTrue(engine._is_speech_active)
        await engine.cleanup()

    async def test_segment_handoff_no_leakage_or_drop(self):
        """Verifies clean audio separation between Segment A and Segment B without dropped onsets."""
        engine = TranscribeEngine(session_id="test_handoff")
        engine.sentence_config.max_duration_sec = 15.0
        engine.sentence_config.max_duration_grace_sec = 2.0
        engine.sentence_config.boundary_candidate_silence_ms = 80

        popped_segments = []

        # Intercept pop_all to inspect audio buffers passed to commits
        original_pop_all = engine._audio_buffer_mgr.pop_all

        def mock_pop_all():
            res = original_pop_all()
            if res[0] is not None and len(res[0]) > 0:
                popped_segments.append(res[0].copy())
            return res

        engine._audio_buffer_mgr.pop_all = mock_pop_all

        engine.on_speech_start()
        initial_utt_id = engine._current_utterance_id

        # Feed 15.0s speech (Segment A)
        chunk_1s = self._create_sine_pcm(1.0, freq=440.0)
        for _ in range(15):
            engine.feed_audio(chunk_1s, vad_state=VAD_STATE_SPEECH)

        self.assertEqual(engine._boundary_state, BoundaryState.FORCED_PENDING)

        # Feed 100ms silence to trigger MAX_DURATION_SAFE
        silence_100ms = self._create_silence_pcm(0.100)
        engine.feed_audio(silence_100ms, vad_state=VAD_STATE_NON_SPEECH)

        # Segment A was popped
        self.assertEqual(len(popped_segments), 1)
        seg_a_samples = len(popped_segments[0])
        self.assertGreaterEqual(seg_a_samples, 15.0 * DEFAULT_SAMPLE_RATE)

        # New utterance ID was generated
        self.assertNotEqual(engine._current_utterance_id, initial_utt_id)
        self.assertTrue(engine._is_speech_active)

        # Now feed 3.0s of Segment B (distinct frequency 880Hz)
        chunk_b = self._create_sine_pcm(1.0, freq=880.0)
        for _ in range(3):
            engine.feed_audio(chunk_b, vad_state=VAD_STATE_SPEECH)

        # Finish stream
        engine.on_speech_end(reason="VAD_SILENCE")

        # Segment B was popped
        self.assertEqual(len(popped_segments), 2)
        seg_b_samples = len(popped_segments[1])
        self.assertAlmostEqual(seg_b_samples / DEFAULT_SAMPLE_RATE, 3.0, delta=0.05)

        # Buffer is now clean
        self.assertEqual(engine._audio_buffer_mgr.duration_sec, 0.0)
        await engine.cleanup()

    async def test_synthetic_runaway_emergency_cut(self):
        """Pathological continuous speech without silence for >17s triggers exactly 1 MAX_DURATION_EMERGENCY."""
        engine = TranscribeEngine(session_id="test_runaway")
        engine.sentence_config.max_duration_sec = 15.0
        engine.sentence_config.max_duration_grace_sec = 2.0
        engine.sentence_config.boundary_candidate_silence_ms = 80

        commit_reasons = []
        original_on_speech_end = engine.on_speech_end

        def mock_on_speech_end(reason="VAD_SILENCE"):
            commit_reasons.append(reason)
            return original_on_speech_end(reason=reason)

        engine.on_speech_end = mock_on_speech_end

        engine.on_speech_start()

        # Feed continuous speech without any silent frames for 18.0s
        chunk_500ms = self._create_sine_pcm(0.50, freq=300.0)
        for i in range(36):  # 36 * 0.5s = 18.0s
            engine.feed_audio(chunk_500ms, vad_state=VAD_STATE_SPEECH)
            dur = (i + 1) * 0.5
            if dur < 15.0:
                self.assertEqual(engine._boundary_state, BoundaryState.NORMAL)
                self.assertEqual(len(commit_reasons), 0)
            elif 15.0 <= dur < 17.0:
                # In grace period
                self.assertEqual(engine._boundary_state, BoundaryState.FORCED_PENDING)
                self.assertEqual(len(commit_reasons), 0)

        # At >= 17.0s, MAX_DURATION_EMERGENCY must have fired exactly once!
        self.assertEqual(commit_reasons.count("MAX_DURATION_EMERGENCY"), 1)
        self.assertEqual(engine._boundary_state, BoundaryState.NORMAL)

        # Post-emergency audio (17s..18s) begins new segment with speech still active
        self.assertTrue(engine._is_speech_active)
        self.assertAlmostEqual(engine._audio_buffer_mgr.duration_sec, 1.0, delta=0.1)

        await engine.cleanup()


if __name__ == "__main__":
    unittest.main()
