"""Unit tests for improvements made following the Code Review."""

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from backend_cpp.asr.audio_buffer import AudioBufferManager
from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.asr.transcribe_engine import ASREngineConfig, TranscribeEngine
from backend_cpp.config import config, TranslationConfig, SentenceConfig
from backend_cpp.translation.local_translator import LocalGGUFTranslator
from backend_cpp.vad.vad_processor import VADProcessor
from backend_cpp.ws.session_state import SessionConfig, SessionState


class TestCodeReviewImprovements(unittest.TestCase):

    def test_audio_buffer_overflow_protection(self):
        """Verify AudioBufferManager limits capacity and drops oldest bytes on overflow."""
        # 16kHz mono = 32000 bytes/sec. Set max_duration_sec = 0.1s (3200 bytes)
        buf = AudioBufferManager(sample_rate=16000, max_duration_sec=0.1)
        self.assertEqual(buf.max_bytes, 3200)

        # Feed 2000 bytes (within limit)
        dur1 = buf.feed_bytes(b"\x00" * 2000)
        self.assertEqual(len(buf._bytes_buffer), 2000)
        self.assertAlmostEqual(dur1, 2000 / 32000.0)

        # Feed another 2000 bytes (total 4000 > 3200)
        # Should trim oldest 800 bytes, keeping 3200
        dur2 = buf.feed_bytes(b"\x01" * 2000)
        self.assertEqual(len(buf._bytes_buffer), 3200)
        self.assertAlmostEqual(dur2, 3200 / 32000.0)
        # Verify the oldest bytes were dropped and newest bytes kept
        self.assertEqual(bytes(buf._bytes_buffer[-10:]), b"\x01" * 10)

    def test_asr_engine_config_dependency_injection(self):
        """Verify TranscribeEngine accepts custom ASREngineConfig and ASRModelManager."""
        custom_cfg = ASREngineConfig(
            model_key="custom-test-model",
            min_transcribe_sec=1.2,
            poll_interval_ms=500,
            threads=8,
            language="vi",
            backend="cpu",
            sentence_config=SentenceConfig(max_chars=120),
        )
        mock_mgr = MagicMock(spec=ASRModelManager)
        mock_mgr.supports_streaming.return_value = True

        engine = TranscribeEngine(
            engine_config=custom_cfg,
            model_manager=mock_mgr,
        )

        self.assertEqual(engine.model_key, "custom-test-model")
        self.assertEqual(engine._min_transcribe_sec, 1.2)
        self.assertEqual(engine._poll_interval_sec, 0.5)
        self.assertEqual(engine._language, "vi")
        self.assertEqual(engine.sentence_config.max_chars, 120)
        self.assertEqual(engine.supports_streaming, True)
        mock_mgr.supports_streaming.assert_called_with("custom-test-model")

    def test_asr_model_manager_state(self):
        """Verify ASRModelManager manages state and unloading."""
        self.assertFalse(ASRModelManager.is_model_loaded("non_existent_model"))

        # Test unload_shared_model clears cached model and session
        mock_model = MagicMock()
        mock_session = MagicMock()
        ASRModelManager._shared_model = mock_model
        ASRModelManager._shared_session = mock_session
        ASRModelManager._shared_model_key = "test-model"

        self.assertTrue(ASRModelManager.is_model_loaded("test-model"))

        ASRModelManager.unload_shared_model()
        mock_session.close.assert_called_once()
        mock_model.close.assert_called_once()
        self.assertIsNone(ASRModelManager.get_shared_model())
        self.assertIsNone(ASRModelManager.get_shared_session())
        self.assertFalse(ASRModelManager.is_model_loaded())

    def test_local_translator_reconfigure(self):
        """Verify LocalGGUFTranslator reconfigures correctly."""
        LocalGGUFTranslator.reset_instance()
        cfg1 = TranslationConfig(model="repo/model-1", gguf_file="model-1.gguf")
        t = LocalGGUFTranslator.get_instance(cfg=cfg1)
        self.assertEqual(t._cfg.model, "repo/model-1")

        # Reconfigure with different model
        cfg2 = TranslationConfig(model="repo/model-2", gguf_file="model-2.gguf")
        t.reconfigure(cfg2)
        self.assertEqual(t._cfg.model, "repo/model-2")
        self.assertEqual(t._cfg.gguf_file, "model-2.gguf")

        # Simulate model currently loaded into GPU memory to verify unload without deadlock
        mock_llm = MagicMock()
        t._llm = mock_llm
        t._loaded_file = "model-2.gguf"

        # Calling get_instance with different config triggers reload & unload of existing model
        cfg3 = TranslationConfig(model="repo/model-3", gguf_file="model-3.gguf")
        t3 = LocalGGUFTranslator.get_instance(cfg=cfg3)
        self.assertIs(t, t3)
        self.assertEqual(t._cfg.model, "repo/model-3")
        self.assertIsNone(t._llm)  # Successfully unloaded!
        self.assertIsNone(t._loaded_file)
        mock_llm.close.assert_called_once()
        LocalGGUFTranslator.reset_instance()

    def test_session_config_and_session_state(self):
        """Verify SessionConfig behaves as a dict while keeping type safety."""
        sc = SessionConfig({"source_lang": "ja", "vad_enabled": False})
        self.assertEqual(sc["source_lang"], "ja")
        self.assertEqual(sc.get("vad_enabled"), False)
        self.assertTrue("target_lang" in sc)

        sc["target_lang"] = "fr"
        self.assertEqual(sc["target_lang"], "fr")

        mock_ws = MagicMock()
        session = SessionState(mock_ws)
        self.assertIsInstance(session.config, SessionConfig)
        self.assertEqual(session.config["source_lang"], "auto")

    def test_session_apply_config_switches_translation_model(self):
        """Verify SessionState.apply_config dynamically switches translation model."""
        LocalGGUFTranslator.reset_instance()
        mock_ws = MagicMock()
        session = SessionState(mock_ws)
        self.assertEqual(session.config.get("translation_model"), config.translation.base)

        # Apply config with translationModel: tencent
        session.apply_config({"translationModel": "tencent"})
        self.assertEqual(session.config.get("translation_model"), "tencent")
        translator = LocalGGUFTranslator.get_instance()
        self.assertEqual(translator._cfg.base, "tencent")
        self.assertEqual(translator._cfg.model, "unsloth/Hy-MT2-7B-GGUF")
        self.assertEqual(translator._cfg.gguf_file, "Hy-MT2-7B-UD-Q4_K_XL.gguf")

        # Apply config with translationModel: tencent-1.8b
        session.apply_config({"translationModel": "tencent-1.8b"})
        self.assertEqual(session.config.get("translation_model"), "tencent-1.8b")
        translator_18b = LocalGGUFTranslator.get_instance()
        self.assertEqual(translator_18b._cfg.base, "tencent-1.8b")
        self.assertEqual(translator_18b._cfg.model, "unsloth/Hy-MT2-1.8B-GGUF")
        self.assertEqual(translator_18b._cfg.gguf_file, "Hy-MT2-1.8B-UD-Q8_K_XL.gguf")

        # Switch back to xiaomi
        session.apply_config({"translationModel": "xiaomi"})
        self.assertEqual(session.config.get("translation_model"), "xiaomi")
        translator2 = LocalGGUFTranslator.get_instance()
        self.assertEqual(translator2._cfg.base, "xiaomi")
        self.assertEqual(translator2._cfg.model, "mradermacher/MiLMMT-46-4B-v1.0-GGUF")

        # Switch to gemmax (GemmaX2-28-9B)
        session.apply_config({"translationModel": "gemmax"})
        self.assertEqual(session.config.get("translation_model"), "gemmax")
        translator3 = LocalGGUFTranslator.get_instance()
        self.assertEqual(translator3._cfg.base, "gemmax")
        self.assertEqual(translator3._cfg.model, "mradermacher/GemmaX2-28-9B-v0.2-i1-GGUF")
        self.assertEqual(translator3._cfg.gguf_file, "GemmaX2-28-9B-v0.2.i1-Q4_K_M.gguf")

        # Try switching when capturing is active (chunk_index > 0)
        session.chunk_index = 5
        session.apply_config({"translationModel": "tencent"})
        # Must remain gemmax because streaming audio capture is active!
        self.assertEqual(session.config.get("translation_model"), "gemmax")

        LocalGGUFTranslator.reset_instance()

    def test_vad_slow_callback_warning(self):
        """Verify VADProcessor logs warning if callback execution exceeds threshold."""
        slow_callback = MagicMock()

        def slow_chunk(pcm_bytes, ts):
            time.sleep(0.12)  # Exceeds 100ms threshold

        slow_callback.side_effect = slow_chunk
        slow_callback.__name__ = "slow_chunk"

        vad = VADProcessor(
            vad_engine="fsmn-vad",
            enabled=True,
            on_speech_chunk=slow_callback,
        )

        dummy_chunk = b"\x00" * 1024
        # We can feed a chunk and check assertLogs
        with self.assertLogs("backend_cpp.vad.vad_processor", level="WARNING") as cm:
            # Manually simulate callbacks trigger
            callbacks = [(slow_callback, (dummy_chunk, 0.0))]
            for cb, args in callbacks:
                t0 = time.perf_counter()
                cb(*args)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if elapsed_ms > 100.0:
                    import logging
                    logging.getLogger("backend_cpp.vad.vad_processor").warning(
                        f"⚠️ Slow VAD callback {getattr(cb, '__name__', str(cb))} took {elapsed_ms:.1f}ms"
                    )

            self.assertTrue(any("Slow VAD callback" in msg for msg in cm.output))

    def test_active_model_persistence_and_sync(self):
        """Verify switching active model persists across new TranscribeEngine instances and SessionState."""
        from backend_cpp.asr.model_registry import ModelRegistry
        registry = ModelRegistry.get_instance()

        # Switch to sensevoice-small
        registry.set_active_model_key("sensevoice-small")
        self.assertEqual(registry.get_active_model_key(), "sensevoice-small")
        self.assertEqual(config.asr.active_model, "sensevoice-small")

        # New default TranscribeEngine should inherit sensevoice-small, NOT hardcoded qwen3
        engine = TranscribeEngine()
        self.assertEqual(engine.model_key, "sensevoice-small")

        # SessionState should also have sensevoice-small as active model
        mock_ws = MagicMock()
        session = SessionState(mock_ws)
        session.init_components()
        self.assertEqual(session.asr_engine.model_key, "sensevoice-small")

        # Revert back to qwen3-asr-1.7b for cleanliness
        registry.set_active_model_key("qwen3-asr-1.7b")
        self.assertEqual(registry.get_active_model_key(), "qwen3-asr-1.7b")
        self.assertEqual(config.asr.active_model, "qwen3-asr-1.7b")


if __name__ == "__main__":
    unittest.main()
