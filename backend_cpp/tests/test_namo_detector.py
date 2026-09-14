"""Unit tests for NamoTurnDetector and its integration with TranscribeEngine."""

import unittest
from unittest.mock import MagicMock, patch

from backend_cpp.asr.namo_detector import NamoTurnDetector
from backend_cpp.config import NamoConfig, config


class TestNamoTurnDetector(unittest.TestCase):
    """Test NamoTurnDetector inference, filtering, and configuration."""

    def test_singleton_accessor(self):
        d1 = NamoTurnDetector.get_shared_instance(confidence_threshold=0.70)
        d2 = NamoTurnDetector.get_shared_instance(confidence_threshold=0.75)
        self.assertIs(d1, d2)
        self.assertEqual(d1.confidence_threshold, 0.75)

    def test_min_tokens_filtering(self):
        detector = NamoTurnDetector.get_shared_instance()
        # Short fragments (< 3 tokens) should immediately return (False, 0.0) without inference
        is_eou, conf = detector.predict_eou("はい", min_tokens=3)
        self.assertFalse(is_eou)
        self.assertEqual(conf, 0.0)

        is_eou, conf = detector.predict_eou("Hi there", min_tokens=3)
        self.assertFalse(is_eou)
        self.assertEqual(conf, 0.0)

    def test_empty_text(self):
        detector = NamoTurnDetector.get_shared_instance()
        is_eou, conf = detector.predict_eou("")
        self.assertFalse(is_eou)
        self.assertEqual(conf, 0.0)

        is_eou, conf = detector.predict_eou("   \n\t ")
        self.assertFalse(is_eou)
        self.assertEqual(conf, 0.0)

    def test_eou_prediction_real(self):
        detector = NamoTurnDetector.get_shared_instance()
        # "Hello, how can I help you today?" is a standard complete sentence
        is_eou, conf = detector.predict_eou("Hello, how can I help you today?", confidence_threshold=0.70)
        self.assertTrue(is_eou)
        self.assertGreaterEqual(conf, 0.70)

        # "I was thinking that maybe we could" is incomplete
        is_eou, conf = detector.predict_eou("I was thinking that maybe we could", confidence_threshold=0.70)
        self.assertFalse(is_eou)

    def test_config_defaults(self):
        cfg = NamoConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.confidence_threshold, 0.70)
        self.assertEqual(cfg.min_tokens, 3)
        self.assertEqual(cfg.require_silence_ms, 120)
        self.assertEqual(cfg.repo_id, "videosdk-live/Namo-Turn-Detector-v1-Multilingual")
        self.assertTrue(cfg.model_dir.replace("\\", "/").endswith("backend_cpp/models/namo"))
        self.assertEqual(cfg.max_length, 512)

    def test_model_dir_loading(self):
        detector = NamoTurnDetector.get_shared_instance()
        self.assertTrue(detector.model_dir.exists())
        self.assertTrue((detector.model_dir / "model_quant.onnx").exists())
        self.assertTrue((detector.model_dir / "tokenizer.json").exists())


if __name__ == "__main__":
    unittest.main()
