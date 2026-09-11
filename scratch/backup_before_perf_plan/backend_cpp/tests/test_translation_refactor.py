import asyncio
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend_cpp.config import TranslationConfig
from backend_cpp.translation import (
    ContextManager,
    LocalGGUFTranslator,
    translate_sentence,
    build_translation_prompt,
    clean_translated_text,
    detect_script,
    resolve_effective_source_lang,
    resolve_lang_name,
    get_prompt_strategy,
)
from backend_cpp.translation.prompt_strategies import (
    PromptStrategy,
    MiLMMTPromptStrategy,
    GemmaXPromptStrategy,
    TencentEnPromptStrategy,
    TencentZhPromptStrategy,
)


class TestTranslationRefactor(unittest.TestCase):

    def setUp(self):
        LocalGGUFTranslator.reset_instance()

    def tearDown(self):
        LocalGGUFTranslator.reset_instance()

    def test_script_detection(self):
        """Test pre-compiled unicode regex script detection."""
        self.assertEqual(detect_script("こんにちは、世界"), "Japanese")
        self.assertEqual(detect_script("안녕하세요"), "Korean")
        self.assertEqual(detect_script("你好，世界"), "Chinese")
        self.assertEqual(detect_script("Привет, мир"), "Russian")
        self.assertEqual(detect_script("Hello, world!"), "English")

    def test_resolve_effective_source_lang(self):
        """Test language resolution and hallucination prevention overrides."""
        # Kana override when configured as en
        self.assertEqual(
            resolve_effective_source_lang("ありがとうございます", "en", use_chinese=False),
            "Japanese"
        )
        # Hangul override
        self.assertEqual(
            resolve_effective_source_lang("감사합니다", "ja", use_chinese=False),
            "Korean"
        )
        # Cyrillic override
        self.assertEqual(
            resolve_effective_source_lang("Спасибо", "en", use_chinese=False),
            "Russian"
        )
        # Auto detect Chinese
        self.assertEqual(
            resolve_effective_source_lang("早上好", "auto", use_chinese=True),
            "中文"
        )

    def test_prompt_strategies_and_stop_tokens(self):
        """Test Strategy Pattern implementations, Protocol conformance, and stop tokens."""
        # 1. Xiaomi MiLMMT Strategy
        strategy_milmmt = get_prompt_strategy("milmmt-4b.gguf", "xiaomi/milmmt", "auto")
        self.assertIsInstance(strategy_milmmt, PromptStrategy)
        self.assertIsInstance(strategy_milmmt, MiLMMTPromptStrategy)
        p1 = strategy_milmmt.build_prompt("Hello", "en", "vi", context="Prev context", use_context=True)
        self.assertIn("Background: Prev context", p1)
        self.assertIn("Translate this from English to Vietnamese:", p1)
        self.assertIn("English: Hello", p1)
        stops_milmmt = strategy_milmmt.get_stop_tokens()
        self.assertIn("\n", stops_milmmt)
        self.assertIn("<end_of_turn>", stops_milmmt)

        # 2. Tencent English Strategy
        strategy_en = get_prompt_strategy("hy-mt.gguf", "tencent/hy-mt", "en")
        self.assertIsInstance(strategy_en, PromptStrategy)
        self.assertIsInstance(strategy_en, TencentEnPromptStrategy)
        p2 = strategy_en.build_prompt("Hello", "en", "vi")
        self.assertIn("<|im_start|>user", p2)
        self.assertIn("Translate the following text from English into Vietnamese", p2)
        stops_en = strategy_en.get_stop_tokens()
        self.assertIn("\n\n", stops_en)
        self.assertNotIn("\n", stops_en)

        # 3. Tencent Chinese Strategy (Default)
        strategy_zh = get_prompt_strategy("hy-mt.gguf", "tencent/hy-mt", "auto")
        self.assertIsInstance(strategy_zh, PromptStrategy)
        self.assertIsInstance(strategy_zh, TencentZhPromptStrategy)
        p3 = strategy_zh.build_prompt("Hello", "en", "vi", context="Ctx", use_context=True)
        self.assertIn("【背景信息】", p3)
        self.assertIn("请结合背景信息将以下文本翻译为越南语", p3)
        stops_zh = strategy_zh.get_stop_tokens()
        self.assertIn("\n\n", stops_zh)
        self.assertNotIn("\n", stops_zh)

    def test_singleton_accessor_and_warning(self):
        """Test singleton pattern and warning when duplicate cfg passed."""
        t1 = LocalGGUFTranslator.get_instance()
        self.assertIsNotNone(t1)

        with self.assertLogs("backend_cpp.translation.local_translator", level="WARNING") as cm:
            t2 = LocalGGUFTranslator.get_instance(cfg=TranslationConfig(base="milmmt"))
            self.assertIs(t1, t2)
            self.assertTrue(any("already exists" in msg for msg in cm.output))

    def test_clean_translated_text(self):
        """Test text cleaning edge cases: think tags, chat tokens, prefixes, quotes."""
        # Strip thought tags
        raw1 = "<think>Let's think step by step...</think> Xin chào thế giới!"
        self.assertEqual(clean_translated_text(raw1), "Xin chào thế giới!")

        # Cut off stop tokens
        raw2 = "Chào bạn<|im_end|> random garbage"
        self.assertEqual(clean_translated_text(raw2), "Chào bạn")

        # Strip prefixes
        raw3 = "Dịch: Hôm nay trời rất đẹp."
        self.assertEqual(clean_translated_text(raw3), "Hôm nay trời rất đẹp.")

        raw4 = "Vietnamese: Cảm ơn bạn rất nhiều."
        self.assertEqual(clean_translated_text(raw4), "Cảm ơn bạn rất nhiều.")

        # Strip enclosing quotes
        raw5 = '"Một câu nói trong ngoặc kép"'
        self.assertEqual(clean_translated_text(raw5), "Một câu nói trong ngoặc kép")

        raw6 = '“Ngoặc kép Unicode”'
        self.assertEqual(clean_translated_text(raw6), "Ngoặc kép Unicode")

    def test_context_manager(self):
        """Test rolling window ContextManager."""
        cm = ContextManager(window_size=2)
        cm.add("Hello", "Xin chào")
        cm.add("How are you?", "Bạn khỏe không?")
        ctx_str = cm.get_context_str()
        self.assertIn("Hello -> Xin chào", ctx_str)
        self.assertIn("How are you? -> Bạn khỏe không?", ctx_str)

        # Exceed window size
        cm.add("Goodbye", "Tạm biệt")
        ctx_str2 = cm.get_context_str()
        self.assertNotIn("Hello -> Xin chào", ctx_str2)
        self.assertIn("Goodbye -> Tạm biệt", ctx_str2)


class TestAsyncTranslator(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        LocalGGUFTranslator.reset_instance()

    def tearDown(self):
        LocalGGUFTranslator.reset_instance()

    async def test_empty_text_returns_empty_status(self):
        res = await translate_sentence("   ")
        self.assertEqual(res["status"], "empty")
        self.assertEqual(res["translated_text"], "")

    async def test_timeout_handling(self):
        """Test that translation timeout triggers graceful timeout error with deterministic sync."""
        cfg = TranslationConfig(base="tencent")
        translator = LocalGGUFTranslator(cfg=cfg)

        release_event = threading.Event()

        # Mock LLM that blocks until release_event is set
        def slow_llm(*args, **kwargs):
            release_event.wait(timeout=5.0)
            return {"choices": [{"text": "Late result"}]}

        translator._llm = slow_llm

        try:
            res = await translator.translate(
                text="Test timeout sentence",
                timeout=0.05
            )
            self.assertEqual(res["status"], "timeout")
            self.assertIn("timed out", res["error"])
            self.assertEqual(res["translated_text"], "Test timeout sentence")
        finally:
            release_event.set()

    async def test_mocked_successful_translation(self):
        """Test mock translation with cleaning."""
        translator = LocalGGUFTranslator()

        def fake_llm(prompt, **kwargs):
            return {"choices": [{"text": "Bản dịch: Xin chào các bạn!\n\n"}]}

        translator._llm = fake_llm

        res = await translator.translate("Hello friends!", target_lang="vi")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["translated_text"], "Xin chào các bạn!")

    async def test_subsequent_request_after_timeout(self):
        """Verify that a subsequent translation request executes properly after a timeout."""
        translator = LocalGGUFTranslator()
        import time

        def slow_llm(*args, **kwargs):
            time.sleep(0.08)
            return {"choices": [{"text": "Late result"}]}

        translator._llm = slow_llm

        res1 = await translator.translate("Slow text", timeout=0.02)
        self.assertEqual(res1["status"], "timeout")

        def quick_llm(*args, **kwargs):
            return {"choices": [{"text": "Quick translation"}]}

        translator._llm = quick_llm
        res2 = await translator.translate("Quick text", timeout=1.0)
        self.assertEqual(res2["status"], "success")
        self.assertEqual(res2["translated_text"], "Quick translation")

    def test_custom_strategy_registration_ocp(self):
        """Test OCP extensibility using register_prompt_strategy."""
        from backend_cpp.translation import register_prompt_strategy, get_prompt_strategy

        @register_prompt_strategy("mock_custom")
        class CustomStrategy:
            def build_prompt(self, text, source_lang, target_lang, context="", use_context=True):
                return f"[CUSTOM] {text}"

            def get_stop_tokens(self):
                return ["<custom_stop>"]

        strat = get_prompt_strategy("custom.gguf", "repo/custom", prompt_style="mock_custom")
        self.assertEqual(strat.build_prompt("Test", "en", "vi"), "[CUSTOM] Test")
        self.assertEqual(strat.get_stop_tokens(), ["<custom_stop>"])

    def test_unload_model_calls_close(self):
        """Test that unload_model calls close() on underlying llm object to free VRAM."""
        translator = LocalGGUFTranslator()
        mock_llm = MagicMock()
        mock_llm.close = MagicMock()
        translator._llm = mock_llm

        translator.unload_model()
        mock_llm.close.assert_called_once()
        self.assertIsNone(translator._llm)

    async def test_lock_timeout_status(self):
        """Verify that translate() returns status 'lock_timeout' when lock cannot be acquired."""
        translator = LocalGGUFTranslator()
        translator._llm = MagicMock()

        # Simulate lock held by another worker
        translator._infer_lock.acquire()
        try:
            res = await translator.translate("Contention text", timeout=0.05)
            self.assertEqual(res["status"], "lock_timeout")
            self.assertEqual(res["translated_text"], "Contention text")
            self.assertIn("error", res)
        finally:
            translator._infer_lock.release()

    def test_prompt_style_precedence_over_model_name(self):
        """Verify explicit prompt_style takes priority even when model_name matches milmmt/gemmax."""
        # 1. When prompt_style is auto, it infers GemmaX
        s_auto = get_prompt_strategy("gemmax-9b.gguf", "xiaomi/gemmax", "auto")
        self.assertIsInstance(s_auto, GemmaXPromptStrategy)

        # 2. When prompt_style is explicitly set to tencent_en, explicit style takes precedence
        s_explicit = get_prompt_strategy("gemmax-9b.gguf", "xiaomi/gemmax", "tencent_en")
        self.assertIsInstance(s_explicit, TencentEnPromptStrategy)

    def test_context_manager_thread_safety(self):
        """Verify concurrent operations on ContextManager do not raise exceptions."""
        import concurrent.futures
        ctx = ContextManager(window_size=5)

        def worker(idx):
            for i in range(50):
                ctx.add(f"Src {idx}-{i}", f"Tgt {idx}-{i}")
                _ = ctx.get_context_str()
            ctx.clear()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

    def test_translation_model_registry_loads_yaml(self):
        """Verify TranslationModelRegistry correctly loads from translation_models.yaml."""
        from backend_cpp.translation.model_registry import TranslationModelRegistry
        reg = TranslationModelRegistry.get_instance()
        self.assertGreaterEqual(len(reg.models), 4)
        self.assertIn("tencent-1.8b", reg.models)
        self.assertIn("tencent", reg.models)
        self.assertIn("xiaomi", reg.models)
        self.assertIn("gemmax", reg.models)

        m18 = reg.get_model("tencent-1.8b")
        self.assertIsNotNone(m18)
        self.assertEqual(m18["model"], "unsloth/Hy-MT2-1.8B-GGUF")
        self.assertEqual(m18["gguf_file"], "Hy-MT2-1.8B-UD-Q8_K_XL.gguf")

    def test_translation_model_registry_aliases(self):
        """Verify alias resolution maps correctly to canonical model keys."""
        from backend_cpp.translation.model_registry import TranslationModelRegistry
        reg = TranslationModelRegistry.get_instance()
        self.assertEqual(reg.resolve_key("hy-mt2-1.8b"), "tencent-1.8b")
        self.assertEqual(reg.resolve_key("tencent-1.8"), "tencent-1.8b")
        self.assertEqual(reg.resolve_key("milmmt"), "xiaomi")
        self.assertEqual(reg.resolve_key("gemma"), "gemmax")
        self.assertEqual(reg.resolve_key("gemmax-9b"), "gemmax")

    def test_translation_model_registry_list_models(self):
        """Verify list_models() returns properly formatted dictionary entries."""
        from backend_cpp.translation.model_registry import TranslationModelRegistry
        reg = TranslationModelRegistry.get_instance()
        items = reg.list_models()
        ids = [item["id"] for item in items]
        self.assertIn("tencent-1.8b", ids)
        for item in items:
            self.assertIn("name", item)
            self.assertIn("desc", item)
            self.assertIn("gguf_file", item)
            self.assertIn("is_downloaded", item)
            self.assertIsInstance(item["is_downloaded"], bool)

    def test_translation_config_with_18b(self):
        """Verify TranslationConfig correctly initializes with tencent-1.8b."""
        cfg = TranslationConfig(base="tencent-1.8b")
        self.assertEqual(cfg.base, "tencent-1.8b")
        self.assertEqual(cfg.model, "unsloth/Hy-MT2-1.8B-GGUF")
        self.assertEqual(cfg.gguf_file, "Hy-MT2-1.8B-UD-Q8_K_XL.gguf")
        self.assertEqual(cfg.prompt_style, "tencent")
        self.assertEqual(cfg.temperature, 0.7)


if __name__ == "__main__":
    unittest.main()

