"""GGUFTranslator: Engine dịch thuật cục bộ qua Llama.cpp (llama-cpp-python).

Hỗ trợ:
- Tự động nạp mô hình GGUF lên GPU VRAM (n_gpu_layers=-1).
- Tối ưu hóa cho 1 session: Thread-safe singleton với infer lock.
- Pre-warm sẵn sàng khi khởi động.
- Chuyển đổi mô hình nóng (Hot-Swap) tức thì.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from backend.utils.cuda import setup_cuda_dll_paths
from backend.utils.logger import logger

# Bắt buộc thiết lập đường dẫn CUDA DLL trước khi nạp llama_cpp
setup_cuda_dll_paths()

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from backend.config import config, TranslationConfig
from backend.translation.base import BaseTranslator
from backend.translation.registry import TranslationModelRegistry
from backend.translation.prompts import get_prompt_strategy

_TRANS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translation_worker")


class GGUFTranslator(BaseTranslator):
    """Engine dịch thuật sử dụng Llama GGUF."""

    _instance: Optional["GGUFTranslator"] = None
    _shared_llm: Optional[Any] = None
    _shared_model_key: Optional[str] = None
    _shared_lock = threading.RLock()
    _infer_lock = threading.RLock()

    @classmethod
    def get_instance(cls, trans_cfg: Optional[TranslationConfig] = None) -> "GGUFTranslator":
        with cls._shared_lock:
            if cls._instance is None:
                cls._instance = GGUFTranslator(trans_cfg)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._shared_lock:
            if cls._instance is not None:
                try:
                    cls._instance.unload_model()
                except Exception as e:
                    logger.debug(f"Translation unload_model notice: {e}")
                cls._instance = None

    def __init__(self, trans_cfg: Optional[TranslationConfig] = None):
        self.cfg = trans_cfg or config.translation
        self.registry = TranslationModelRegistry.get_instance()
        self.canonical_key = self.registry.resolve_key(self.cfg.base)
        self.prompt_strategy = get_prompt_strategy(self.cfg.prompt_style)

    def load_model(self) -> None:
        """Nạp model GGUF lên GPU nếu chưa nạp."""
        with self.__class__._shared_lock:
            if (
                self.__class__._shared_llm is not None
                and self.__class__._shared_model_key == self.canonical_key
            ):
                return

            if Llama is None:
                raise RuntimeError("llama-cpp-python chưa được cài đặt!")

            gguf_path = self.registry.resolve_gguf_path(self.canonical_key)
            if not os.path.exists(gguf_path):
                raise FileNotFoundError(f"Không tìm thấy file GGUF Translation tại: {gguf_path}")

            info = self.registry.get_model(self.canonical_key) or {}
            logger.info(
                f"Đang nạp mô hình dịch GGUF '{self.canonical_key}' từ: {gguf_path}",
                extra={"module_tag": "TRANSLATE"},
            )

            # Giải phóng model cũ nếu có
            if self.__class__._shared_llm is not None:
                try:
                    if hasattr(self.__class__._shared_llm, "close"):
                        self.__class__._shared_llm.close()
                except Exception:
                    pass
                del self.__class__._shared_llm
                self.__class__._shared_llm = None
                import gc
                gc.collect()

            n_gpu_layers = self.cfg.n_gpu_layers if self.cfg.n_gpu_layers is not None else -1
            llm = Llama(
                model_path=gguf_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=512,
                n_batch=256,
                n_threads=4,
                verbose=False,
            )

            self.__class__._shared_llm = llm
            self.__class__._shared_model_key = self.canonical_key
            self.prompt_strategy = get_prompt_strategy(info.get("prompt_style", "tencent"))

            logger.info(
                f"Nạp thành công mô hình dịch '{self.canonical_key}' trên GPU (n_ctx=512, n_batch=256)",
                extra={"module_tag": "TRANSLATE"},
            )

    @classmethod
    def shutdown_executors(cls, wait: bool = False) -> None:
        """Dừng các luồng worker executor của Translation."""
        try:
            _TRANS_EXECUTOR.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            pass

    def unload_model(self) -> None:
        """Giải phóng model dịch khỏi GPU an toàn."""
        with self.__class__._shared_lock:
            with self.__class__._infer_lock:
                if self.__class__._shared_llm is not None:
                    try:
                        if hasattr(self.__class__._shared_llm, "close"):
                            self.__class__._shared_llm.close()
                        del self.__class__._shared_llm
                    except Exception:
                        pass
                    self.__class__._shared_llm = None
                    self.__class__._shared_model_key = None
                    import gc
                    gc.collect()
                    logger.info("Đã giải phóng mô hình dịch khỏi GPU VRAM.", extra={"module_tag": "TRANSLATE"})


    def prewarm(self) -> None:
        """Prewarm mô hình dịch bằng một câu ngắn."""
        try:
            self.load_model()
            self._translate_sync("Hello", source_lang="en", target_lang="vi")
            logger.info(f"Pre-warm hoàn tất cho Translation model '{self.canonical_key}'", extra={"module_tag": "TRANSLATE"})
        except Exception as e:
            logger.warning(f"Pre-warm Translation warning: {e}", extra={"module_tag": "TRANSLATE"})

    def _translate_sync(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> Dict[str, Any]:
        """Thực hiện dịch đồng bộ trên worker thread."""
        if not text or not text.strip():
            return {"translated_text": "", "elapsed_ms": 0.0}

        self.load_model()
        llm = self.__class__._shared_llm
        if llm is None:
            return {"translated_text": text, "elapsed_ms": 0.0}

        info = self.registry.get_model(self.canonical_key) or {}
        temperature = self.cfg.temperature if self.cfg.temperature is not None else info.get("temperature", 0.7)
        top_p = self.cfg.top_p if self.cfg.top_p is not None else info.get("top_p", 0.6)
        top_k = self.cfg.top_k if self.cfg.top_k is not None else info.get("top_k", 20)
        repetition_penalty = self.cfg.repetition_penalty if self.cfg.repetition_penalty is not None else info.get("repetition_penalty", 1.05)

        prompt = self.prompt_strategy.build_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            use_context=self.cfg.use_context,
        )
        stop_tokens = self.prompt_strategy.get_stop_tokens()

        with self.__class__._infer_lock:
            t0 = time.perf_counter()
            output = llm(
                prompt,
                max_tokens=self.cfg.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stop=stop_tokens,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            raw_text = output["choices"][0]["text"].strip()
            # Dọn dẹp khoảng trắng
            clean_out = raw_text.replace("<|im_end|>", "").strip()

            return {
                "translated_text": clean_out,
                "elapsed_ms": elapsed_ms,
                "tokens": output.get("usage", {}).get("completion_tokens", 0),
            }

    async def translate_sentence(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> Dict[str, Any]:
        """Async wrapper gọi hàm dịch trên thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _TRANS_EXECUTOR,
            self._translate_sync,
            text,
            source_lang,
            target_lang,
            context,
        )

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> Dict[str, Any]:
        """Alias cho translate_sentence."""
        return await self.translate_sentence(text, source_lang, target_lang, context)


def get_translator(cfg: Optional[TranslationConfig] = None) -> GGUFTranslator:
    """Helper lấy translator singleton."""
    return GGUFTranslator.get_instance(cfg)


def reset_translator() -> None:
    """Helper reset translator."""
    GGUFTranslator.reset_instance()


# Alias tương thích ngược
GGUFTranslationEngine = GGUFTranslator
get_translation_engine = get_translator
reset_translation_engine = reset_translator


