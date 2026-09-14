"""Local GGUF Translation Engine using llama-cpp-python with GPU acceleration.

Models are loaded and stored strictly in backend_cpp/models/.
"""

import asyncio
import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

from backend_cpp.config import MODELS_DIR, TranslationConfig
from backend_cpp.utils.perf_profiler import perf
from backend_cpp.translation.cleaner import clean_translated_text
from backend_cpp.translation.interfaces import PromptStrategy, TranslationEngine
from backend_cpp.translation.lang_utils import (
    _LANG_PAIRS,
    LANG_NAME_MAP_ZH,
    LANG_NAME_MAP_EN,
    resolve_lang_name,
    detect_script,
    resolve_effective_source_lang,
)
from backend_cpp.translation.prompt_strategies import (
    BaseChatMLStrategy,
    MiLMMTPromptStrategy,
    TencentEnPromptStrategy,
    TencentZhPromptStrategy,
    get_prompt_strategy,
    build_translation_prompt,
    register_prompt_strategy,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LocalGGUFTranslator",
    "PromptStrategy",
    "TranslationEngine",
    "BaseChatMLStrategy",
    "MiLMMTPromptStrategy",
    "TencentEnPromptStrategy",
    "TencentZhPromptStrategy",
    "get_prompt_strategy",
    "build_translation_prompt",
    "register_prompt_strategy",
    "clean_translated_text",
    "detect_script",
    "resolve_effective_source_lang",
    "resolve_lang_name",
    "_LANG_PAIRS",
    "LANG_NAME_MAP_ZH",
    "LANG_NAME_MAP_EN",
]


class LocalGGUFTranslator:
    """Thread-safe GGUF translation engine with non-blocking streaming and timeout cancellation."""

    _instance: Optional["LocalGGUFTranslator"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls, cfg: Optional[TranslationConfig] = None) -> "LocalGGUFTranslator":
        """Thread-safe singleton accessor with automatic reconfiguration."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(cfg=cfg)
            elif cfg is not None:
                logger.warning(
                    "LocalGGUFTranslator instance already exists; passed 'cfg' will reconfigure the instance. "
                    "Call reset_instance() first if a fresh instance is needed."
                )
                if cfg != cls._instance._cfg:
                    cls._instance.reconfigure(cfg)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton and unload model to free GPU memory."""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.unload_model()
                cls._instance = None

    def reconfigure(self, new_cfg: TranslationConfig) -> None:
        """Dynamically reconfigure translator parameters, unloading model if necessary."""
        with self._load_lock:
            with self._infer_lock:
                needs_reload = (
                    self._llm is not None
                    and (
                        self._cfg.model != new_cfg.model
                        or self._cfg.gguf_file != new_cfg.gguf_file
                        or self._cfg.n_gpu_layers != new_cfg.n_gpu_layers
                    )
                )
                if needs_reload:
                    logger.info("LocalGGUFTranslator: Model file or GPU layers changed. Unloading current model...")
                    self._unload_model_unlocked()
                self._cfg = new_cfg
                logger.info(f"LocalGGUFTranslator: Reconfigured successfully (model={new_cfg.model}).")

    def __init__(self, cfg: Optional[TranslationConfig] = None, models_dir: Optional[Path] = None):
        if cfg is None:
            from backend_cpp.config import config
            self._cfg = config.translation
        else:
            self._cfg = cfg
        self._models_dir = models_dir or MODELS_DIR
        self._llm = None
        self._loaded_file: Optional[str] = None
        self._load_lock = threading.RLock()
        self._infer_lock = threading.RLock()
        self._async_load_lock: Optional[asyncio.Lock] = None

    def _resolve_gguf_path(self) -> Optional[str]:
        """Strictly locate or auto-download GGUF translation model in models directory."""
        target_file = self._cfg.gguf_file
        repo_id = self._cfg.model
        self._models_dir.mkdir(parents=True, exist_ok=True)

        local_path = self._models_dir / target_file
        if local_path.exists():
            return str(local_path)

        # Check alternative naming locally before downloading (e.g. i1 vs non-i1)
        alt_names = []
        if ".i1-" in target_file:
            alt_names.append(target_file.replace(".i1-", "."))
            alt_names.append(target_file.replace(".i1-", "-"))
        elif ".Q4_K_M" in target_file:
            alt_names.append(target_file.replace(".Q4_K_M", ".i1-Q4_K_M"))
            alt_names.append(target_file.replace("-Q4_K_M", ".i1-Q4_K_M"))
        for alt in alt_names:
            alt_path = self._models_dir / alt
            if alt_path.exists():
                logger.info(f"Using locally available translation model: {alt_path.name}")
                return str(alt_path)

        # 1. Try to auto-download configured model from HuggingFace
        try:
            logger.info(f"📥 Downloading translation model '{target_file}' from '{repo_id}' to {self._models_dir}...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=target_file,
                local_dir=str(self._models_dir),
            )
            if downloaded and os.path.exists(downloaded):
                logger.info(f"✅ Successfully downloaded translation model: {downloaded}")
                return downloaded
        except Exception as e:
            logger.error(f"⚠️ Could not auto-download translation model '{target_file}' from '{repo_id}': {e}")

        # 2. Fallback only if configured model is completely unavailable: check for existing .gguf models
        for f in self._models_dir.glob("*.gguf"):
            if any(k in f.name.lower() for k in ("hy-mt", "milmmt", "gemmax", "gemma", "translate")):
                logger.warning(f"⚠️ Fallback to available translation model in folder: {f.name}")
                return str(f)

        return None

    def load_model(self) -> None:
        """Load GGUF model into memory and offload to GPU with thread-safe double-checked lock."""
        if self._llm is not None:
            return

        with self._load_lock:
            if self._llm is not None:
                return

            # Lazily ensure CUDA/Vulkan DLL paths right before loading model
            from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths
            setup_cuda_dll_paths()

            model_path = self._resolve_gguf_path()
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"No translation GGUF model found in {self._models_dir}. Translation will be bypassed.")
                return

            try:
                from llama_cpp import Llama

                logger.info(
                    f"⚡ Loading Local GGUF Translator: {model_path} (n_gpu_layers={self._cfg.n_gpu_layers})..."
                )
                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=self._cfg.n_gpu_layers,
                    n_ctx=getattr(self._cfg, "n_ctx", 2048),
                    n_batch=getattr(self._cfg, "n_batch", 512),
                    verbose=False,
                )
                self._llm = llm
                self._loaded_file = model_path
                logger.info("✅ Local GGUF Translator successfully loaded on GPU!")
            except Exception as e:
                logger.error(f"Failed to load GGUF translation model: {e}", exc_info=True)

    def _unload_model_unlocked(self) -> None:
        """Internal helper to unload model when caller already holds locks."""
        if self._llm is not None:
            logger.info("Releasing Local GGUF Translator resources...")
            try:
                if hasattr(self._llm, "close"):
                    self._llm.close()
                del self._llm
            except Exception:
                pass
            self._llm = None
            self._loaded_file = None
            gc.collect()

    def unload_model(self) -> None:
        """Explicitly unload model and release GPU memory/contexts."""
        with self._load_lock:
            with self._infer_lock:
                self._unload_model_unlocked()

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Translate text asynchronously using high-performance GPU C++ inference.

        Concurrency & Timeout Note:
            - `timeout` specifies the maximum time allowed for acquiring `_infer_lock`
              and awaiting token generation.
            - If `_infer_lock` acquisition times out, `status: "lock_timeout"` is returned immediately.
            - If `asyncio.wait_for` cancels due to total timeout, the asyncio task is canceled,
              while the underlying C++ inference in the worker thread runs to completion
              (native C++ execution cannot be preemptively aborted).
        """
        if not text or not text.strip():
            return {"translated_text": "", "status": "empty"}

        # Non-blocking model loading guarded by async lock to avoid redundant thread submissions
        if self._llm is None:
            if self._async_load_lock is None:
                self._async_load_lock = asyncio.Lock()
            async with self._async_load_lock:
                if self._llm is None:
                    await asyncio.to_thread(self.load_model)

        if self._llm is None:
            logger.warning("Translation model not loaded, returning original text")
            return {"translated_text": text, "status": "model_not_loaded"}

        strategy = get_prompt_strategy(
            model_name=self._cfg.gguf_file,
            repo_name=self._cfg.model,
            prompt_style=self._cfg.prompt_style,
        )
        prompt = strategy.build_prompt(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            context=context,
            use_context=self._cfg.use_context,
        )
        stop_tokens = strategy.get_stop_tokens()

        def _infer() -> Tuple[str, float, float, int]:
            lock_timeout = min(timeout, 10.0) if (timeout is not None and timeout > 0) else -1
            t_lock_start = time.perf_counter()
            acquired = self._infer_lock.acquire(timeout=lock_timeout)
            lock_wait_ms = (time.perf_counter() - t_lock_start) * 1000.0
            if not acquired:
                logger.warning(
                    f"Translation infer_lock timed out waiting for lock after {lock_timeout}s "
                    f"(waited {lock_wait_ms:.1f}ms)"
                )
                return "", lock_wait_ms, 0.0, 0
            try:
                llm = self._llm
                if llm is None:
                    return "", lock_wait_ms, 0.0, 0

                t_infer_start = time.perf_counter()
                out = llm(
                    prompt,
                    max_tokens=self._cfg.max_tokens,
                    temperature=self._cfg.temperature,
                    top_p=self._cfg.top_p,
                    top_k=self._cfg.top_k,
                    repeat_penalty=self._cfg.repetition_penalty,
                    stop=stop_tokens,
                )
                infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

                raw_text = ""
                completion_tokens = 0
                if isinstance(out, dict):
                    raw_text = out.get("choices", [{}])[0].get("text", "")
                    usage = out.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)
                else:
                    raw_text = str(out)

                if completion_tokens == 0 and raw_text:
                    completion_tokens = max(1, len(raw_text.split()))

                return raw_text, lock_wait_ms, infer_ms, completion_tokens
            finally:
                self._infer_lock.release()

        try:
            infer_task = asyncio.to_thread(_infer)
            if timeout is not None and timeout > 0:
                # Overall timeout covers both lock wait and generation
                raw_out, lock_wait_ms, infer_ms, comp_tokens = await asyncio.wait_for(
                    infer_task, timeout=timeout * 2.0
                )
            else:
                raw_out, lock_wait_ms, infer_ms, comp_tokens = await infer_task

            # Record telemetry
            perf.record_metric("translation", "lock_wait_ms", lock_wait_ms)
            if infer_ms > 0:
                perf.record_metric("translation", "infer_ms", infer_ms)
                tokens_per_sec = (comp_tokens / max(0.001, infer_ms / 1000.0))
                perf.record_metric("translation", "tokens_per_sec", tokens_per_sec)
                perf.increment_counter("translation.total_translations")

                if perf.enabled and infer_ms > 150.0:
                    logger.debug(
                        f"🌐 [PERF_TRANS] Infer: {infer_ms:.1f}ms | LockWait: {lock_wait_ms:.1f}ms | "
                        f"Tokens: {comp_tokens} | Speed: {tokens_per_sec:.1f} tok/s"
                    )

            if not raw_out and lock_wait_ms > 0 and infer_ms == 0:
                logger.warning(
                    f"Translation skipped due to lock contention/timeout ({lock_wait_ms:.1f}ms) for: '{text[:40]}...'"
                )
                return {
                    "translated_text": text,
                    "status": "lock_timeout",
                    "error": f"Inference lock acquisition timed out after {lock_wait_ms:.1f}ms",
                }

            cleaned = clean_translated_text(raw_out)
            return {
                "translated_text": cleaned or text,
                "status": "success",
            }
        except asyncio.TimeoutError:
            logger.error(f"Translation inference timed out after {timeout}s for text: '{text[:50]}...'")
            return {
                "translated_text": text,
                "status": "timeout",
                "error": f"Translation timed out after {timeout}s",
            }
        except asyncio.CancelledError:
            # Re-raise cancellation immediately so calling tasks/workers can terminate cleanly
            raise
        except Exception as e:
            logger.error(f"Translation inference error: {e}", exc_info=True)
            return {"translated_text": text, "status": "error", "error": str(e)}
