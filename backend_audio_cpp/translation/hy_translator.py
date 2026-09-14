"""Hunyuan-MT2 (Hy-MT2) 7B GPU Translation Engine for backend_audio_cpp.

Communicates with llama-cpp-python using full CUDA GPU offload (RTX 5060 Ti),
delivering high-speed, high-accuracy machine translation with structured logging.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Union

from backend_audio_cpp.translation.cleaner import clean_translated_text
from backend_audio_cpp.translation.cuda_loader import ensure_cuda_dlls_loaded
from backend_audio_cpp.translation.prompt_builder import (
    STOP_TOKENS,
    build_translation_prompt,
    detect_script,
)

logger = logging.getLogger("backend_audio_cpp.translation")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = (
    _PROJECT_ROOT / "backend_audio_cpp" / "models" / "Hy-MT2-7B-UD-Q4_K_XL.gguf"
)


@dataclass
class TranslationConfig:
    """Configuration for Hunyuan-MT2 Translation Engine."""
    model_path: str = str(_DEFAULT_MODEL_PATH)
    n_gpu_layers: int = -1  # Full GPU offload
    n_ctx: int = 2048
    n_threads: int = 4
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class TranslationResult:
    """Telemetry and outcome of a translation request."""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    tokens: int
    latency_ms: float
    tps: float


class HyMTTranslator:
    """High-performance Hunyuan-MT2 GPU Translator wrapping llama-cpp-python."""

    _instance: Optional["HyMTTranslator"] = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: Optional[TranslationConfig] = None) -> "HyMTTranslator":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._llm = None
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize llama_cpp with CUDA acceleration."""
        ensure_cuda_dlls_loaded()

        model_file = Path(self.config.model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Translation model not found at: {model_file}\n"
                "Please place Hy-MT2-7B-UD-Q4_K_XL.gguf in backend_audio_cpp/models/"
            )

        from llama_cpp import Llama

        logger.info(
            f"[TRANSLATE] Loading Hy-MT2 7B into GPU (n_gpu_layers={self.config.n_gpu_layers}, n_ctx={self.config.n_ctx})..."
        )
        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=str(model_file),
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            verbose=False,
        )
        load_dur = time.perf_counter() - t0
        logger.info(f"[TRANSLATE] Model loaded into GPU memory in {load_dur:.2f}s!")

    def translate(
        self,
        text: str,
        target_lang: str = "vi",
        source_lang: str = "auto",
        utt_id: Union[int, str] = 0,
        context: str = "",
    ) -> TranslationResult:
        """Translate text with structured logging.

        Args:
            text: Input sentence to translate.
            target_lang: Target language code ('vi', 'en', 'zh', etc.).
            source_lang: Source language code or 'auto'.
            utt_id: Utterance identifier for log correlation.
            context: Optional prior background context.

        Returns:
            TranslationResult with text, timing, tokens, and TPS.
        """
        clean_input = text.strip()
        if not clean_input:
            return TranslationResult(
                original_text="",
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                tokens=0,
                latency_ms=0.0,
                tps=0.0,
            )

        eff_source = source_lang if source_lang != "auto" else detect_script(clean_input)

        # Structured Log: START
        logger.info(
            f'[TRANSLATE] START [utt_{utt_id}]: "{clean_input}" ({eff_source} -> {target_lang})'
        )

        prompt = build_translation_prompt(
            text=clean_input,
            target_lang=target_lang,
            source_lang=eff_source,
            context=context,
        )

        t0 = time.perf_counter()
        with self._lock:
            resp = self._llm(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=STOP_TOKENS,
            )
        elapsed_sec = time.perf_counter() - t0
        latency_ms = elapsed_sec * 1000.0

        raw_output = resp["choices"][0]["text"].strip()
        translated_text = clean_translated_text(raw_output)
        tokens = resp["usage"].get("completion_tokens", len(translated_text.split()))
        tps = tokens / elapsed_sec if elapsed_sec > 0 else 0.0

        # Structured Log: DONE
        logger.info(
            f'[TRANSLATE] DONE [utt_{utt_id}]: "{clean_input}" -> "{translated_text}" | '
            f'tokens={tokens} | latency={latency_ms:.1f}ms | speed={tps:.1f} tps'
        )

        return TranslationResult(
            original_text=clean_input,
            translated_text=translated_text,
            source_lang=eff_source,
            target_lang=target_lang,
            tokens=tokens,
            latency_ms=latency_ms,
            tps=tps,
        )
