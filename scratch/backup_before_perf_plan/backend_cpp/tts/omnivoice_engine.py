"""Native PyTorch OmniVoice TTS Engine for backend_cpp (Sub-0.6s Real-Time Synthesis)."""

import asyncio
import gc
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import numpy as np
import torch

from backend_cpp.config import config, PROJECT_ROOT
from backend_cpp.tts.base import TTSEngine
from backend_cpp.tts.audio_processor import AudioProcessor
from backend_cpp.tts.voice_manager import VoiceManager

logger = logging.getLogger(__name__)

BACKEND_CPP_DIR = Path(__file__).resolve().parent.parent
LEGACY_MODELS_DIR = PROJECT_ROOT / "backend" / "models"


class OmniVoiceTTS:
    """Singleton TTS Engine wrapping PyTorch OmniVoice for ultra-fast in-process synthesis.

    Implements the TTSEngine protocol.
    """

    _instance: Optional["OmniVoiceTTS"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "OmniVoiceTTS":
        """Thread-safe singleton accessor."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = OmniVoiceTTS()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Unload and tear down the singleton instance for lifecycle management."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload_model()
                cls._instance = None

    def __init__(self):
        self.sample_rate = 24000
        self.device = config.tts.device if hasattr(config, "tts") and config.tts.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._is_loaded = False
        self._cudnn_disabled = False
        self._voice_prompt_cache: Dict[Tuple[str, str], Any] = {}
        self._init_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _resolve_model_path(self) -> str:
        """Locate configured model or fallback to local HF snapshot / repo id."""
        cfg_model = config.tts.model if hasattr(config, "tts") and config.tts.model else "splendor1811/omnivoice-vietnamese"

        # 1. Prioritize explicit local file/directory if configured and existing
        if cfg_model and os.path.exists(cfg_model):
            return cfg_model

        # 2. Check local HF snapshot cache in legacy backend/models or backend_cpp/models
        snap_dir = LEGACY_MODELS_DIR / "huggingface" / "models--splendor1811--omnivoice-vietnamese" / "snapshots"
        if snap_dir.exists():
            for child in snap_dir.iterdir():
                if child.is_dir() and (child / "config.json").exists():
                    return str(child)

        return cfg_model or "splendor1811/omnivoice-vietnamese"

    def is_model_ready(self) -> bool:
        """Check if OmniVoice model weights are resolvable locally or via Hugging Face."""
        p = self._resolve_model_path()
        return bool(p and (os.path.exists(p) or "splendor1811" in p))

    def _run_generate(self, gen_kwargs: dict) -> Any:
        """Execute model.generate with inference_mode and persistent cuDNN fallback."""
        with torch.inference_mode():
            if self._cudnn_disabled:
                with torch.backends.cudnn.flags(enabled=False):
                    return self.model.generate(**gen_kwargs)

            try:
                return self.model.generate(**gen_kwargs)
            except RuntimeError as e:
                if "CUDNN" in str(e):
                    logger.warning(
                        f"⚠️ [TTS] cuDNN mismatch detected ({e}). "
                        f"Disabling cuDNN permanently for TTS to maintain optimal latency."
                    )
                    self._cudnn_disabled = True
                    with torch.backends.cudnn.flags(enabled=False):
                        return self.model.generate(**gen_kwargs)
                raise

    def _get_voice_clone_prompt(self, ref_audio_path: str, ref_text: str) -> Optional[Any]:
        """Fetch or create and cache a reusable VoiceClonePrompt for the given voice reference.

        Eliminates repeated disk I/O, audio decoding, silence trimming, and prompt embedding,
        saving ~70ms on every subsequent sentence.
        """
        if self.model is None or not hasattr(self.model, "create_voice_clone_prompt"):
            return None

        cache_key = (ref_audio_path, ref_text)
        if cache_key in self._voice_prompt_cache:
            return self._voice_prompt_cache[cache_key]

        try:
            prompt = self.model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_text)
            self._voice_prompt_cache[cache_key] = prompt
            logger.debug(f"🎙️ [TTS] Cached VoiceClonePrompt for reference voice: {Path(ref_audio_path).name}")
            return prompt
        except Exception as e:
            logger.debug(f"[TTS] Failed to create VoiceClonePrompt, falling back to raw ref_audio: {e}")
            return None

    def load_model(self) -> None:
        """Load and warm up the OmniVoice model in GPU memory with zero autograd overhead."""
        with self._init_lock:
            if self._is_loaded and self.model is not None:
                return

            try:
                from omnivoice import OmniVoice
            except ImportError:
                logger.error("[TTS] 'omnivoice' package is not installed.")
                raise RuntimeError("omnivoice package is not installed")

            model_path = self._resolve_model_path()
            logger.info(f"🔄 [TTS] Loading PyTorch OmniVoice model from: '{model_path}' on {self.device}...")
            t0 = time.perf_counter()

            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.model = OmniVoice.from_pretrained(
                model_path,
                device_map=self.device if self.device != "cpu" else None,
                dtype=torch_dtype,
            )

            # Warmup with small token under inference_mode (and pre-cache default voice prompt)
            try:
                ref_audio_path, ref_text = VoiceManager.resolve_voice(config.tts.default_voice)
                if os.path.exists(ref_audio_path):
                    cached_prompt = self._get_voice_clone_prompt(ref_audio_path, ref_text)
                    if cached_prompt is not None:
                        warmup_kwargs = {
                            "text": "Sẵn sàng.",
                            "voice_clone_prompt": cached_prompt,
                            "num_step": 4,
                        }
                    else:
                        warmup_kwargs = {
                            "text": "Sẵn sàng.",
                            "ref_audio": ref_audio_path,
                            "ref_text": ref_text,
                            "num_step": 4,
                        }
                    with self._infer_lock:
                        _ = self._run_generate(warmup_kwargs)
                        if torch.cuda.is_available() and "cuda" in str(self.device):
                            torch.cuda.synchronize()
            except Exception as e:
                logger.debug(f"[TTS] Warmup notice: {e}")

            self._is_loaded = True
            elapsed = time.perf_counter() - t0
            logger.info(f"✅ [TTS] PyTorch OmniVoice loaded and warmed up in {elapsed:.2f}s!")

    async def prewarm(self) -> bool:
        """Prewarm model in background."""
        if not self._is_loaded:
            await asyncio.to_thread(self.load_model)
        return self._is_loaded

    def synthesize_sync(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Synchronously synthesize text to speech via voice cloning."""
        if not text or not text.strip():
            return None, 0.0

        if not self._is_loaded:
            self.load_model()

        ref_audio_path, ref_text = VoiceManager.resolve_voice(voice_id or config.tts.default_voice)
        if not os.path.exists(ref_audio_path):
            logger.warning(f"Reference voice audio not found: {ref_audio_path}")
            return None, 0.0

        clean_text = text.strip()
        voice_name = Path(ref_audio_path).name

        # Use cached VoiceClonePrompt if supported (saves ~70ms audio preprocessing latency)
        cached_prompt = self._get_voice_clone_prompt(ref_audio_path, ref_text)
        if cached_prompt is not None:
            gen_kwargs = {
                "text": clean_text,
                "voice_clone_prompt": cached_prompt,
                "num_step": max(4, int(getattr(config.tts, "num_inference_steps", 8))),
            }
        else:
            gen_kwargs = {
                "text": clean_text,
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
                "num_step": max(4, int(getattr(config.tts, "num_inference_steps", 8))),
            }

        t_start = time.perf_counter()
        with self._infer_lock:
            audio_output = self._run_generate(gen_kwargs)
            if torch.cuda.is_available() and "cuda" in str(self.device):
                torch.cuda.synchronize()

        infer_time = time.perf_counter() - t_start

        # 1. Convert to 1D float32 numpy array
        audio_np = AudioProcessor.convert_to_numpy(audio_output)

        # 2. Time-stretching (Speed adjustment) without pitch shift
        effective_speed = float(speed if speed is not None else getattr(config.tts, "speed", 1.0) or 1.0)
        if abs(effective_speed - 1.0) >= 0.02 and len(audio_np) > 0:
            audio_np = AudioProcessor.apply_time_stretch(
                audio_np, speed=effective_speed, sample_rate=self.sample_rate
            )

        # 3. Audio Normalization & Volume Scaling
        vol = float(getattr(config.tts, "volume", 1.0) or 1.0)
        audio_np = AudioProcessor.normalize_audio(audio_np, volume=vol, target_peak=0.95)

        num_samples = len(audio_np)
        duration_sec = num_samples / self.sample_rate if self.sample_rate > 0 else 0.0

        # 4. Base64 WAV encoding
        audio_b64 = AudioProcessor.encode_wav_to_base64(audio_np, self.sample_rate)

        elapsed_ms = int(infer_time * 1000)
        rtf = (infer_time / duration_sec) if duration_sec > 0 else 0.0

        logger.info(
            f"🔊 [TTS CLONE] ({voice_name} in {elapsed_ms}ms, {duration_sec:.2f}s, speed={effective_speed:.2f}x, RTF: {rtf:.3f}): '{clean_text}'"
        )
        return audio_b64, duration_sec

    async def synthesize_clone(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Asynchronously synthesize text to speech in thread pool."""
        return await asyncio.to_thread(self.synthesize_sync, text, voice_id, speed)

    def unload_model(self) -> None:
        """Release the TTS model from memory and completely free GPU/VRAM atomically."""
        with self._init_lock:
            with self._infer_lock:
                self._is_loaded = False
                self.model = None
                self._voice_prompt_cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        logger.info("🗑️ [TTS] OmniVoice model unloaded and VRAM freed.")
