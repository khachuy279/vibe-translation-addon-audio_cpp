"""Engine tổng hợp giọng nói Voice Cloning OmniVoice (C++ GGUF Native / PyTorch Fallback).

Hỗ trợ:
- Tích hợp Native omnivoice.cpp C-ABI binding qua ctypes (tiết kiệm ~60% VRAM, khởi động < 300ms).
- Sử dụng mô hình GGUF: omnivoice-base-Q8_0.gguf + omnivoice-tokenizer-F32.gguf.
- Tự động nạp Voice Clone từ tệp .rvq siêu nhẹ (3.2 KB) bỏ qua codec encoding runtime.
- Hỗ trợ Fallback sang PyTorch Native nếu cần.
- Tối ưu hóa 1 session: Thread-safe double-checked locking với RLock, zero VRAM spikes.
"""

import asyncio
import gc
import os
from pathlib import Path
import threading
import time
from typing import Optional, Tuple, Any, Dict
import numpy as np

from backend.config import config, MODELS_DIR
from backend.tts.base import BaseTTSEngine
from backend.tts.audio_processor import AudioProcessor
from backend.tts.voice_manager import VoiceManager
from backend.tts.bindings import OmniVoiceCppEngine
from backend.utils.logger import get_logger

logger = get_logger("tts.omnivoice")


class OmniVoiceTTS(BaseTTSEngine):
    """Singleton TTS Engine quản lý tổng hợp giọng nói độ trễ thấp."""

    _instance: Optional["OmniVoiceTTS"] = None
    _lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "OmniVoiceTTS":
        """Truy cập Singleton instance thread-safe."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = OmniVoiceTTS()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Giải phóng hoàn toàn Singleton instance khi shutdown."""
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance.unload_model()
                except Exception as e:
                    logger.debug(f"TTS unload_model notice: {e}")
                cls._instance = None

    def __init__(self):
        self.sample_rate = 24000
        self.engine_type = getattr(config.tts, "engine", "omnivoice.cpp")
        self.cpp_engine: Optional[OmniVoiceCppEngine] = None
        self.pytorch_model = None
        self._is_loaded = False
        self._voice_prompt_cache: Dict[Tuple[str, str], Any] = {}
        self._init_lock = threading.RLock()
        self._infer_lock = threading.RLock()

    def _resolve_cpp_model_paths(self) -> Tuple[str, str]:
        """Xác định đường dẫn mô hình GGUF và Tokenizer."""
        model_name = getattr(config.tts, "model", "omnivoice-base-Q8_0.gguf")
        codec_name = getattr(config.tts, "codec_model", "omnivoice-tokenizer-F32.gguf")

        # 1. Đường dẫn tuyệt đối nếu tồn tại
        if os.path.exists(model_name) and os.path.exists(codec_name):
            return model_name, codec_name

        # 2. Tìm trong MODELS_DIR
        model_path = str(MODELS_DIR / Path(model_name).name)
        codec_path = str(MODELS_DIR / Path(codec_name).name)
        return model_path, codec_path

    def is_model_ready(self) -> bool:
        """Kiểm tra xem mô hình có sẵn sàng hay không."""
        model_p, codec_p = self._resolve_cpp_model_paths()
        return bool(os.path.exists(model_p) and os.path.exists(codec_p))

    def load_model(self) -> None:
        """Nạp mô hình OmniVoice C++ GGUF (hoặc PyTorch) vào bộ nhớ."""
        with self._init_lock:
            if self._is_loaded:
                return

            t0 = time.perf_counter()
            model_path, codec_path = self._resolve_cpp_model_paths()

            if os.path.exists(model_path) and os.path.exists(codec_path):
                # 1. Nạp qua C++ Native omnivoice.cpp
                logger.info(
                    f"🔄 Đang nạp omnivoice.cpp GGUF C-ABI Engine ({Path(model_path).name} + {Path(codec_path).name})..."
                )
                self.cpp_engine = OmniVoiceCppEngine()
                self.cpp_engine.init_context(
                    model_path=model_path,
                    codec_path=codec_path,
                    use_fa=True,
                )
                self.engine_type = "omnivoice.cpp"

                # Warmup nhẹ qua voice sample mặc định
                try:
                    ref_audio, ref_txt, ref_rvq = VoiceManager.resolve_voice_extended(config.tts.default_voice)
                    if ref_rvq or ref_audio:
                        with self._infer_lock:
                            _, _ = self.cpp_engine.synthesize(
                                text="Sẵn sàng.",
                                ref_rvq_path=ref_rvq,
                                ref_text=ref_txt,
                                ref_wav_path=ref_audio if not ref_rvq else None,
                                num_steps=2,
                            )
                except Exception as e:
                    logger.debug(f"Warmup notice: {e}")

                self._is_loaded = True
                elapsed = time.perf_counter() - t0
                logger.info(f"✅ omnivoice.cpp GGUF C-ABI Engine đã nạp và warm-up hoàn tất trong {elapsed:.2f}s!")

            else:
                # 2. Fallback sang PyTorch Native nếu file GGUF không tồn tại
                logger.info("🔄 GGUF không tìm thấy, fallback sang PyTorch OmniVoice Native...")
                try:
                    import torch
                    from omnivoice import OmniVoice

                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    self.pytorch_model = OmniVoice.from_pretrained(
                        "splendor1811/omnivoice-vietnamese",
                        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                        dtype=torch_dtype,
                    )
                    self.engine_type = "omnivoice"
                    self._is_loaded = True
                    elapsed = time.perf_counter() - t0
                    logger.info(f"✅ PyTorch OmniVoice đã nạp hoàn tất trong {elapsed:.2f}s!")
                except Exception as e:
                    logger.error(f"Không thể nạp PyTorch OmniVoice: {e}")
                    raise RuntimeError(f"Không thể khởi tạo TTS Engine: {e}")

    async def prewarm(self) -> bool:
        """Khởi động và nạp sẵn mô hình trong tiến trình nền."""
        if not self._is_loaded:
            await asyncio.to_thread(self.load_model)
        return self._is_loaded

    def synthesize_sync(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Tổng hợp giọng nói đồng bộ qua cơ chế Voice Cloning."""
        if not text or not text.strip():
            return None, 0.0

        if not self._is_loaded:
            self.load_model()

        clean_text = text.strip()
        ref_audio_path, ref_text, ref_rvq_path = VoiceManager.resolve_voice_extended(
            voice_id or config.tts.default_voice
        )
        voice_name = Path(ref_audio_path).name if ref_audio_path else "default"
        num_steps = max(1, int(getattr(config.tts, "num_inference_steps", 8)))

        t_start = time.perf_counter()

        # 1. Tổng hợp âm thanh qua C++ Engine hoặc PyTorch Fallback
        if self.cpp_engine is not None:
            with self._infer_lock:
                audio_np, _ = self.cpp_engine.synthesize(
                    text=clean_text,
                    ref_rvq_path=ref_rvq_path,
                    ref_text=ref_text,
                    ref_wav_path=ref_audio_path if not ref_rvq_path else None,
                    num_steps=num_steps,
                )
        elif self.pytorch_model is not None:
            import torch
            gen_kwargs = {
                "text": clean_text,
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
                "num_step": num_steps,
            }
            with self._infer_lock:
                with torch.inference_mode():
                    audio_out = self.pytorch_model.generate(**gen_kwargs)
                    audio_np = AudioProcessor.convert_to_numpy(audio_out)
                    del audio_out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            logger.error("Không có TTS engine nào sẵn sàng!")
            return None, 0.0

        infer_time = time.perf_counter() - t_start

        # 2. Điều chỉnh tốc độ (Time-stretching) nếu cần
        effective_speed = float(speed if speed is not None else getattr(config.tts, "speed", 1.0) or 1.0)
        if abs(effective_speed - 1.0) >= 0.02 and len(audio_np) > 0:
            audio_np = AudioProcessor.apply_time_stretch(
                audio_np, speed=effective_speed, sample_rate=self.sample_rate
            )

        # 3. Chuẩn hóa đỉnh biên độ và âm lượng
        vol = float(getattr(config.tts, "volume", 1.0) or 1.0)
        audio_np = AudioProcessor.normalize_audio(audio_np, volume=vol, target_peak=0.95)

        num_samples = len(audio_np)
        duration_sec = num_samples / self.sample_rate if self.sample_rate > 0 else 0.0

        # 4. Mã hóa Base64 WAV PCM 16-bit
        audio_b64 = AudioProcessor.encode_wav_to_base64(audio_np, self.sample_rate)

        elapsed_ms = int(infer_time * 1000)
        rtf = (infer_time / duration_sec) if duration_sec > 0 else 0.0

        backend_tag = "omnivoice.cpp" if self.cpp_engine is not None else "PyTorch"
        logger.info(
            f"🔊 [TTS CLONE ({backend_tag})] ({voice_name} in {elapsed_ms}ms, {duration_sec:.2f}s, speed={effective_speed:.2f}x, RTF: {rtf:.3f}): '{clean_text}'"
        )
        return audio_b64, duration_sec

    async def synthesize_clone(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Tổng hợp giọng nói bất đồng bộ không làm block asyncio event loop."""
        return await asyncio.to_thread(self.synthesize_sync, text, voice_id, speed)

    def unload_model(self) -> None:
        """Giải phóng hoàn toàn mô hình khỏi RAM và VRAM GPU."""
        with self._init_lock:
            with self._infer_lock:
                self._is_loaded = False
                if self.cpp_engine is not None:
                    try:
                        self.cpp_engine.close()
                    except Exception:
                        pass
                    self.cpp_engine = None

                if self.pytorch_model is not None:
                    try:
                        del self.pytorch_model
                    except Exception:
                        pass
                    self.pytorch_model = None

                self._voice_prompt_cache.clear()
                gc.collect()
        logger.info("🗑️ [TTS] Đã giải phóng OmniVoice model và dọn sạch bộ nhớ.")
