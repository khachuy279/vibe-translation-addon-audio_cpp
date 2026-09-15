"""Engine tổng hợp giọng nói Voice Cloning OmniVoice PyTorch Native cho Backend.

Đặc điểm tối ưu:
- Singleton Pattern với Thread-safe double-checked locking.
- Tự động nạp mô hình từ HuggingFace cache cục bộ hoặc tải tự động.
- Caching VoiceClonePrompt: Tiết kiệm ~70ms tiền xử lý âm thanh mẫu cho mỗi câu.
- Cơ chế tự động xử lý cuDNN mismatch (cuDNN auto-fallback) để đảm bảo không bị lỗi crash.
- Thực thi hoàn toàn trong `torch.inference_mode()` loại bỏ overhead tính gradient.
- Đo đạc chi tiết RTF, thời gian xử lý (ms) và thời lượng audio sinh ra (s).
"""

import asyncio
import gc
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import numpy as np
import torch

from backend.config import config, MODELS_DIR
from backend.tts.base import BaseTTSEngine
from backend.tts.audio_processor import AudioProcessor
from backend.tts.voice_manager import VoiceManager
from backend.utils.logger import get_logger

logger = get_logger("tts.omnivoice")


class OmniVoiceTTS(BaseTTSEngine):
    """Singleton TTS Engine bao bọc PyTorch OmniVoice phục vụ tổng hợp giọng nói độ trễ thấp."""

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
        self.device = config.tts.device if hasattr(config, "tts") and config.tts.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._is_loaded = False
        self._cudnn_disabled = False
        self._voice_prompt_cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()  # LRU, max 8 entries
        self._voice_prompt_cache_max: int = 8
        self._init_lock = threading.RLock()
        self._infer_lock = threading.RLock()

    def _resolve_model_path(self) -> str:
        """Xác định đường dẫn mô hình (ưu tiên cache cục bộ, fallback HF repo id)."""
        cfg_model = config.tts.model if hasattr(config, "tts") and config.tts.model else "splendor1811/omnivoice-vietnamese"

        # 1. Đường dẫn tệp/thư mục cục bộ rõ ràng nếu tồn tại
        if cfg_model and os.path.exists(cfg_model):
            return cfg_model

        # 2. Kiểm tra trong MODELS_DIR / huggingface
        snap_dirs = [
            MODELS_DIR / "huggingface" / "models--splendor1811--omnivoice-vietnamese" / "snapshots",
        ]
        for s_dir in snap_dirs:
            if s_dir.exists():
                for child in s_dir.iterdir():
                    if child.is_dir() and (child / "config.json").exists():
                        return str(child)

        return cfg_model or "splendor1811/omnivoice-vietnamese"

    def is_model_ready(self) -> bool:
        """Kiểm tra xem mô hình có sẵn sàng hay không."""
        p = self._resolve_model_path()
        return bool(p and (os.path.exists(p) or "splendor1811" in p))

    def _run_generate(self, gen_kwargs: dict) -> Any:
        """Thực thi model.generate trong inference_mode và xử lý cuDNN an toàn."""
        with torch.inference_mode():
            if self._cudnn_disabled:
                with torch.backends.cudnn.flags(enabled=False):
                    return self.model.generate(**gen_kwargs)

            try:
                return self.model.generate(**gen_kwargs)
            except RuntimeError as e:
                if "CUDNN" in str(e):
                    logger.warning(
                        f"⚠️ Phát hiện xung đột cuDNN ({e}). "
                        f"Tự động tắt cuDNN cho TTS để duy trì độ ổn định tối đa."
                    )
                    self._cudnn_disabled = True
                    with torch.backends.cudnn.flags(enabled=False):
                        return self.model.generate(**gen_kwargs)
                raise

    def _get_voice_clone_prompt(self, ref_audio_path: str, ref_text: str) -> Optional[Any]:
        """Tạo hoặc lấy từ cache VoiceClonePrompt tái sử dụng cho mẫu giọng tham chiếu.

        Tiết kiệm ~70ms cho mỗi câu nói tiếp theo do không phải lặp lại Disk I/O,
        giải mã âm thanh và trích xuất embedding.
        """
        if self.model is None or not hasattr(self.model, "create_voice_clone_prompt"):
            return None

        cache_key = (ref_audio_path, ref_text)
        if cache_key in self._voice_prompt_cache:
            # LRU: promote to most-recently-used position
            self._voice_prompt_cache.move_to_end(cache_key)
            return self._voice_prompt_cache[cache_key]

        try:
            prompt = self.model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_text)
            self._voice_prompt_cache[cache_key] = prompt
            # LRU eviction: xoa entry cu nhat neu vuot maxsize
            if len(self._voice_prompt_cache) > self._voice_prompt_cache_max:
                evicted_key, _ = self._voice_prompt_cache.popitem(last=False)
                logger.debug(f"Evicted voice cache: {Path(evicted_key[0]).name}")
            logger.debug(f"Da luu cache VoiceClonePrompt cho: {Path(ref_audio_path).name} (cache={len(self._voice_prompt_cache)})")
            return prompt
        except Exception as e:
            logger.debug(f"Khong the tao VoiceClonePrompt cache ({e}), fallback sang raw ref_audio.")
            return None

    def load_model(self) -> None:
        """Nạp và warm-up mô hình OmniVoice vào GPU memory."""
        with self._init_lock:
            if self._is_loaded and self.model is not None:
                return

            try:
                from omnivoice import OmniVoice
            except ImportError:
                logger.error("Chưa cài đặt thư viện 'omnivoice'.")
                raise RuntimeError("Thư viện omnivoice chưa được cài đặt")

            model_path = self._resolve_model_path()
            logger.info(f"🔄 Đang nạp PyTorch OmniVoice model từ: '{model_path}' trên thiết bị {self.device}...")
            t0 = time.perf_counter()

            torch_dtype = torch.float16 if self.device != "cpu" and "cuda" in str(self.device) else torch.float32
            if torch.cuda.is_available() and "cuda" in str(self.device):
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass

            self.model = OmniVoice.from_pretrained(
                model_path,
                device_map=self.device if self.device != "cpu" else None,
                dtype=torch_dtype,
            )

            # Warmup với đoạn văn bản ngắn và tiền nạp cache VoiceClonePrompt
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
                logger.debug(f"Warmup notice: {e}")

            # Thu hồi bộ nhớ đệm sau nạp và warmup
            gc.collect()
            if torch.cuda.is_available() and "cuda" in str(self.device):
                torch.cuda.empty_cache()

            self._is_loaded = True
            elapsed = time.perf_counter() - t0
            logger.info(f"PyTorch OmniVoice đã nạp và warm-up hoàn tất trong {elapsed:.2f}s!")

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

        ref_audio_path, ref_text = VoiceManager.resolve_voice(voice_id or config.tts.default_voice)
        if not os.path.exists(ref_audio_path):
            logger.warning(f"Không tìm thấy file mẫu giọng: {ref_audio_path}")
            return None, 0.0

        clean_text = text.strip()
        voice_name = Path(ref_audio_path).name

        # Dùng VoiceClonePrompt đã được cache nếu khả dụng
        cached_prompt = self._get_voice_clone_prompt(ref_audio_path, ref_text)
        num_steps = max(4, int(getattr(config.tts, "num_inference_steps", 8)))
        if cached_prompt is not None:
            gen_kwargs = {
                "text": clean_text,
                "voice_clone_prompt": cached_prompt,
                "num_step": num_steps,
            }
        else:
            gen_kwargs = {
                "text": clean_text,
                "ref_audio": ref_audio_path,
                "ref_text": ref_text,
                "num_step": num_steps,
            }

        t_start = time.perf_counter()
        with self._infer_lock:
            audio_output = self._run_generate(gen_kwargs)
            if torch.cuda.is_available() and "cuda" in str(self.device):
                torch.cuda.synchronize()

        infer_time = time.perf_counter() - t_start

        # 1. Chuyển đổi sang mảng 1D float32 numpy
        audio_np = AudioProcessor.convert_to_numpy(audio_output)
        del audio_output

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
        """Tổng hợp giọng nói bất đồng bộ không làm block asyncio event loop."""
        return await asyncio.to_thread(self.synthesize_sync, text, voice_id, speed)

    def unload_model(self) -> None:
        """Giải phóng hoàn toàn mô hình khỏi RAM và VRAM GPU."""
        with self._init_lock:
            with self._infer_lock:
                self._is_loaded = False
                if self.model is not None:
                    try:
                        del self.model
                    except Exception:
                        pass
                self.model = None
                self._voice_prompt_cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
        logger.info("🗑️ [TTS] Đã giải phóng OmniVoice model và dọn sạch VRAM.")
