"""Python ctypes C-ABI Binding cho thư viện C++ omnivoice.cpp.

Cung cấp:
- Đầy đủ kiểu dữ liệu cấu trúc POD (ov_init_params, ov_tts_params, ov_audio).
- Helper đọc và giải nén tệp .rvq (11-bit packed LSB-first) sang mảng numpy int32.
- Class quản lý Context: OmniVoiceContext (thread-safe, zero-copy, non-blocking).
"""

import ctypes
from ctypes import (
    c_int,
    c_float,
    c_char_p,
    c_bool,
    c_uint64,
    c_void_p,
    c_int32,
    POINTER,
    Structure,
    CFUNCTYPE,
)
import os
from pathlib import Path
import threading
from typing import Optional, Tuple, Union, List
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger("tts.bindings")

# ABI Version của omnivoice.h
OV_ABI_VERSION = 3


class OVInitParams(Structure):
    _fields_ = [
        ("abi_version", c_int),
        ("model_path", c_char_p),
        ("codec_path", c_char_p),
        ("use_fa", c_bool),
        ("clamp_fp16", c_bool),
    ]


class OVAudio(Structure):
    _fields_ = [
        ("samples", POINTER(c_float)),
        ("n_samples", c_int),
        ("sample_rate", c_int),
        ("channels", c_int),
    ]


OV_CANCEL_CB = CFUNCTYPE(c_bool, c_void_p)
OV_CHUNK_CB = CFUNCTYPE(c_bool, POINTER(c_float), c_int, c_void_p)
OV_LOG_CB = CFUNCTYPE(None, c_int, c_char_p, c_void_p)


class OVTTSParams(Structure):
    _fields_ = [
        ("abi_version", c_int),
        ("text", c_char_p),
        ("lang", c_char_p),
        ("instruct", c_char_p),
        ("T_override", c_int),
        ("chunk_duration_sec", c_float),
        ("chunk_threshold_sec", c_float),
        ("denoise", c_bool),
        ("preprocess_prompt", c_bool),
        ("mg_num_step", c_int),
        ("mg_guidance_scale", c_float),
        ("mg_t_shift", c_float),
        ("mg_layer_penalty_factor", c_float),
        ("mg_position_temperature", c_float),
        ("mg_class_temperature", c_float),
        ("mg_seed", c_uint64),
        ("ref_audio_tokens", POINTER(c_int32)),
        ("ref_T", c_int),
        ("ref_audio_24k", POINTER(c_float)),
        ("ref_n_samples", c_int),
        ("ref_text", c_char_p),
        ("dump_dir", c_char_p),
        ("cancel", OV_CANCEL_CB),
        ("cancel_user_data", c_void_p),
        ("on_chunk", OV_CHUNK_CB),
        ("on_chunk_user_data", c_void_p),
        ("postproc", c_bool),
    ]


def _find_omnivoice_dll() -> Optional[str]:
    """Tìm đường dẫn tệp omnivoice.dll trên hệ thống."""
    here = Path(__file__).resolve().parent
    root = here.parent.parent

    candidate_paths = [
        root / "backend" / "bin" / "omnivoice.dll",
        root / "external" / "omnivoice.cpp" / "build" / "Release" / "omnivoice.dll",
        root / "external" / "omnivoice.cpp" / "build" / "omnivoice.dll",
        root / "backend" / "omnivoice.dll",
    ]

    for p in candidate_paths:
        if p.exists():
            return str(p)
    return None


def unpack_rvq_file(path: Union[str, Path], k: int = 8, code_bits: int = 11) -> Tuple[np.ndarray, int]:
    """Giải nén tệp .rvq thành mảng int32 [K * T] và số frames T."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp RVQ: {path}")

    raw_bytes = p.read_bytes()
    size = len(raw_bytes)
    if size == 0:
        raise ValueError("Tệp RVQ rỗng")

    total_bits = size * 8
    n_codes = total_bits // code_bits
    if n_codes == 0 or (n_codes % k) != 0:
        raise ValueError(f"Dữ liệu RVQ không hợp lệ (n_codes={n_codes}, K={k})")

    mask = (1 << code_bits) - 1
    codes = np.zeros(n_codes, dtype=np.int32)

    acc = 0
    bits_in_acc = 0
    in_pos = 0

    for i in range(n_codes):
        while bits_in_acc < code_bits and in_pos < size:
            acc |= int(raw_bytes[in_pos]) << bits_in_acc
            in_pos += 1
            bits_in_acc += 8
        codes[i] = acc & mask
        acc >>= code_bits
        bits_in_acc -= code_bits

    n_frames = n_codes // k
    return codes, n_frames


class OmniVoiceCppEngine:
    """Wrapper Python gọi C-ABI omnivoice.dll (Thread-safe)."""

    def __init__(self, dll_path: Optional[str] = None):
        self.dll_path = dll_path or _find_omnivoice_dll()
        if not self.dll_path or not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"Không tìm thấy omnivoice.dll! Vui lòng biên dịch trước.")

        # Thêm thư mục DLL vào path tìm kiếm
        dll_dir = os.path.dirname(os.path.abspath(self.dll_path))
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

        self._lib = ctypes.CDLL(self.dll_path)
        self._setup_function_signatures()
        self._ctx = None
        self._lock = threading.RLock()
        self._is_initialized = False

    def _setup_function_signatures(self) -> None:
        """Khai báo kiểu tham số và kiểu trả về cho C functions."""
        # const char * ov_version(void);
        self._lib.ov_version.argtypes = []
        self._lib.ov_version.restype = c_char_p

        # const char * ov_last_error(void);
        self._lib.ov_last_error.argtypes = []
        self._lib.ov_last_error.restype = c_char_p

        # void ov_init_default_params(struct ov_init_params * p);
        self._lib.ov_init_default_params.argtypes = [POINTER(OVInitParams)]
        self._lib.ov_init_default_params.restype = None

        # struct ov_context * ov_init(const struct ov_init_params * params);
        self._lib.ov_init.argtypes = [POINTER(OVInitParams)]
        self._lib.ov_init.restype = c_void_p

        # void ov_free(struct ov_context * ov);
        self._lib.ov_free.argtypes = [c_void_p]
        self._lib.ov_free.restype = None

        # void ov_tts_default_params(struct ov_tts_params * p);
        self._lib.ov_tts_default_params.argtypes = [POINTER(OVTTSParams)]
        self._lib.ov_tts_default_params.restype = None

        # int ov_synthesize(struct ov_context * ov, const struct ov_tts_params * params, struct ov_audio * audio);
        self._lib.ov_synthesize.argtypes = [c_void_p, POINTER(OVTTSParams), POINTER(OVAudio)]
        self._lib.ov_synthesize.restype = c_int

        # void ov_audio_free(struct ov_audio * a);
        self._lib.ov_audio_free.argtypes = [POINTER(OVAudio)]
        self._lib.ov_audio_free.restype = None

    def get_version(self) -> str:
        """Lấy phiên bản build của omnivoice.cpp."""
        v = self._lib.ov_version()
        return v.decode("utf-8") if v else "unknown"

    def init_context(
        self,
        model_path: str,
        codec_path: str,
        use_fa: bool = True,
        clamp_fp16: bool = False,
    ) -> bool:
        """Khởi tạo context mô hình OmniVoice C++."""
        with self._lock:
            if self._is_initialized and self._ctx is not None:
                return True

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Không tìm thấy model GGUF: {model_path}")
            if not os.path.exists(codec_path):
                raise FileNotFoundError(f"Không tìm thấy codec GGUF: {codec_path}")

            iparams = OVInitParams()
            self._lib.ov_init_default_params(ctypes.byref(iparams))
            iparams.abi_version = OV_ABI_VERSION
            iparams.model_path = model_path.encode("utf-8")
            iparams.codec_path = codec_path.encode("utf-8")
            iparams.use_fa = use_fa
            iparams.clamp_fp16 = clamp_fp16

            logger.info(f"Đang nạp omnivoice.cpp context từ: {model_path}")
            ctx = self._lib.ov_init(ctypes.byref(iparams))
            if not ctx:
                err = self._lib.ov_last_error()
                err_msg = err.decode("utf-8") if err else "Lỗi không xác định khi gọi ov_init"
                logger.error(f"ov_init thất bại: {err_msg}")
                raise RuntimeError(f"ov_init thất bại: {err_msg}")

            self._ctx = ctx
            self._is_initialized = True
            logger.info(f"✅ Nạp thành công omnivoice.cpp C-ABI engine (v{self.get_version()})")
            return True

    def synthesize(
        self,
        text: str,
        ref_rvq_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_wav_path: Optional[str] = None,
        lang: str = "",
        instruct: str = "",
        num_steps: int = 8,
        guidance_scale: float = 2.0,
        seed: int = 42,
    ) -> Tuple[np.ndarray, int]:
        """Tổng hợp giọng nói từ văn bản (Thread-safe).

        Trả về:
            (samples_np, sample_rate): Mảng float32 1 chiều và tần số lấy mẫu (24000).
        """
        if not text or not text.strip():
            return np.zeros((0,), dtype=np.float32), 24000

        with self._lock:
            if not self._is_initialized or self._ctx is None:
                raise RuntimeError("OmniVoiceCppEngine chưa được khởi tạo qua init_context!")

            tparams = OVTTSParams()
            self._lib.ov_tts_default_params(ctypes.byref(tparams))
            tparams.abi_version = OV_ABI_VERSION
            tparams.text = text.strip().encode("utf-8")
            tparams.lang = (lang or "").encode("utf-8")
            tparams.instruct = (instruct or "").encode("utf-8")
            tparams.mg_num_step = max(1, int(num_steps))
            tparams.mg_guidance_scale = float(guidance_scale)
            tparams.mg_seed = c_uint64(seed)

            # Cấu hình Voice Cloning qua RVQ (ưu tiên) hoặc WAV
            rvq_codes_holder = None
            if ref_rvq_path and os.path.exists(ref_rvq_path):
                codes, n_frames = unpack_rvq_file(ref_rvq_path)
                rvq_codes_holder = (c_int32 * len(codes))(*codes)
                tparams.ref_audio_tokens = rvq_codes_holder
                tparams.ref_T = n_frames
                if ref_text:
                    tparams.ref_text = ref_text.encode("utf-8")
                tparams.denoise = True
                tparams.preprocess_prompt = True

            elif ref_wav_path and os.path.exists(ref_wav_path):
                # Fallback qua file WAV nếu cần
                import soundfile as sf
                audio_data, sr = sf.read(ref_wav_path, dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                audio_c = (c_float * len(audio_data))(*audio_data)
                tparams.ref_audio_24k = audio_c
                tparams.ref_n_samples = len(audio_data)
                if ref_text:
                    tparams.ref_text = ref_text.encode("utf-8")
                tparams.denoise = True
                tparams.preprocess_prompt = True

            out_audio = OVAudio()
            rc = self._lib.ov_synthesize(self._ctx, ctypes.byref(tparams), ctypes.byref(out_audio))
            if rc != 0:
                err = self._lib.ov_last_error()
                err_msg = err.decode("utf-8") if err else f"Lỗi mã {rc}"
                logger.error(f"ov_synthesize thất bại: {err_msg}")
                raise RuntimeError(f"ov_synthesize thất bại: {err_msg}")

            try:
                n_samples = out_audio.n_samples
                sr = out_audio.sample_rate or 24000
                if n_samples > 0 and bool(out_audio.samples):
                    # Zero-copy hoặc copy mảng float32 vào numpy
                    buf = np.ctypeslib.as_array(out_audio.samples, shape=(n_samples,))
                    result_np = buf.copy()
                else:
                    result_np = np.zeros((0,), dtype=np.float32)
            finally:
                self._lib.ov_audio_free(ctypes.byref(out_audio))

            return result_np, sr

    def close(self) -> None:
        """Giải phóng hoàn toàn context và bộ nhớ."""
        with self._lock:
            if self._ctx is not None:
                try:
                    self._lib.ov_free(self._ctx)
                except Exception as e:
                    logger.debug(f"ov_free notice: {e}")
                self._ctx = None
            self._is_initialized = False
            logger.info("🗑️ Đã giải phóng OmniVoiceCppEngine context.")
