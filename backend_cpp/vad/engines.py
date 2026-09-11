"""Official VAD Inference Engines (Zero ONNX manual parsing).

Implements official integrations for:
1. FireRedStreamVad (fireredvad official package by Xiaohongshu / FireRedTeam)
2. Silero VAD (silero-vad official package with VADIterator)
3. FSMN-VAD (funasr official package by Alibaba DAMO Academy)

All engines are managed via VADEngineFactory and loaded from backend_cpp/models/.
"""

import abc
import importlib.resources as impresources
import logging
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from backend_cpp.config import MODELS_DIR, config
from backend_cpp.vad.stream_state import VADStreamState

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VADResult:
    """Standardized VAD frame detection result with backward-compatible tuple unpacking."""

    is_speech: bool
    probability: float
    event: Optional[str] = None  # 'START', 'END', or None

    def __iter__(self):
        """Enable tuple unpacking: is_speech, prob, event = result"""
        yield self.is_speech
        yield self.probability
        yield self.event

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        if idx == 0:
            return self.is_speech
        elif idx == 1:
            return self.probability
        elif idx == 2:
            return self.event
        raise IndexError("VADResult index out of range")



class SileroModelProbe:
    """Proxy wrapping Silero JIT model to capture exact continuous probabilities without duplicate inference."""

    def __init__(self, model: Any):
        self._model = model
        self.last_prob: float = 0.0

    def __call__(self, x: torch.Tensor, sr: int) -> torch.Tensor:
        out = self._model(x, sr)
        try:
            self.last_prob = float(out.item())
        except Exception:
            self.last_prob = 0.0
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "firered-vad": config.vad.threshold,
    "silero-vad": config.vad.threshold,
    "fsmn-vad": config.vad.threshold,
}
SUPPORTED_VAD_ENGINES: Tuple[str, ...] = tuple(DEFAULT_THRESHOLDS.keys())


class BaseVADEngine(abc.ABC):
    """Abstract base class for official streaming VAD inference engines."""

    name: str = ""
    default_threshold: float = config.vad.threshold
    native_frame_samples: int = 512  # Number of samples per native step

    @abc.abstractmethod
    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        """Create a fresh session-isolated VADStreamState."""
        pass

    @abc.abstractmethod
    def is_speech(
        self,
        chunk_float32: np.ndarray,
        state: VADStreamState,
        threshold: float,
    ) -> VADResult:
        """Process audio chunk [-1.0, 1.0] and update state in-place.

        Returns:
            VADResult containing is_speech, probability, and optional_event ('START', 'END', None).
        """
        pass


class FireRedOfficialVADEngine(BaseVADEngine):
    """Official FireRed Stream-VAD engine powered by the fireredvad package."""

    name = "firered-vad"
    default_threshold = config.vad.firered.threshold
    native_frame_samples = 400  # 25ms @ 16kHz

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self._ensure_model_files()

        from fireredvad.core.audio_feat import AudioFeat
        from fireredvad.core.detect_model import DetectModel

        cmvn_path = str(self.model_dir / "cmvn.ark")
        self.audio_feat = AudioFeat(cmvn_path)
        self.vad_model = DetectModel.from_pretrained(str(self.model_dir))
        self.vad_model.eval()
        self.vad_model.cpu()
        logger.info(f"🔊 Loaded Official FireRed Stream-VAD Engine from: {self.model_dir}")

    def _ensure_model_files(self) -> None:
        """Ensure Stream-VAD model weights and cmvn.ark exist in backend_cpp/models."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        cmvn_file = self.model_dir / "cmvn.ark"
        model_file = self.model_dir / "model.pth.tar"

        if not cmvn_file.exists() or not model_file.exists():
            from huggingface_hub import hf_hub_download
            logger.info("Downloading official FireRed Stream-VAD model files to backend_cpp/models...")
            parent_dir = self.model_dir.parent
            hf_hub_download("FireRedTeam/FireRedVAD", "Stream-VAD/cmvn.ark", local_dir=str(parent_dir))
            hf_hub_download("FireRedTeam/FireRedVAD", "Stream-VAD/model.pth.tar", local_dir=str(parent_dir))

    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        from fireredvad.core.stream_vad_postprocessor import StreamVadPostprocessor

        cfg = config.vad.firered
        active_thresh = threshold if threshold is not None else cfg.threshold

        state = VADStreamState()
        state.firered_postprocessor = StreamVadPostprocessor(
            smooth_window_size=cfg.smooth_window_size,
            speech_threshold=active_thresh,
            pad_start_frame=cfg.pad_start_frame,
            min_speech_frame=cfg.min_speech_frame,
            max_speech_frame=2000,
            min_silence_frame=cfg.min_silence_frame,
        )
        state.firered_caches = None
        return state

    def is_speech(
        self,
        chunk_float32: np.ndarray,
        state: VADStreamState,
        threshold: float,
    ) -> VADResult:

        if state.firered_postprocessor is None:
            init_s = self.create_initial_state(threshold)
            state.firered_postprocessor = init_s.firered_postprocessor
            state.firered_caches = init_s.firered_caches

        # Ensure threshold is up to date in postprocessor (live dynamic update)
        if threshold is not None and state.firered_postprocessor.speech_threshold != threshold:
            state.firered_postprocessor.speech_threshold = float(threshold)

        # Convert float32 [-1.0, 1.0] to int16 format expected by fireredvad
        if len(chunk_float32) != self.native_frame_samples:
            if len(chunk_float32) < self.native_frame_samples:
                chunk_float32 = np.pad(chunk_float32, (0, self.native_frame_samples - len(chunk_float32)))
            else:
                chunk_float32 = chunk_float32[: self.native_frame_samples]

        chunk_int16 = (np.clip(chunk_float32, -1.0, 1.0) * 32767.0).astype(np.int16)

        feat, _ = self.audio_feat.extract(chunk_int16)
        with torch.no_grad():
            probs, state.firered_caches = self.vad_model.forward(
                feat.unsqueeze(0), caches=state.firered_caches
            )

        raw_prob = probs.squeeze().tolist()
        if isinstance(raw_prob, list):
            raw_prob = float(raw_prob[-1]) if raw_prob else 0.0
        else:
            raw_prob = float(raw_prob)

        frame_result = state.firered_postprocessor.process_one_frame(raw_prob)

        event = None
        if frame_result.is_speech_start:
            event = "START"
        elif frame_result.is_speech_end:
            event = "END"

        return VADResult(
            is_speech=bool(frame_result.is_speech),
            probability=float(frame_result.smoothed_prob),
            event=event,
        )


class SileroOfficialVADEngine(BaseVADEngine):
    """Official Silero VAD v5 engine powered by the silero-vad package."""

    name = "silero-vad"
    default_threshold = config.vad.silero.threshold
    native_frame_samples = 512  # 32ms @ 16kHz

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self._ensure_model_file()

        from silero_vad.utils_vad import init_jit_model
        # Pre-verify model loads cleanly
        self._prototype_model = init_jit_model(str(self.model_path))
        logger.info(f"🔊 Loaded Official Silero VAD JIT Model from: {self.model_path}")

    def _ensure_model_file(self) -> None:
        """Ensure silero_vad.jit exists in backend_cpp/models."""
        if not self.model_path.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                src = str(impresources.files("silero_vad.data").joinpath("silero_vad.jit"))
                shutil.copy(src, self.model_path)
            except Exception as e:
                logger.warning(f"Could not copy bundled silero_vad.jit: {e}. Downloading...")
                import torch
                torch.hub.download_url_to_file(
                    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit",
                    str(self.model_path),
                )

    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        from silero_vad.utils_vad import init_jit_model
        from silero_vad import VADIterator

        cfg = config.vad.silero
        active_thresh = threshold if threshold is not None else config.vad.threshold

        state = VADStreamState()
        # Session-isolated JIT model clone with probability probe (~1.5MB RAM)
        state.silero_model = init_jit_model(str(self.model_path))
        state.silero_probe = SileroModelProbe(state.silero_model)
        state.silero_iterator = VADIterator(
            model=state.silero_probe,
            threshold=active_thresh,
            sampling_rate=16000,
            min_silence_duration_ms=cfg.min_silence_duration_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )
        return state

    def is_speech(
        self,
        chunk_float32: np.ndarray,
        state: VADStreamState,
        threshold: float,
    ) -> VADResult:
        if state.silero_iterator is None or state.silero_model is None:
            init_s = self.create_initial_state(threshold)
            state.silero_model = init_s.silero_model
            state.silero_probe = init_s.silero_probe
            state.silero_iterator = init_s.silero_iterator

        if len(chunk_float32) != self.native_frame_samples:
            if len(chunk_float32) < self.native_frame_samples:
                chunk_float32 = np.pad(chunk_float32, (0, self.native_frame_samples - len(chunk_float32)))
            else:
                chunk_float32 = chunk_float32[: self.native_frame_samples]

        tensor_chunk = torch.from_numpy(chunk_float32).float()

        # Update threshold if changed (live dynamic update)
        if threshold is not None and state.silero_iterator.threshold != threshold:
            state.silero_iterator.threshold = float(threshold)

        with torch.no_grad():
            res = state.silero_iterator(tensor_chunk)

        event = None
        if res:
            if "start" in res:
                event = "START"
            elif "end" in res:
                event = "END"

        # Speech active if iterator is currently triggered
        is_speech_active = bool(state.silero_iterator.triggered)
        # Real continuous probability captured by model probe
        prob = state.silero_probe.last_prob if state.silero_probe is not None else (0.9 if is_speech_active else 0.1)

        return VADResult(is_speech=is_speech_active, probability=prob, event=event)


class FsmnOfficialVADEngine(BaseVADEngine):
    """Official Alibaba DAMO FunASR FSMN-VAD engine powered by the funasr package."""

    name = "fsmn-vad"
    default_threshold = config.vad.fsmn.speech_noise_thres
    native_frame_samples = 960  # 60ms @ 16kHz

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self._ensure_model_files()

        from funasr import AutoModel
        self.model = AutoModel(
            model=str(self.model_dir),
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )


        cfg = config.vad.fsmn
        self.model.model.vad_opts.output_frame_probs = True
        self.model.model.vad_opts.speech_noise_thres = float(config.vad.threshold)
        self.model.model.vad_opts.max_end_silence_time = int(cfg.max_end_silence_time)
        self.model.model.vad_opts.speech_to_sil_time_thres = int(cfg.speech_to_sil_time_thres)
        self.model.model.vad_opts.sil_to_speech_time_thres = int(cfg.sil_to_speech_time_thres)

        logger.info(f"🔊 Loaded Official FSMN-VAD Engine from: {self.model_dir}")

    def _ensure_model_files(self) -> None:
        """Ensure FSMN-VAD model weights and configs exist in backend_cpp/models/fsmn_vad."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        required = ["model.pt", "am.mvn", "config.yaml", "configuration.json"]
        if not all((self.model_dir / f).exists() for f in required):
            from huggingface_hub import snapshot_download
            logger.info("Downloading official FSMN-VAD model files to backend_cpp/models/fsmn_vad...")
            snapshot_download("funasr/fsmn-vad", local_dir=str(self.model_dir))

    def create_initial_state(self, threshold: Optional[float] = None) -> VADStreamState:
        cfg = config.vad.fsmn
        active_thresh = threshold if threshold is not None else cfg.speech_noise_thres
        state = VADStreamState()
        state.fsmn_cache = {}
        self.model.model.init_cache(
            state.fsmn_cache,
            speech_noise_thres=float(active_thresh),
            max_end_silence_time=int(cfg.max_end_silence_time),
            speech_to_sil_time_thres=int(cfg.speech_to_sil_time_thres),
            sil_to_speech_time_thres=int(cfg.sil_to_speech_time_thres),
        )
        state.fsmn_in_speech = False
        return state

    def is_speech(
        self,
        chunk_float32: np.ndarray,
        state: VADStreamState,
        threshold: float,
    ) -> VADResult:
        if not state.fsmn_cache:
            init_s = self.create_initial_state(threshold)
            state.fsmn_cache = init_s.fsmn_cache

        # Ensure threshold is up to date in cache stats (live dynamic update)
        if threshold is not None and "stats" in state.fsmn_cache:
            stats = state.fsmn_cache["stats"]
            if getattr(stats, "speech_noise_thres", None) != threshold:
                stats.speech_noise_thres = float(threshold)

        if len(chunk_float32) != self.native_frame_samples:
            if len(chunk_float32) < self.native_frame_samples:
                chunk_float32 = np.pad(chunk_float32, (0, self.native_frame_samples - len(chunk_float32)))
            else:
                chunk_float32 = chunk_float32[: self.native_frame_samples]

        tensor_chunk = torch.from_numpy(chunk_float32).float()

        res = self.model.generate(
            input=[tensor_chunk],
            cache=state.fsmn_cache,
            is_final=False,
            chunk_size=60,
            dynamic_silence=False,
            disable_pbar=True,
            disable_log=True,
        )


        signals = res[0].get("value", []) if res else []
        event = None

        for sig in signals:
            if sig[0] >= 0 and sig[1] == -1:
                event = "START"
                state.fsmn_in_speech = True
            elif sig[0] == -1 and sig[1] >= 0:
                event = "END"
                state.fsmn_in_speech = False
            elif sig[0] >= 0 and sig[1] >= 0:
                event = "START"
                state.fsmn_in_speech = False

        # Frame speech state: active speech segment without accumulated silence
        stats = state.fsmn_cache.get("stats")
        sil_cnt = getattr(stats, "continous_silence_frame_count", 0) if stats else 0
        is_speech_frame = bool(state.fsmn_in_speech and sil_cnt == 0)

        # Extract exact continuous frame probability from frame_probs if available
        prob = 0.9 if is_speech_frame else 0.1
        extracted_real_prob = False
        if stats is not None and getattr(stats, "frame_probs", None):
            last_fp = stats.frame_probs[-1]
            if hasattr(last_fp, "score"):
                prob = float(last_fp.score)
                extracted_real_prob = True

        if not extracted_real_prob and not getattr(self, "_warned_frame_probs", False):
            logger.warning(
                "FSMN-VAD: Could not extract continuous score from stats.frame_probs; falling back to binary probability."
            )
            self._warned_frame_probs = True

        return VADResult(is_speech=is_speech_frame, probability=prob, event=event)



class VADEngineFactory:
    """Thread-safe factory and pool for official VAD inference engines."""

    _engines: Dict[str, BaseVADEngine] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_engine(cls, engine_name: str) -> BaseVADEngine:
        """Get or initialize a shared official VAD engine singleton.

        An unknown name falls back to the current default WITH A WARNING. It used to fall back
        to ``firered-vad`` silently, which is the most expensive engine (~8x silero), so a
        config typo quietly cost CPU and was indistinguishable from a deliberate choice in the
        logs.
        """
        engine = (engine_name or "fsmn-vad").lower().strip()
        if engine not in DEFAULT_THRESHOLDS:
            logger.warning(
                "Unknown VAD engine %r requested; falling back to 'fsmn-vad' "
                "(supported: %s)",
                engine,
                ", ".join(sorted(DEFAULT_THRESHOLDS)),
            )
            engine = "fsmn-vad"

        with cls._lock:
            if engine in cls._engines:
                return cls._engines[engine]

            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            if engine == "firered-vad":
                model_dir = MODELS_DIR / "firered_stream" / "Stream-VAD"
                instance = FireRedOfficialVADEngine(model_dir)

            elif engine == "silero-vad":
                model_path = MODELS_DIR / "silero_vad.jit"
                instance = SileroOfficialVADEngine(model_path)

            else:  # fsmn-vad
                model_dir = MODELS_DIR / "fsmn_vad"
                instance = FsmnOfficialVADEngine(model_dir)

            cls._engines[engine] = instance
            return instance

    @classmethod
    def reset_pool(cls) -> None:
        """Clear cached engine singletons (useful for testing or reloading)."""
        with cls._lock:
            cls._engines.clear()


# Backward compatibility aliases
SileroVADEngine = SileroOfficialVADEngine
FsmnVADEngine = FsmnOfficialVADEngine
FireRedVADEngine = FireRedOfficialVADEngine
