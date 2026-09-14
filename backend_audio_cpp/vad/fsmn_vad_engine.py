"""High-Performance Streaming FSMN-VAD Engine for backend_audio_cpp.

Zero PyTorch dependency, pure NumPy + ONNX Runtime implementation of Alibaba FunASR FSMN-VAD.
High sensitivity on whispered speech, ASMR, soft voice, and multilingual audio.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from backend_audio_cpp.vad.base_vad import (
    BaseVADEngine,
    BaseVADStreamState,
    VADConfig,
    VADResult,
)

logger = logging.getLogger("backend_audio_cpp.vad.fsmn")


def _create_mel_filterbank(sr: int = 16000, n_fft: int = 512, n_mels: int = 80, fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    """Create Kaldi/HTK-compatible triangular Mel filterbank matrix in pure NumPy."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fft_bins = np.floor((n_fft + 1) * hz_pts / sr).astype(np.int32)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for i in range(n_mels):
        left = fft_bins[i]
        center = fft_bins[i + 1]
        right = fft_bins[i + 2]

        if center > left:
            for j in range(left, center):
                weights[i, j] = (j - left) / (center - left)
        if right > center:
            for j in range(center, right):
                weights[i, j] = (right - j) / (right - center)

    return weights


class FsmnVADStreamState(BaseVADStreamState):
    """Session-isolated streaming recurrent state for FSMN-VAD."""

    def __init__(self, config: VADConfig):
        super().__init__(config)
        # 4 FSMN cache buffers with shape (1, 128, 19, 1)
        self.in_cache0 = np.zeros((1, 128, 19, 1), dtype=np.float32)
        self.in_cache1 = np.zeros((1, 128, 19, 1), dtype=np.float32)
        self.in_cache2 = np.zeros((1, 128, 19, 1), dtype=np.float32)
        self.in_cache3 = np.zeros((1, 128, 19, 1), dtype=np.float32)

        # Audio sliding context buffer for STFT framing (400 samples = 25ms @ 16kHz)
        self.audio_context = np.zeros(240, dtype=np.float32)
        # Mel sliding buffer for LFR (lfr_m=5 stacking)
        self.mel_history: list = []

    def reset(self) -> None:
        super().reset()
        self.in_cache0.fill(0.0)
        self.in_cache1.fill(0.0)
        self.in_cache2.fill(0.0)
        self.in_cache3.fill(0.0)
        self.audio_context.fill(0.0)
        self.mel_history.clear()


class FsmnVADEngine(BaseVADEngine):
    """High-performance FSMN-VAD engine with zero PyTorch dependency."""

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        models_dir = Path("backend_audio_cpp/models")
        self.model_path = Path(self.config.model_path) if self.config.model_path else (models_dir / "fsmn_vad.onnx")
        self.mvn_path = models_dir / "vad.mvn"

        if not self.model_path.exists():
            raise FileNotFoundError(f"FSMN-VAD model not found at: {self.model_path}")

        # Precompute Mel filterbank & Hamming window for Kaldi Fbank
        self.win_length = 400   # 25ms
        self.hop_length = 160   # 10ms
        self.n_fft = 512
        self.n_mels = 80
        self.window = np.hamming(self.win_length).astype(np.float32)
        self.mel_basis = _create_mel_filterbank(16000, self.n_fft, self.n_mels, 20.0, 8000.0)

        # Load CMVN shift & scale
        self.shift, self.scale = self._load_cmvn(self.mvn_path)

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.active_provider = self.session.get_providers()[0]
        self._log(f"Loaded FSMN-VAD from {self.model_path} [Provider: {self.active_provider}]", force_info=True)

    def _log(self, message: str, force_info: bool = False) -> None:
        if self.config.debug or force_info:
            logger.info(message)
        else:
            logger.debug(message)

    def _load_cmvn(self, mvn_file: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse Kaldi CMVN AddShift and Rescale parameters."""
        shift = np.zeros(400, dtype=np.float32)
        scale = np.ones(400, dtype=np.float32)
        if not mvn_file.exists():
            return shift, scale

        with open(mvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "<AddShift>" in line and i + 1 < len(lines):
                    parts = lines[i + 1].split("[")[1].split("]")[0].strip().split()
                    shift = np.array([float(x) for x in parts], dtype=np.float32)
                elif "<Rescale>" in line and i + 1 < len(lines):
                    parts = lines[i + 1].split("[")[1].split("]")[0].strip().split()
                    scale = np.array([float(x) for x in parts], dtype=np.float32)

        return shift, scale

    def create_state(self) -> FsmnVADStreamState:
        return FsmnVADStreamState(self.config)

    def _compute_fbank_frame(self, chunk_400: np.ndarray) -> np.ndarray:
        """Compute 80-dim log Mel energy for a single 400-sample (25ms) frame."""
        # Pre-emphasis (0.97)
        pre = np.empty_like(chunk_400)
        pre[0] = chunk_400[0]
        pre[1:] = chunk_400[1:] - 0.97 * chunk_400[:-1]

        # Windowing & Real FFT
        windowed = pre * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)
        power = np.abs(spec) ** 2

        # Mel filterbank dot product
        mel = np.dot(self.mel_basis, power)
        return np.log(np.maximum(mel, 1e-5)).astype(np.float32)

    def _infer_chunk(self, frame_float32: np.ndarray, state: FsmnVADStreamState) -> float:
        """Process 512 samples (32ms), extract streaming Fbank + LFR, and run ONNX model."""
        # Combine with previous 240 samples to form three 400-sample windows with hop 160
        # Total samples = 240 + 512 = 752 samples
        # Window 1: [0:400]
        # Window 2: [160:560]
        # Window 3: [320:720]
        # Remaining 32 samples + [480:720] -> 240 samples context for next frame
        full_audio = np.concatenate([state.audio_context, frame_float32])
        state.audio_context = full_audio[-240:]

        mel_frames = []
        for start_idx in range(0, len(full_audio) - self.win_length + 1, self.hop_length):
            w = full_audio[start_idx : start_idx + self.win_length]
            mel = self._compute_fbank_frame(w)
            mel_frames.append(mel)

        if not mel_frames:
            return 0.0

        for m in mel_frames:
            state.mel_history.append(m)
        if len(state.mel_history) > 10:
            state.mel_history = state.mel_history[-10:]

        # LFR stacking (5 frames = 400 dims)
        lfr_feats = []
        for i in range(len(mel_frames)):
            # Pick 5 frames from history
            hist_len = len(state.mel_history)
            frame_mels = []
            for k in range(5):
                pos = hist_len - len(mel_frames) + i + k
                idx = max(0, min(pos, hist_len - 1))
                frame_mels.append(state.mel_history[idx])
            lfr_feats.append(np.concatenate(frame_mels))

        lfr_arr = np.array(lfr_feats, dtype=np.float32)
        norm_feats = (lfr_arr + self.shift) * self.scale
        norm_feats = np.expand_dims(norm_feats, axis=0)  # (1, T, 400)

        ort_inputs = {
            "speech": norm_feats,
            "in_cache0": state.in_cache0,
            "in_cache1": state.in_cache1,
            "in_cache2": state.in_cache2,
            "in_cache3": state.in_cache3,
        }

        outs = self.session.run(None, ort_inputs)
        logits = outs[0]
        state.in_cache0 = outs[1][:, :, :19, :]
        state.in_cache1 = outs[2][:, :, :19, :]
        state.in_cache2 = outs[3][:, :, :19, :]
        state.in_cache3 = outs[4][:, :, :19, :]

        # Softmax speech probability: 1.0 - prob(silence_pdf_0)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        sil_prob = probs[0, -1, 0]
        speech_prob = float(1.0 - sil_prob)

        return speech_prob

    def process_frame(
        self,
        frame_pcm16: bytes,
        timestamp_sec: float,
        state: FsmnVADStreamState,
    ) -> VADResult:
        """Process 512 samples of 16-bit PCM mono (1024 bytes)."""
        # Monotonic timestamp & Seek recovery check
        if state.last_timestamp_sec > 0.0 and timestamp_sec < (state.last_timestamp_sec - 0.5):
            self._log(
                f"[FSMN-VAD] Time regression detected ({state.last_timestamp_sec:.2f}s -> {timestamp_sec:.2f}s). "
                f"Resetting VAD stream state for seek recovery."
            )
            state.reset()
        state.last_timestamp_sec = timestamp_sec

        expected_bytes = state.frame_samples * 2  # 1024 bytes
        if len(frame_pcm16) != expected_bytes:
            return VADResult(
                timestamp_sec=timestamp_sec,
                is_speech=state.is_speech_active,
                probability=0.0,
                event=None,
                reason="INVALID_FRAME_SIZE",
            )

        pcm_int16 = np.frombuffer(frame_pcm16, dtype=np.int16)
        frame_float32 = pcm_int16.astype(np.float32) / 32768.0

        prob = self._infer_chunk(frame_float32, state)
        threshold = self.config.threshold

        event: Optional[str] = None
        reason: Optional[str] = None
        duration_sec = 0.0
        speech_start_sec = state.speech_start_time
        speech_end_sec: Optional[float] = None
        frame_dur = state.frame_samples / state.sample_rate  # 0.032s

        if prob >= threshold:
            # Current frame is SPEECH
            state.last_speech_time = timestamp_sec
            state.silence_start_time = None

            if not state.is_speech_active:
                state.is_speech_active = True
                state.current_utterance_id += 1
                state.speech_start_time = timestamp_sec
                speech_start_sec = timestamp_sec
                event = "SPEECH_START"
                self._log(
                    f"[FSMN-VAD] >> SPEECH START at {timestamp_sec:.2f}s "
                    f"(prob={prob:.2f}, threshold={threshold:.2f}, utt_id={state.current_utterance_id})"
                )

            speech_dur = (timestamp_sec - state.speech_start_time) + frame_dur
            if speech_dur >= self.config.max_speech_duration_sec:
                event = "SPEECH_END"
                reason = "MAX_SPEECH_DURATION_REACHED"
                duration_sec = speech_dur
                speech_end_sec = timestamp_sec
                self._log(
                    f"[FSMN-VAD] << SPEECH END at {timestamp_sec:.2f}s (duration={duration_sec:.2f}s, utt_id={state.current_utterance_id}) "
                    f"| Reason: MAX_SPEECH_DURATION_REACHED ({self.config.max_speech_duration_sec:.1f}s limit)"
                )
                state.is_speech_active = False
                state.speech_start_time = None
                state.last_speech_time = None

        else:
            # Current frame is SILENCE
            if state.is_speech_active:
                if state.silence_start_time is None:
                    state.silence_start_time = timestamp_sec

                silence_dur = timestamp_sec - state.silence_start_time
                if silence_dur >= self.config.min_silence_duration_sec:
                    last_speech = state.last_speech_time or timestamp_sec
                    start_speech = state.speech_start_time or timestamp_sec
                    raw_duration = (last_speech - start_speech) + frame_dur

                    event = "SPEECH_END"
                    reason = "SILENCE_TIMEOUT"
                    duration_sec = max(frame_dur, raw_duration)
                    speech_end_sec = last_speech
                    self._log(
                        f"[FSMN-VAD] << SPEECH END at {timestamp_sec:.2f}s (duration={duration_sec:.2f}s, utt_id={state.current_utterance_id}) "
                        f"| Reason: SILENCE_TIMEOUT (silence={silence_dur:.2f}s >= {self.config.min_silence_duration_sec:.2f}s)"
                    )

                    state.is_speech_active = False
                    state.speech_start_time = None
                    state.last_speech_time = None
                    state.silence_start_time = None

        return VADResult(
            timestamp_sec=timestamp_sec,
            is_speech=state.is_speech_active,
            probability=prob,
            event=event,
            reason=reason,
            utterance_id=state.current_utterance_id if state.is_speech_active or event == "SPEECH_END" else None,
            duration_sec=duration_sec,
            speech_start_sec=speech_start_sec,
            speech_end_sec=speech_end_sec,
        )

    def flush(self, timestamp_sec: float, state: FsmnVADStreamState, auto_reset: bool = True) -> Optional[VADResult]:
        """Force flush remaining audio buffer at end of audio stream."""
        res: Optional[VADResult] = None
        if state.is_speech_active and state.speech_start_time is not None:
            last_time = state.last_speech_time or timestamp_sec
            dur = (last_time - state.speech_start_time) + (state.frame_samples / state.sample_rate)
            self._log(
                f"[FSMN-VAD] << SPEECH END at {timestamp_sec:.2f}s (duration={dur:.2f}s, utt_id={state.current_utterance_id}) "
                f"| Reason: STREAM_EOF"
            )
            res = VADResult(
                timestamp_sec=timestamp_sec,
                is_speech=False,
                probability=0.0,
                event="SPEECH_END",
                reason="STREAM_EOF",
                utterance_id=state.current_utterance_id,
                duration_sec=dur,
                speech_start_sec=state.speech_start_time,
                speech_end_sec=last_time,
            )

        if auto_reset:
            state.reset()
        return res
