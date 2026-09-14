"""High-Performance Streaming Silero VAD v5 Engine for backend_audio_cpp.

Zero PyTorch dependency, pure NumPy + ONNX Runtime with 64-sample context buffer,
512-sample frame slicing (32ms @ 16kHz), and structured debug logging (SPEECH_START, SPEECH_END + reason).
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger("backend_audio_cpp.vad")


@dataclass
class VADConfig:
    """Configuration for Silero VAD."""
    model_path: str = "backend_audio_cpp/models/silero_vad.onnx"
    sample_rate: int = 16000
    frame_samples: int = 512               # 32ms at 16kHz
    context_samples: int = 64              # Silero v5 temporal context
    threshold: float = 0.50                # Speech detection probability threshold
    min_silence_duration_sec: float = 0.55 # Silence duration to trigger SPEECH_END
    min_speech_duration_sec: float = 0.20  # Minimum speech duration to qualify as valid utterance
    max_speech_duration_sec: float = 8.00  # Hard cutoff to prevent runaway sentences


@dataclass
class VADResult:
    """Result of processing a single frame or chunk."""
    timestamp_sec: float
    is_speech: bool
    probability: float
    event: Optional[str] = None            # "SPEECH_START", "SPEECH_END", or None
    reason: Optional[str] = None           # "SILENCE_TIMEOUT", "MAX_SPEECH_DURATION_REACHED", "STREAM_EOF"
    utterance_id: Optional[int] = None
    duration_sec: float = 0.0              # Valid during SPEECH_END
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None


class VADStreamState:
    """Session-isolated streaming state for Silero VAD."""

    def __init__(self, config: VADConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_samples = config.frame_samples
        self.context_samples = config.context_samples

        # Model recurrent state (2, 1, 128) and context (1, 64) for Silero VAD v5
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.context_samples), dtype=np.float32)

        # Speech tracking state
        self.is_speech_active: bool = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.current_utterance_id: int = 0
        self.silence_start_time: Optional[float] = None

    def reset(self) -> None:
        """Reset internal states between sessions."""
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.context_samples), dtype=np.float32)
        self.is_speech_active = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None


class SileroVADEngine:
    """High-performance Silero VAD v5 engine with structured debug logging."""

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.model_path = Path(self.config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found at: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPU Execution is optimal for Silero (< 0.2ms latency per 32ms frame, keeps GPU 100% free)
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.active_provider = self.session.get_providers()[0]
        logger.info(f"Loaded Silero VAD from {self.model_path} [Provider: {self.active_provider}]")

    def create_state(self) -> VADStreamState:
        """Create a fresh isolated session state."""
        return VADStreamState(self.config)

    def _infer_frame(self, frame_float32: np.ndarray, state: VADStreamState) -> float:
        """Run single 512-sample frame through ONNX model with context buffer."""
        chunk = frame_float32.reshape(1, -1)
        inp = np.concatenate([state.context, chunk], axis=1)
        state.context = inp[:, -state.context_samples :]

        sr = np.array(state.sample_rate, dtype=np.int64)

        ort_inputs = {
            "input": inp,
            "state": state.state,
            "sr": sr,
        }

        out, state_out = self.session.run(None, ort_inputs)
        state.state = state_out

        return float(out[0][0])

    def process_frame(
        self,
        frame_pcm16: bytes,
        timestamp_sec: float,
        state: VADStreamState,
    ) -> VADResult:
        """Process exactly 512 samples of 16-bit PCM mono (1024 bytes).

        Args:
            frame_pcm16: Exactly 1024 bytes (512 int16 samples).
            timestamp_sec: Timestamp of current frame.
            state: Active VADStreamState.

        Returns:
            VADResult with detection state, probability, and optional events.
        """
        pcm_int16 = np.frombuffer(frame_pcm16, dtype=np.int16)
        if len(pcm_int16) != state.frame_samples:
            if len(pcm_int16) < state.frame_samples:
                pcm_int16 = np.pad(pcm_int16, (0, state.frame_samples - len(pcm_int16)))
            else:
                pcm_int16 = pcm_int16[: state.frame_samples]

        frame_float32 = pcm_int16.astype(np.float32) / 32768.0

        prob = self._infer_frame(frame_float32, state)
        threshold = self.config.threshold

        event: Optional[str] = None
        reason: Optional[str] = None
        duration_sec = 0.0
        speech_start_sec = state.speech_start_time
        speech_end_sec: Optional[float] = None

        if prob >= threshold:
            # Current frame is SPEECH
            state.last_speech_time = timestamp_sec
            state.silence_start_time = None

            if not state.is_speech_active:
                # Transition: SILENCE -> SPEECH
                state.is_speech_active = True
                state.current_utterance_id += 1
                state.speech_start_time = timestamp_sec
                speech_start_sec = timestamp_sec
                event = "SPEECH_START"
                logger.info(
                    f"[VAD] >> SPEECH START at {timestamp_sec:.2f}s "
                    f"(prob={prob:.2f}, threshold={threshold:.2f}, utt_id={state.current_utterance_id})"
                )

            # Check for MAX_SPEECH_DURATION_REACHED
            speech_dur = timestamp_sec - state.speech_start_time
            if speech_dur >= self.config.max_speech_duration_sec:
                event = "SPEECH_END"
                reason = "MAX_SPEECH_DURATION_REACHED"
                duration_sec = speech_dur
                speech_end_sec = timestamp_sec
                logger.info(
                    f"[VAD] << SPEECH END at {timestamp_sec:.2f}s (duration={duration_sec:.2f}s, utt_id={state.current_utterance_id}) "
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
                    # Transition: SPEECH -> SILENCE
                    event = "SPEECH_END"
                    reason = "SILENCE_TIMEOUT"
                    duration_sec = (state.last_speech_time or timestamp_sec) - (state.speech_start_time or 0.0)
                    speech_end_sec = state.last_speech_time
                    logger.info(
                        f"[VAD] << SPEECH END at {timestamp_sec:.2f}s (duration={duration_sec:.2f}s, utt_id={state.current_utterance_id}) "
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

    def flush(self, timestamp_sec: float, state: VADStreamState) -> Optional[VADResult]:
        """Force flush when audio stream ends."""
        if state.is_speech_active and state.speech_start_time is not None:
            last_time = state.last_speech_time or timestamp_sec
            dur = last_time - state.speech_start_time
            logger.info(
                f"[VAD] << SPEECH END at {timestamp_sec:.2f}s (duration={dur:.2f}s, utt_id={state.current_utterance_id}) "
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
            state.reset()
            return res
        state.reset()
        return None
