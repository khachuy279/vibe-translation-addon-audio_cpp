"""OmniVoice TTS Engine for backend_audio_cpp.

Communicates with native audio.cpp CUDA runtime via HTTP (/v1/audio/speech),
providing fast zero-shot voice cloning using reference audio samples from backend_audio_cpp/voices/.
"""

from dataclasses import dataclass
import io
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union
import wave

import requests

from backend_audio_cpp.tts.voice_manager import VoiceManager, VoiceProfile

logger = logging.getLogger("backend_audio_cpp.tts")


@dataclass
class TTSConfig:
    """Configuration for OmniVoice TTS Engine."""
    server_url: str = "http://127.0.0.1:8089"
    model_id: str = "omnivoice"
    default_sample_rate: int = 24000
    timeout_sec: float = 30.0


@dataclass
class TTSSynthesisResult:
    """Outcome of a TTS synthesis request."""
    audio_wav_bytes: bytes
    sample_rate: int
    duration_sec: float
    synth_time_ms: float
    rtf: float
    voice_id: str
    voice_name: str


class OmniVoiceTTSEngine:
    """TTS Engine wrapping native audio.cpp CUDA OmniVoice zero-shot voice cloning."""

    _instance: Optional["OmniVoiceTTSEngine"] = None

    @classmethod
    def get_instance(cls, config: Optional[TTSConfig] = None) -> "OmniVoiceTTSEngine":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        voice_manager: Optional[VoiceManager] = None,
    ):
        self.config = config or TTSConfig()
        self.voice_manager = voice_manager or VoiceManager.get_instance()
        self.server_url = self.config.server_url.rstrip("/")
        self.speech_endpoint = f"{self.server_url}/v1/audio/speech"

    def list_available_voices(self) -> list[Dict[str, Any]]:
        """Return catalog of registered voice samples."""
        return self.voice_manager.list_voices()

    def get_wav_duration_and_sr(self, wav_bytes: bytes) -> tuple[float, int]:
        """Extract duration in seconds and sample rate from in-memory WAV container."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sr = wf.getframerate()
                nframes = wf.getnframes()
                dur = nframes / float(sr) if sr > 0 else 0.0
                return dur, sr
        except Exception:
            return 0.0, self.config.default_sample_rate

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        utt_id: Union[int, str] = 0,
    ) -> TTSSynthesisResult:
        """Synthesize speech using cloned voice sample.

        Args:
            text: Text to synthesize.
            voice_id: Optional voice profile ID (defaults to primary voice).
            utt_id: Utterance identifier for logging.

        Returns:
            TTSSynthesisResult with WAV bytes, duration, timing, and RTF.
        """
        clean_text = text.strip()
        if not clean_text:
            return TTSSynthesisResult(
                audio_wav_bytes=b"",
                sample_rate=self.config.default_sample_rate,
                duration_sec=0.0,
                synth_time_ms=0.0,
                rtf=0.0,
                voice_id="",
                voice_name="",
            )

        profile: Optional[VoiceProfile] = self.voice_manager.get_voice(voice_id)
        resolved_voice_path = (
            str(profile.audio_path.resolve()).replace("\\", "/")
            if profile and profile.audio_path
            else ""
        )
        ref_text = profile.reference_text if profile else ""
        voice_label = profile.name if profile else "default"

        # Structured Log: START
        logger.info(f'[TTS] START [utt_{utt_id}]: "{clean_text}" | voice={voice_label}')

        payload = {
            "model": self.config.model_id,
            "input": clean_text,
            "voice_ref": resolved_voice_path,
            "voice": resolved_voice_path,
        }
        if ref_text:
            payload["reference_text"] = ref_text

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                self.speech_endpoint,
                json=payload,
                timeout=self.config.timeout_sec,
            )
            elapsed_sec = time.perf_counter() - t0
            synth_time_ms = elapsed_sec * 1000.0

            if resp.status_code != 200:
                logger.error(
                    f"[TTS] Server error {resp.status_code}: {resp.text}"
                )
                return TTSSynthesisResult(
                    audio_wav_bytes=b"",
                    sample_rate=self.config.default_sample_rate,
                    duration_sec=0.0,
                    synth_time_ms=synth_time_ms,
                    rtf=0.0,
                    voice_id=profile.id if profile else "",
                    voice_name=voice_label,
                )

            wav_bytes = resp.content
            duration_sec, sample_rate = self.get_wav_duration_and_sr(wav_bytes)
            rtf = elapsed_sec / duration_sec if duration_sec > 0 else 0.0

            # Structured Log: DONE
            logger.info(
                f'[TTS] DONE [utt_{utt_id}] -> WAV {sample_rate // 1000}kHz '
                f'(duration={duration_sec:.2f}s) | synth_time={synth_time_ms:.1f}ms | rtf={rtf:.3f}'
            )

            return TTSSynthesisResult(
                audio_wav_bytes=wav_bytes,
                sample_rate=sample_rate,
                duration_sec=duration_sec,
                synth_time_ms=synth_time_ms,
                rtf=rtf,
                voice_id=profile.id if profile else "",
                voice_name=voice_label,
            )

        except Exception as e:
            logger.error(f"[TTS] Request exception: {e}")
            return TTSSynthesisResult(
                audio_wav_bytes=b"",
                sample_rate=self.config.default_sample_rate,
                duration_sec=0.0,
                synth_time_ms=0.0,
                rtf=0.0,
                voice_id=profile.id if profile else "",
                voice_name=voice_label,
            )
