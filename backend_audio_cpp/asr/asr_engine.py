"""Audio.cpp Native ASR Engine with Dynamic Live Model Switching.

Supports seamless runtime model switching between:
- Qwen3-ASR 1.7B (GGUF Q8_0)
- Nemotron 3.5 ASR Streaming 0.6B (GGUF Q8_0)
- Voxtral Mini 4B Realtime 2602 (GGUF Q4_K)
and any future models registered in backend_audio_cpp/models.yaml.
"""

import io
import logging
import os
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from backend_audio_cpp.asr.model_registry import ASRModelRegistry

logger = logging.getLogger("backend_audio_cpp.asr")

_RE_SPECIAL_TAGS = re.compile(r"<\|.*?\|>|<[^>]+>")


def clean_asr_text(raw_text: str) -> str:
    """Strip special tokens, system prompt artifacts, and extra whitespace."""
    if not raw_text:
        return ""
    text = _RE_SPECIAL_TAGS.sub("", raw_text)
    text = re.sub(
        r"(?m)^(?:system|user|assistant|language\s+\w+)\s*[:]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


@dataclass
class ASRConfig:
    """Configuration for Audio.cpp ASR Engine."""
    server_url: str = "http://127.0.0.1:8089"
    default_model: str = "qwen3-asr-1.7b"
    sample_rate: int = 16000
    timeout_sec: float = 20.0
    enable_normalization: bool = True


class AudioCppASREngine:
    """ASR Engine wrapping native audio.cpp CUDA runtime with dynamic live switching."""

    _instance: Optional["AudioCppASREngine"] = None

    @classmethod
    def get_instance(
        cls,
        config: Optional[ASRConfig] = None,
        registry: Optional[ASRModelRegistry] = None,
    ) -> "AudioCppASREngine":
        if cls._instance is None:
            cls._instance = cls(config, registry)
        return cls._instance

    def __init__(
        self,
        config: Optional[ASRConfig] = None,
        registry: Optional[ASRModelRegistry] = None,
    ):
        self.config = config or ASRConfig()
        self.registry = registry or ASRModelRegistry.get_instance()
        self.server_url = self.config.server_url.rstrip("/")
        self.transcribe_endpoint = f"{self.server_url}/v1/audio/transcriptions"

        # Set default active model
        self.active_model_id = self.registry.get_active_model_key()
        self._ensure_server_available()

    def get_active_model(self) -> str:
        """Return canonical ID of currently active ASR model."""
        return self.active_model_id

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Return catalog of models with their download and active state."""
        return self.registry.list_models()

    def unload_model(self, model_id: str) -> bool:
        """Unload specific model from audio.cpp server VRAM via POST /v1/models/unload."""
        if not model_id:
            return False
        try:
            resp = requests.post(
                f"{self.server_url}/v1/models/unload",
                json={"id": model_id},
                timeout=3.0,
            )
            if resp.status_code == 200:
                logger.info(f"[ASR] Successfully unloaded model '{model_id}' from VRAM")
                return True
            else:
                logger.warning(
                    f"[ASR] Unload '{model_id}' status {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            logger.warning(f"[ASR] Error unloading model '{model_id}': {e}")
        return False

    def switch_model(self, model_key_or_alias: str) -> str:
        """Switch ASR model live without restarting the backend.

        Args:
            model_key_or_alias: Model key (e.g. 'nemotron-3.5-streaming', 'voxtral', 'qwen3').

        Returns:
            Canonical model key switched to.
        """
        old_model = self.active_model_id
        canonical_key = self.registry.set_active_model_key(model_key_or_alias)

        if old_model and old_model != canonical_key:
            logger.info(f"[ASR] Unloading previous model '{old_model}' from VRAM...")
            self.unload_model(old_model)

        self.active_model_id = canonical_key

        logger.info(
            f"[ASR] LIVE MODEL SWITCH: '{old_model}' -> '{canonical_key}' "
            f"(name: {self.registry.get_model(canonical_key).get('name', canonical_key)})"
        )
        return canonical_key

    def _ensure_server_available(self) -> None:
        """Verify audio.cpp server is reachable, or launch it if needed."""
        health_url = f"{self.server_url}/health"
        try:
            resp = requests.get(health_url, timeout=2.0)
            if resp.status_code == 200:
                logger.info(f"Connected to audio.cpp server at {self.server_url}")
                return
        except Exception:
            pass

        logger.warning(
            f"audio.cpp server not responding at {self.server_url}. Attempting auto-start..."
        )
        self._launch_in_process_server()

    def _launch_in_process_server(self) -> None:
        """Start audiocpp_server.exe in background if not already running."""
        here = Path(__file__).resolve().parent.parent
        bin_dir = here / "bin"
        server_exe = bin_dir / "audiocpp_server.exe"
        cfg_file = here / "config_server.json"

        # Ensure server config contains all models
        self.registry.generate_server_config()

        if not server_exe.exists():
            raise FileNotFoundError(f"audiocpp_server.exe not found at {server_exe}")

        cmd = [
            str(server_exe),
            "--config", str(cfg_file),
            "--ui-management",
            "--max-loaded-models", "1",
        ]
        subprocess.Popen(
            cmd,
            cwd=str(bin_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait up to 20s for server to become healthy
        health_url = f"{self.server_url}/health"
        for _ in range(40):
            time.sleep(0.5)
            try:
                r = requests.get(health_url, timeout=1.0)
                if r.status_code == 200:
                    logger.info(f"Successfully launched audio.cpp server on {self.server_url}")
                    return
            except Exception:
                pass

        raise RuntimeError("Failed to start audio.cpp server within 20 seconds.")

    def pcm_to_wav_bytes(self, pcm16_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """Convert raw PCM16 mono bytes into an in-memory WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16_bytes)
        return buf.getvalue()

    def transcribe_chunk(
        self,
        pcm16_bytes: bytes,
        utt_id: Union[int, str] = 0,
        is_partial: bool = True,
        model: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Transcribe PCM audio buffer with the active or requested model.

        Args:
            pcm16_bytes: Raw 16-bit PCM mono bytes.
            utt_id: Active utterance ID for logging.
            is_partial: True if this is an intermediate preview, False if final.
            model: Optional model key override for this chunk.
            sample_rate: Sample rate of PCM bytes (default config.sample_rate).

        Returns:
            (clean_text, timing_dict)
        """
        if not pcm16_bytes or len(pcm16_bytes) < 1024:
            return "", {}

        if self.config.enable_normalization:
            from backend_audio_cpp.asr.speech_normalizer import normalize_pcm_bytes
            pcm16_bytes = normalize_pcm_bytes(pcm16_bytes)

        sr = sample_rate or self.config.sample_rate
        target_model = self.registry.resolve_key(model) if model else self.active_model_id
        audio_sec = len(pcm16_bytes) / (sr * 2.0)
        wav_bytes = self.pcm_to_wav_bytes(pcm16_bytes, sr)

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": target_model}

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                self.transcribe_endpoint,
                files=files,
                data=data,
                timeout=self.config.timeout_sec,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                logger.error(
                    f"[ASR][{target_model}] Error {resp.status_code} from server: {resp.text}"
                )
                return "", {}

            res_json = resp.json()
            raw_text = res_json.get("text", "")
            clean_text = clean_asr_text(raw_text)

            timing = res_json.get("timing", {})
            wall_ms = timing.get("wall_ms", elapsed_ms)
            rtf = timing.get("rtf", (elapsed_ms / 1000.0) / audio_sec if audio_sec > 0 else 0.0)

            if is_partial:
                logger.info(
                    f'[ASR][{target_model}] PARTIAL [utt_{utt_id}]: "{clean_text}" | '
                    f'poll_ms={wall_ms:.1f}ms | rtf={rtf:.3f}'
                )
            else:
                logger.info(
                    f'[ASR][{target_model}] FINAL UTT [utt_{utt_id}]: "{clean_text}" | '
                    f'total_audio={audio_sec:.2f}s | asr_time={wall_ms:.1f}ms | rtf={rtf:.3f}'
                )

            return clean_text, {
                "model": target_model,
                "audio_sec": audio_sec,
                "wall_ms": wall_ms,
                "rtf": rtf,
            }

        except Exception as e:
            logger.error(f"[ASR][{target_model}] Request exception: {e}")
            return "", {}


# Backward compatibility alias
Qwen3ASREngine = AudioCppASREngine
