"""Streaming Audio Simulator mimicking Firefox Extension AudioWorklet capture & transport."""

import asyncio
import json
import math
import struct
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import AsyncIterator, Generator, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

TARGET_SAMPLE_RATE = 16000
EXTENSION_BASELINE_CHUNK_MS = 64  # 1024 samples @ 16kHz = 64ms


@dataclass
class AudioChunk:
    """Simulated binary audio chunk ready to send to backend."""
    chunk_index: int
    capture_timestamp: float  # Audio position relative to stream start in seconds
    pcm_bytes: bytes          # 16kHz mono 16-bit signed PCM bytes
    duration_ms: float
    is_trailing_silence: bool = False

    def to_format_a_packet(self) -> bytes:
        """Pack chunk into Format A binary WebSocket frame matching extension_firefox.

        [4-byte uint32 LE header length][JSON header bytes][16-bit PCM bytes]
        """
        header_dict = {
            "type": "audio_chunk",
            "chunkIndex": self.chunk_index,
            "captureTimestamp": self.capture_timestamp,
        }
        header_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
        header_len = len(header_bytes)
        return struct.pack("<I", header_len) + header_bytes + self.pcm_bytes

    def to_format_b_packet(self) -> bytes:
        """Pack chunk into Format B binary WebSocket frame (float64 timestamp + PCM)."""
        return struct.pack("<d", self.capture_timestamp) + self.pcm_bytes


class StreamingAudioSimulator:
    """High-fidelity audio stream simulator for Firefox Extension.

    Performs:
    1. Robust audio file loading (supporting any sample rate, bit depth, channel count).
    2. Polyphase downmixing & resampling to 16,000 Hz 16-bit mono PCM.
    3. Strict chunk segmentation (10ms, 20ms, 25ms, 40ms, 50ms, 64ms, 100ms, 200ms).
    4. Real-time pacing (1.0x) or accelerated pacing (0.25x, 0.5x, 2.0x, 5.0x, inf).
    5. Trailing silence generation ensuring complete VAD utterance commits.
    """

    SUPPORTED_CHUNK_MS = [10, 20, 25, 40, 50, 64, 100, 200]

    def __init__(
        self,
        audio_source: Union[str, Path, bytes, np.ndarray],
        chunk_ms: int = EXTENSION_BASELINE_CHUNK_MS,
        speed: float = 1.0,
        trailing_silence_sec: float = 1.5,
        source_sample_rate: Optional[int] = None,
    ):
        self.chunk_ms = chunk_ms
        self.speed = speed
        self.trailing_silence_sec = trailing_silence_sec

        # 1. Load and resample to 16kHz mono int16 PCM
        self.pcm16_bytes, self.source_duration_sec = self._prepare_pcm(
            audio_source, source_sample_rate=source_sample_rate
        )

        # 2. Compute chunking parameters
        self.chunk_samples = int(TARGET_SAMPLE_RATE * (self.chunk_ms / 1000.0))
        self.chunk_bytes = self.chunk_samples * 2  # 2 bytes per int16 sample

        # 3. Trailing silence
        self.silence_samples = int(TARGET_SAMPLE_RATE * self.trailing_silence_sec)
        self.silence_bytes = b"\x00\x00" * self.silence_samples

    @staticmethod
    def _prepare_pcm(
        audio_source: Union[str, Path, bytes, np.ndarray],
        source_sample_rate: Optional[int] = None,
    ) -> Tuple[bytes, float]:
        """Convert arbitrary audio source into 16kHz mono 16-bit PCM bytes."""
        if isinstance(audio_source, (str, Path)):
            path = Path(audio_source)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

            data, sr = sf.read(str(path), dtype="float32")
        elif isinstance(audio_source, np.ndarray):
            data = audio_source.astype(np.float32)
            sr = source_sample_rate or TARGET_SAMPLE_RATE
        elif isinstance(audio_source, bytes):
            # Assume already 16kHz 16-bit mono PCM if bytes passed
            dur = len(audio_source) / (TARGET_SAMPLE_RATE * 2.0)
            return audio_source, dur
        else:
            raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

        # Convert to mono if multi-channel
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        source_dur = len(data) / float(sr)

        # Resample to 16,000 Hz if needed
        if sr != TARGET_SAMPLE_RATE:
            g = gcd(TARGET_SAMPLE_RATE, sr)
            up = TARGET_SAMPLE_RATE // g
            down = sr // g
            data_resampled = resample_poly(data, up, down)
        else:
            data_resampled = data

        # Sanitize and convert to 16-bit linear PCM
        data_clamped = np.clip(data_resampled, -1.0, 1.0)
        pcm16 = (data_clamped * 32767.0).astype(np.int16).tobytes()

        return pcm16, source_dur

    def iter_chunks(self) -> Generator[AudioChunk, None, None]:
        """Synchronously yield all chunks including trailing silence."""
        pcm = self.pcm16_bytes
        total_len = len(pcm)
        chunk_bytes = self.chunk_bytes
        chunk_idx = 1
        offset = 0

        # Speech chunks
        while offset < total_len:
            end = min(offset + chunk_bytes, total_len)
            chunk_data = pcm[offset:end]

            # Zero-pad final partial chunk if necessary for alignment
            if len(chunk_data) < chunk_bytes:
                chunk_data = chunk_data + b"\x00" * (chunk_bytes - len(chunk_data))

            cap_ts = offset / (TARGET_SAMPLE_RATE * 2.0)
            yield AudioChunk(
                chunk_index=chunk_idx,
                capture_timestamp=cap_ts,
                pcm_bytes=chunk_data,
                duration_ms=self.chunk_ms,
                is_trailing_silence=False,
            )
            chunk_idx += 1
            offset = end

        # Trailing silence chunks
        silence_total = len(self.silence_bytes)
        sil_offset = 0
        while sil_offset < silence_total:
            end = min(sil_offset + chunk_bytes, silence_total)
            chunk_data = self.silence_bytes[sil_offset:end]
            if len(chunk_data) < chunk_bytes:
                chunk_data = chunk_data + b"\x00" * (chunk_bytes - len(chunk_data))

            cap_ts = (total_len + sil_offset) / (TARGET_SAMPLE_RATE * 2.0)
            yield AudioChunk(
                chunk_index=chunk_idx,
                capture_timestamp=cap_ts,
                pcm_bytes=chunk_data,
                duration_ms=self.chunk_ms,
                is_trailing_silence=True,
            )
            chunk_idx += 1
            sil_offset = end

    async def stream_paced(self) -> AsyncIterator[AudioChunk]:
        """Asynchronously stream chunks with accurate clock pacing according to `speed`.

        speed:
        - 1.0: Realtime playback pacing (e.g. 64ms chunk takes ~64ms).
        - 2.0: 2x speed (half sleep duration).
        - float('inf') or <= 0: unthrottled (burst / backpressure test).
        """
        step_sec = self.chunk_ms / 1000.0
        use_pacing = not (math.isinf(self.speed) or self.speed <= 0)
        delay_sec = (step_sec / self.speed) if use_pacing else 0.0

        t0 = time.perf_counter()
        sent_chunks = 0

        for chunk in self.iter_chunks():
            yield chunk
            sent_chunks += 1

            if use_pacing and delay_sec > 0:
                expected_elapsed = sent_chunks * delay_sec
                actual_elapsed = time.perf_counter() - t0
                sleep_needed = expected_elapsed - actual_elapsed
                if sleep_needed > 0.0005:
                    await asyncio.sleep(sleep_needed)
                elif sleep_needed < -0.1:
                    # Minor yield to prevent event loop starvation under heavy lag
                    await asyncio.sleep(0.0)
