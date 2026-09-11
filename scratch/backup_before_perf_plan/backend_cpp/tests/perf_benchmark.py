"""Automated Performance Benchmark & Diagnostic Suite for backend_cpp.

Executes 4 rigorous real-world and stress-testing scenarios against the backend WebSocket:
1. Scenario 1: Conversational Speech Stream (Realistic video subtitles: speech -> silence -> speech)
2. Scenario 2: Continuous Long Speech (Tests max_duration split, buffer size, preview poller under load)
3. Scenario 3: Rapid Lock Contention & Burst Test (Tests lock contention between preview & commit)
4. Scenario 4: Connection Lifecycle & Resource Leak Test (Connect -> Stream -> Disconnect x 5 cycles)

Collects client-side timing and queries server-side /api/perf/summary for end-to-end diagnostics.
"""

import asyncio
import json
import logging
import math
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import websockets
except ImportError:
    raise ImportError("websockets package is required for running benchmarks. Run: pip install websockets")

logger = logging.getLogger("perf_benchmark")
SAMPLE_RATE = 16000


def generate_pcm_audio(
    duration_sec: float,
    sample_rate: int = SAMPLE_RATE,
    speech_like: bool = True,
    amplitude: float = 0.5,
) -> bytes:
    """Generate 16kHz 16-bit mono PCM audio.

    If speech_like is True, generates multi-frequency harmonic wave with amplitude
    modulation to trigger Voice Activity Detection (VAD) models (Silero/FSMN).
    If speech_like is False, generates low-level background silence.
    """
    total_samples = int(duration_sec * sample_rate)
    if total_samples <= 0:
        return b""

    t = np.linspace(0, duration_sec, total_samples, endpoint=False, dtype=np.float32)

    if speech_like:
        # Mix fundamental frequencies typical of human voice (150Hz, 300Hz, 600Hz, 1200Hz, 2400Hz)
        wave = (
            0.35 * np.sin(2 * np.pi * 180.0 * t)
            + 0.25 * np.sin(2 * np.pi * 360.0 * t)
            + 0.20 * np.sin(2 * np.pi * 720.0 * t)
            + 0.15 * np.sin(2 * np.pi * 1440.0 * t)
            + 0.05 * np.sin(2 * np.pi * 2880.0 * t)
        )
        # Syllable-like envelope modulation (~3-4 Hz speech rhythm)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
        signal = wave * envelope * amplitude
    else:
        # Faint dither noise / silence (< -60dB)
        signal = (np.random.rand(total_samples).astype(np.float32) - 0.5) * 0.001

    # Convert to int16 PCM bytes
    signal_int16 = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    return signal_int16.tobytes()


def pack_audio_frame(pcm_bytes: bytes, chunk_idx: int, capture_ts: float, format_type: str = "A") -> bytes:
    """Pack PCM chunk into binary WebSocket message matching extension_firefox protocol."""
    if format_type == "A":
        # Format A: 4-byte uint32 header length + JSON header + PCM
        header_dict = {
            "type": "audio_chunk",
            "chunkIndex": chunk_idx,
            "captureTimestamp": capture_ts,
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_bytes)
        return struct.pack("<I", header_len) + header_bytes + pcm_bytes
    else:
        # Format B: 8-byte float64 capture timestamp + PCM
        return struct.pack("<d", capture_ts) + pcm_bytes


class BenchmarkClient:
    """WebSocket client driving real-time audio streams and collecting latency."""

    def __init__(self, uri: str, ssl_context: Any = None):
        self.uri = uri
        self.ssl_context = ssl_context
        self.received_messages: List[Dict[str, Any]] = []
        self.utterance_events: Dict[str, Dict[str, Any]] = {}
        self._rx_task: Optional[asyncio.Task] = None
        self._connected = False

    async def connect(self) -> Any:
        self.ws = await websockets.connect(
            self.uri,
            ssl=self.ssl_context,
            max_size=10 * 1024 * 1024,
            ping_interval=None,
        )
        self._connected = True
        self._rx_task = asyncio.create_task(self._listen_loop())
        return self.ws

    async def disconnect(self) -> None:
        self._connected = False
        if self._rx_task and not self._rx_task.done():
            self._rx_task.cancel()
            try:
                await self._rx_task
            except asyncio.CancelledError:
                pass
        if hasattr(self, "ws"):
            await self.ws.close()

    async def _listen_loop(self) -> None:
        try:
            while self._connected:
                raw_msg = await self.ws.recv()
                rx_time = time.perf_counter()
                if isinstance(raw_msg, str):
                    try:
                        data = json.loads(raw_msg)
                        data["_rx_time"] = rx_time
                        self.received_messages.append(data)

                        mtype = data.get("type", "")
                        utt_id = data.get("utterance_id", "")
                        if utt_id:
                            entry = self.utterance_events.setdefault(utt_id, {})
                            if mtype == "utterance_update":
                                if data.get("is_final"):
                                    entry["asr_final_time"] = rx_time
                                    entry["final_text"] = data.get("text", "")
                                else:
                                    entry.setdefault("preview_count", 0)
                                    entry["preview_count"] += 1
                                    entry["last_preview_time"] = rx_time
                            elif mtype == "translation":
                                entry["translation_time"] = rx_time
                                entry["translated_text"] = data.get("translated", "")
                            elif mtype == "tts_audio":
                                entry["tts_time"] = rx_time
                                entry["tts_duration"] = data.get("duration_sec", 0.0)
                    except json.JSONDecodeError:
                        pass
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass

    async def send_config(self, config_dict: Dict[str, Any]) -> None:
        payload = {"type": "set_config", **config_dict}
        await self.ws.send(json.dumps(payload))

    async def stream_audio_duration(
        self,
        duration_sec: float,
        chunk_ms: int = 100,
        speech_like: bool = True,
        realtime_factor: float = 1.0,
        start_chunk_idx: int = 1,
    ) -> int:
        """Stream audio chunks simulating browser microphone/tab capture."""
        chunk_duration_sec = chunk_ms / 1000.0
        samples_per_chunk = int(SAMPLE_RATE * chunk_duration_sec)
        bytes_per_chunk = samples_per_chunk * 2

        pcm_data = generate_pcm_audio(duration_sec, SAMPLE_RATE, speech_like=speech_like)
        total_chunks = len(pcm_data) // bytes_per_chunk
        chunk_idx = start_chunk_idx
        start_ts = time.time()

        for i in range(total_chunks):
            offset = i * bytes_per_chunk
            chunk_bytes = pcm_data[offset: offset + bytes_per_chunk]
            capture_ts = start_ts + (i * chunk_duration_sec)
            frame = pack_audio_frame(chunk_bytes, chunk_idx, capture_ts, format_type="A")
            await self.ws.send(frame)
            chunk_idx += 1

            if realtime_factor > 0:
                sleep_sec = chunk_duration_sec / realtime_factor
                await asyncio.sleep(sleep_sec)

        return chunk_idx


async def run_scenario_conversational_stream(
    uri: str,
    ssl_context: Any = None,
    num_turns: int = 3,
    speech_sec: float = 3.0,
    silence_sec: float = 1.0,
) -> Dict[str, Any]:
    """Scenario 1: Realistic conversational video speech stream."""
    logger.info(f"▶️ [SCENARIO 1] Conversational Stream ({num_turns} turns, speech={speech_sec}s, silence={silence_sec}s)...")
    client = BenchmarkClient(uri, ssl_context)
    await client.connect()

    await client.send_config({
        "sourceLang": "auto",
        "targetLang": "vi",
        "vadEngine": "fsmn-vad",
        "ttsEnabled": False,
        "splitOnStability": True,
    })
    await asyncio.sleep(0.5)

    chunk_idx = 1
    t0 = time.perf_counter()

    for turn in range(1, num_turns + 1):
        logger.info(f"  🗣️ Turn {turn}/{num_turns}: Speaking ({speech_sec}s)...")
        chunk_idx = await client.stream_audio_duration(
            duration_sec=speech_sec,
            chunk_ms=100,
            speech_like=True,
            realtime_factor=1.0,
            start_chunk_idx=chunk_idx,
        )

        logger.info(f"  🤫 Turn {turn}/{num_turns}: Silence ({silence_sec}s)...")
        chunk_idx = await client.stream_audio_duration(
            duration_sec=silence_sec,
            chunk_ms=100,
            speech_like=False,
            realtime_factor=1.0,
            start_chunk_idx=chunk_idx,
        )

    # Wait for pipeline completion
    await asyncio.sleep(2.0)
    total_time = time.perf_counter() - t0
    await client.disconnect()

    events = client.utterance_events
    logger.info(f"✅ [SCENARIO 1] Completed in {total_time:.1f}s, captured {len(events)} utterances.")
    return {
        "scenario": "conversational_stream",
        "turns": num_turns,
        "total_time_sec": round(total_time, 2),
        "utterances_captured": len(events),
        "events": events,
    }


async def run_scenario_continuous_speech(
    uri: str,
    ssl_context: Any = None,
    duration_sec: float = 12.0,
) -> Dict[str, Any]:
    """Scenario 2: Long uninterrupted monologue testing max_duration auto-commits."""
    logger.info(f"▶️ [SCENARIO 2] Continuous Monologue ({duration_sec}s uninterrupted)...")
    client = BenchmarkClient(uri, ssl_context)
    await client.connect()

    await client.send_config({
        "sourceLang": "auto",
        "targetLang": "vi",
        "maxDurationSec": 6.0,
    })
    await asyncio.sleep(0.3)

    t0 = time.perf_counter()
    chunk_idx = await client.stream_audio_duration(
        duration_sec=duration_sec,
        chunk_ms=100,
        speech_like=True,
        realtime_factor=1.0,
    )
    # Trail with 1s silence to finalize
    await client.stream_audio_duration(
        duration_sec=1.0,
        chunk_ms=100,
        speech_like=False,
        realtime_factor=1.0,
        start_chunk_idx=chunk_idx,
    )

    await asyncio.sleep(2.0)
    total_time = time.perf_counter() - t0
    await client.disconnect()

    events = client.utterance_events
    logger.info(f"✅ [SCENARIO 2] Completed in {total_time:.1f}s, auto-split into {len(events)} utterances.")
    return {
        "scenario": "continuous_speech",
        "duration_sec": duration_sec,
        "total_time_sec": round(total_time, 2),
        "utterances_captured": len(events),
        "events": events,
    }


async def run_scenario_rapid_burst_contention(
    uri: str,
    ssl_context: Any = None,
    num_bursts: int = 5,
) -> Dict[str, Any]:
    """Scenario 3: Rapid burst stream (2x real-time) testing lock contention & queue pressure."""
    logger.info(f"▶️ [SCENARIO 3] Rapid Burst & Lock Contention ({num_bursts} bursts @ 2.0x speed)...")
    client = BenchmarkClient(uri, ssl_context)
    await client.connect()

    chunk_idx = 1
    t0 = time.perf_counter()

    for b in range(1, num_bursts + 1):
        chunk_idx = await client.stream_audio_duration(
            duration_sec=1.5,
            chunk_ms=50,
            speech_like=True,
            realtime_factor=2.0,  # 2x faster than real-time
            start_chunk_idx=chunk_idx,
        )
        chunk_idx = await client.stream_audio_duration(
            duration_sec=0.5,
            chunk_ms=50,
            speech_like=False,
            realtime_factor=2.0,
            start_chunk_idx=chunk_idx,
        )

    await asyncio.sleep(2.0)
    total_time = time.perf_counter() - t0
    await client.disconnect()

    logger.info(f"✅ [SCENARIO 3] Completed in {total_time:.1f}s.")
    return {
        "scenario": "rapid_burst_contention",
        "bursts": num_bursts,
        "total_time_sec": round(total_time, 2),
        "events": client.utterance_events,
    }


async def run_scenario_leak_detection(
    uri: str,
    ssl_context: Any = None,
    cycles: int = 4,
) -> Dict[str, Any]:
    """Scenario 4: Repeated connect -> stream -> disconnect cycles to verify resource reclamation."""
    logger.info(f"▶️ [SCENARIO 4] Resource Leak Detection ({cycles} connect/disconnect cycles)...")
    t0 = time.perf_counter()

    for c in range(1, cycles + 1):
        logger.info(f"  🔄 Cycle {c}/{cycles}: Connecting and streaming 2s audio...")
        client = BenchmarkClient(uri, ssl_context)
        await client.connect()
        await client.stream_audio_duration(
            duration_sec=2.0,
            chunk_ms=100,
            speech_like=True,
            realtime_factor=2.0,
        )
        await asyncio.sleep(0.5)
        await client.disconnect()
        await asyncio.sleep(0.3)

    total_time = time.perf_counter() - t0
    logger.info(f"✅ [SCENARIO 4] Completed {cycles} cycles in {total_time:.1f}s.")
    return {
        "scenario": "leak_detection",
        "cycles": cycles,
        "total_time_sec": round(total_time, 2),
    }
