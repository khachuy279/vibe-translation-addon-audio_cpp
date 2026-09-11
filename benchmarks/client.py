"""WebSocket client simulating Firefox extension connection, configuration, and binary streaming."""

import asyncio
import json
import logging
import ssl
import time
import urllib.request
from typing import Any, Dict, List, Optional

import websockets

from benchmarks.metrics import TimelineCollector
from benchmarks.simulator import StreamingAudioSimulator

logger = logging.getLogger("benchmarks.client")


class BenchmarkSession:
    """A benchmark execution session over WebSocket."""

    def __init__(
        self,
        server_ws_url: str,
        http_base_url: Optional[str] = None,
        session_config: Optional[Dict[str, Any]] = None,
    ):
        self.server_ws_url = server_ws_url
        self.http_base_url = http_base_url or server_ws_url.replace("wss://", "https://").replace("ws://", "http://").rstrip("/ws").rstrip("/")
        self.session_config = session_config or {}

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._rx_task: Optional[asyncio.Task] = None
        self._connected = False
        self.received_messages: List[Dict[str, Any]] = []

        # SSL context for local self-signed certificates
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def connect(self) -> None:
        """Connect WebSocket to server."""
        ssl_ctx = self.ssl_context if self.server_ws_url.startswith("wss://") else None
        self.ws = await websockets.connect(
            self.server_ws_url,
            ssl=ssl_ctx,
            max_size=20 * 1024 * 1024,
            ping_interval=None,
        )
        self._connected = True
        self._rx_task = asyncio.create_task(self._rx_loop())

    async def disconnect(self) -> None:
        """Close WebSocket session and cancel receiver."""
        self._connected = False
        if self._rx_task and not self._rx_task.done():
            self._rx_task.cancel()
            try:
                await self._rx_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def send_config(self, config_dict: Dict[str, Any]) -> None:
        """Send configuration JSON to backend."""
        payload = {"type": "set_config", **config_dict}
        if self.ws:
            await self.ws.send(json.dumps(payload))

    async def _rx_loop(self) -> None:
        """Background loop receiving incoming WebSocket frames from server."""
        try:
            while self._connected and self.ws:
                msg = await self.ws.recv()
                rx_time = time.perf_counter()
                if isinstance(msg, str):
                    try:
                        data = json.loads(msg)
                        data["_rx_mono"] = rx_time
                        self.received_messages.append(data)
                    except json.JSONDecodeError:
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"BenchmarkSession rx error: {e}")

    def get_server_perf_summary(self) -> Optional[Dict[str, Any]]:
        """Query server /api/perf/summary."""
        try:
            url = f"{self.http_base_url}/api/perf/summary"
            req = urllib.request.Request(url, headers={"User-Agent": "BenchmarkHarness"})
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=3.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass
        return None

    def reset_server_perf(self) -> None:
        """Reset server-side performance counters."""
        try:
            url = f"{self.http_base_url}/api/perf/reset"
            req = urllib.request.Request(url, data=b"{}", headers={"User-Agent": "BenchmarkHarness"})
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=3.0) as resp:
                pass
        except Exception:
            pass

    async def stream_audio(
        self,
        simulator: StreamingAudioSimulator,
        timeline: TimelineCollector,
        drain_timeout_sec: float = 3.0,
    ) -> None:
        """Stream chunks from simulator into WebSocket while tracking timeline events."""
        if not self.ws:
            raise RuntimeError("WebSocket not connected")

        timeline.note_stream_started()

        # Send initial config
        if self.session_config:
            await self.send_config(self.session_config)
            await asyncio.sleep(0.1)

        # Index of next message to dispatch to timeline
        msg_cursor = 0

        async for chunk in simulator.stream_paced():
            # Check for arrived messages during streaming
            while msg_cursor < len(self.received_messages):
                msg = self.received_messages[msg_cursor]
                msg_cursor += 1
                timeline.note_subtitle_received(msg)

            frame_data = chunk.to_format_a_packet()
            await self.ws.send(frame_data)
            timeline.note_chunk_sent(
                sample_idx=chunk.chunk_index * simulator.chunk_samples,
                audio_ts=chunk.capture_timestamp,
                is_silence=chunk.is_trailing_silence,
            )

        # Drain loop: wait for backend to process and commit final utterance
        t_drain_start = time.perf_counter()
        while time.perf_counter() - t_drain_start < drain_timeout_sec:
            while msg_cursor < len(self.received_messages):
                msg = self.received_messages[msg_cursor]
                msg_cursor += 1
                timeline.note_subtitle_received(msg)

            # Check if we got final commits
            if timeline.finals:
                # Give small extra grace for trailing translation/TTS
                await asyncio.sleep(0.3)
                while msg_cursor < len(self.received_messages):
                    msg = self.received_messages[msg_cursor]
                    msg_cursor += 1
                    timeline.note_subtitle_received(msg)
                break

            await asyncio.sleep(0.05)

        # Final sweep
        while msg_cursor < len(self.received_messages):
            msg = self.received_messages[msg_cursor]
            msg_cursor += 1
            timeline.note_subtitle_received(msg)
