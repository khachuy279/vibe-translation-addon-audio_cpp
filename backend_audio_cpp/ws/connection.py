"""Thread-safe & async-safe WebSocket connection wrapper."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union
from fastapi import WebSocket

logger = logging.getLogger("backend_audio_cpp.ws.connection")


class SafeWebSocketConnection:
    """Safely synchronizes concurrent writes over a FastAPI WebSocket connection."""

    def __init__(self, ws: WebSocket):
        self._ws: WebSocket = ws
        self._send_lock: asyncio.Lock = asyncio.Lock()
        self._closed: bool = False

    @property
    def raw_ws(self) -> WebSocket:
        return self._ws

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def accept(self) -> None:
        await self._ws.accept()

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        """Send JSON payload safely under lock."""
        if self._closed:
            return False
        try:
            text = json.dumps(payload, ensure_ascii=False)
            async with self._send_lock:
                if self._closed:
                    return False
                await self._ws.send_text(text)
            return True
        except Exception as e:
            logger.debug(f"Error sending JSON payload: {e}")
            self._closed = True
            return False

    async def send_text(self, text: str) -> bool:
        """Send raw text safely under lock."""
        if self._closed:
            return False
        try:
            async with self._send_lock:
                if self._closed:
                    return False
                await self._ws.send_text(text)
            return True
        except Exception as e:
            logger.debug(f"Error sending text: {e}")
            self._closed = True
            return False

    async def send_bytes(self, data: bytes) -> bool:
        """Send binary data safely under lock."""
        if self._closed:
            return False
        try:
            async with self._send_lock:
                if self._closed:
                    return False
                await self._ws.send_bytes(data)
            return True
        except Exception as e:
            logger.debug(f"Error sending binary: {e}")
            self._closed = True
            return False

    async def receive(self) -> Dict[str, Any]:
        """Receive message from websocket."""
        return await self._ws.receive()

    async def close(self, code: int = 1000) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close(code=code)
        except Exception:
            pass
