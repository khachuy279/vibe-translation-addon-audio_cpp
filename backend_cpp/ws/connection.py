"""Thread-safe and task-safe WebSocket connection wrapper for backend_cpp."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class SafeWebSocketConnection:
    """Encapsulates a FastAPI WebSocket connection to ensure serialized, race-condition-free writes.

    ASGI and Starlette do not permit concurrent calls to WebSocket.send_* across tasks.
    This wrapper enforces an asyncio.Lock on all outbound sends and tracks connection state.
    """

    def __init__(self, ws: WebSocket):
        self._ws: WebSocket = ws
        self._send_lock: asyncio.Lock = asyncio.Lock()
        self._is_closed: bool = False

    @property
    def raw_ws(self) -> WebSocket:
        """Direct reference to underlying FastAPI WebSocket."""
        return self._ws

    @property
    def is_closed(self) -> bool:
        """True if client disconnected or connection has been closed."""
        return self._is_closed

    async def accept(self) -> None:
        """Accept WebSocket handshake."""
        await self._ws.accept()

    async def receive(self) -> Dict[str, Any]:
        """Receive incoming raw ASGI message frame."""
        return await self._ws.receive()

    async def send_text(self, text: str) -> bool:
        """Send raw text message serialized under send lock."""
        if self._is_closed:
            return False

        try:
            async with self._send_lock:
                if self._is_closed:
                    return False
                await self._ws.send_text(text)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self._is_closed = True
            return False
        except Exception as e:
            logger.debug(f"SafeWebSocketConnection send_text error: {e}")
            self._is_closed = True
            return False

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        """Serialize payload to JSON and send safely under send lock."""
        if self._is_closed:
            return False
        try:
            raw_text = json.dumps(payload)
            return await self.send_text(raw_text)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize payload to JSON: {e}")
            return False

    async def close(self, code: int = 1000, reason: Optional[str] = None) -> None:
        """Gracefully close connection with an optional close reason."""
        if self._is_closed:
            return
        self._is_closed = True
        try:
            if reason:
                await self._ws.close(code=code, reason=reason)
            else:
                await self._ws.close(code=code)
        except Exception:
            pass
