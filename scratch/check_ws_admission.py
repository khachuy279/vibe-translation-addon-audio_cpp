"""Diagnostic: does the backend admission control reject a second WebSocket session?

The P1-04 guard (``config.ws.max_sessions = 1``) closes extra sessions with code 1013.
If a session ever lingers (e.g. a stale browser tab, or the extension's background
bridge holding one open), every later START from the extension would fail to connect.

This script connects twice and reports what the server does. It does not send audio,
so no models are loaded. Both sockets are closed before exiting.

Usage:
    python scratch/check_ws_admission.py
"""

import asyncio
import ssl
import sys

URL = "wss://localhost:8765/ws"


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _probe(websockets, label: str, hold: float):
    """Open one connection and report the outcome."""
    try:
        async with websockets.connect(URL, ssl=_ssl_context(), open_timeout=5) as ws:
            print(f"[{label}] handshake OK (admitted)")
            try:
                await asyncio.wait_for(ws.recv(), timeout=hold)
                print(f"[{label}] received a frame from server")
            except asyncio.TimeoutError:
                print(f"[{label}] no server frame within {hold}s (expected: server waits for audio)")
            except Exception as exc:  # connection closed by server
                print(f"[{label}] server closed the connection: {type(exc).__name__}: {exc}")
            return "admitted"
    except Exception as exc:
        print(f"[{label}] handshake FAILED: {type(exc).__name__}: {exc}")
        return f"rejected: {type(exc).__name__}"


async def main() -> int:
    try:
        import websockets
    except ImportError:
        print("websockets library not installed; install with: pip install websockets")
        return 2

    print(f"Probing {URL}\n")

    # Connection 1: occupy the single allowed slot.
    first = await _probe(websockets, "first", hold=1.0)

    # Connection 2: should be rejected while the first is still open.
    async def hold_first():
        try:
            async with websockets.connect(URL, ssl=_ssl_context(), open_timeout=5) as ws:
                print("[holder] handshake OK; holding the slot for 3s")
                await asyncio.sleep(3.0)
                print("[holder] releasing")
        except Exception as exc:
            print(f"[holder] handshake FAILED: {type(exc).__name__}: {exc}")

    holder = asyncio.create_task(hold_first())
    await asyncio.sleep(1.0)
    await _probe(websockets, "second (while first held)", hold=1.0)
    await holder

    # Connection after release: should be admitted again.
    await asyncio.sleep(0.5)
    after = await _probe(websockets, "after release", hold=1.0)

    print("\nSummary")
    print(f"  first          : {first}")
    print(f"  after release  : {after}")
    print(
        "\nExpected with the NEWEST-WINS policy (config.ws.max_sessions = 1):\n"
        "  * 'first' is admitted.\n"
        "  * 'second (while first held)' gets its handshake accepted and then the\n"
        "    OLDER connection is closed with 1013 'superseded_by_new_session'.\n"
        "  * 'after release' is admitted.\n"
        "A newcomer is never rejected, so a stale session can no longer lock START out."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
