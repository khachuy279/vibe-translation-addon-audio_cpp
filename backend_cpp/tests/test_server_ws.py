"""WebSocket integration test simulating extension_firefox connecting to backend_cpp."""

import asyncio
import json
import os
import struct
import sys
import wave
import websockets

import ssl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

@pytest.mark.asyncio
async def test_ws_client():
    cert_path_cpp = os.path.join(PROJECT_ROOT, "backend_cpp", "cert.pem")
    cert_path_old = os.path.join(PROJECT_ROOT, "backend", "cert.pem")
    use_ssl = os.path.exists(cert_path_cpp) or os.path.exists(cert_path_old)
    uri = "wss://127.0.0.1:8765/ws" if use_ssl else "ws://127.0.0.1:8765/ws"
    print(f"Connecting to {uri} (use_ssl={use_ssl})...")

    ssl_ctx = None
    if use_ssl:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        async with websockets.connect(uri, ssl=ssl_ctx) as ws:
            print("Connected to WebSocket!")

            # 1. Send set_config
            config_msg = {
                "type": "set_config",
                "sourceLang": "auto",
                "targetLang": "vi",
                "vadThreshold": 0.45,
                "silenceDurationMs": 400,
            }
            await ws.send(json.dumps(config_msg))
            print("Sent set_config")

            # 2. Send ping
            await ws.send(json.dumps({"type": "ping", "timestamp": 12345}))
            pong = await ws.recv()
            pong_data = json.loads(pong)
            print("Received ping response:", pong_data)
            assert pong_data.get("type") == "pong"

            # 3. Stream audio chunks (Format B: 8-byte float timestamp + Int16 PCM)
            wav_path = os.path.join(PROJECT_ROOT, "wav_test", "OSR_us_000_0010_16k.wav")
            with wave.open(wav_path, "rb") as wf:
                raw_bytes = wf.readframes(16000 * 5)  # 5 seconds

            chunk_samples = 320  # 20ms chunks like extension
            chunk_bytes = chunk_samples * 2

            received_updates = []

            async def _receiver():
                try:
                    while True:
                        msg_raw = await ws.recv()
                        data = json.loads(msg_raw)
                        msg_type = data.get("type")
                        if msg_type == "utterance_update":
                            received_updates.append(data)
                            text = data.get("text", "")
                            trans = data.get("translated", "")
                            is_fin = data.get("is_final", False)
                            print(f"  <- WS [utterance_update] is_final={is_fin} | text='{text}' | trans='{trans}'")
                        elif msg_type == "translation":
                            received_updates.append(data)
                            print(f"  <- WS [translation] id={data.get('sentence_id')} | trans='{data.get('translated')}'")
                            if data.get("translated"):
                                print("Received translation from server!")
                                break
                except asyncio.CancelledError:
                    pass

            recv_task = asyncio.create_task(_receiver())

            print(f"Streaming {len(raw_bytes)} bytes of audio chunks to /ws...")
            ts = 0.0
            for off in range(0, len(raw_bytes), chunk_bytes):
                pcm = raw_bytes[off:off + chunk_bytes]
                header = struct.pack("<d", ts)
                packet = header + pcm
                await ws.send(packet)
                ts += 0.02
                await asyncio.sleep(0.01)

            # Send silence (1.4s) to trigger VAD speech end
            print("Sending silence to trigger VAD speech end...")
            silence_pcm = b"\x00" * chunk_bytes
            for _ in range(70):
                header = struct.pack("<d", ts)
                await ws.send(header + silence_pcm)
                ts += 0.02
                await asyncio.sleep(0.02)

            try:
                await asyncio.wait_for(recv_task, timeout=12.0)
            except asyncio.TimeoutError:
                print("Receiver timeout waiting for translation")
                recv_task.cancel()

            print(f"Total updates received: {len(received_updates)}")
            assert len(received_updates) > 0, "No utterance_updates received!"
            print("WebSocket client test passed successfully!")
    except (ConnectionRefusedError, OSError) as e:
        pytest.skip(f"Live server not running ({e})")

if __name__ == "__main__":
    asyncio.run(test_ws_client())
