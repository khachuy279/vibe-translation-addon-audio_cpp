"""Dịch vụ nền HTTP micro-daemon cho OmniVoice C++ GGUF.

Chức năng:
- Chạy trên một tiến trình riêng biệt (Port cục bộ 127.0.0.1).
- Hoàn toàn độc lập không bị xung đột DLL với transcribe_cpp / llama_cpp.
- Nạp OmniVoice C-ABI DLL và giữ mô hình GGUF thường trú trong RAM (Zero Reload).
- Phục vụ API tổng hợp giọng nói cực nhanh qua HTTP /binary audio.
"""

import sys
import os
import json
import base64
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import threading
import numpy as np
import soundfile as sf
import io

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.tts.bindings import OmniVoiceCppEngine
from backend.utils.logger import get_logger

logger = get_logger("tts.daemon")

_engine: OmniVoiceCppEngine = None
_engine_lock = threading.RLock()


class TTSHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Không in access log rác vào console
        pass

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            with _engine_lock:
                loaded = _engine is not None and getattr(_engine, "_is_initialized", False)
                self._send_json(200, {"status": "ok", "loaded": loaded})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        global _engine
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            req = json.loads(post_body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        if self.path == "/init":
            model_path = req.get("model_path")
            codec_path = req.get("codec_path")
            use_fa = req.get("use_fa", True)
            clamp_fp16 = req.get("clamp_fp16", False)

            with _engine_lock:
                try:
                    if _engine is None:
                        _engine = OmniVoiceCppEngine()
                    _engine.init_context(
                        model_path=model_path,
                        codec_path=codec_path,
                        use_fa=use_fa,
                        clamp_fp16=clamp_fp16,
                    )
                    self._send_json(200, {"status": "ok", "version": _engine.get_version()})
                except Exception as e:
                    logger.error(f"Lỗi init model: {e}")
                    self._send_json(500, {"error": str(e)})

        elif self.path == "/synthesize":
            with _engine_lock:
                if _engine is None:
                    self._send_json(503, {"error": "Mô hình chưa được khởi tạo với /init"})
                    return

                text = req.get("text", "")
                ref_rvq = req.get("ref_rvq")
                ref_text = req.get("ref_text")
                ref_wav = req.get("ref_wav")
                lang = req.get("lang", "")
                instruct = req.get("instruct", "")
                num_steps = req.get("num_steps", 2)
                guidance_scale = req.get("guidance_scale", 2.0)
                seed = req.get("seed", 42)

                try:
                    audio_np, sr = _engine.synthesize(
                        text=text,
                        ref_rvq_path=ref_rvq,
                        ref_text=ref_text,
                        ref_wav_path=ref_wav,
                        lang=lang,
                        instruct=instruct,
                        num_steps=num_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                    )

                    # Xuất WAV bytes trực tiếp
                    wav_io = io.BytesIO()
                    sf.write(wav_io, audio_np, sr, format="WAV", subtype="PCM_16")
                    wav_bytes = wav_io.getvalue()

                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(wav_bytes)))
                    self.send_header("X-Sample-Rate", str(sr))
                    self.send_header("X-Num-Samples", str(len(audio_np)))
                    self.end_headers()
                    self.wfile.write(wav_bytes)

                except Exception as e:
                    logger.error(f"Lỗi synthesize: {e}")
                    self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Not Found"})


def run_daemon(host: str = "127.0.0.1", port: int = 18888):
    server = HTTPServer((host, port), TTSHandler)
    logger.info(f"🚀 OmniVoice TTS Daemon đang lắng nghe tại http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("🛑 OmniVoice TTS Daemon đã dừng.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18888)
    args = parser.parse_args()
    run_daemon(host=args.host, port=args.port)
