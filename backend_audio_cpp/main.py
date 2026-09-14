"""Bilingual Subtitle Backend – Entry Point using native audio.cpp & Hy-MT2."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend_audio_cpp.config import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_VAD_ENGINES,
    config,
)
from backend_audio_cpp.asr.asr_engine import AudioCppASREngine
from backend_audio_cpp.asr.model_registry import ModelRegistry
from backend_audio_cpp.translation.hy_translator import HyMTTranslator
from backend_audio_cpp.translation.model_registry import TranslationModelRegistry
from backend_audio_cpp.tts.voice_manager import VoiceManager
from backend_audio_cpp.tts.omnivoice_engine import OmniVoiceTTSEngine
from backend_audio_cpp.vad.vad_engine import SileroVADEngine
from backend_audio_cpp.utils.ssl_utils import ensure_ssl_certificates
from backend_audio_cpp.ws.ws_handler import handle_ws

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend_audio_cpp")

# Windows console UTF-8 setup
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


class SwitchModelRequest(BaseModel):
    model_id: Optional[str] = None
    asr_engine: Optional[str] = None
    vad_engine: Optional[str] = None
    vad_threshold: Optional[float] = None
    silence_duration_ms: Optional[int] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    translation_model: Optional[str] = None
    min_words_to_commit: Optional[int] = None
    tts_enabled: Optional[bool] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm native audio.cpp server, VAD, ASR, and Hunyuan-MT2 Translation."""
    logger.info("🔥 [STARTUP] Pre-warming native audio.cpp runtime, VAD & GPU Translation models...")

    def _warmup():
        try:
            # 1. Ensure audio.cpp server is up and prewarm ASR
            asr = AudioCppASREngine.get_instance()
            logger.info(f"✅ [STARTUP] ASR Engine ready (active: {asr.get_active_model()})")

            # 2. Prewarm Translation model on GPU
            translator = HyMTTranslator.get_instance()
            logger.info("✅ [STARTUP] Hunyuan-MT2 Translation model ready on GPU!")

            # 3. Prewarm Silero VAD on CPU
            _ = SileroVADEngine()
            logger.info("✅ [STARTUP] Silero VAD model ready on CPU!")

            # 4. Prewarm OmniVoice TTS
            _ = OmniVoiceTTSEngine.get_instance()
            logger.info("✅ [STARTUP] OmniVoice TTS Engine ready!")

        except Exception as e:
            logger.warning(f"Startup warmup notice: {e}", exc_info=True)

    await asyncio.to_thread(_warmup)
    yield
    logger.info("🛑 [SHUTDOWN] Releasing backend resources...")


app = FastAPI(
    title="Bilingual Subtitle Backend (audio.cpp)",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow CORS for browser extensions and localhost
_CORS_ORIGIN_REGEX = (
    r"http://localhost(:\d+)?"
    r"|https://localhost(:\d+)?"
    r"|moz-extension://[0-9a-f-]+"
    r"|chrome-extension://[0-9a-z-]+"
    r"|edge-extension://[0-9a-z-]+"
    r"|extension://[0-9a-z-]+"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "backend_audio_cpp",
        "engine": "audio.cpp",
        "active_model": ModelRegistry.get_instance().get_active_model_key(),
    }


@app.get("/", response_class=HTMLResponse)
async def root_page():
    """SSL verification landing page for Firefox extension."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Bilingual Subtitle Backend (audio.cpp)</title></head>
    <body style="font-family: system-ui, sans-serif; text-align: center; padding-top: 60px; background: #0f172a; color: #f8fafc;">
        <h1 style="color: #38bdf8;">✅ Bilingual Subtitle Backend (audio.cpp)</h1>
        <p style="font-size: 16px;">Backend đang hoạt động an toàn ở chế độ <strong>HTTPS / WSS</strong> trên port 8765.</p>
        <p style="color: #4ade80; font-weight: bold; font-size: 18px;">Chứng chỉ SSL đã được xác nhận thành công!</p>
        <p style="color: #94a3b8; font-size: 14px;">Bạn có thể đóng tab này và bắt đầu sử dụng Extension trên Firefox.</p>
    </body>
    </html>
    """


def _build_config_response(include_catalog: bool = True) -> Dict[str, Any]:
    """Build unified backend configuration response for extension popup."""
    asr_registry = ModelRegistry.get_instance()
    active_asr_key = asr_registry.get_active_model_key()
    active_asr_info = asr_registry.get_model(active_asr_key) or {}
    loaded_model_name = active_asr_info.get("name", active_asr_key)

    trans_registry = TranslationModelRegistry.get_instance()
    active_trans_key = trans_registry.active_model_key

    vm = VoiceManager.get_instance()

    resp: Dict[str, Any] = {
        "status": "ok",
        "engine": active_asr_key,
        "asr_engine": active_asr_key,
        "active_model": active_asr_key,
        "loaded_model": loaded_model_name,
        "resolved_vad": "silero-vad",
        "vad_engine": "silero-vad",
        "available_vad_engines": SUPPORTED_VAD_ENGINES,
        "vad_silence_duration_ms": config.vad.silence_duration_ms,
        "silence_duration_ms": config.vad.silence_duration_ms,
        "vad_threshold": config.vad.threshold,
        "min_words_to_commit": config.sentence.min_words_to_commit,
        "source_lang": config.asr.language,
        "supported_languages": SUPPORTED_LANGUAGES,
        "translation": {
            "model": active_trans_key,
            "translation_model": active_trans_key,
            "base": active_trans_key,
            "gguf_file": config.translation.gguf_file,
            "target_lang": config.translation.target_lang,
            "source_lang": config.translation.source_lang,
            "available_models": trans_registry.list_models(),
        },
        "tts": {
            "enabled": config.tts.enabled,
            "engine": config.tts.engine,
            "model": config.tts.model,
            "speed": config.tts.speed,
            "default_voice": config.tts.default_voice,
            "voices": vm.list_voices(),
        },
    }

    if include_catalog:
        available_models = asr_registry.list_models()
        resp["available_models"] = available_models
        resp["available_asr_engines"] = [m["id"] for m in available_models]

    return resp


@app.get("/api/config")
async def get_backend_config():
    """Return full backend configuration catalog."""
    return _build_config_response(include_catalog=True)


@app.get("/api/voices")
async def get_voices_list():
    """Return available voice clone samples found in backend_audio_cpp/voices."""
    return {"status": "ok", "voices": VoiceManager.get_instance().list_voices()}


@app.post("/api/config")
@app.post("/api/switch-engine")
async def update_backend_config(req: SwitchModelRequest):
    """Hot swap ASR model or update VAD/Translation/TTS parameters live."""
    target_model = req.model_id or req.asr_engine
    if target_model:
        try:
            asr = AudioCppASREngine.get_instance()
            canonical_key = asr.switch_model(target_model)
            logger.info(f"Switched active ASR model to '{canonical_key}'")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if req.vad_threshold is not None:
        config.vad.threshold = req.vad_threshold
        logger.info(f"Updated default vad_threshold to {config.vad.threshold}")

    if req.silence_duration_ms is not None:
        config.vad.silence_duration_ms = req.silence_duration_ms
        logger.info(f"Updated default silence_duration_ms to {config.vad.silence_duration_ms}")

    if req.min_words_to_commit is not None:
        config.sentence.min_words_to_commit = max(0, req.min_words_to_commit)
        logger.info(f"Updated default min_words_to_commit to {config.sentence.min_words_to_commit}")

    if req.target_lang is not None:
        config.translation.target_lang = req.target_lang

    if req.source_lang is not None:
        config.asr.language = req.source_lang

    if req.translation_model is not None:
        tm = req.translation_model.lower().strip()
        trans_reg = TranslationModelRegistry.get_instance()
        try:
            canonical_trans = trans_reg.set_active_model_key(tm)
            config.translation.base = canonical_trans
            logger.info(f"Switched active translation model to '{canonical_trans}'")
        except Exception as e:
            logger.warning(f"Failed to switch translation model: {e}")

    if req.tts_enabled is not None:
        config.tts.enabled = req.tts_enabled

    if req.tts_voice is not None:
        config.tts.default_voice = req.tts_voice

    if req.tts_speed is not None:
        config.tts.speed = req.tts_speed

    return _build_config_response(include_catalog=True)


@app.post("/api/tts/prewarm")
async def prewarm_tts():
    """Explicit endpoint to prewarm configured TTS engine."""
    _ = OmniVoiceTTSEngine.get_instance()
    config.tts.enabled = True
    return {"status": "ok", "prewarmed": True}


@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(ws: WebSocket):
    """Main WebSocket endpoint matching Firefox extension protocol."""
    await handle_ws(ws)


def main():
    """Server runner with SSL certificates support on port 8765."""
    cert_path, key_path = ensure_ssl_certificates()

    ssl_kwargs = {
        "ssl_certfile": cert_path,
        "ssl_keyfile": key_path,
    }
    logger.info(f"🔒 WSS (SSL) enabled with cert: {cert_path}")
    logger.info(f"🚀 Starting backend_audio_cpp on https://{config.ws.host}:{config.ws.port}")

    uvicorn.run(
        app,
        host=config.ws.host,
        port=config.ws.port,
        ws_ping_interval=config.ws.ping_interval,
        ws_ping_timeout=config.ws.ping_timeout,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
