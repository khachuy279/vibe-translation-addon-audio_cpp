"""Bilingual Subtitle Backend CPP – Entry Point using transcribe.cpp."""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Ensure project root is in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Disable tqdm / HF progress bars to prevent '\r' from overwriting console log lines
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from backend_cpp.config import config, SUPPORTED_LANGUAGES
from backend_cpp.vad import SUPPORTED_VAD_ENGINES
from backend_cpp.asr.model_registry import ModelRegistry
from backend_cpp.asr.transcribe_engine import TranscribeEngine
from backend_cpp.translation import get_translator, reset_translator, TranslationModelRegistry
from backend_cpp.tts import VoiceManager, OmniVoiceTTS
from backend_cpp.utils.cuda_utils import setup_cuda_dll_paths
from backend_cpp.utils.perf_profiler import perf
from backend_cpp.ws.ws_handler import handle_ws

# Setup logging.
# Default is INFO: DEBUG on a realtime pipeline is expensive (every VAD frame / preview
# cycle can emit records) and the audit flagged it as I/O overhead on the hot path.
# Opt in explicitly for diagnostics with:  BS_LOG_LEVEL=DEBUG
_LOG_LEVEL_NAME = os.environ.get("BS_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backend_cpp")
logger.info(f"Log level: {logging.getLevelName(_LOG_LEVEL)} (set BS_LOG_LEVEL=DEBUG for verbose output)")

# Windows DLL registration
setup_cuda_dll_paths()

# Configure UTF-8 for Windows console to prevent emoji/charmap crash
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass

# Track active background tasks to prevent garbage collection and catch unhandled exceptions
_background_tasks: set = set()


def track_background_task(coro, name: str = "background_task") -> asyncio.Task:
    """Create and track an asyncio task to prevent early GC and log unhandled exceptions."""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(
        lambda t: (
            _background_tasks.discard(t),
            logger.error(f"Task '{name}' failed: {t.exception()}", exc_info=t.exception())
            if not t.cancelled() and t.exception()
            else None,
        )
    )
    return task


class SwitchModelRequest(BaseModel):
    model_id: Optional[str] = None
    asr_engine: Optional[str] = None
    vad_engine: Optional[str] = None
    vad_threshold: Optional[float] = None
    silence_duration_ms: Optional[int] = None
    hangover_ms: Optional[int] = None
    pre_speech_buffer_ms: Optional[int] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    translation_model: Optional[str] = None
    min_words_to_commit: Optional[int] = None
    tts_enabled: Optional[bool] = None
    tts_voice: Optional[str] = None
    tts_speed: Optional[float] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm transcribe.cpp model, VAD model & local GGUF translation model on startup."""
    logger.info("🔥 [STARTUP] Pre-warming transcribe.cpp ASR, VAD & Local Translation models...")

    def _warmup():
        # 1. Prewarm ASR model on GPU
        try:
            engine = TranscribeEngine()
            engine.prewarm()
            logger.info("✅ [STARTUP] ASR model pre-warmed!")
        except Exception as e:
            logger.warning(f"Startup ASR pre-warming warning: {e}", exc_info=True)

        # 2. Prewarm Translation model on GPU
        try:
            translator = get_translator()
            translator.load_model()
            logger.info("✅ [STARTUP] Translation model pre-warmed!")
        except Exception as e:
            logger.warning(f"Startup Translation pre-warming warning: {e}", exc_info=True)

        # 3. Prewarm VAD model on CPU
        try:
            from backend_cpp.vad.vad_processor import VADProcessor
            vad = VADProcessor(vad_engine=config.vad.vad_engine)
            vad._ensure_model()
            logger.info(f"✅ [STARTUP] VAD ({config.vad.vad_engine}) pre-warmed!")
        except Exception as e:
            logger.warning(f"Startup VAD pre-warming warning: {e}", exc_info=True)

        logger.info("🚀 [STARTUP] Model warmup sequence complete!")

    await asyncio.to_thread(_warmup)
    yield
    logger.info("🛑 [SHUTDOWN] Releasing GPU, translation & TTS model resources...")
    for t in list(_background_tasks):
        if not t.done():
            t.cancel()
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)

    TranscribeEngine.shutdown_executors(wait=False)
    # Offload: unload waits for in-flight native inference before closing the model.
    if not await asyncio.to_thread(TranscribeEngine.unload_shared_model):
        logger.warning(
            "[SHUTDOWN] ASR model unload was skipped because the inference lock stayed busy; "
            "native resources will be released on process exit."
        )
    reset_translator()
    OmniVoiceTTS.reset_instance()


app = FastAPI(title="Bilingual Subtitle Backend (transcribe.cpp)", version="1.0.0", lifespan=lifespan)

# Allow CORS for Firefox, Chrome, Edge extensions and localhost
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
        "service": "backend_cpp",
        "engine": "transcribe.cpp",
        "active_model": ModelRegistry.get_instance().get_active_model_key(),
    }


@app.get("/", response_class=HTMLResponse)
async def root_page():
    """SSL verification landing page."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Bilingual Subtitle Backend CPP</title></head>
    <body style="font-family: system-ui, sans-serif; text-align: center; padding-top: 60px; background: #0f172a; color: #f8fafc;">
        <h1 style="color: #38bdf8;">✅ Bilingual Subtitle Backend CPP</h1>
        <p style="font-size: 16px;">Backend đang hoạt động ở chế độ bảo mật <strong>HTTPS / WSS</strong>.</p>
        <p style="color: #4ade80; font-weight: bold; font-size: 18px;">Chứng chỉ SSL đã được xác nhận thành công!</p>
        <p style="color: #94a3b8; font-size: 14px;">Bạn có thể đóng tab này và bắt đầu sử dụng Extension trên Firefox.</p>
    </body>
    </html>
    """


def _build_config_response(include_catalog: bool = True) -> Dict[str, Any]:
    """Build unified backend configuration response for GET and POST endpoints."""
    registry = ModelRegistry.get_instance()
    active_key = registry.get_active_model_key()
    active_info = registry.get_model_info(active_key) or {}
    loaded_model_name = active_info.get("name", active_key)

    resp: Dict[str, Any] = {
        "status": "ok",
        "engine": active_key,
        "asr_engine": active_key,
        "active_model": active_key,
        "loaded_model": loaded_model_name,
        "resolved_vad": config.vad.vad_engine,
        "vad_engine": config.vad.vad_engine,
        "available_vad_engines": list(SUPPORTED_VAD_ENGINES),
        "vad_silence_duration_ms": config.vad.silence_duration_ms,
        "silence_duration_ms": config.vad.silence_duration_ms,
        "vad_threshold": config.vad.threshold,
        "vad_hangover_ms": config.vad.hangover_ms,
        "vad_pre_speech_buffer_ms": config.vad.pre_speech_buffer_ms,
        "vad_profiles": {k: v.model_dump() for k, v in config.vad.engine_profiles.items()},
        "min_words_to_commit": config.sentence.min_words_to_commit,
        "source_lang": config.asr.language,
        "supported_languages": SUPPORTED_LANGUAGES,
        "translation": {
            "model": config.translation.model,
            "gguf_file": config.translation.gguf_file,
            "target_lang": config.translation.target_lang,
            "source_lang": config.translation.source_lang,
            "base": config.translation.base,
            "translation_model": config.translation.base,
            "available_models": TranslationModelRegistry.get_instance().list_models(),
        },
        "tts": {
            "enabled": config.tts.enabled,
            "engine": config.tts.engine,
            "model": config.tts.model,
            "speed": config.tts.speed,
            "default_voice": config.tts.default_voice,
            "voices": VoiceManager.get_available_voices(),
        },
    }

    if include_catalog:
        available_models = registry.list_models()
        resp["available_models"] = available_models
        resp["available_asr_engines"] = [m["id"] for m in available_models]

    return resp


@app.get("/api/config")
async def get_backend_config():
    """Return backend configuration, active model, and available models."""
    return _build_config_response(include_catalog=True)


@app.get("/api/voices")
async def get_voices_list():
    """Return available voice clone samples found in backend_cpp/voices."""
    return {"status": "ok", "voices": VoiceManager.get_available_voices()}


@app.post("/api/config")
@app.post("/api/switch-engine")
async def update_backend_config(req: SwitchModelRequest):
    """Hot swap ASR model or update VAD/translation/TTS configuration on the fly."""
    registry = ModelRegistry.get_instance()
    target_model = req.model_id or req.asr_engine

    if target_model:
        try:
            registry.set_active_model_key(target_model)
            # Unload old shared model and load new one.
            # Offload to a thread: unload waits for in-flight inference to finish and
            # must not block the event loop.
            unloaded = await asyncio.to_thread(TranscribeEngine.unload_shared_model)
            if not unloaded:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "An ASR inference is currently in flight; the model was left untouched. "
                        "Retry the switch in a moment."
                    ),
                )
            engine = TranscribeEngine(target_model)
            await asyncio.to_thread(engine.prewarm)
            logger.info(f"Switched active model to '{target_model}'")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if req.vad_engine is not None:
        ve = req.vad_engine.lower().strip()
        if ve in ("none", "off", "disabled"):
            raise HTTPException(
                status_code=400,
                detail=f"VAD is mandatory and cannot be disabled. Supported engines: {list(SUPPORTED_VAD_ENGINES)}",
            )
        elif ve in SUPPORTED_VAD_ENGINES:
            config.vad.enabled = True
            old_ve = config.vad.vad_engine
            if ve != old_ve:
                profile = config.vad.apply_engine_profile(ve)
                logger.info(f"Switched default VAD engine to '{ve}' with optimal profile: {profile}")
            else:
                config.vad.vad_engine = ve
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown VAD engine '{ve}'. Supported engines: {list(SUPPORTED_VAD_ENGINES)}",
            )

    if req.vad_threshold is not None:
        config.vad.threshold = req.vad_threshold
        logger.info(f"Updated default vad_threshold to {config.vad.threshold}")
    if req.silence_duration_ms is not None:
        config.vad.silence_duration_ms = req.silence_duration_ms
        logger.info(f"Updated default silence_duration_ms to {config.vad.silence_duration_ms}")
    if req.hangover_ms is not None:
        config.vad.hangover_ms = req.hangover_ms
        logger.info(f"Updated default hangover_ms to {config.vad.hangover_ms}")
    if req.pre_speech_buffer_ms is not None:
        config.vad.pre_speech_buffer_ms = req.pre_speech_buffer_ms
        logger.info(f"Updated default pre_speech_buffer_ms to {config.vad.pre_speech_buffer_ms}")
    if req.min_words_to_commit is not None:
        config.sentence.min_words_to_commit = max(0, req.min_words_to_commit)
        logger.info(f"Updated default min_words_to_commit to {config.sentence.min_words_to_commit}")
    if req.target_lang is not None:
        config.translation.target_lang = req.target_lang
    if req.source_lang is not None:
        config.asr.language = req.source_lang

    if req.translation_model is not None:
        tm = req.translation_model.lower().strip()
        trans_registry = TranslationModelRegistry.get_instance()
        canonical_key = trans_registry.resolve_key(tm)
        model_info = trans_registry.get_model(canonical_key)
        if model_info is not None:
            try:
                from backend_cpp.config import TranslationConfig
                config.translation.base = canonical_key
                new_trans_cfg = TranslationConfig(
                    base=canonical_key,
                    target_lang=config.translation.target_lang,
                    source_lang=config.translation.source_lang,
                )
                translator = get_translator(new_trans_cfg)
                await asyncio.to_thread(translator.load_model)
                logger.info(f"Switched active translation model to '{canonical_key}' ({new_trans_cfg.gguf_file})")
            except Exception as e:
                logger.error(f"Failed to switch translation model to '{canonical_key}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to load translation model '{canonical_key}': {str(e)}")
        else:
            logger.warning(f"Unknown translation model requested: '{req.translation_model}'")

    if req.tts_enabled is not None:
        config.tts.enabled = req.tts_enabled
        if req.tts_enabled:
            from backend_cpp.tts import get_tts_engine
            track_background_task(get_tts_engine().prewarm(), name="tts_prewarm_post_config")
    if req.tts_voice is not None:
        config.tts.default_voice = req.tts_voice
    if req.tts_speed is not None:
        config.tts.speed = req.tts_speed

    return _build_config_response(include_catalog=True)


@app.post("/api/tts/prewarm")
async def prewarm_tts_endpoint():
    """Explicit endpoint to prewarm configured TTS engine on GPU."""
    from backend_cpp.tts import get_tts_engine
    success = await get_tts_engine().prewarm()
    if success:
        config.tts.enabled = True
    return {"status": "ok" if success else "error", "prewarmed": success}


@app.get("/api/perf/summary")
async def get_performance_summary():
    """Return realtime performance telemetry, resource usage, and bottleneck alerts."""
    return perf.generate_report()


@app.post("/api/perf/reset")
async def reset_performance_metrics():
    """Reset all performance counters and latency metrics."""
    perf.reset()
    return {"status": "ok", "message": "Performance metrics have been reset."}


@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(ws: WebSocket):
    """Main WebSocket endpoint matching extension_firefox protocol."""
    await handle_ws(ws)


import time
from backend_cpp.ws.session_state import ACTIVE_SESSIONS, LAST_DISCONNECTED_REPORT


@app.get("/api/telemetry/transport")
async def get_transport_telemetry():
    """Return P4-A audio transport telemetry for active session or last closed session."""
    from backend_cpp.ws import session_state
    if session_state.ACTIVE_SESSIONS:
        latest_sess = list(session_state.ACTIVE_SESSIONS.values())[-1]
        summary = latest_sess.transport_telemetry.get_summary()
        summary["session_id"] = latest_sess.session_id
        summary["status"] = "active"
        summary["duration_sec"] = round(time.time() - latest_sess.connected_at, 2)
        return summary
    elif session_state.LAST_DISCONNECTED_REPORT:
        report = dict(session_state.LAST_DISCONNECTED_REPORT)
        report["status"] = "disconnected"
        return report
    else:
        return {"status": "no_session", "message": "No transport session recorded"}


def main():
    """Server runner with SSL certificates support and fallback."""
    import argparse
    parser = argparse.ArgumentParser(description="Bilingual Subtitle Backend")
    parser.add_argument("--no-ssl", action="store_true", help="Run without SSL (plain HTTP/WS mode)")
    args, _ = parser.parse_known_args()

    ssl_kwargs = {}
    if not args.no_ssl:
        from backend_cpp.utils.ssl_utils import ensure_ssl_certificates
        cert_path, key_path = ensure_ssl_certificates()
        ssl_kwargs = {
            "ssl_certfile": cert_path,
            "ssl_keyfile": key_path,
        }
        logger.info(f"🔒 WSS (SSL) enabled with cert: {cert_path}")
    else:
        logger.warning("⚠️ Running in plain HTTP/WS mode (--no-ssl)")

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
