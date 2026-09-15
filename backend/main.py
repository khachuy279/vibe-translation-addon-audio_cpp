"""Điểm khởi chạy chính (Entry Point) của Hệ Thống Backend Real-Time Audio Processing.

Kiến trúc:
- FastAPI Web Framework với Lifespan quản lý nạp/giải phóng mô hình GPU/CPU.
- Hỗ trợ đầy đủ CORS cho Firefox, Chrome, Edge Extension và Localhost.
- WebSocket Streaming Endpoint: /ws (tối ưu hóa độ trễ cho 1 session).
- Quản lý REST API: Hot-switch ASR/Translation/VAD/TTS runtime.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Đảm bảo đường dẫn gốc dự án trong sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Tối ưu hóa Intel OpenMP / PyTorch / MKL để triệt tiêu hiện tượng busy-spin gây 100% CPU trên đa nhân
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# Tắt thanh tiến trình tqdm của HuggingFace để không làm rác log console
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Thiết lập giới hạn luồng PyTorch CPU sớm nhất có thể
try:
    import torch
    torch.set_num_threads(2)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(2)
except ImportError:
    pass

# Windows CUDA DLL setup
from backend.utils.cuda import setup_cuda_dll_paths
setup_cuda_dll_paths()

from backend.config import config, SUPPORTED_LANGUAGES, TranslationConfig
from backend.vad import SUPPORTED_VAD_ENGINES, VADProcessor
from backend.asr.registry import ModelRegistry
from backend.asr.engine import TranscribeEngine
from backend.translation.engine import GGUFTranslationEngine, get_translation_engine, reset_translation_engine
from backend.translation.registry import TranslationModelRegistry
from backend.tts import VoiceManager, OmniVoiceTTS, get_tts_engine
from backend.core.metrics import metrics_collector
from backend.ws.handler import handle_ws
from backend.utils.logger import get_logger

logger = get_logger("main")

_background_tasks: set = set()


def track_background_task(coro, name: str = "background_task") -> asyncio.Task:
    """Tạo và theo dõi tác vụ bất đồng bộ trong background tránh bị Garbage Collector thu hồi sớm."""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(
        lambda t: (
            _background_tasks.discard(t),
            logger.error(f"Task '{name}' gặp lỗi: {t.exception()}", exc_info=t.exception())
            if not t.cancelled() and t.exception()
            else None,
        )
    )
    return task


class SwitchModelRequest(BaseModel):
    """Mô hình nhận request cập nhật cấu hình nóng qua REST API."""
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
    """Khởi tạo và làm ấm trước (Pre-warm) song song toàn bộ các mô hình khi máy chủ khởi động."""
    logger.info("[STARTUP] Đang nạp và làm ấm (Pre-warm) song song ASR, Translation & VAD...")

    def _prewarm_asr():
        try:
            TranscribeEngine().prewarm()
            logger.info("[STARTUP] ASR model đã được nạp & làm ấm thành công!", extra={"module_tag": "ASR"})
        except Exception as e:
            logger.warning(f"[STARTUP] Cảnh báo làm ấm ASR: {e}", exc_info=True, extra={"module_tag": "ASR"})

    def _prewarm_translation():
        try:
            get_translation_engine().load_model()
            logger.info("[STARTUP] Translation model đã được nạp & làm ấm thành công!", extra={"module_tag": "TRANSLATE"})
        except Exception as e:
            logger.warning(f"[STARTUP] Cảnh báo làm ấm Translation: {e}", exc_info=True, extra={"module_tag": "TRANSLATE"})

    def _prewarm_vad():
        try:
            vad = VADProcessor(vad_engine=config.vad.vad_engine)
            vad.feed_chunk(bytes(800))
            logger.info("[STARTUP] VAD engine đã được làm ấm thành công!", extra={"module_tag": "VAD"})
        except Exception as e:
            logger.warning(f"[STARTUP] Cảnh báo làm ấm VAD: {e}", exc_info=True, extra={"module_tag": "VAD"})

    results = await asyncio.gather(
        asyncio.to_thread(_prewarm_asr),
        asyncio.to_thread(_prewarm_translation),
        asyncio.to_thread(_prewarm_vad),
        return_exceptions=True,
    )
    for res in results:
        if isinstance(res, Exception):
            logger.warning(f"[STARTUP] Lỗi thành phần trong quá trình prewarm: {res}", extra={"module_tag": "MAIN"})

    logger.info("[STARTUP] Toàn bộ mô hình đã được làm ấm song song và sẵn sàng phục vụ!", extra={"module_tag": "MAIN"})
    yield
    logger.info("[SHUTDOWN] Đang giải phóng toàn bộ tài nguyên GPU & RAM...")
    for t in list(_background_tasks):
        if not t.done():
            t.cancel()

    try:
        TranscribeEngine.shutdown_executors(wait=False)
        TranscribeEngine.unload_shared_model()
    except Exception as e:
        logger.debug(f"ASR cleanup notice: {e}")

    try:
        from backend.translation.engine import GGUFTranslator
        GGUFTranslator.shutdown_executors(wait=False)
        reset_translation_engine()
    except Exception as e:
        logger.debug(f"Translation cleanup notice: {e}")

    try:
        OmniVoiceTTS.reset_instance()
    except Exception as e:
        logger.debug(f"TTS cleanup notice: {e}")

    try:
        from backend.ws.handler import shutdown_vad_executor
        shutdown_vad_executor(wait=False)
    except Exception as e:
        logger.debug(f"VAD cleanup notice: {e}")

    logger.info("[SHUTDOWN] Hoàn tất tắt máy chủ an toàn.")



app = FastAPI(
    title="Vibe Translation Backend (Refactored Modular)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Regex cho Web Extensions
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
    """Endpoint kiểm tra sức khỏe hệ thống."""
    return {
        "status": "ok",
        "service": "backend_modular",
        "asr_model": ModelRegistry.get_instance().get_active_model_key(),
        "vad_engine": config.vad.vad_engine,
        "translation_model": config.translation.base,
        "tts_model": config.tts.model,
    }


@app.get("/", response_class=HTMLResponse)
async def root_page():
    """Trang xác nhận chứng chỉ SSL/WSS cho trình duyệt."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Vibe Translation Backend Modular</title></head>
    <body style="font-family: system-ui, sans-serif; text-align: center; padding-top: 60px; background: #0f172a; color: #f8fafc;">
        <h1 style="color: #38bdf8;">✅ Vibe Translation Backend Modular v2.0</h1>
        <p style="font-size: 16px;">Backend đang hoạt động ở chế độ bảo mật <strong>HTTPS / WSS</strong>.</p>
        <p style="color: #4ade80; font-weight: bold; font-size: 18px;">Chứng chỉ SSL đã được xác nhận thành công!</p>
        <p style="color: #94a3b8; font-size: 14px;">Bạn có thể đóng tab này và bắt đầu sử dụng Extension trên Firefox.</p>
    </body>
    </html>
    """


def _build_config_response(include_catalog: bool = True) -> Dict[str, Any]:
    """Tạo đối tượng phản hồi cấu hình tập trung cho Client."""
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
    """Trả về cấu hình hiện tại và danh mục mô hình ASR/Translation/TTS."""
    return _build_config_response(include_catalog=True)


@app.get("/api/voices")
async def get_voices_list():
    """Trả về danh sách các mẫu giọng clone có sẵn."""
    return {"status": "ok", "voices": VoiceManager.get_available_voices()}


@app.post("/api/config")
@app.post("/api/switch-engine")
async def update_backend_config(req: SwitchModelRequest):
    """Cập nhật cấu hình runtime động hoặc hot-swap mô hình."""
    registry = ModelRegistry.get_instance()
    target_model = req.model_id or req.asr_engine

    if target_model:
        try:
            registry.set_active_model_key(target_model)
            TranscribeEngine.unload_shared_model()
            engine = TranscribeEngine(target_model)
            await asyncio.to_thread(engine.prewarm)
            logger.info(f"Đã chuyển đổi ASR Model sang: '{target_model}'")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if req.vad_engine is not None:
        ve = req.vad_engine.lower().strip()
        if ve in SUPPORTED_VAD_ENGINES:
            config.vad.vad_engine = ve
            logger.info(f"Đã chuyển VAD engine sang: '{ve}'")

    if req.vad_threshold is not None:
        config.vad.threshold = req.vad_threshold
    if req.silence_duration_ms is not None:
        config.vad.silence_duration_ms = req.silence_duration_ms
    if req.min_words_to_commit is not None:
        config.sentence.min_words_to_commit = max(0, req.min_words_to_commit)
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
                config.translation.base = canonical_key
                new_trans_cfg = TranslationConfig(
                    base=canonical_key,
                    target_lang=config.translation.target_lang,
                    source_lang=config.translation.source_lang,
                )
                translator = get_translation_engine(new_trans_cfg)
                await asyncio.to_thread(translator.load_model)
                logger.info(f"Đã chuyển mô hình dịch sang: '{canonical_key}'")
            except Exception as e:
                logger.error(f"Lỗi chuyển mô hình dịch sang '{canonical_key}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

    if req.tts_enabled is not None:
        config.tts.enabled = req.tts_enabled
        if req.tts_enabled:
            track_background_task(get_tts_engine().prewarm(), name="tts_prewarm_post_config")
    if req.tts_voice is not None:
        config.tts.default_voice = req.tts_voice
    if req.tts_speed is not None:
        config.tts.speed = req.tts_speed

    # Thông báo rõ ràng trên console khi popup cập nhật giá trị
    updated_items = []
    if req.model_id or req.asr_engine:
        updated_items.append(f"asr='{target_model}'")
    if req.vad_engine is not None:
        updated_items.append(f"vad='{req.vad_engine}'")
    if req.vad_threshold is not None:
        updated_items.append(f"threshold={req.vad_threshold}")
    if req.silence_duration_ms is not None:
        updated_items.append(f"silence={req.silence_duration_ms}ms")
    if req.min_words_to_commit is not None:
        updated_items.append(f"min_words={req.min_words_to_commit}")
    if req.source_lang is not None:
        updated_items.append(f"src='{req.source_lang}'")
    if req.target_lang is not None:
        updated_items.append(f"tgt='{req.target_lang}'")
    if req.translation_model is not None:
        updated_items.append(f"trans='{req.translation_model}'")
    if req.tts_enabled is not None:
        updated_items.append(f"tts={req.tts_enabled}")
    if req.tts_voice is not None:
        updated_items.append(f"voice='{req.tts_voice}'")
    if req.tts_speed is not None:
        updated_items.append(f"speed={req.tts_speed}")

    if updated_items:
        logger.info(
            f"[POPUP CONFIG UPDATE] Thay đổi từ Extension Popup: {', '.join(updated_items)}",
            extra={"module_tag": "CONFIG"},
        )

    return _build_config_response(include_catalog=True)


@app.post("/api/tts/prewarm")
async def prewarm_tts_endpoint():
    """Làm ấm mô hình TTS trên GPU."""
    success = await get_tts_engine().prewarm()
    if success:
        config.tts.enabled = True
    return {"status": "ok" if success else "error", "prewarmed": success}


@app.get("/api/metrics")
async def get_metrics():
    """Lấy báo cáo đo lường hiệu năng và cảnh báo điểm nghẽn thời gian thực."""
    return metrics_collector.generate_report()


@app.websocket("/ws")
@app.websocket("/")
async def websocket_endpoint(ws: WebSocket):
    """Endpoint kết nối WebSocket trực tiếp từ Firefox Extension."""
    await handle_ws(ws)


def main():
    """Khởi chạy máy chủ Backend kèm hỗ trợ chứng chỉ WSS tự ký."""
    from backend.utils.ssl import ensure_ssl_certificates
    cert_path, key_path = ensure_ssl_certificates()

    ssl_kwargs = {
        "ssl_certfile": cert_path,
        "ssl_keyfile": key_path,
    }
    logger.info(f"Chế độ WSS (SSL) kích hoạt với cert: {cert_path}")

    try:
        uvicorn.run(
            app,
            host=config.ws.host,
            port=config.ws.port,
            ws_ping_interval=config.ws.ping_interval,
            ws_ping_timeout=config.ws.ping_timeout,
            timeout_graceful_shutdown=1,
            **ssl_kwargs,
        )
    except KeyboardInterrupt:
        logger.info("👋 Nhận tín hiệu ngắt (Ctrl+C). Đã dừng máy chủ an toàn.")



if __name__ == "__main__":
    main()
