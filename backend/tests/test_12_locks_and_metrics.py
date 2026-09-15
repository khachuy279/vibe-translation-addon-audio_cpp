"""Test tầng A — an toàn luồng, lock ordering, metrics bounded.

Tập trung vào các phát hiện rủi ro cao:
- P1.8b / F-04: race use-after-free khi giải phóng model ASR (lock ordering).
- P4.4  / F-15: `metrics._checkpoints` tăng vô hạn.
- P0.1: instrumentation gauge phải hoạt động.
- P4.1: logger chỉ flush theo chính sách (không flush mỗi record).
"""

import threading
import time

import pytest

from backend.asr.engine import TranscribeEngine
from backend.core.metrics import MetricsCollector, metrics_collector


# --------------------------------------------------------------- lock ordering (P1.8b)
def test_unload_shared_model_waits_for_inference_lock(monkeypatch):
    """P1.8b: `unload_shared_model` PHẢI chờ `_infer_lock` trước khi đóng native handle.

    Trước đây nó chỉ giữ `_shared_lock`, nên có thể `close()` session native trong khi
    một inference đang chạy ⇒ use-after-free (crash tiến trình).
    """
    order = []
    closed = threading.Event()

    class FakeSession:
        def close(self):
            order.append("session.close")

    class FakeModel:
        def close(self):
            order.append("model.close")

    with TranscribeEngine._shared_lock:
        TranscribeEngine._shared_session = FakeSession()
        TranscribeEngine._shared_model = FakeModel()
        TranscribeEngine._shared_model_key = "fake"

    # Giữ _infer_lock trong thread chính để mô phỏng inference đang chạy
    TranscribeEngine._infer_lock.acquire()
    try:
        def _unload():
            TranscribeEngine.unload_shared_model()
            closed.set()

        t = threading.Thread(target=_unload, daemon=True)
        t.start()
        # Trong lúc inference đang giữ lock, unload KHÔNG được đóng gì cả
        time.sleep(0.15)
        assert not closed.is_set(), "unload_shared_model đã chạy khi inference còn đang giữ lock!"
        assert order == [], f"native handle bị đóng khi inference đang chạy: {order}"
    finally:
        TranscribeEngine._infer_lock.release()

    t.join(timeout=3.0)
    assert closed.is_set(), "unload phải hoàn tất sau khi nhả _infer_lock"
    assert "session.close" in order


def test_lock_order_is_consistent_no_deadlock():
    """Thứ tự lock chuẩn `_infer_lock` -> `_shared_lock` không được gây deadlock."""
    acquired = []

    def _worker():
        for _ in range(50):
            with TranscribeEngine._infer_lock:
                with TranscribeEngine._shared_lock:
                    acquired.append(1)

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "deadlock khi lấy lock theo thứ tự chuẩn"
    assert len(acquired) == 200


def test_engine_has_no_reverse_lock_acquisition():
    """Chốt chặn hồi quy: không được có chỗ nào lấy `_shared_lock` rồi `_infer_lock`."""
    import inspect
    import re

    from backend.asr import engine as engine_mod

    src = inspect.getsource(engine_mod)
    # Tìm pattern: with ..._shared_lock:  ... with ..._infer_lock:
    nested = re.findall(
        r"with\s+\w*\.?_shared_lock\s*:(.{0,400}?)with\s+\w*\.?_infer_lock\s*:",
        src,
        flags=re.DOTALL,
    )
    assert not nested, (
        "Phát hiện thứ tự lock NGƯỢC (_shared_lock -> _infer_lock) trong asr/engine.py. "
        "Thứ tự duy nhất được phép là _infer_lock -> _shared_lock."
    )


def test_two_hundred_model_switches_during_inference(monkeypatch):
    """K10: **200 lần** đổi model NGAY TRONG lúc inference đang chạy — 0 use-after-free.

    Mô phỏng đúng thứ tự lock của `_run_inference_sync` thật (giữ `_infer_lock` suốt lúc
    suy luận) rồi cho `unload_shared_model()` chạy song song trên luồng khác. Nếu
    `unload_shared_model` đóng native handle mà không chờ `_infer_lock`, session giả sẽ
    thấy cờ "đang suy luận" còn bật ⇒ ghi nhận vi phạm (thực tế là crash tiến trình).
    """
    import asyncio

    import numpy as np

    from backend.tests.fakes import FakeInferenceEngine

    engine = FakeInferenceEngine(text_fn=lambda n: "x")
    state = {"running": False}
    violations = []

    def _infer(pcm):  # giống `_run_inference_sync` thật: giữ `_infer_lock` khi suy luận
        with TranscribeEngine._infer_lock:
            state["running"] = True
            time.sleep(0.002)
            state["running"] = False
        return "ok"

    class FakeSession:
        def close(self):
            if state["running"]:
                violations.append("close() khi inference đang chạy (use-after-free)")

    class FakeModel:
        def close(self):
            pass

    monkeypatch.setattr(engine, "_run_inference_sync", _infer, raising=False)
    monkeypatch.setattr(engine, "_preload_model", lambda force_warm=False: None, raising=False)

    saved = (
        TranscribeEngine._shared_session,
        TranscribeEngine._shared_model,
        TranscribeEngine._shared_model_key,
    )
    try:
        for i in range(200):
            with TranscribeEngine._shared_lock:
                TranscribeEngine._shared_session = FakeSession()
                TranscribeEngine._shared_model = FakeModel()
                TranscribeEngine._shared_model_key = "fake"

            def _run_inference():
                asyncio.run(engine._infer_with_watchdog(np.zeros(1600, dtype=np.float32)))

            t = threading.Thread(target=_run_inference, daemon=True)
            t.start()
            time.sleep(0.0005)  # để inference kịp vào `_infer_lock`
            TranscribeEngine.unload_shared_model()
            t.join(timeout=5.0)
            assert not t.is_alive(), f"lần {i}: inference không kết thúc sau khi unload"
    finally:
        TranscribeEngine._shared_session, TranscribeEngine._shared_model, TranscribeEngine._shared_model_key = saved

    assert not violations, f"{len(violations)} vi phạm: {violations[:3]}"


# ----------------------------------------------------------------- metrics (P4.4)
def test_metrics_checkpoints_are_bounded():
    """P4.4/F-15: `_checkpoints` phải bounded, không tăng vô hạn theo số session."""
    m = MetricsCollector(max_history=10)
    for i in range(5000):
        m.record_checkpoint(f"session_start_{i:08x}")
    from backend.core.metrics import _MAX_CHECKPOINTS

    assert len(m._checkpoints) <= _MAX_CHECKPOINTS, (
        f"_checkpoints giữ {len(m._checkpoints)} mục (trần {_MAX_CHECKPOINTS}) — memory leak"
    )
    report = m.generate_report()
    assert report["checkpoints_retained"] <= _MAX_CHECKPOINTS


def test_metrics_gauges_record_and_read():
    """P0.1: gauge ghi đè giá trị cũ và đọc lại được (queue depth)."""
    m = MetricsCollector()
    m.record_gauge("queue", "translation_depth", 3)
    assert m.get_gauge("queue", "translation_depth") == 3
    m.record_gauge("queue", "translation_depth", 0)
    assert m.get_gauge("queue", "translation_depth") == 0
    assert m.get_gauge("queue", "khong_ton_tai", 7) == 7
    report = m.generate_report()
    assert "gauges" in report
    assert report["gauges"]["queue.translation_depth"] == 0


def test_metrics_latency_percentiles_work():
    """Gauge mới không được làm hỏng thống kê percentile cũ."""
    m = MetricsCollector()
    for v in (10.0, 20.0, 30.0, 40.0, 100.0):
        m.record_metric("asr", "preview_ms", v)
    stats = m.get_stage_stats("asr.preview_ms")
    assert stats["count"] == 5
    assert stats["p50_ms"] == 30.0
    assert stats["max_ms"] == 100.0


def test_snapshot_pipeline_contains_hot_path_stages():
    """P0.1: `/api/metrics/pipeline` phải phơi ra các stage hot path (F-22)."""
    m = MetricsCollector()
    m.record_metric("asr", "preview_ms", 12.0)
    m.record_metric("vad", "chunk_ms", 0.4)
    m.record_gauge("queue", "tts_depth", 1)
    snap = m.snapshot_pipeline()
    assert snap["stages"]["asr.preview_ms"]["count"] == 1
    assert snap["stages"]["vad.chunk_ms"]["count"] == 1
    assert snap["gauges"]["queue.tts_depth"] == 1


# ------------------------------------------------------------------- logger (P4.1)
def test_logger_flush_is_not_per_record(monkeypatch):
    """P4.1: `SafeStreamHandler` chỉ flush khi >= WARNING hoặc quá interval."""
    import logging

    from backend.utils.logger import SafeStreamHandler

    flushes = []
    handler = SafeStreamHandler(stream=open("NUL", "w", encoding="utf-8") if hasattr(__import__("os"), "name") else None)
    monkeypatch.setattr(handler, "flush", lambda: flushes.append(time.monotonic()))

    SafeStreamHandler._last_flush = time.monotonic()

    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
    for _ in range(20):
        handler.emit(rec)
    assert len(flushes) == 0, "INFO liên tiếp không được flush mỗi record"

    warn = logging.LogRecord("t", logging.WARNING, __file__, 1, "warn", None, None)
    handler.emit(warn)
    assert len(flushes) == 1, "WARNING phải được flush ngay"


def test_backend_logger_level_is_info():
    """P4.1: logger 'backend' không còn ở mức DEBUG (tránh log hot path)."""
    import logging

    from backend.utils.logger import logger as backend_logger

    assert backend_logger.level == logging.INFO
