"""Regression tests for the ASR lifecycle barrier (audit finding P0-01).

The defect these tests guard against
------------------------------------
``unload_shared_model()`` / ``ensure_model()`` used to hold only ``_shared_lock``
while closing the native model and session. Because ``_run_inference()`` releases
``_shared_lock`` *before* acquiring ``_shared_infer_lock``, an unload could destroy
the native objects while a worker thread was still inside ``session.run()`` -- a
use-after-free class defect in the C++/GGML runtime.

The fix introduces an explicit lifecycle barrier: every close/load transition takes
``_shared_lock`` **then** ``_shared_infer_lock``, so it cannot overlap inference.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from backend_cpp.asr.model_manager import ASRModelManager
from backend_cpp.config import config


@pytest.fixture(autouse=True)
def reset_shared_state():
    """Reset shared model/session state and the infer-lock ownership record."""

    def _reset():
        ASRModelManager._shared_model = None
        ASRModelManager._shared_session = None
        ASRModelManager._shared_model_key = None
        ASRModelManager._shared_supports_streaming = False
        ASRModelManager._infer_lock_state.depth = 0

    _reset()
    yield
    _reset()


def _install_mock_model(model_key: str = "test-model"):
    """Populate the shared slot with a mock model+session and return both."""
    model = MagicMock(name="model")
    session = MagicMock(name="session")
    ASRModelManager._shared_model = model
    ASRModelManager._shared_session = session
    ASRModelManager._shared_model_key = model_key
    return model, session


# ---------------------------------------------------------------------------
# Barrier: unload must wait for in-flight inference
# ---------------------------------------------------------------------------
def test_unload_waits_for_inflight_inference():
    """close() must happen only AFTER the worker releases the inference lock."""
    model, session = _install_mock_model()

    order = []
    session.close.side_effect = lambda: order.append("close_session")
    model.close.side_effect = lambda: order.append("close_model")

    inference_started = threading.Event()

    def fake_inference():
        assert ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0)
        try:
            inference_started.set()
            time.sleep(0.3)
            order.append("inference_done")
        finally:
            ASRModelManager.release_infer_lock()

    worker = threading.Thread(target=fake_inference, daemon=True)
    worker.start()
    assert inference_started.wait(2.0), "worker never acquired the inference lock"

    assert ASRModelManager.unload_shared_model(timeout=5.0) is True
    worker.join(5.0)
    assert not worker.is_alive()

    # This is the core assertion of the whole lifecycle barrier.
    assert order.index("inference_done") < order.index("close_session")
    assert order.index("inference_done") < order.index("close_model")

    assert ASRModelManager.get_shared_model() is None
    assert ASRModelManager.get_shared_session() is None


def test_unload_times_out_and_leaves_resources_untouched():
    """On timeout nothing may be closed -- the caller can safely retry."""
    model, session = _install_mock_model()

    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_inference_lock():
        assert ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0)
        try:
            lock_held.set()
            release_lock.wait(timeout=5.0)
        finally:
            ASRModelManager.release_infer_lock()

    holder = threading.Thread(target=hold_inference_lock, daemon=True)
    holder.start()
    try:
        assert lock_held.wait(2.0)

        assert ASRModelManager.unload_shared_model(timeout=0.2) is False

        session.close.assert_not_called()
        model.close.assert_not_called()
        assert ASRModelManager.get_shared_model() is model
        assert ASRModelManager.get_shared_session() is session
    finally:
        release_lock.set()
        holder.join(5.0)


def test_unload_is_noop_when_nothing_loaded():
    """An empty manager unloads successfully without acquiring the lock."""
    assert ASRModelManager.unload_shared_model(timeout=1.0) is True
    assert ASRModelManager.get_shared_model() is None


def test_unload_releases_both_session_and_model():
    model, session = _install_mock_model()

    assert ASRModelManager.unload_shared_model(timeout=5.0) is True

    session.close.assert_called_once()
    model.close.assert_called_once()
    assert ASRModelManager.is_model_loaded() is False


# ---------------------------------------------------------------------------
# Barrier: model swap must wait for in-flight inference
# ---------------------------------------------------------------------------
def test_ensure_model_swap_waits_for_inference(monkeypatch):
    """A model swap must not close the outgoing model while inference is running."""
    import backend_cpp.asr.model_manager as model_manager_module

    old_model, old_session = _install_mock_model("old-model")

    order = []
    old_session.close.side_effect = lambda: order.append("close_old_session")
    old_model.close.side_effect = lambda: order.append("close_old_model")

    new_model = MagicMock(name="new_model")
    new_model.capabilities.supports_streaming = False

    fake_transcribe = MagicMock(name="transcribe_cpp")
    fake_transcribe.Model.return_value = new_model
    monkeypatch.setattr(model_manager_module, "transcribe_cpp", fake_transcribe)

    manager = ASRModelManager(registry=MagicMock())
    manager.registry.ensure_model.return_value = "dummy.gguf"

    inference_started = threading.Event()

    def inference():
        assert ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0)
        try:
            inference_started.set()
            time.sleep(0.3)
            order.append("inference_done")
        finally:
            ASRModelManager.release_infer_lock()

    worker = threading.Thread(target=inference, daemon=True)
    worker.start()
    assert inference_started.wait(2.0)

    swapped = manager.ensure_model("new-model", backend="auto")
    worker.join(5.0)

    assert swapped is new_model
    assert order.index("inference_done") < order.index("close_old_session")
    assert order.index("inference_done") < order.index("close_old_model")
    assert ASRModelManager.get_shared_model() is new_model


def test_ensure_model_reuses_loaded_model_without_locking(monkeypatch):
    """Fast path: an already-loaded matching model is returned as-is."""
    import backend_cpp.asr.model_manager as model_manager_module

    model, _ = _install_mock_model("same-model")
    monkeypatch.setattr(
        model_manager_module, "transcribe_cpp", MagicMock(name="transcribe_cpp")
    )

    manager = ASRModelManager(registry=MagicMock())
    assert manager.ensure_model("same-model") is model
    manager.registry.ensure_model.assert_not_called()


# ---------------------------------------------------------------------------
# Ownership invariant on ensure_session()
# ---------------------------------------------------------------------------
def test_ensure_session_works_while_holding_infer_lock():
    """The documented call contract: hold the lock, get a session."""
    manager = ASRModelManager(registry=MagicMock())
    model = MagicMock(name="model")

    assert ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0)
    try:
        session = manager.ensure_session(model, threads=2)
    finally:
        ASRModelManager.release_infer_lock()

    assert session is model.session.return_value
    model.session.assert_called_once_with(n_threads=2)


def test_ensure_session_warns_without_lock_by_default(monkeypatch, caplog):
    """Default behaviour: warn (do not break existing callers)."""
    import logging

    monkeypatch.setattr(config.debug, "strict_lock_checks", False)

    manager = ASRModelManager(registry=MagicMock())
    model = MagicMock(name="model")

    with caplog.at_level(logging.WARNING, logger="backend_cpp.asr.model_manager"):
        session = manager.ensure_session(model, threads=1)

    assert session is model.session.return_value
    assert any("without holding the ASR inference lock" in rec.message for rec in caplog.records)


def test_ensure_session_raises_in_strict_mode(monkeypatch):
    """Strict mode turns an invariant violation into a hard error."""
    monkeypatch.setattr(config.debug, "strict_lock_checks", True)

    manager = ASRModelManager(registry=MagicMock())
    with pytest.raises(RuntimeError, match="without holding the ASR inference lock"):
        manager.ensure_session(MagicMock(name="model"))


def test_ensure_session_replaces_session_for_different_model():
    """Switching the backing model closes the stale session (under the lock)."""
    manager = ASRModelManager(registry=MagicMock())

    first_model = MagicMock(name="model_a")
    stale_session = MagicMock(name="stale_session")
    stale_session._model = MagicMock(name="some_other_model")
    ASRModelManager._shared_session = stale_session

    assert ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0)
    try:
        session = manager.ensure_session(first_model, threads=1)
    finally:
        ASRModelManager.release_infer_lock()

    stale_session.close.assert_called_once()
    assert session is first_model.session.return_value


# ---------------------------------------------------------------------------
# Deadlock / stress
# ---------------------------------------------------------------------------
def test_no_deadlock_between_inference_and_unload():
    """Lock ordering (shared -> infer) must never produce a deadlock cycle."""
    model, session = _install_mock_model("stress-model")

    errors = []
    stop = threading.Event()

    def inference_worker():
        try:
            for _ in range(80):
                if stop.is_set():
                    break
                if ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0):
                    try:
                        time.sleep(0.0005)
                    finally:
                        ASRModelManager.release_infer_lock()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def unload_worker():
        try:
            for _ in range(40):
                if stop.is_set():
                    break
                ASRModelManager.unload_shared_model(timeout=5.0)
                # Re-arm so the next iteration still has something to release.
                with ASRModelManager._shared_lock:
                    if ASRModelManager._shared_model is None:
                        ASRModelManager._shared_model = model
                        ASRModelManager._shared_session = session
                        ASRModelManager._shared_model_key = "stress-model"
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=inference_worker, daemon=True) for _ in range(3)]
    threads.append(threading.Thread(target=unload_worker, daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join(30.0)
    stop.set()

    assert not errors, f"unexpected errors during stress run: {errors}"
    assert all(not t.is_alive() for t in threads), "deadlock: worker threads never finished"


def test_infer_lock_depth_tracks_acquire_and_release():
    """The ownership record must return to zero so the invariant stays meaningful."""
    assert ASRModelManager._infer_lock_depth() == 0

    assert ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0)
    assert ASRModelManager._infer_lock_depth() == 1

    # ``_shared_infer_lock`` is a plain (non-reentrant) Lock, so a nested acquire
    # from the same thread must fail rather than succeed. Depth therefore stays at 1.
    assert ASRModelManager.acquire_infer_lock(blocking=False) is False
    assert ASRModelManager._infer_lock_depth() == 1

    ASRModelManager.release_infer_lock()
    assert ASRModelManager._infer_lock_depth() == 0

    # Ownership is tracked per-thread.
    observed = []
    ASRModelManager.acquire_infer_lock(blocking=True, timeout=5.0)
    try:
        other = threading.Thread(
            target=lambda: observed.append(ASRModelManager._infer_lock_depth()), daemon=True
        )
        other.start()
        other.join(5.0)
    finally:
        ASRModelManager.release_infer_lock()

    assert observed == [0]
    assert ASRModelManager._infer_lock_depth() == 0
