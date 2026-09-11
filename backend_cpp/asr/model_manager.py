"""Dedicated model and session lifecycle manager for transcribe.cpp ASR models."""

import logging
import threading
from typing import Any, Optional

try:
    import transcribe_cpp
except ImportError:
    transcribe_cpp = None

from backend_cpp.config import config
from backend_cpp.asr.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ASRModelManager:
    """Manages shared transcribe_cpp Model and Session instances.

    Responsible for:
    - Thread-safe model loading, backend resolution, and caching.
    - Inference session caching and thread allocation.
    - Concurrency control (shared lock, inference lock, commit prioritization).
    - Resource deallocation (unloading models and closing sessions from VRAM/RAM).
    """

    _shared_model: Optional[Any] = None
    _shared_model_key: Optional[str] = None
    _shared_session: Optional[Any] = None
    _shared_supports_streaming: bool = False

    _shared_lock: threading.Lock = threading.Lock()
    _shared_infer_lock: threading.Lock = threading.Lock()
    _commit_waiting: int = 0
    _commit_lock: threading.Lock = threading.Lock()

    # Thread-local record of how many times the *current thread* holds
    # ``_shared_infer_lock``. It lets ``ensure_session()`` verify the ownership
    # invariant documented there.
    #
    # IMPORTANT: always use ``acquire_infer_lock()`` / ``release_infer_lock()``
    # instead of touching ``_shared_infer_lock`` directly, otherwise the
    # ownership record silently goes out of sync.
    _infer_lock_state: threading.local = threading.local()

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry.get_instance()

    # ------------------------------------------------------------------
    # Inference lock helpers (single source of truth for lock ownership)
    # ------------------------------------------------------------------
    @classmethod
    def _infer_lock_depth(cls) -> int:
        """Number of times the current thread currently holds ``_shared_infer_lock``."""
        return int(getattr(cls._infer_lock_state, "depth", 0))

    @classmethod
    def acquire_infer_lock(cls, blocking: bool = True, timeout: float = -1.0) -> bool:
        """Acquire ``_shared_infer_lock`` and record thread ownership.

        Args:
            blocking: Wait for the lock when True, return immediately otherwise.
            timeout: Seconds to wait when ``blocking`` is True. ``-1`` waits forever.

        Returns:
            True when the lock was acquired, False on timeout / immediate failure.
        """
        if blocking:
            if timeout is not None and timeout > 0:
                acquired = cls._shared_infer_lock.acquire(blocking=True, timeout=timeout)
            else:
                acquired = cls._shared_infer_lock.acquire(blocking=True)
        else:
            acquired = cls._shared_infer_lock.acquire(blocking=False)

        if acquired:
            cls._infer_lock_state.depth = cls._infer_lock_depth() + 1
        return acquired

    @classmethod
    def release_infer_lock(cls) -> None:
        """Release ``_shared_infer_lock`` and drop the thread ownership record."""
        depth = cls._infer_lock_depth()
        if depth > 0:
            cls._infer_lock_state.depth = depth - 1
        cls._shared_infer_lock.release()

    @classmethod
    def get_shared_model(cls) -> Optional[Any]:
        """Return the shared model.

        NOTE: Takes ``_shared_lock``. Never call this while already holding
        ``_shared_lock`` (it is not reentrant), and never call it while holding
        ``_shared_infer_lock`` (that would invert the documented lock order).
        """
        with cls._shared_lock:
            return cls._shared_model

    @classmethod
    def get_shared_session(cls) -> Optional[Any]:
        """Return the shared inference session. See ``get_shared_model()`` for locking notes."""
        with cls._shared_lock:
            return cls._shared_session

    @classmethod
    def is_model_loaded(cls, model_key: Optional[str] = None) -> bool:
        with cls._shared_lock:
            if cls._shared_model is None:
                return False
            if model_key is not None:
                return cls._shared_model_key == model_key
            return True

    @classmethod
    def supports_streaming(cls, model_key: str) -> bool:
        with cls._shared_lock:
            if cls._shared_model is not None and cls._shared_model_key == model_key:
                return cls._shared_supports_streaming
        return ModelRegistry.get_instance().is_streaming_model(model_key)

    @classmethod
    def _close_shared_locked(cls) -> None:
        """Close the shared native session and model.

        LIFECYCLE BARRIER: the caller MUST hold BOTH ``_shared_lock`` and
        ``_shared_infer_lock``. Holding both guarantees no worker thread is inside
        ``session.run()`` / ``session.stream()`` while the native objects are being
        destroyed, which would otherwise be a use-after-free crash.
        """
        if cls._shared_session is not None:
            try:
                cls._shared_session.close()
            except Exception:
                pass
            cls._shared_session = None

        if cls._shared_model is not None:
            try:
                cls._shared_model.close()
            except Exception:
                pass
            cls._shared_model = None
            cls._shared_model_key = None
            cls._shared_supports_streaming = False
            logger.info("ASRModelManager: Shared model closed and freed from memory.")

    @classmethod
    def unload_shared_model(cls, timeout: Optional[float] = None) -> bool:
        """Release the shared model and cached session from VRAM/RAM.

        Waits for in-flight native inference to complete before closing anything, so
        that a model swap/unload can never overlap an active ``session.run()``.

        Args:
            timeout: Seconds to wait for the inference lock. Defaults to
                ``config.asr.unload_lock_timeout_sec``.

        Returns:
            True when the resources were released (or were already free).
            False when the inference lock could not be acquired in time; in that case
            NOTHING is closed, because destroying a model that is still in use would
            crash the native runtime. The caller may retry later.
        """
        if timeout is None:
            timeout = getattr(config.asr, "unload_lock_timeout_sec", 15.0)

        with cls._shared_lock:
            if cls._shared_model is None and cls._shared_session is None:
                return True

            acquired = cls.acquire_infer_lock(blocking=True, timeout=timeout)
            if not acquired:
                logger.error(
                    "ASRModelManager: could not acquire the ASR inference lock within %.1fs; "
                    "skipping model unload to avoid closing a model that is still in use. "
                    "Retry once inference completes.",
                    timeout,
                )
                return False

            try:
                cls._close_shared_locked()
            finally:
                cls.release_infer_lock()

        return True

    def ensure_model(self, model_key: str, backend: Optional[str] = None) -> Any:
        """Load and cache transcribe.cpp model if not already loaded.

        Model cleanup + load is a lifecycle transition and is therefore serialized
        against native inference via the lifecycle barrier (``_shared_lock`` THEN
        ``_shared_infer_lock``), so a model swap can never close a model that a
        worker thread is still running inference on.
        """
        with self._shared_lock:
            if (
                self.__class__._shared_model is not None
                and self.__class__._shared_model_key == model_key
            ):
                return self.__class__._shared_model

            cls = self.__class__
            load_timeout = getattr(config.asr, "model_load_lock_timeout_sec", 30.0)
            acquired = cls.acquire_infer_lock(blocking=True, timeout=load_timeout)
            if not acquired:
                raise RuntimeError(
                    f"ASRModelManager: could not acquire the ASR inference lock within "
                    f"{load_timeout:.1f}s to swap model '{model_key}'. "
                    f"Model swap aborted; the previous model was left untouched."
                )

            try:
                # Cleanup previous session and model (never overlaps inference here)
                cls._close_shared_locked()

                model_path = self.registry.ensure_model(model_key)
                logger.info(f"Loading transcribe.cpp model '{model_key}' from: {model_path}...")

                backend_choice = backend or getattr(config.asr, "backend", "auto")
                if transcribe_cpp is not None and backend_choice != "auto" and hasattr(transcribe_cpp, "backend_available"):
                    if not transcribe_cpp.backend_available(backend_choice):
                        avail = [b.kind for b in transcribe_cpp.backends()]
                        logger.warning(
                            f"ASR backend '{backend_choice}' is not available in transcribe.cpp (available: {avail}). "
                            f"Falling back to 'auto'."
                        )
                        backend_choice = "auto"

                if transcribe_cpp is not None:
                    if backend_choice == "auto":
                        model = transcribe_cpp.Model(model_path)
                    else:
                        model = transcribe_cpp.Model(model_path, backend=backend_choice)
                else:
                    raise RuntimeError("transcribe_cpp library is not installed or available.")

                cls._shared_model = model
                cls._shared_model_key = model_key
                cls._shared_supports_streaming = bool(
                    getattr(model.capabilities, "supports_streaming", False)
                )

                logger.info(
                    f"✅ transcribe.cpp model '{model_key}' loaded! "
                    f"(Arch: {getattr(model, 'arch', 'unknown')}, Variant: {getattr(model, 'variant', 'unknown')}, "
                    f"Backend: {getattr(model, 'backend', 'unknown')}, "
                    f"Streaming: {cls._shared_supports_streaming})"
                )
                return model
            finally:
                cls.release_infer_lock()

    def ensure_session(self, model: Any, threads: int = 4) -> Any:
        """Reuse the cached session, or instantiate a new one for ``model``.

        LIFECYCLE CONTRACT:
            The caller MUST hold ``ASRModelManager._shared_infer_lock`` (acquired via
            :meth:`acquire_infer_lock`). This serializes session instantiation,
            parameter tuning and session replacement against model unload/swap.

            A session is closed here whenever the backing model object changes, so
            running without the lock would race with ``unload_shared_model()``.

        Set ``config.debug.strict_lock_checks = True`` to turn the contract into a
        hard error instead of a warning.
        """
        cls = self.__class__

        if cls._infer_lock_depth() == 0:
            message = (
                "ensure_session() called without holding the ASR inference lock. "
                "This can race with model unload/swap; acquire it via "
                "ASRModelManager.acquire_infer_lock() first."
            )
            if getattr(config.debug, "strict_lock_checks", False):
                raise RuntimeError(message)
            logger.warning(message)

        if (
            cls._shared_session is None
            or getattr(cls._shared_session, "_model", None) is not model
        ):
            if cls._shared_session is not None:
                try:
                    cls._shared_session.close()
                except Exception:
                    pass
                cls._shared_session = None
            cls._shared_session = model.session(n_threads=threads)
        return cls._shared_session
