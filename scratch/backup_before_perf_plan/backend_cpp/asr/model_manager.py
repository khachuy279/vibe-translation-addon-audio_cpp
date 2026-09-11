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

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry.get_instance()

    @classmethod
    def get_shared_model(cls) -> Optional[Any]:
        return cls._shared_model

    @classmethod
    def get_shared_session(cls) -> Optional[Any]:
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
    def unload_shared_model(cls) -> None:
        """Release shared model and cached session from VRAM/RAM."""
        with cls._shared_lock:
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

    def ensure_model(self, model_key: str, backend: Optional[str] = None) -> Any:
        """Load and cache transcribe.cpp model if not already loaded."""
        with self._shared_lock:
            if (
                self.__class__._shared_model is not None
                and self.__class__._shared_model_key == model_key
            ):
                return self.__class__._shared_model

            # Cleanup previous session and model
            if self.__class__._shared_session is not None:
                try:
                    self.__class__._shared_session.close()
                except Exception:
                    pass
                self.__class__._shared_session = None

            if self.__class__._shared_model is not None:
                try:
                    self.__class__._shared_model.close()
                except Exception:
                    pass
                self.__class__._shared_model = None

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

            self.__class__._shared_model = model
            self.__class__._shared_model_key = model_key
            self.__class__._shared_supports_streaming = bool(
                getattr(model.capabilities, "supports_streaming", False)
            )

            logger.info(
                f"✅ transcribe.cpp model '{model_key}' loaded! "
                f"(Arch: {getattr(model, 'arch', 'unknown')}, Variant: {getattr(model, 'variant', 'unknown')}, "
                f"Backend: {getattr(model, 'backend', 'unknown')}, "
                f"Streaming: {self.__class__._shared_supports_streaming})"
            )
            return model

    def ensure_session(self, model: Any, threads: int = 4) -> Any:
        """Reuse cached session or instantiate a new session for inference.

        Thread-safety requirement:
            Caller MUST hold ASRModelManager._shared_infer_lock to serialize
            inference session instantiation, parameter tuning, and lifecycle.
        """
        if (
            self.__class__._shared_session is None
            or getattr(self.__class__._shared_session, "_model", None) is not model
        ):
            if self.__class__._shared_session is not None:
                try:
                    self.__class__._shared_session.close()
                except Exception:
                    pass
                self.__class__._shared_session = None
            self.__class__._shared_session = model.session(n_threads=threads)
        return self.__class__._shared_session
