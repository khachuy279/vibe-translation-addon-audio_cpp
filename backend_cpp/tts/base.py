"""Base abstractions and protocols for Text-to-Speech engines in backend_cpp."""

from typing import Optional, Tuple, Protocol, runtime_checkable


@runtime_checkable
class TTSEngine(Protocol):
    """Protocol defining common interface for all TTS engines (DIP / OCP)."""

    sample_rate: int

    def is_model_ready(self) -> bool:
        """Check if the TTS model weights/resources are available."""
        ...

    def load_model(self) -> None:
        """Load and warm up the TTS model into memory/device."""
        ...

    def unload_model(self) -> None:
        """Release TTS model resources from memory and free GPU/VRAM."""
        ...

    def synthesize_sync(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Synchronously synthesize text to base64 audio and duration in seconds."""
        ...

    async def synthesize_clone(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[Optional[str], float]:
        """Asynchronously synthesize text to speech via thread pool."""
        ...
