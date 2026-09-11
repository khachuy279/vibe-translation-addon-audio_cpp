"""Core protocols and interfaces for the translation module."""

from typing import Dict, Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class PromptStrategy(Protocol):
    """Protocol defining prompt formatting and stop token strategy for translation models."""

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = True,
    ) -> str:
        """Construct the prompt formatted for the target model."""
        ...

    def get_stop_tokens(self) -> list[str]:
        """Return model-appropriate stop tokens for generation."""
        ...


@runtime_checkable
class TranslationEngine(Protocol):
    """Protocol defining a generic translation engine (GGUF, ONNX, Cloud API, etc.)."""

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Translate text asynchronously."""
        ...

    def load_model(self) -> None:
        """Load model resources into memory/GPU."""
        ...

    def unload_model(self) -> None:
        """Explicitly unload model resources and free memory/VRAM."""
        ...
