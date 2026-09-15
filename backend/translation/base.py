"""Abstract Base Class cho các Translation Engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTranslator(ABC):
    """Interface tiêu chuẩn cho engine dịch thuật."""

    @abstractmethod
    def load_model(self) -> None:
        """Nạp model GGUF lên GPU VRAM."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Giải phóng model khỏi GPU VRAM."""
        pass

    @abstractmethod
    async def translate_sentence(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "vi",
        context: str = "",
    ) -> Dict[str, Any]:
        """Dịch một câu từ source_lang sang target_lang."""
        pass
