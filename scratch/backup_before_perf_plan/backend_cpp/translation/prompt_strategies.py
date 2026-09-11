"""Strategy pattern implementations for translation prompts and stop tokens."""

import logging
from typing import Dict, Type, Optional, Callable
from backend_cpp.translation.interfaces import PromptStrategy
from backend_cpp.translation.lang_utils import (
    resolve_effective_source_lang,
    resolve_lang_name,
)

logger = logging.getLogger(__name__)

# Registry for Open/Closed Principle extension
_STRATEGY_REGISTRY: Dict[str, Callable[[], PromptStrategy]] = {}


def register_prompt_strategy(name: str):
    """Decorator to register custom prompt strategy factories for OCP extensibility."""
    def decorator(cls_or_factory):
        _STRATEGY_REGISTRY[name.lower().strip()] = cls_or_factory
        return cls_or_factory
    return decorator


class BaseChatMLStrategy:
    """Base strategy providing common ChatML stop tokens to ensure DRY."""

    def get_stop_tokens(self) -> list[str]:
        return [
            "<|im_end|>", "<|im_end>", "<|im_start|>", "<|im_start",
            "<|endoftext|>", "</s>", "\n\n"
        ]


@register_prompt_strategy("milmmt")
class MiLMMTPromptStrategy:
    """Prompt format for Xiaomi MiLMMT models."""

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = True,
    ) -> str:
        tgt_name = resolve_lang_name(target_lang, use_chinese=False)
        src_name = resolve_effective_source_lang(text, source_lang, use_chinese=False)

        if context and use_context:
            return (
                f"Background: {context.strip()}\n"
                f"Translate this from {src_name} to {tgt_name}:\n"
                f"{src_name}: {text.strip()}\n"
                f"{tgt_name}:"
            )
        return (
            f"Translate this from {src_name} to {tgt_name}:\n"
            f"{src_name}: {text.strip()}\n"
            f"{tgt_name}:"
        )

    def get_stop_tokens(self) -> list[str]:
        return [
            "<|im_end|>", "<|im_end>", "<|im_start|>", "<|im_start",
            "<|endoftext|>", "</s>", "\n", "<end_of_turn>", "<eos>"
        ]


@register_prompt_strategy("gemmax")
@register_prompt_strategy("gemmax2")
@register_prompt_strategy("gemma")
class GemmaXPromptStrategy(MiLMMTPromptStrategy):
    """Prompt format for Xiaomi GemmaX2-28-9B models (using official 28-lang translation template)."""

    def get_stop_tokens(self) -> list[str]:
        return [
            "<end_of_turn>", "<eos>", "<|endoftext|>", "</s>",
            "<|im_end|>", "\n",
        ]


# Static system prefixes to maximize KV Cache hits with LlamaRAMCache
SYSTEM_PREFIX_EN = (
    "<|im_start|>system\n"
    "You are a professional machine translation engine. "
    "Translate user inputs accurately. "
    "Output only the translated text without explanations.<|im_end|>\n"
)

SYSTEM_PREFIX_ZH = (
    "<|im_start|>system\n"
    "你是一个专业的高质量翻译引擎，只需要输出翻译后的结果，不要额外解释。<|im_end|>\n"
)


@register_prompt_strategy("tencent_en")
@register_prompt_strategy("en")
class TencentEnPromptStrategy(BaseChatMLStrategy):
    """English prompt format using ChatML template (system prefix disabled by default for lowest latency)."""

    def __init__(self, use_system_prefix: bool = False):
        self.use_system_prefix = use_system_prefix

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = True,
    ) -> str:
        tgt_name = resolve_lang_name(target_lang, use_chinese=False)
        src_name = resolve_effective_source_lang(text, source_lang, use_chinese=False)

        if context and use_context:
            user_content = (
                f"[Background Information]\n"
                f"{context.strip()}\n\n"
                f"Please translate the following text from {src_name} into {tgt_name}, taking the provided background information into consideration. "
                f"Note that you should only output the translated result without any additional explanation:\n\n"
                f"{text.strip()}"
            )
        else:
            user_content = (
                f"Translate the following text from {src_name} into {tgt_name}. "
                f"Note that you should only output the translated result without any additional explanation:\n\n"
                f"{text.strip()}"
            )

        prefix = SYSTEM_PREFIX_EN if self.use_system_prefix else ""
        return f"{prefix}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


@register_prompt_strategy("tencent_zh")
@register_prompt_strategy("zh")
@register_prompt_strategy("tencent")
class TencentZhPromptStrategy(BaseChatMLStrategy):
    """Official Chinese prompt format for Tencent Hy-MT models (fastest and most accurate)."""

    def __init__(self, use_system_prefix: bool = False):
        self.use_system_prefix = use_system_prefix

    def build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: str = "",
        use_context: bool = True,
    ) -> str:
        # Note: Hy-MT official Chinese template omits source_lang as the model natively auto-detects source language (Any-to-Any).
        tgt_name = resolve_lang_name(target_lang, use_chinese=True)

        if context and use_context:
            user_content = (
                f"【背景信息】\n"
                f"{context.strip()}\n\n"
                f"请结合背景信息将以下文本翻译为{tgt_name}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
                f"{text.strip()}"
            )
        else:
            user_content = (
                f"将以下文本翻译为{tgt_name}，注意只需要输出翻译后的结果，不要额外解释：\n\n"
                f"{text.strip()}"
            )

        prefix = SYSTEM_PREFIX_ZH if self.use_system_prefix else ""
        return f"{prefix}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def get_prompt_strategy(
    model_name: str,
    repo_name: str,
    prompt_style: str,
    use_system_prefix: bool = False,
) -> PromptStrategy:
    """Factory selecting prompt strategy conforming to model requirements, extensible via registry."""
    model_name_lower = (model_name or "").lower()
    repo_name_lower = (repo_name or "").lower()
    style_key = (prompt_style or "").lower().strip()

    # 1. Explicit prompt style requested by caller takes precedence over automatic filename inference
    if style_key and style_key not in ("auto", "default", ""):
        if style_key in _STRATEGY_REGISTRY:
            return _STRATEGY_REGISTRY[style_key]()
        if style_key in ("en", "english", "tencent_en"):
            return TencentEnPromptStrategy(use_system_prefix=use_system_prefix)
        if style_key in ("zh", "chinese", "tencent_zh", "tencent"):
            return TencentZhPromptStrategy(use_system_prefix=use_system_prefix)

    # 2. Automatic model-specific inference based on filenames/repo
    if "milmmt" in model_name_lower or "milmmt" in repo_name_lower:
        return MiLMMTPromptStrategy()

    if any(k in model_name_lower or k in repo_name_lower for k in ("gemmax", "gemma")):
        return GemmaXPromptStrategy()

    return TencentZhPromptStrategy(use_system_prefix=use_system_prefix)


def build_translation_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    context: str = "",
    model_name: Optional[str] = None,
    repo_name: Optional[str] = None,
    prompt_style: Optional[str] = None,
    use_context: Optional[bool] = None,
) -> str:
    """Backward-compatible helper constructing prompt according to selected strategy."""
    if any(v is None for v in (model_name, repo_name, prompt_style, use_context)):
        try:
            from backend_cpp.config import config
            cfg_trans = config.translation
        except Exception:
            cfg_trans = None
    else:
        cfg_trans = None

    m_name = model_name if model_name is not None else (getattr(cfg_trans, "gguf_file", "") if cfg_trans else "")
    r_name = repo_name if repo_name is not None else (getattr(cfg_trans, "model", "") if cfg_trans else "")
    p_style = prompt_style if prompt_style is not None else (getattr(cfg_trans, "prompt_style", "auto") if cfg_trans else "auto")
    u_context = use_context if use_context is not None else (getattr(cfg_trans, "use_context", True) if cfg_trans else True)

    strategy = get_prompt_strategy(m_name, r_name, p_style)
    return strategy.build_prompt(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        context=context,
        use_context=u_context,
    )
