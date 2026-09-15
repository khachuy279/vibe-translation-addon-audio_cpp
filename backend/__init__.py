"""Bilingual Subtitle Backend Package."""

from backend.config import config, load_config
from backend.utils.logger import logger, get_logger

__version__ = "2.0.0"
__all__ = ["config", "load_config", "logger", "get_logger"]
