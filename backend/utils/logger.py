"""Module logging tập trung cho Backend.

Hỗ trợ:
- Định dạng log chuyên nghiệp với màu sắc ANSI trên Windows và Linux.
- Định danh module rõ ràng, in ra console với emoji trực quan.
- Độ chính xác microsecond/millisecond cho phân tích độ trễ real-time.
- Ngăn chặn lỗi charmap/emoji trên Windows console.
"""

import logging
import os
import sys
import time
from typing import Optional

# Cấu hình UTF-8 cho stdout/stderr trên Windows để tránh crash emoji
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass

# Bảng mã màu ANSI
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# Emoji cho từng cấp độ và module
LEVEL_ICONS = {
    logging.DEBUG: "🔍",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🚨",
}

MODULE_COLORS = {
    "CORE": LogColors.CYAN,
    "VAD": LogColors.GREEN,
    "ASR": LogColors.BLUE,
    "ASR_PREVIEW": LogColors.DIM + LogColors.BLUE,
    "ASR_COMMIT": LogColors.BOLD + LogColors.BLUE,
    "TRANSLATE": LogColors.MAGENTA,
    "TTS": LogColors.YELLOW,
    "WS": LogColors.CYAN,
    "METRICS": LogColors.GREEN,
}


class ColoredFormatter(logging.Formatter):
    """Formatter tùy biến thêm màu sắc và biểu tượng trực quan."""

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR")))

    def format(self, record: logging.LogRecord) -> str:
        created = record.created
        msec = int((created - int(created)) * 1000)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created)) + f".{msec:03d}"
        
        icon = LEVEL_ICONS.get(record.levelno, "•")
        levelname = record.levelname
        
        module_tag = getattr(record, "module_tag", record.name.replace("backend.", "").upper())
        
        if self.use_color:
            level_color = {
                logging.DEBUG: LogColors.DIM,
                logging.INFO: LogColors.GREEN,
                logging.WARNING: LogColors.YELLOW,
                logging.ERROR: LogColors.RED,
                logging.CRITICAL: LogColors.BG_RED + LogColors.WHITE,
            }.get(record.levelno, LogColors.RESET)
            
            tag_color = MODULE_COLORS.get(module_tag, LogColors.CYAN)
            
            header = (
                f"{LogColors.DIM}{time_str}{LogColors.RESET} "
                f"{level_color}[{levelname:<5}]{LogColors.RESET} "
                f"{tag_color}[{module_tag}]{LogColors.RESET} {icon} "
            )
        else:
            header = f"{time_str} [{levelname:<5}] [{module_tag}] {icon} "
            
        message = record.getMessage()
        
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                message = f"{message}\n{record.exc_text}"
                
        return f"{header}{message}"


def get_logger(name: str = "backend", level: int = logging.INFO) -> logging.Logger:
    """Khởi tạo hoặc lấy logger cấu hình chuẩn cho Backend."""
    logger_instance = logging.getLogger(name)
    
    if not logger_instance.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter(use_color=True))
        logger_instance.addHandler(handler)
        logger_instance.setLevel(level)
        logger_instance.propagate = False
        
    return logger_instance


logger = get_logger("backend", level=logging.DEBUG)
