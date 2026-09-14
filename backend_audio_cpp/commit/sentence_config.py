"""Sentence segmentation and commit configuration for backend_audio_cpp."""

from dataclasses import dataclass


@dataclass
class SentenceConfig:
    """Configuration for sentence boundary detection and commit triggering."""
    max_chars: int = 150
    max_duration_sec: float = 8.0
    min_words_to_commit: int = 2
    split_on_stability: bool = True          # Ngắt câu khi preview text ổn định qua nhiều chu kỳ poll
    stability_duration_sec: float = 1.0      # Thời gian (giây) preview text giữ nguyên để chốt câu
    stability_threshold_polls: int = 3       # Số lần poll tối thiểu giữ nguyên kết quả
