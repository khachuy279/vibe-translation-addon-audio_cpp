"""Named constants for ASR processing pipeline."""

# Audio stream properties
DEFAULT_SAMPLE_RATE: int = 16000

# Minimum audio duration (in seconds) required before running inference
MIN_INFERENCE_AUDIO_SEC: float = 0.20

# Deduplication cache settings
RECENT_COMMITS_CACHE_SEC: float = 6.0
DEDUP_WINDOW_SEC: float = 4.0
DEDUP_SUBSTRING_RATIO: float = 0.65
# Jaccard threshold is intentionally strict (0.80) to avoid false positives
# with short overlapping words (e.g., "Yes" vs "Yes please").
# Substring containment (DEDUP_SUBSTRING_RATIO=0.65) handles the "nearly equal" case.
DEDUP_JACCARD_THRESHOLD: float = 0.80

# Prefix stripping window (in seconds)
PREFIX_STRIP_WINDOW_SEC: float = 4.0
