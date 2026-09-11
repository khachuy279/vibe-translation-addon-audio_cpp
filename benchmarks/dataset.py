"""Dataset discovery, metadata inference, and ground truth loading."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf


@dataclass
class DatasetPair:
    """Benchmark dataset audio/reference pair with rich metadata."""
    pair_id: str
    wav_path: Path
    txt_path: Path
    raw_reference: str
    normalized_reference: str
    inferred_language: str
    inferred_condition: str
    inferred_speed: str
    inferred_duration_sec: float
    actual_duration_sec: float
    channels: int
    sample_rate: int
    bit_depth: int
    format_subtype: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "wav_name": self.wav_path.name,
            "txt_name": self.txt_path.name,
            "language": self.inferred_language,
            "condition": self.inferred_condition,
            "speed": self.inferred_speed,
            "duration_sec": round(self.actual_duration_sec, 2),
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "raw_reference": self.raw_reference,
            "normalized_reference": self.normalized_reference,
        }


def normalize_text_reference(text: str, language: str = "en") -> str:
    """Normalize reference text for WER/CER calculation without altering raw ground truth."""
    if not text:
        return ""
    t = text.strip()
    # Normalize unicode whitespace
    t = re.sub(r"\s+", " ", t)

    lang_lower = (language or "").lower()
    if lang_lower in ("zh", "chinese", "ja", "japanese"):
        # For CJK, remove common fullwidth and ASCII punctuation
        t = re.sub(r"[，。！？、；：“”‘’（）《》【】…—\.,!\?:;\"\'\(\)\[\]\-]", "", t)
        t = re.sub(r"\s+", "", t)
    else:
        # Western languages: lower-case, remove outer punctuation while keeping letters/numbers/spaces
        t = t.lower()
        t = re.sub(r"[^\w\s\']", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t


def infer_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract metadata (language, condition, speed, duration) from filename."""
    base = Path(filename).stem
    meta: Dict[str, str] = {
        "language": "UNKNOWN",
        "condition": "CLEAN",
        "speed": "NORMAL",
        "duration_sec": "0.0",
    }

    lower = base.lower()

    # Inferred language
    if "cross_lingual" in lower or ("english" in lower and ("french" in lower or "spanish" in lower)):
        meta["language"] = "multi"
    elif "chinese" in lower:
        meta["language"] = "zh"
    elif "japanese" in lower:
        meta["language"] = "ja"
    elif "russian" in lower:
        meta["language"] = "ru"
    elif "english" in lower:
        meta["language"] = "en"
    elif "french" in lower:
        meta["language"] = "fr"
    elif "spanish" in lower:
        meta["language"] = "es"

    # Inferred condition
    if "multiple_kinds_of_noise" in lower:
        meta["condition"] = "HEAVY_MULTI_NOISE"
    elif "noise" in lower:
        meta["condition"] = "NOISY"
    elif "low_speech_quality" in lower:
        meta["condition"] = "LOW_QUALITY"
    elif "fast_speed" in lower:
        meta["condition"] = "FAST_SPEECH"

    # Inferred speed
    if "fast_speed" in lower:
        meta["speed"] = "FAST"
    elif "slow_speed" in lower:
        meta["speed"] = "SLOW"

    # Inferred duration from regex e.g. "11s", "28s", "88s"
    dur_match = re.search(r"(\d+)s(?:ec)?", lower)
    if dur_match:
        meta["duration_sec"] = float(dur_match.group(1))

    return meta


def discover_dataset(directory: str | Path = "wav_test") -> List[DatasetPair]:
    """Scan directory for matching *.wav and *.txt pairs."""
    dir_path = Path(directory)
    if not dir_path.is_absolute():
        dir_path = Path.cwd() / dir_path

    if not dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dir_path}")

    wav_files = sorted(dir_path.glob("*.wav"))
    dataset: List[DatasetPair] = []

    for wav_file in wav_files:
        base_name = wav_file.stem
        txt_file = dir_path / f"{base_name}.txt"

        raw_ref = ""
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
                raw_ref = f.read()

        meta = infer_metadata_from_filename(base_name)
        lang = meta["language"]

        norm_ref = normalize_text_reference(raw_ref, language=lang)

        # Inspect WAV actual technical audio properties via soundfile
        try:
            info = sf.info(str(wav_file))
            actual_dur = info.duration
            channels = info.channels
            sample_rate = info.samplerate
            subtype = info.subtype
            # Map subtype to approximate bit depth
            bit_depth = 16
            if "PCM_24" in subtype:
                bit_depth = 24
            elif "PCM_32" in subtype or "FLOAT" in subtype:
                bit_depth = 32
            elif "PCM_16" in subtype:
                bit_depth = 16
            elif "PCM_U8" in subtype:
                bit_depth = 8
        except Exception:
            actual_dur = float(meta["duration_sec"])
            channels = 1
            sample_rate = 16000
            bit_depth = 16
            subtype = "PCM_16"

        pair = DatasetPair(
            pair_id=base_name,
            wav_path=wav_file,
            txt_path=txt_file,
            raw_reference=raw_ref,
            normalized_reference=norm_ref,
            inferred_language=meta["language"],
            inferred_condition=meta["condition"],
            inferred_speed=meta["speed"],
            inferred_duration_sec=float(meta["duration_sec"]),
            actual_duration_sec=actual_dur,
            channels=channels,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            format_subtype=subtype,
        )
        dataset.append(pair)

    return dataset
