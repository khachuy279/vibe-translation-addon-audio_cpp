"""Voice Manager for audio.cpp OmniVoice Voice Cloning.

Manages reference voice files in backend_cpp/voices/ and metadata from voices.json.
Provides fast hot-swapping of clone voices without server or model restart.
"""

import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from backend_cpp.config import config

logger = logging.getLogger(__name__)

BACKEND_CPP_DIR = Path(__file__).resolve().parent.parent


class VoiceManager:
    """Manages voice clone samples with dynamic discovery, metadata parsing, and thread-safety."""

    _cached_voices: Optional[List[Dict[str, Any]]] = None
    _lock = threading.Lock()

    @classmethod
    def get_voices_dir(cls) -> Path:
        """Return configured voices directory or default fallback."""
        configured = getattr(getattr(config, "tts", None), "voices_dir", None)
        if configured:
            p = Path(configured)
            if p.exists() or p.parent.exists():
                return p
        return BACKEND_CPP_DIR / "voices"

    @classmethod
    def get_available_voices(cls, force_reload: bool = False) -> List[Dict[str, Any]]:
        """Scan voices directory and load voices.json metadata thread-safely."""
        with cls._lock:
            if cls._cached_voices is not None and not force_reload:
                return cls._cached_voices

            voices_dir = cls.get_voices_dir()
            voices_dir.mkdir(parents=True, exist_ok=True)
            voices_config_file = voices_dir / "voices.json"

            voices: List[Dict[str, Any]] = []
            registered_files = set()

            # 1. Load from voices.json if present
            if voices_config_file.exists():
                try:
                    with open(voices_config_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and item.get("audio"):
                                    audio_path = item["audio"]
                                    p = Path(audio_path)
                                    full_path = p if p.is_absolute() else (voices_dir / p.name)
                                    if full_path.exists():
                                        item_id = item.get("id") or full_path.name
                                        voices.append({
                                            "id": item_id,
                                            "name": item.get("name") or item_id,
                                            "audio": str(full_path),
                                            "text": item.get("text") or "",
                                        })
                                        registered_files.add(full_path.name.lower())
                except Exception as e:
                    logger.warning(f"[VoiceManager] Failed to read voices.json: {e}")

            # 2. Auto-discover any .wav files in voices directory not yet registered
            try:
                for file_path in sorted(voices_dir.glob("*.wav")):
                    if file_path.name.lower() not in registered_files:
                        clean_name = file_path.stem.replace("_", " ").replace("-", " ").title()
                        # Check if a sibling .txt transcript exists
                        sibling_txt = file_path.with_suffix(".txt")
                        ref_text = ""
                        if sibling_txt.exists():
                            try:
                                ref_text = sibling_txt.read_text(encoding="utf-8").strip()
                            except Exception:
                                pass

                        voices.append({
                            "id": file_path.name,
                            "name": f"🎙️ {clean_name} ({file_path.name})",
                            "audio": str(file_path),
                            "text": ref_text,
                        })
                        registered_files.add(file_path.name.lower())
            except Exception as e:
                logger.warning(f"[VoiceManager] Error scanning voices dir: {e}")

            # 3. Fallback default if completely empty
            if not voices:
                default_wav = voices_dir / "speaker_01_0039.wav"
                if default_wav.exists():
                    voices.append({
                        "id": "speaker_01_0039.wav",
                        "name": "👩 Giọng nữ trẻ, ngọt ngào",
                        "audio": str(default_wav),
                        "text": "Suốt quãng đời còn lại, bởi vì chúng ta không thể nào hợp lý hoá nó, mà cũng không thể nào thuyết phục được chính mình.",
                    })
                else:
                    logger.warning(
                        f"[VoiceManager] No voice audio files found in {voices_dir}. "
                        "Voice cloning requires at least one .wav reference file."
                    )

            cls._cached_voices = voices
            return voices

    @classmethod
    def resolve_voice(cls, voice_id_or_path: Optional[str]) -> Tuple[str, str]:
        """Resolve voice_id or audio path to (abs_audio_path, ref_text).

        Avoids mixing mismatched reference text with custom audio to prevent hallucination.
        """
        voices = cls.get_available_voices()
        if not voices:
            logger.warning("[VoiceManager] No available voices registered or found.")
            return "", ""

        if not voice_id_or_path:
            return voices[0]["audio"], voices[0]["text"]

        target = str(voice_id_or_path).strip().lower()
        target_name = Path(target).name.lower()

        # Check registered voices
        for v in voices:
            if v["id"].lower() == target or Path(v["audio"]).name.lower() == target_name:
                return v["audio"], v["text"]

        # Direct file path fallback
        p = Path(voice_id_or_path)
        if not p.is_absolute():
            p = cls.get_voices_dir() / p.name

        if p.exists():
            # Check for sibling .txt transcript
            sibling_txt = p.with_suffix(".txt")
            ref_text = ""
            if sibling_txt.exists():
                try:
                    ref_text = sibling_txt.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            else:
                logger.warning(
                    f"[VoiceManager] Custom voice '{p.name}' provided without transcript (.txt). "
                    f"Using empty ref_text to avoid cross-speaker hallucination."
                )
            return str(p), ref_text

        # Default fallback to first available registered voice
        return voices[0]["audio"], voices[0]["text"]
