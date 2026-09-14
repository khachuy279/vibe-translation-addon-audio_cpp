"""Voice Catalog Manager for backend_audio_cpp.

Manages reference voice samples located in backend_audio_cpp/voices/
used for zero-shot voice cloning with OmniVoice.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("backend_audio_cpp.tts.voices")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VOICES_DIR = _PROJECT_ROOT / "backend_audio_cpp" / "voices"
_VOICES_JSON = _VOICES_DIR / "voices.json"


@dataclass
class VoiceProfile:
    """Metadata for a reference voice clone sample."""
    id: str
    name: str
    audio_path: Path
    reference_text: str

    def exists(self) -> bool:
        return self.audio_path.exists()


class VoiceManager:
    """Manages available reference audio samples for voice cloning."""

    _instance: Optional["VoiceManager"] = None

    @classmethod
    def get_instance(cls) -> "VoiceManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, voices_dir: Optional[Path] = None):
        self.voices_dir = voices_dir or _VOICES_DIR
        self.catalog: Dict[str, VoiceProfile] = {}
        self.default_voice_id: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        """Scan voices.json and directory to populate available voice profiles."""
        self.catalog.clear()
        if not self.voices_dir.exists():
            self.voices_dir.mkdir(parents=True, exist_ok=True)

        json_file = self.voices_dir / "voices.json"
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    for item in entries:
                        v_id = item.get("id") or item.get("name", "")
                        raw_audio = item.get("audio", "")
                        audio_path = (
                            Path(raw_audio)
                            if Path(raw_audio).is_absolute()
                            else (_PROJECT_ROOT / raw_audio)
                        )
                        # Fallback to voices_dir if relative path points elsewhere
                        if not audio_path.exists():
                            audio_path = self.voices_dir / Path(raw_audio).name

                        self.catalog[v_id] = VoiceProfile(
                            id=v_id,
                            name=item.get("name", v_id),
                            audio_path=audio_path,
                            reference_text=item.get("text", ""),
                        )
            except Exception as e:
                logger.error(f"Error reading voices.json: {e}")

        # Also register any standalone .wav files in voices_dir not in voices.json
        for wav_file in self.voices_dir.glob("*.wav"):
            fname = wav_file.name
            if fname not in self.catalog:
                self.catalog[fname] = VoiceProfile(
                    id=fname,
                    name=fname.replace(".wav", "").replace("_", " ").title(),
                    audio_path=wav_file,
                    reference_text="",
                )

        if self.catalog:
            self.default_voice_id = next(iter(self.catalog.keys()))
        logger.info(f"Loaded {len(self.catalog)} voice profile(s) from {self.voices_dir}")

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return list of voice profiles for UI dropdown."""
        return [
            {
                "id": v.id,
                "name": v.name,
                "audio_path": str(v.audio_path),
                "has_audio": v.exists(),
                "is_default": (v.id == self.default_voice_id),
            }
            for v in self.catalog.values()
        ]

    def get_voice(self, voice_id_or_name: Optional[str] = None) -> Optional[VoiceProfile]:
        """Resolve voice profile by ID, filename, or fallback to default."""
        if not voice_id_or_name:
            voice_id_or_name = self.default_voice_id

        if not voice_id_or_name:
            return None

        # Direct match
        if voice_id_or_name in self.catalog:
            return self.catalog[voice_id_or_name]

        # Match by filename
        for k, v in self.catalog.items():
            if Path(k).name == Path(voice_id_or_name).name:
                return v

        # Fallback to default
        if self.default_voice_id and self.default_voice_id in self.catalog:
            return self.catalog[self.default_voice_id]

        return None
