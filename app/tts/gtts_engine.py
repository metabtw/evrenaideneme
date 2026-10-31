"""gTTS tabanlı çevrimiçi Türkçe seslendirme (M3 iskeleti).

M3'te tamamlanacak: MP3 üretimi, oynatma, geçici dosya temizliği.
"""

from __future__ import annotations

from app.config import TTSSettings

from .base import SpeechEngine, SpeechUnavailableError


class GttsEngine(SpeechEngine):
    """``gTTS`` + oynatıcı motoru (M3)."""

    def __init__(self, settings: TTSSettings | None = None) -> None:
        self.settings = settings or TTSSettings()
        raise SpeechUnavailableError(
            "gTTS motoru M3 fazında eklenecek. Şimdilik TTS_ENGINE=none kullanın."
        )

    @property
    def name(self) -> str:
        return "gtts"

    def speak(self, text: str) -> None:  # pragma: no cover - M3
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - M3
        raise NotImplementedError

    def is_busy(self) -> bool:  # pragma: no cover - M3
        return False

    def close(self) -> None:  # pragma: no cover - M3
        return None
