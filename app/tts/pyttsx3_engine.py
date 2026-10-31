"""pyttsx3 tabanlı çevrimdışı Türkçe seslendirme (M3 iskeleti).

M3'te tamamlanacak: ``tr`` ses seçimi, cümle kuyruğu, ``stop()`` ile kesme.
Şimdilik sınıf yalnızca iskelet; oluşturulursa anlaşılır hata verir.
"""

from __future__ import annotations

from app.config import TTSSettings

from .base import SpeechEngine, SpeechUnavailableError


class Pyttsx3Engine(SpeechEngine):
    """``pyttsx3`` motoru (M3)."""

    def __init__(self, settings: TTSSettings | None = None) -> None:
        self.settings = settings or TTSSettings()
        raise SpeechUnavailableError(
            "pyttsx3 motoru M3 fazında eklenecek. Şimdilik TTS_ENGINE=none kullanın."
        )

    @property
    def name(self) -> str:
        return "pyttsx3"

    def speak(self, text: str) -> None:  # pragma: no cover - M3
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - M3
        raise NotImplementedError

    def is_busy(self) -> bool:  # pragma: no cover - M3
        return False

    def close(self) -> None:  # pragma: no cover - M3
        return None
