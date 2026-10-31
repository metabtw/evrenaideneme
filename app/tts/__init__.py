"""Sesli geri bildirim katmanı.

Motor seçimi::

    engine = create_speech_engine(settings.tts)   # none -> sessiz motor
"""

from __future__ import annotations

from app.config import TTSSettings

from .base import SpeechEngine, SpeechError, SpeechUnavailableError, normalize_text
from .null_engine import NullSpeechEngine

__all__ = [
    "NullSpeechEngine",
    "SpeechEngine",
    "SpeechError",
    "SpeechUnavailableError",
    "create_speech_engine",
    "normalize_text",
]


def create_speech_engine(settings: TTSSettings | None = None) -> SpeechEngine:
    """Ayardaki motoru oluşturur.

    M3'e kadar yalnızca ``none`` çalışır; diğer motorlar anlaşılır bir
    :class:`SpeechUnavailableError` fırlatır (uygulama bu hatayı yakalayıp
    sessiz moda düşebilir).
    """
    settings = settings or TTSSettings()

    if settings.engine == "none":
        return NullSpeechEngine()
    if settings.engine == "pyttsx3":
        from .pyttsx3_engine import Pyttsx3Engine  # noqa: PLC0415 - isteğe bağlı

        return Pyttsx3Engine(settings)
    if settings.engine == "gtts":
        from .gtts_engine import GttsEngine  # noqa: PLC0415 - isteğe bağlı

        return GttsEngine(settings)

    raise SpeechUnavailableError(f"Bilinmeyen TTS motoru: {settings.engine!r}")
