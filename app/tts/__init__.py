"""Sesli geri bildirim katmanı (giyilebilir asistan).

İki katman vardır:

* :class:`SpeechEngine` — tek metni okuyan **motor** (pyttsx3, gTTS, sessiz).
* :class:`TextToSpeech` — kuyruk, kesme (barge-in), tekrar bastırma gibi
  **politikayı** uygulayan servis. Uygulama kodu bunu kullanır.

    from app.tts import TextToSpeech, Priority, create_speech_engine

    tts = TextToSpeech(settings.tts, engine=create_speech_engine(settings.tts))
    tts.start()
    tts.say("Masada bir fincan var.")
    tts.say_critical("Dikkat, sandalye çok yakın.")
"""

from __future__ import annotations

from app.config import TTSSettings
from app.utils.logging import get_logger

from .base import SpeechEngine, SpeechError, SpeechUnavailableError, normalize_text
from .null_engine import NullSpeechEngine
from .service import Priority, TextToSpeech, split_sentences, to_speech_text

logger = get_logger(__name__)

__all__ = [
    "NullSpeechEngine",
    "Priority",
    "SpeechEngine",
    "SpeechError",
    "SpeechUnavailableError",
    "TextToSpeech",
    "create_speech_engine",
    "create_tts",
    "normalize_text",
    "split_sentences",
    "to_speech_text",
]


def create_speech_engine(settings: TTSSettings | None = None) -> SpeechEngine:
    """Ayardaki motoru oluşturur.

    Motor kurulamazsa (eksik ses paketi, ses aygıtı yok) **uygulama düşmez**;
    sessiz motora düşülür ve durum loglanır. Görme engelli bir kullanıcı için
    uygulamanın tamamen kapanması, sessiz çalışmasından daha kötüdür.
    """
    settings = settings or TTSSettings()

    if settings.engine == "none":
        return NullSpeechEngine()

    if settings.engine not in ("pyttsx3", "pyttsx3-proc", "gtts"):
        raise SpeechUnavailableError(f"Bilinmeyen TTS motoru: {settings.engine!r}")

    try:
        if settings.engine == "pyttsx3-proc":
            from .subprocess_engine import SubprocessSpeechEngine  # noqa: PLC0415

            return SubprocessSpeechEngine(settings)

        if settings.engine == "pyttsx3":
            from .pyttsx3_engine import Pyttsx3Engine  # noqa: PLC0415 - isteğe bağlı

            return Pyttsx3Engine(settings)

        from .gtts_engine import GttsEngine  # noqa: PLC0415 - isteğe bağlı

        return GttsEngine(settings)
    except SpeechUnavailableError as exc:
        logger.error("TTS motoru kullanılamıyor (%s): %s", settings.engine, exc)
        return NullSpeechEngine()


def create_tts(
    settings: TTSSettings | None = None,
    *,
    engine: SpeechEngine | None = None,
) -> TextToSpeech:
    """Hazır (başlatılmış) bir :class:`TextToSpeech` servisi döndürür."""
    settings = settings or TTSSettings()
    selected = engine if engine is not None else create_speech_engine(settings)
    return TextToSpeech(settings, engine=selected).start()
