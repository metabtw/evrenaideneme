"""pyttsx3 tabanlı çevrimdışı Türkçe seslendirme motoru.

Tasarım notları:
    * Motor **kendi thread'inde** oluşturulur ve kullanılır. Windows'ta SAPI5
      COM nesnesi thread'e bağlıdır; motoru başka thread'den sürmek kilitlenme
      üretir. Bu yüzden ``speak`` çağrısı, motoru yaratan thread'den yapılmalıdır
      (bkz. :class:`app.tts.service.TextToSpeech`).
    * Türkçe ses paketi yoksa İngilizce ses ile okumaya çalışmak anlaşılmaz
      olur; bu durumda net bir uyarı verilir ve kullanıcıya çözüm sunulur.
    * ``runAndWait`` bloklar; ``stop()`` çağrısı bu bekleme sırasında etkilidir.
"""

from __future__ import annotations

from typing import Any

from app.config import TTSSettings
from app.utils.logging import get_logger

from .base import SpeechEngine, SpeechError, SpeechUnavailableError

logger = get_logger(__name__)

#: Türkçe ses paketlerinde geçen ipuçları (SAPI5, eSpeak, macOS).
_TURKISH_HINTS = ("turkish", "türkçe", "tr-tr", "tr_tr", "tolga", "filiz", "turkce")


def _import_pyttsx3() -> Any:
    try:
        import pyttsx3  # noqa: PLC0415 - ağır/isteğe bağlı bağımlılık
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise SpeechUnavailableError(
            "pyttsx3 kurulu değil. Kurulum: pip install -r requirements.txt"
        ) from exc
    return pyttsx3


class Pyttsx3Engine(SpeechEngine):
    """``pyttsx3`` (SAPI5 / NSSpeechSynthesizer / eSpeak) motoru."""

    def __init__(self, settings: TTSSettings | None = None) -> None:
        self.settings = settings or TTSSettings()
        self._engine: Any | None = None
        self._voice_name = ""
        self.warning = ""

    # -- yaşam döngüsü -----------------------------------------------------

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        pyttsx3 = _import_pyttsx3()
        try:
            engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001 - sürücü/ses aygıtı yok
            raise SpeechUnavailableError(
                f"Ses motoru başlatılamadı (ses aygıtı bulunamadı?). Ayrıntı: {exc}"
            ) from exc

        try:
            engine.setProperty("rate", self.settings.rate)
            engine.setProperty("volume", self.settings.volume)
        except Exception:  # noqa: BLE001 - bazı sürücüler bu özellikleri yok sayar
            logger.debug("Ses hızı/sesi ayarlanamadı", exc_info=True)

        voice, warning = self._select_voice(engine)
        if voice is not None:
            try:
                engine.setProperty("voice", voice)
            except Exception:  # noqa: BLE001
                logger.debug("Ses seçilemedi", exc_info=True)
        if warning:
            self.warning = warning
            logger.warning("%s", warning)

        self._engine = engine
        return engine

    def _select_voice(self, engine: Any) -> tuple[str | None, str]:
        """Türkçe ses bulmayı dener; bulamazsa uyarı metni döndürür."""
        try:
            voices = engine.getProperty("voices") or []
        except Exception:  # noqa: BLE001
            return None, ""

        for voice in voices:
            haystack = " ".join(
                str(getattr(voice, attr, "")) for attr in ("name", "id", "languages")
            ).lower()
            if any(hint in haystack for hint in _TURKISH_HINTS):
                self._voice_name = str(getattr(voice, "name", "türkçe"))
                logger.info("Türkçe ses seçildi: %s", self._voice_name)
                return str(getattr(voice, "id", "")) or None, ""

        return None, (
            "Türkçe ses paketi bulunamadı; mevcut sesle okunacak ve telaffuz "
            "bozuk olabilir. Windows: Ayarlar > Zaman ve Dil > Konuşma > "
            "Ses ekle (Türkçe) — veya `TTS_ENGINE=gtts` kullanın."
        )

    # -- SpeechEngine ------------------------------------------------------

    @property
    def name(self) -> str:
        if self._voice_name:
            return f"pyttsx3 ({self._voice_name})"
        return "pyttsx3"

    def speak(self, text: str) -> None:
        engine = self._ensure_engine()
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001 - sürücü hataları çeşitli
            raise SpeechError(f"Seslendirme başarısız: {exc}") from exc

    def stop(self) -> None:
        engine = self._engine
        if engine is None:
            return
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            logger.debug("Motor durdurulamadı", exc_info=True)

    def is_busy(self) -> bool:
        engine = self._engine
        if engine is None:
            return False
        try:
            return bool(engine.isBusy())
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is None:
            return
        try:
            engine.stop()
            del engine
        except Exception:  # noqa: BLE001
            logger.debug("Motor kapatılırken hata", exc_info=True)
