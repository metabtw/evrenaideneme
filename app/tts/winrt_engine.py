"""WinRT (OneCore) ses motoru — Windows'ta gerçek Türkçe ses.

Neden bu motor?
    Windows 10/11'de kullanıcı Ayarlar > Konuşma'dan Türkçe ses eklediğinde
    ses **OneCore** deposuna kurulur (``Speech_OneCore``). ``pyttsx3`` ise
    **SAPI5** deposunu okur; bu yüzden Türkçe ses kurulu olsa bile yalnızca
    İngilizce sesi görür (ölçüldü: SAPI5'te yalnızca Zira, OneCore'da Tolga).
    Kayıt defteri köprüsü de işe yaramaz: OneCore seslerinin motoru SAPI5'ten
    yüklenemez (farklı CLSID).

    Doğru yol ``Windows.Media.SpeechSynthesis`` API'sidir (WinRT). Çevrimdışı
    çalışır, kurulu Türkçe sesi kullanır ve WAV üretir.

Tasarım:
    * Sentez ~10 ms sürer; seslendirme (oynatma) bloklar ve TTS worker
      thread'inde yapılır (ana döngü etkilenmez).
    * ``stop()`` oynatmayı ``winsound`` üzerinden anında keser.
    * Hız/ses seviyesi SSML ``prosody`` ile verilir (WinRT'de doğrudan
      özellik yok).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from app.config import TTSSettings
from app.utils.logging import get_logger

from .base import SpeechEngine, SpeechError, SpeechUnavailableError

logger = get_logger(__name__)

#: Tercih sırası: kadın/erkek Türkçe sesler (Windows'ta kurulu olan neyse).
_PREFERRED = ("tolga", "filiz", "turkish", "tr-tr", "türkçe")


def _winrt_modules() -> tuple[Any, Any]:
    try:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer  # noqa: PLC0415
        from winrt.windows.storage.streams import DataReader  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise SpeechUnavailableError(
            "WinRT ses API'si yok. Kurulum: "
            "pip install winrt-Windows.Media.SpeechSynthesis winrt-Windows.Storage.Streams"
        ) from exc
    return SpeechSynthesizer, DataReader


def _winsound() -> Any:
    try:
        import winsound  # noqa: PLC0415 - yalnızca Windows
    except ImportError as exc:  # pragma: no cover
        raise SpeechUnavailableError(
            "winsound bulunamadı; bu motor yalnızca Windows'ta çalışır."
        ) from exc
    return winsound


class WinRTEngine(SpeechEngine):
    """OneCore Türkçe seslerle çevrimdışı sentez (Windows)."""

    def __init__(self, settings: TTSSettings | None = None) -> None:
        self.settings = settings or TTSSettings()
        self.warning = ""
        self._voice_name = ""
        self._busy = threading.Event()
        self._stop_requested = threading.Event()
        self._synthesizer: Any = None

    # -- kurulum -----------------------------------------------------------

    def _ensure_synthesizer(self) -> Any:
        if self._synthesizer is not None:
            return self._synthesizer

        speech_synthesizer, _ = _winrt_modules()
        try:
            synth = speech_synthesizer()
        except Exception as exc:  # noqa: BLE001 - WinRT her türlü hata verir
            raise SpeechUnavailableError(f"WinRT ses motoru başlatılamadı: {exc}") from exc

        voice = self._pick_voice(synth)
        if voice is None:
            raise SpeechUnavailableError(
                "Türkçe ses bulunamadı. Windows: Ayarlar > Zaman ve Dil > "
                "Konuşma > Ses ekle > Türkçe (Türkiye)."
            )

        try:
            synth.voice = voice
            self._voice_name = str(getattr(voice, "display_name", "türkçe"))
        except Exception as exc:  # noqa: BLE001
            raise SpeechUnavailableError(f"Ses seçilemedi: {exc}") from exc

        logger.info("WinRT Türkçe ses: %s", self._voice_name)
        self._synthesizer = synth
        return synth

    def _pick_voice(self, synth: Any) -> Any | None:
        # ``all_voices`` sınıf özniteliğidir (instance'da bulunmaz).
        try:
            voices = list(type(synth).all_voices)
        except Exception:  # noqa: BLE001
            try:
                voices = list(synth.all_voices)  # eski sürümler
            except Exception:
                return None
        if not voices:
            return None

        # 1) Dil kodu "tr" ile başlayanlar
        turkish = [
            voice
            for voice in voices
            if str(getattr(voice, "language", "")).lower().startswith("tr")
        ]
        if not turkish:
            return None
        # 2) Tercih listesine göre sırala
        for hint in _PREFERRED:
            for voice in turkish:
                haystack = (
                    f"{getattr(voice, 'display_name', '')} {getattr(voice, 'id', '')}"
                ).lower()
                if hint in haystack:
                    return voice
        return turkish[0]

    # -- SpeechEngine ------------------------------------------------------

    @property
    def name(self) -> str:
        return f"winrt ({self._voice_name})" if self._voice_name else "winrt"

    def speak(self, text: str) -> None:
        synth = self._ensure_synthesizer()
        winsound = _winsound()
        self._stop_requested.clear()
        self._busy.set()
        try:
            audio = self._synthesize(synth, text)
            if self._stop_requested.is_set() or not audio:
                return
            winsound.PlaySound(audio, winsound.SND_MEMORY)
        except SpeechError:
            raise
        except Exception as exc:  # noqa: BLE001 - oynatma aygıtı hataları
            raise SpeechError(f"Seslendirme başarısız: {exc}") from exc
        finally:
            self._busy.clear()

    def _synthesize(self, synth: Any, text: str) -> bytes:
        """Metni WAV baytlarına çevirir (SSML ile hız/ses)."""
        _, data_reader = _winrt_modules()
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{self.settings.language}">'
            f'<prosody rate="{self._rate_percent()}%" volume="{int(self.settings.volume * 100)}">'
            f"{_escape(text)}</prosody></speak>"
        )

        async def _run() -> bytes:
            stream = await synth.synthesize_ssml_to_stream_async(ssml)
            size = int(stream.size)
            if size <= 0:
                return b""
            reader = data_reader(stream.get_input_stream_at(0))
            await reader.load_async(size)
            buffer = reader.read_buffer(size)
            return _buffer_to_bytes(buffer, size)

        try:
            return asyncio.run(_run())
        except RuntimeError as exc:  # zaten çalışan event loop (beklenmez)
            raise SpeechError(f"WinRT sentez hatası: {exc}") from exc

    def _rate_percent(self) -> int:
        """``TTS_RATE`` (kelime/dk) → SSML yüzdesi. 180 wpm ≈ %0."""
        return max(-50, min(100, int((self.settings.rate - 180) / 1.8)))

    def stop(self) -> None:
        self._stop_requested.set()
        try:
            winsound = _winsound()
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:  # noqa: BLE001
            logger.debug("Oynatma durdurulamadı", exc_info=True)

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def close(self) -> None:
        self.stop()
        self._synthesizer = None


def _buffer_to_bytes(buffer: Any, size: int) -> bytes:
    """WinRT ``IBuffer`` → ``bytes``."""
    try:
        return bytes(memoryview(buffer))
    except TypeError:
        pass
    # Bazı sürümlerde doğrudan bytes() çalışmaz; DataReader ile oku.
    from winrt.windows.storage.streams import DataReader  # noqa: PLC0415

    reader = DataReader.from_buffer(buffer)
    return bytes(reader.read_buffer(size))


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
