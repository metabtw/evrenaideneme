"""Süreç-yalıtımlı pyttsx3 motoru (gerçek zamanlı sistemler için).

Neden ayrı bir **süreç**?
    ``pyttsx3`` (Windows'ta SAPI5/COM) ``runAndWait()`` sırasında GIL'i uzun
    süre tutar. Thread içinde çalıştırmak ana döngüyü 18 FPS'ten 7 FPS'e
    düşürdü (ölçüldü: 12 sn'lik koşu 32 sn sürdü). Python thread'leri aynı
    GIL'i paylaştığı için thread yalıtımı bu sorunu çözmez; süreç yalıtımı
    çözer: çocuk süreç kendi GIL'inde bloklanır, ana döngü akışı korur.

Kesme (barge-in) stratejisi:
    ``runAndWait()`` dışarıdan kesilemez. Bu yüzden ``stop()`` çocuk süreci
    sonlandırır ve bir sonraki ``speak()`` çağrısında yeniden başlatır.
    Yeniden başlatma ~0.3-1 sn sürer; 20 sn'lik bir donmadan çok daha iyidir.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from contextlib import suppress
from typing import Any

from app.config import TTSSettings
from app.utils.logging import get_logger

from .base import SpeechEngine, SpeechError, SpeechUnavailableError

logger = get_logger(__name__)

_TURKISH_HINTS = ("turkish", "türkçe", "tr-tr", "tr_tr", "tolga", "filiz", "turkce")
_START_TIMEOUT = 10.0
_SPEAK_TIMEOUT = 120.0


def _child_main(commands: Any, events: Any, settings: dict[str, Any]) -> None:  # pragma: no cover
    """Çocuk sürecin gövdesi: motoru kurar, komutları işler."""
    try:
        import pyttsx3  # noqa: PLC0415
    except ImportError as exc:
        events.put(("error", f"pyttsx3 kurulu değil: {exc}"))
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", settings["rate"])
        engine.setProperty("volume", settings["volume"])
        _select_voice(engine, events)
    except Exception as exc:  # noqa: BLE001
        events.put(("error", f"Ses motoru başlatılamadı: {exc}"))
        return

    events.put(("ready", ""))
    while True:
        try:
            command = commands.get()
        except (EOFError, OSError):
            break
        if command is None or command[0] == "quit":
            break
        if command[0] == "say":
            _, sequence, text = command
            try:
                engine.say(text)
                engine.runAndWait()
                events.put(("done", sequence))
            except Exception as exc:  # noqa: BLE001
                events.put(("error", f"Seslendirme başarısız: {exc}"))
                events.put(("done", sequence))
    with suppress(Exception):
        engine.stop()


def _select_voice(engine: Any, events: Any) -> None:  # pragma: no cover
    """Türkçe ses varsa seçer; yoksa uyarı olayı gönderir."""
    try:
        voices = engine.getProperty("voices") or []
    except Exception:  # noqa: BLE001
        return
    for voice in voices:
        haystack = " ".join(
            str(getattr(voice, attr, "")) for attr in ("name", "id", "languages")
        ).lower()
        if any(hint in haystack for hint in _TURKISH_HINTS):
            try:
                engine.setProperty("voice", voice.id)
                events.put(("voice", str(getattr(voice, "name", "türkçe"))))
            except Exception:  # noqa: BLE001
                pass
            return
    events.put(
        (
            "warning",
            "Türkçe ses paketi bulunamadı; telaffuz bozuk olabilir. "
            "Windows: Ayarlar > Zaman ve Dil > Konuşma > Ses ekle (Türkçe).",
        )
    )


class SubprocessSpeechEngine(SpeechEngine):
    """pyttsx3'ü ayrı süreçte çalıştıran motor (GIL yalıtımı)."""

    def __init__(self, settings: TTSSettings | None = None) -> None:
        self.settings = settings or TTSSettings()
        self.warning = ""
        self._context = mp.get_context("spawn")
        self._commands: Any = None
        self._events: Any = None
        self._process: Any = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._voice = ""

    # -- süreç yönetimi ----------------------------------------------------

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._commands = self._context.Queue()
        self._events = self._context.Queue()
        payload = {
            "rate": self.settings.rate,
            "volume": self.settings.volume,
            "language": self.settings.language,
        }
        process = self._context.Process(
            target=_child_main,
            args=(self._commands, self._events, payload),
            daemon=True,
            name="evrenaidd-tts-proc",
        )
        process.start()
        self._process = process

        try:
            kind, message = self._events.get(timeout=_START_TIMEOUT)
        except queue.Empty:
            self._terminate()
            raise SpeechUnavailableError(
                "Ses süreci zamanında başlamadı (ses aygıtı meşgul olabilir)."
            ) from None

        if kind == "error":
            self._terminate()
            raise SpeechUnavailableError(message)
        if kind == "voice":
            self._voice = message
            logger.info("Türkçe ses seçildi: %s", message)
        elif kind == "warning":
            self.warning = message
            logger.warning("%s", message)
        elif kind == "ready":
            logger.info("Ses süreci hazır (ayrı süreç, GIL yalıtımlı)")

    def _terminate(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            logger.debug("Ses süreci sonlandırılamadı", exc_info=True)

    # -- SpeechEngine ------------------------------------------------------

    @property
    def name(self) -> str:
        return f"pyttsx3-proc ({self._voice})" if self._voice else "pyttsx3-proc"

    def speak(self, text: str) -> None:
        with self._lock:
            self._ensure_process()
            process = self._process
            self._sequence += 1
            sequence = self._sequence
            assert self._commands is not None
            self._commands.put(("say", sequence, text))

            deadline = time.monotonic() + _SPEAK_TIMEOUT
            while True:
                # Kısa aralıklarla bekle: stop() süreci öldürdüyse hemen çık.
                if process is None or not process.is_alive():
                    raise SpeechError("Seslendirme kesildi")
                if time.monotonic() > deadline:
                    self.stop()
                    raise SpeechError("Seslendirme zaman aşımına uğradı")

                try:
                    kind, message = self._events.get(timeout=0.2)
                except queue.Empty:
                    continue
                if kind == "error":
                    raise SpeechError(message)
                if kind == "done" and message == sequence:
                    return

    def stop(self) -> None:
        """Süreci **asin kron** olarak sonlandırır (çağıranı bloklamaz).

        ``runAndWait`` dışarıdan kesilemez; süreç öldürülür ve bir sonraki
        ``speak`` çağrısında yeniden başlatılır. Sonlandırma ayrı bir thread'de
        yapılır: aksi halde kritik uyarı anında ana döngü saniyelerce kilitlenir
        (ölçüldü: p95 2000 ms).
        """
        process, self._process = self._process, None
        if process is None:
            return
        threading.Thread(
            target=self._reap, args=(process,), name="evrenaidd-tts-reap", daemon=True
        ).start()

    @staticmethod
    def _reap(process: Any) -> None:
        """Süreci kısa bir bekleme sonrası zorla sonlandırır."""
        try:
            process.join(timeout=0.3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        except Exception:  # noqa: BLE001
            logger.debug("Ses süreci toplanamadı", exc_info=True)

    def is_busy(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def close(self) -> None:
        with self._lock:
            commands, self._commands = self._commands, None
            if commands is not None:
                with suppress(Exception):
                    commands.put(("quit",))
            process, self._process = self._process, None
            if process is not None:
                try:
                    if process.is_alive():
                        process.join(timeout=1.5)
                    if process.is_alive():
                        process.terminate()
                except Exception:  # noqa: BLE001
                    logger.debug("Ses süreci kapatılamadı", exc_info=True)
