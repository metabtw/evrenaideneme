"""Giyilebilir asistan için TTS servisi: kuyruk + tek thread + kesme (barge-in).

Neden ayrı bir servis sınıfı?
    ``SpeechEngine`` arayüzü senkron ve saftır (tek metin okur). Giyilebilir
    cihazda ise üç davranış kritiktir ve bunlar politika işidir, motor işi değil:

    1. **Bloklamama** — seslendirme saniyelerce sürer, ana döngü bekleyemez.
    2. **Kesme (barge-in)** — "Dikkat, araba çok yakın" uyarısı, sürmekte olan
       uzun bir açıklamayı anında kesebilmelidir. Görme engelli kullanıcı için
       bu bir tercih değil, güvenlik gereğidir.
    3. **Tekrar bastırma** — aynı uyarı saniyede 30 kez okunmamalı.

    ``TextToSpeech`` bu politikayı uygular, motoru değiştirilebilir tutar.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum

from app.config import TTSSettings
from app.utils.logging import get_logger

from .base import SpeechEngine, SpeechError, normalize_text

logger = get_logger(__name__)


class Priority(IntEnum):
    """Düşük değer = daha önemli. Kritik uyarı kuyruğu öne alır."""

    CRITICAL = 0
    NORMAL = 1
    BACKGROUND = 2


@dataclass(order=True, slots=True)
class _Utterance:
    priority: int
    sequence: int
    text: str = field(compare=False)
    interrupt: bool = field(default=False, compare=False)


class TextToSpeech:
    """Konuşma kuyruğunu yöneten, thread-güvenli yüksek seviye TTS.

    Örnek::

        tts = TextToSpeech(settings.tts, engine=Pyttsx3Engine(settings.tts))
        tts.start()
        tts.say("Masada bir fincan var.")
        tts.say("Dikkat, sandalye çok yakın.", priority=Priority.CRITICAL)
        tts.stop()          # sürmekte olanı kes
        tts.close()
    """

    def __init__(
        self,
        settings: TTSSettings | None = None,
        *,
        engine: SpeechEngine | None = None,
        dedupe_window: float = 2.0,
        name: str = "evrenaidd-tts",
    ) -> None:
        self.settings = settings or TTSSettings()
        self._engine = engine
        self._dedupe_window = dedupe_window
        self._name = name

        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._queue: list[_Utterance] = []
        self._sequence = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._interrupt_event = threading.Event()

        self._speaking = False
        self._current: str = ""
        self._recent: dict[str, float] = {}
        self._spoken_count = 0
        self._enabled = self.settings.engine != "none"

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> TextToSpeech:
        """Konuşma thread'ini başlatır (motor zaten hazır olmalı)."""
        if self._thread is not None and self._thread.is_alive():
            return self
        if self._engine is None:
            logger.warning("TTS motoru yok; sesli geri bildirim devre dışı")
            self._enabled = False
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.info("TTS hazır (%s)", getattr(self._engine, "name", "bilinmiyor"))
        return self

    def close(self, timeout: float = 3.0) -> None:
        """Kuyruğu boşaltır, thread'i durdurur ve motoru kapatır."""
        self._stop_event.set()
        self.stop()
        with self._wakeup:
            self._queue.clear()
            self._wakeup.notify_all()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("TTS thread'i %.1f sn içinde durmadı", timeout)
        self._thread = None

        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:  # noqa: BLE001 - kapanış hatası akışı etkilemesin
                logger.debug("TTS motoru kapatılırken hata", exc_info=True)

    def __enter__(self) -> TextToSpeech:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- genel API ---------------------------------------------------------

    def say(
        self,
        text: str,
        *,
        priority: Priority = Priority.NORMAL,
        interrupt: bool = False,
    ) -> bool:
        """Metni kuyruğa ekler.

        Args:
            text: Okunacak metin (Türkçe).
            priority: Düşük değer önce okunur.
            interrupt: ``True`` ise sürmekte olan konuşma kesilir (kritik uyarı).

        Returns:
            Kuyruğa alındıysa ``True``; boş/tekrar/sessiz mod ise ``False``.
        """
        if not self._enabled or self._engine is None:
            return False

        clean = normalize_text(text)
        if not clean:
            return False

        now = time.monotonic()

        # Kesme: sürmekte olan konuşmayı durdur ve bekleyenleri at, SONRA ekle.
        # (Ters sırada yapılırsa yeni kritik mesaj da silinirdi.)
        if interrupt:
            self._preempt()

        with self._wakeup:
            last = self._recent.get(clean)
            if last is not None and now - last < self._dedupe_window:
                logger.debug("Tekrar eden metin atlandı: %s", clean[:40])
                return False
            self._recent[clean] = now
            # Sözlük şişmesin: eski kayıtları temizle.
            if len(self._recent) > 128:
                threshold = now - self._dedupe_window * 4
                self._recent = {k: v for k, v in self._recent.items() if v >= threshold}

            self._sequence += 1
            self._queue.append(
                _Utterance(
                    priority=int(priority),
                    sequence=self._sequence,
                    text=clean,
                    interrupt=interrupt,
                )
            )
            self._queue.sort(key=lambda item: (item.priority, item.sequence))
            self._wakeup.notify()

        return True

    def _preempt(self) -> None:
        """Sürmekte olan konuşmayı keser, bekleyen kuyruğu boşaltır."""
        with self._wakeup:
            self._interrupt_event.set()
            self._queue.clear()
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Motor durdurulurken hata", exc_info=True)

    def say_critical(self, text: str) -> bool:
        """Kritik uyarı: kuyruğu öne alır ve mevcut konuşmayı keser."""
        return self.say(text, priority=Priority.CRITICAL, interrupt=True)

    def stop(self) -> None:
        """Sürmekte olan konuşmayı keser ve bekleyen kuyruğu boşaltır."""
        self._preempt()

    def set_enabled(self, enabled: bool) -> bool:
        """Sesli geri bildirimi açar/kapatır; yeni durumu döndürür."""
        with self._wakeup:
            self._enabled = bool(enabled) and self._engine is not None
            if not self._enabled:
                self._queue.clear()
            enabled_now = self._enabled
        if not enabled_now:
            self.stop()
        logger.info("Sesli geri bildirim: %s", "açık" if enabled_now else "kapalı")
        return enabled_now

    def toggle(self) -> bool:
        return self.set_enabled(not self._enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._speaking or bool(self._queue)

    @property
    def current_text(self) -> str:
        with self._lock:
            return self._current

    @property
    def spoken_count(self) -> int:
        with self._lock:
            return self._spoken_count

    # -- thread gövdesi ----------------------------------------------------

    def _run(self) -> None:
        assert self._engine is not None
        while not self._stop_event.is_set():
            utterance = self._take()
            if utterance is None:
                return

            self._interrupt_event.clear()
            with self._lock:
                self._speaking = True
                self._current = utterance.text

            try:
                for chunk in split_sentences(utterance.text):
                    if self._interrupt_event.is_set() or self._stop_event.is_set():
                        break
                    self._engine.speak(chunk)
            except SpeechError as exc:
                logger.error("Seslendirme hatası: %s", exc)
            except Exception:  # noqa: BLE001 - TTS asla akışı düşürmemeli
                logger.exception("Seslendirmede beklenmeyen hata")
            finally:
                with self._lock:
                    self._speaking = False
                    self._current = ""
                    self._spoken_count += 1
                    self._wakeup.notify_all()

    def _take(self) -> _Utterance | None:
        with self._wakeup:
            while not self._queue and not self._stop_event.is_set():
                self._wakeup.wait(timeout=0.2)
            if self._stop_event.is_set():
                return None
            return self._queue.pop(0)


def split_sentences(text: str, *, max_chars: int = 180) -> list[str]:
    """Uzun metni cümlelere böler.

    Neden? ``stop()`` cümle sınırında daha hızlı etkili olur ve uzun tek blok
    motorlarda takılmaya yol açabilir. Cümle sonu yoksa kelime sınırından böler.
    """
    text = normalize_text(text)
    if not text:
        return []

    sentences: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in ".!?…" and len(current.strip()) > 1:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue
        words = sentence.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > max_chars and line:
                chunks.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            chunks.append(line)
    return chunks


def to_speech_text(text: str, *, max_chars: int = 600) -> str:
    """VLM yanıtını konuşmaya uygun hale getirir.

    Madde işaretleri, markdown ve fazla boşluk temizlenir; uzun yanıt kısaltılır.
    """
    clean = normalize_text(text.replace("#", " ").replace("*", " ").replace("`", " "))
    clean = clean.replace("- ", "").replace("•", "")
    clean = normalize_text(clean)
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    last_stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    return cut[: last_stop + 1] if last_stop > 80 else cut
