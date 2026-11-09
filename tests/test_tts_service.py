"""TextToSpeech testleri: kuyruk, öncelik, kesme, tekrar bastırma, cümle bölme."""

from __future__ import annotations

import threading
import time

from app.tts import Priority, TextToSpeech, split_sentences, to_speech_text


class RecordingEngine:
    """Konuşulan metinleri kaydeden sahte motor.

    ``hold=True`` ise ``speak`` çağrısı test ``block.set()`` diyene kadar
    bekler; kuyruk/kesme davranışını deterministik sınamak için kullanılır.
    """

    def __init__(self, *, hold: bool = False) -> None:
        self.spoken: list[str] = []
        self._speaking = threading.Event()
        self.stop_calls = 0
        self.closed = False
        self.block = threading.Event()
        if not hold:
            self.block.set()

    @property
    def name(self) -> str:
        return "recording"

    def speak(self, text: str) -> None:
        self._speaking.set()
        self.block.wait(timeout=5.0)
        self.spoken.append(text)
        self._speaking.clear()

    def stop(self) -> None:
        self.stop_calls += 1

    def is_busy(self) -> bool:
        return self._speaking.is_set()

    def close(self) -> None:
        self.closed = True


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_say_speaks_text_on_background_thread() -> None:
    engine = RecordingEngine()
    with TextToSpeech(engine=engine) as tts:
        assert tts.say("Merhaba dünya.") is True
        assert _wait(lambda: engine.spoken == ["Merhaba dünya."])

    assert engine.closed is True


def test_empty_or_whitespace_text_is_rejected() -> None:
    engine = RecordingEngine()
    with TextToSpeech(engine=engine) as tts:
        assert tts.say("") is False
        assert tts.say("   \n ") is False
        time.sleep(0.05)
    assert engine.spoken == []


def test_duplicate_text_is_suppressed_within_window() -> None:
    engine = RecordingEngine()
    with TextToSpeech(engine=engine, dedupe_window=5.0) as tts:
        assert tts.say("Aynı uyarı.") is True
        assert tts.say("Aynı uyarı.") is False
        assert _wait(lambda: engine.spoken == ["Aynı uyarı."])


def test_priority_orders_queue() -> None:
    engine = RecordingEngine(hold=True)
    with TextToSpeech(engine=engine) as tts:
        tts.say("birinci", priority=Priority.NORMAL)
        assert _wait(lambda: engine._speaking.is_set())  # ilk metin okunuyor
        tts.say("arka plan", priority=Priority.BACKGROUND)
        tts.say("kritik", priority=Priority.CRITICAL)
        engine.block.set()
        assert _wait(lambda: len(engine.spoken) >= 3, timeout=5.0)

    assert engine.spoken[0] == "birinci"
    assert engine.spoken[1] == "kritik"
    assert engine.spoken[2] == "arka plan"


def test_critical_interrupts_current_speech() -> None:
    engine = RecordingEngine(hold=True)
    with TextToSpeech(engine=engine) as tts:
        tts.say("uzun açıklama", priority=Priority.NORMAL)
        assert _wait(lambda: engine._speaking.is_set())
        tts.say_critical("Dikkat, araba çok yakın.")
        assert engine.stop_calls >= 1

        engine.block.set()
        assert _wait(lambda: any("Dikkat" in s for s in engine.spoken), timeout=5.0)


def test_stop_clears_queue_and_calls_engine_stop() -> None:
    engine = RecordingEngine()
    with TextToSpeech(engine=engine) as tts:
        tts.say("bir")
        tts.say("iki")
        tts.stop()
        assert engine.stop_calls == 1
        assert tts.is_busy is False


def test_toggle_enabled() -> None:
    engine = RecordingEngine()
    with TextToSpeech(engine=engine) as tts:
        assert tts.enabled is True
        assert tts.toggle() is False  # artık kapalı
        assert tts.enabled is False
        assert tts.say("sessiz") is False
        assert tts.toggle() is True  # yeniden açık
        assert tts.enabled is True


def test_disabled_when_engine_missing() -> None:
    tts = TextToSpeech(engine=None)
    tts.start()
    assert tts.enabled is False
    assert tts.say("bir şey") is False
    tts.close()


def test_engine_error_does_not_kill_worker() -> None:
    class BrokenEngine(RecordingEngine):
        def speak(self, text: str) -> None:
            raise RuntimeError("ses aygıtı koptu")

    engine = BrokenEngine()
    with TextToSpeech(engine=engine) as tts:
        tts.say("birinci")
        time.sleep(0.1)
        assert tts.is_running is True
        tts.say("ikinci")
        time.sleep(0.1)
        assert tts.is_running is True


def test_split_sentences() -> None:
    assert split_sentences("Bir. İki! Üç?") == ["Bir.", "İki!", "Üç?"]
    assert split_sentences("") == []

    long_text = "kelime " * 60
    chunks = split_sentences(long_text, max_chars=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_to_speech_text_cleans_markdown_and_truncates() -> None:
    cleaned = to_speech_text("## Başlık\n- madde bir\n- madde iki")
    assert "#" not in cleaned
    assert "- " not in cleaned

    long_text = "Cümle. " * 200
    assert len(to_speech_text(long_text, max_chars=100)) <= 101


def test_worker_thread_stops_cleanly() -> None:
    engine = RecordingEngine()
    tts = TextToSpeech(engine=engine)
    tts.start()
    assert tts.is_running is True
    tts.close()
    assert tts.is_running is False
    tts.close()  # idempotent
