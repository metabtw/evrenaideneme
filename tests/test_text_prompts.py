"""Metin, etiket ve prompt yardımcıları."""

from __future__ import annotations

import pytest
from app.analysis import AnalysisMode, VLMUnavailableError, build_prompt, create_vlm_client
from app.detection.labels import ascii_label, turkish_label
from app.tts import (
    NullSpeechEngine,
    SpeechUnavailableError,
    create_speech_engine,
)
from app.utils.text import to_ascii, wrap_text


def test_to_ascii_replaces_turkish_characters() -> None:
    assert to_ascii("İZLENİYOR") == "IZLENIYOR"
    assert to_ascii("çğıöşü ÇĞİÖŞÜ") == "cgiosu CGIOSU"


def test_wrap_text_respects_max_chars() -> None:
    lines = wrap_text("kısa bir cümle buraya", max_chars=10)
    assert all(len(line) <= 10 for line in lines)
    assert " ".join(lines) == "kısa bir cümle buraya"
    assert wrap_text("", 10) == [""]


def test_labels_have_turkish_and_ascii_variants() -> None:
    assert turkish_label(0) == "kişi"
    assert turkish_label(2) == "araba"
    assert turkish_label(79) == "diş fırçası"
    assert ascii_label(0) == "kisi"
    assert turkish_label(999) == "sinif-999"
    assert turkish_label(999, "nesne") == "nesne"


def test_build_prompt_modes() -> None:
    describe = build_prompt(AnalysisMode.DESCRIBE, model="llava:7b")
    assert "Türkçe" in describe
    assert "3 cümle" in describe

    ocr = build_prompt("ocr")
    assert "yazıyı" in ocr

    count = build_prompt("count", target="bardak")
    assert "bardak" in count
    assert "Sayı:" in count


def test_build_prompt_validation() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen analiz modu"):
        build_prompt("video")
    with pytest.raises(ValueError, match="target"):
        build_prompt(AnalysisMode.COUNT)


def test_vlm_client_is_explicitly_unavailable() -> None:
    client = create_vlm_client()
    assert "hazır değil" in client.name
    with pytest.raises(VLMUnavailableError, match="M2"):
        client.describe(b"jpeg", "açıkla")
    client.close()


def test_null_speech_engine() -> None:
    from app.config import TTSSettings

    silent = create_speech_engine(TTSSettings(engine="none"))
    assert isinstance(silent, NullSpeechEngine)
    silent.speak("merhaba")
    assert silent.spoken == ["merhaba"]
    assert silent.is_busy() is False
    silent.stop()
    silent.close()


def test_real_tts_engines_are_explicitly_pending() -> None:
    from app.config import TTSSettings

    with pytest.raises(SpeechUnavailableError, match="M3"):
        create_speech_engine(TTSSettings(engine="pyttsx3"))
    with pytest.raises(SpeechUnavailableError, match="M3"):
        create_speech_engine(TTSSettings(engine="gtts"))
