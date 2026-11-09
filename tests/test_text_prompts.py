"""Metin, etiket ve prompt yardımcıları."""

from __future__ import annotations

import pytest
from app.analysis import AnalysisMode, VLMUnavailableError, build_prompt, create_vlm_client
from app.detection.labels import ascii_label, turkish_label
from app.tts import (
    NullSpeechEngine,
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


def test_create_vlm_client_picks_provider() -> None:
    from app.config import VLMSettings

    ollama = create_vlm_client(VLMSettings(provider="ollama", model="llava:7b"))
    assert ollama.name == "ollama:llava:7b"

    # Bulut sağlayıcı anahtarsız oluşturulamaz: uygulama düşmez, yer tutucu döner.
    placeholder = create_vlm_client(VLMSettings(provider="openai-compatible"))
    assert placeholder.name == "vlm:yok"
    with pytest.raises(VLMUnavailableError, match="VLM_API_BASE"):
        placeholder.describe(b"jpeg", "açıkla")
    placeholder.close()


def test_null_speech_engine() -> None:
    from app.config import TTSSettings

    silent = create_speech_engine(TTSSettings(engine="none"))
    assert isinstance(silent, NullSpeechEngine)
    silent.speak("merhaba")
    assert silent.spoken == ["merhaba"]
    assert silent.is_busy() is False
    silent.stop()
    silent.close()


def test_speech_engine_creation_never_raises() -> None:
    """Motor kurulamasa bile uygulama düşmez: sessiz motora ya da gerçek motora düşer."""
    from app.config import TTSSettings

    for engine_name in ("none", "pyttsx3", "gtts"):
        engine = create_speech_engine(TTSSettings(engine=engine_name))
        assert hasattr(engine, "speak")
        assert engine.name
