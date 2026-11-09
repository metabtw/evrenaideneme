"""Yapılandırma yükleme ve doğrulama testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.config import ConfigurationError, Settings, get_settings, reset_settings_cache


@pytest.fixture
def empty_env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    return path


def test_defaults_are_safe(empty_env: Path) -> None:
    settings = Settings.load(env_file=empty_env, environ={})

    assert settings.camera.index == 0
    assert settings.camera.width == 1280
    assert settings.camera.height == 720
    assert settings.camera.target_fps == 30
    assert settings.detection.model_path == "yolov8n.pt"
    assert settings.detection.confidence == pytest.approx(0.35)
    assert settings.detection.classes == ()
    assert settings.vlm.provider == "ollama"
    assert settings.vlm.host == "http://localhost:11434"
    assert settings.tts.engine == "pyttsx3-proc"
    assert settings.ui.trigger_key == "s"
    assert settings.log.level == "INFO"
    assert settings.loaded_env_file == empty_env


def test_env_overrides(empty_env: Path) -> None:
    settings = Settings.load(
        env_file=empty_env,
        environ={
            "APP_CAMERA_INDEX": "1",
            "APP_FRAME_WIDTH": "640",
            "APP_TARGET_FPS": "60",
            "APP_CONFIDENCE": "0,5",  # virgüllü ondalık da kabul edilir
            "APP_CLASSES": "0, 2,16",
            "APP_SHOW_FPS": "false",
            "APP_TRIGGER_KEY": "S",
            "TTS_ENGINE": "none",
            "LOG_LEVEL": "debug",
        },
    )

    assert settings.camera.index == 1
    assert settings.camera.width == 640
    assert settings.camera.target_fps == 60
    assert settings.detection.confidence == pytest.approx(0.5)
    assert settings.detection.classes == (0, 2, 16)
    assert settings.detection.classes_or_none == [0, 2, 16]
    assert settings.ui.show_fps is False
    assert settings.ui.trigger_key == "s"
    assert settings.tts.engine == "none"
    assert settings.log.level == "DEBUG"


def test_dotenv_file_is_loaded_and_environ_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APP_CAMERA_INDEX=2\nAPP_FRAME_WIDTH=800\nOLLAMA_MODEL=qwen2.5vl:7b\n",
        encoding="utf-8",
    )

    settings = Settings.load(env_file=env_file, environ={"APP_FRAME_WIDTH": "1024"})

    assert settings.camera.index == 2
    assert settings.camera.width == 1024
    assert settings.vlm.model == "qwen2.5vl:7b"


def test_missing_env_file_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="bulunamadı"):
        Settings.load(env_file=tmp_path / "yok.env", environ={})


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("APP_TARGET_FPS", "0", "APP_TARGET_FPS"),
        ("APP_TARGET_FPS", "hızlı", "APP_TARGET_FPS"),
        ("APP_CONFIDENCE", "1.5", "APP_CONFIDENCE"),
        ("APP_CONFIDENCE", "abc", "APP_CONFIDENCE"),
        ("APP_FRAME_WIDTH", "10", "APP_FRAME_WIDTH"),
        ("TTS_ENGINE", "magic", "TTS_ENGINE"),
        ("APP_SHOW_FPS", "belki", "APP_SHOW_FPS"),
        ("APP_TRIGGER_KEY", "abc", "APP_TRIGGER_KEY"),
        ("APP_CLASSES", "0,kedi", "APP_CLASSES"),
    ],
)
def test_invalid_values_raise_with_variable_name(
    empty_env: Path, key: str, value: str, expected: str
) -> None:
    with pytest.raises(ConfigurationError, match=expected):
        Settings.load(env_file=empty_env, environ={key: value})


def test_api_key_is_masked_in_repr_and_redacted(empty_env: Path) -> None:
    settings = Settings.load(
        env_file=empty_env,
        environ={"VLM_API_KEY": "cok-gizli-anahtar", "VLM_PROVIDER": "openai-compatible"},
    )

    assert "cok-gizli-anahtar" not in repr(settings.vlm)
    assert settings.vlm.redacted()["api_key"] == "***"
    assert settings.vlm.is_cloud is True


def test_device_helpers(empty_env: Path) -> None:
    auto = Settings.load(env_file=empty_env, environ={}).detection
    assert auto.device_or_none is None

    cpu = Settings.load(env_file=empty_env, environ={"APP_DEVICE": "cpu"}).detection
    assert cpu.device_or_none == "cpu"


def test_get_settings_is_cached() -> None:
    reset_settings_cache()
    first = get_settings()
    second = get_settings()
    assert first is second

    assert get_settings(reload=True) is not first
    reset_settings_cache()
