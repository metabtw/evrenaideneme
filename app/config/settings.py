"""`.env` -> dataclass yapılandırma yükleyicisi.

Tasarım kuralları:
    * Her ayar tek bir yerde tanımlıdır ve ``.env.example`` ile birebir eşleşir.
    * Geçersiz değer sessizce yutulmaz; hangi değişkenin bozuk olduğunu söyleyen
      Türkçe bir :class:`ConfigurationError` fırlatılır.
    * Sırlar (API anahtarı) ``repr`` çıktısında maskelenir.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

# app/config/settings.py -> app/config -> app -> proje kökü
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(ValueError):
    """Geçersiz veya eksik yapılandırma değeri."""


# ---------------------------------------------------------------------------
# Ortam değişkeni okuyucuları (doğrulama + anlaşılır hata)
# ---------------------------------------------------------------------------

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "evet", "e"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", "hayır", "hayir", "h"})


def _raw(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _str(env: Mapping[str, str], key: str, default: str) -> str:
    return _raw(env, key) or default


def _int(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = _raw(env, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigurationError(
            f"{key} tam sayı olmalı (örn. {default}); alınan değer: {raw!r}"
        ) from None
    return _check_bounds(key, value, minimum, maximum)


def _float(
    env: Mapping[str, str],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _raw(env, key)
    if raw is None:
        return default
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        raise ConfigurationError(
            f"{key} ondalık sayı olmalı (örn. {default}); alınan değer: {raw!r}"
        ) from None
    return _check_bounds(key, value, minimum, maximum)


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _raw(env, key)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{key} true/false (veya 1/0) olmalı; alınan değer: {raw!r}")


def _choice(
    env: Mapping[str, str],
    key: str,
    default: str,
    choices: frozenset[str],
) -> str:
    """Değeri (büyük/küçük harf duyarsız) doğrular ve kanonik biçimini döndürür."""
    value = _str(env, key, default).lower()
    for choice in choices:
        if choice.lower() == value:
            return choice
    joined = ", ".join(sorted(choices))
    raise ConfigurationError(f"{key} şu değerlerden biri olmalı: {joined}; alınan değer: {value!r}")


def _int_tuple(env: Mapping[str, str], key: str) -> tuple[int, ...]:
    raw = _raw(env, key)
    if raw is None:
        return ()
    try:
        return tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        raise ConfigurationError(
            f"{key} virgülle ayrılmış sınıf kimlikleri olmalı (örn. 0,2,16); alınan değer: {raw!r}"
        ) from None


def _check_bounds(
    key: str,
    value: float,
    minimum: float | None,
    maximum: float | None,
) -> Any:
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{key} en az {minimum} olmalı; alınan değer: {value}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{key} en fazla {maximum} olmalı; alınan değer: {value}")
    return value


# ---------------------------------------------------------------------------
# Ayar grupları
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraSettings:
    """`APP_*` kamera ayarları."""

    index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    backend: str = "auto"
    reconnect_attempts: int = 3
    reconnect_delay: float = 0.5

    def __post_init__(self) -> None:
        if self.width < 160 or self.height < 120:
            raise ConfigurationError(
                "APP_FRAME_WIDTH / APP_FRAME_HEIGHT çok küçük "
                f"({self.width}x{self.height}); en az 160x120 olmalı"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> CameraSettings:
        return cls(
            index=_int(env, "APP_CAMERA_INDEX", 0, minimum=0, maximum=32),
            width=_int(env, "APP_FRAME_WIDTH", 1280, minimum=160, maximum=7680),
            height=_int(env, "APP_FRAME_HEIGHT", 720, minimum=120, maximum=4320),
            target_fps=_int(env, "APP_TARGET_FPS", 30, minimum=1, maximum=240),
            backend=_choice(
                env,
                "APP_BACKEND",
                "auto",
                frozenset({"auto", "dshow", "msmf", "avfoundation", "v4l2"}),
            ),
            reconnect_attempts=_int(env, "APP_RECONNECT_ATTEMPTS", 3, minimum=0, maximum=20),
            reconnect_delay=_float(env, "APP_RECONNECT_DELAY", 0.5, minimum=0.0, maximum=10.0),
        )


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    """`APP_*` YOLOv8 tespit ayarları."""

    model_path: str = "yolov8n.pt"
    confidence: float = 0.35
    iou: float = 0.45
    image_size: int = 640
    device: str = "auto"
    max_detections: int = 100
    classes: tuple[int, ...] = ()

    @property
    def device_or_none(self) -> str | None:
        """Ultralytics'e verilecek cihaz; ``auto`` ise ``None`` (otomatik seçim)."""
        return None if self.device == "auto" else self.device

    @property
    def classes_or_none(self) -> list[int] | None:
        return list(self.classes) if self.classes else None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> DetectionSettings:
        return cls(
            model_path=_str(env, "APP_MODEL_PATH", "yolov8n.pt"),
            confidence=_float(env, "APP_CONFIDENCE", 0.35, minimum=0.0, maximum=1.0),
            iou=_float(env, "APP_IOU", 0.45, minimum=0.0, maximum=1.0),
            image_size=_int(env, "APP_IMG_SIZE", 640, minimum=32, maximum=4096),
            device=_str(env, "APP_DEVICE", "auto").lower(),
            max_detections=_int(env, "APP_MAX_DETECTIONS", 100, minimum=1, maximum=1000),
            classes=_int_tuple(env, "APP_CLASSES"),
        )


@dataclass(frozen=True, slots=True)
class VLMSettings:
    """Vision LLM (Ollama / OpenAI uyumlu) ayarları."""

    provider: str = "ollama"
    host: str = "http://localhost:11434"
    model: str = "llava:7b"
    timeout_seconds: float = 120.0
    max_tokens: int = 256
    temperature: float = 0.2
    api_base: str = ""
    api_key: str = field(default="", repr=False)
    api_model: str = ""

    @property
    def is_cloud(self) -> bool:
        return self.provider == "openai-compatible"

    def redacted(self) -> dict[str, Any]:
        """Log'lanabilir kopya; API anahtarı maskelenir."""
        data = {
            "provider": self.provider,
            "host": self.host,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_base": self.api_base,
            "api_key": "***" if self.api_key else "",
            "api_model": self.api_model,
        }
        return data

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> VLMSettings:
        provider = _choice(
            env,
            "VLM_PROVIDER",
            "ollama",
            frozenset({"ollama", "openai-compatible"}),
        )
        return cls(
            provider=provider,
            host=_str(env, "OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            model=_str(env, "OLLAMA_MODEL", "llava:7b"),
            timeout_seconds=_float(env, "VLM_TIMEOUT_SECONDS", 120.0, minimum=1.0, maximum=3600.0),
            max_tokens=_int(env, "VLM_MAX_TOKENS", 256, minimum=1, maximum=8192),
            temperature=_float(env, "VLM_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0),
            api_base=_str(env, "VLM_API_BASE", "").rstrip("/"),
            api_key=_str(env, "VLM_API_KEY", ""),
            api_model=_str(env, "VLM_MODEL", ""),
        )


@dataclass(frozen=True, slots=True)
class TTSSettings:
    """Sesli geri bildirim ayarları (M3)."""

    engine: str = "pyttsx3"
    language: str = "tr"
    rate: int = 180
    volume: float = 1.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> TTSSettings:
        return cls(
            engine=_choice(env, "TTS_ENGINE", "pyttsx3", frozenset({"pyttsx3", "gtts", "none"})),
            language=_str(env, "TTS_LANGUAGE", "tr"),
            rate=_int(env, "TTS_RATE", 180, minimum=50, maximum=400),
            volume=_float(env, "TTS_VOLUME", 1.0, minimum=0.0, maximum=1.0),
        )


@dataclass(frozen=True, slots=True)
class UISettings:
    """OpenCV penceresi ve HUD ayarları."""

    window_name: str = "EvrenAid"
    show_fps: bool = True
    trigger_key: str = "s"
    hud_enabled: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> UISettings:
        trigger_key = _str(env, "APP_TRIGGER_KEY", "s").lower()
        if len(trigger_key) != 1:
            raise ConfigurationError(
                f"APP_TRIGGER_KEY tek bir karakter olmalı; alınan değer: {trigger_key!r}"
            )
        return cls(
            window_name=_str(env, "APP_WINDOW_NAME", "EvrenAid"),
            show_fps=_bool(env, "APP_SHOW_FPS", True),
            trigger_key=trigger_key,
            hud_enabled=_bool(env, "APP_HUD_ENABLED", True),
        )


@dataclass(frozen=True, slots=True)
class LogSettings:
    """Loglama ayarları."""

    level: str = "INFO"
    file: str = "logs/app.log"
    console: bool = True

    @property
    def path(self) -> Path:
        path = Path(self.file)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> LogSettings:
        return cls(
            level=_choice(
                env,
                "LOG_LEVEL",
                "INFO",
                frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}),
            ),
            file=_str(env, "LOG_FILE", "logs/app.log"),
            console=_bool(env, "LOG_CONSOLE", True),
        )


# ---------------------------------------------------------------------------
# Kök ayar nesnesi
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Settings:
    """Tüm uygulama ayarlarının değişmez (immutable) anlık görüntüsü."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    vlm: VLMSettings = field(default_factory=VLMSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    ui: UISettings = field(default_factory=UISettings)
    log: LogSettings = field(default_factory=LogSettings)
    loaded_env_file: Path | None = None

    @classmethod
    def load(
        cls,
        env_file: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        """Ayarları yükler.

        Args:
            env_file: Açıkça verilen `.env` yolu. Verilmezse önce çalışma
                dizinindeki, sonra proje kökündeki `.env` aranır.
            environ: Testler için ortam değişkeni haritası. Verilmezse
                ``os.environ`` kullanılır. Verilen harita `.env` değerlerini
                geçersiz kılar (gerçek ortam davranışıyla aynı).
        """
        file_values, loaded_path = _read_env_file(env_file)
        base_env = dict(os.environ if environ is None else environ)
        merged: dict[str, str] = {**file_values, **base_env}

        return cls(
            camera=CameraSettings.from_env(merged),
            detection=DetectionSettings.from_env(merged),
            vlm=VLMSettings.from_env(merged),
            tts=TTSSettings.from_env(merged),
            ui=UISettings.from_env(merged),
            log=LogSettings.from_env(merged),
            loaded_env_file=loaded_path,
        )


def _read_env_file(
    env_file: str | Path | None,
) -> tuple[dict[str, str], Path | None]:
    if env_file is not None:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise ConfigurationError(f"Belirtilen .env dosyası bulunamadı: {path}")
    else:
        candidates = (Path.cwd() / ".env", PROJECT_ROOT / ".env")
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return {}, None

    values = {key: value for key, value in dotenv_values(path).items() if value is not None}
    return values, path


_cached_settings: Settings | None = None


def get_settings(*, reload: bool = False) -> Settings:
    """Süreç genelinde tek bir :class:`Settings` örneği döndürür."""
    global _cached_settings
    if _cached_settings is None or reload:
        _cached_settings = Settings.load()
    return _cached_settings


def reset_settings_cache() -> None:
    """Önbelleği temizler (testler ve ``--reload`` için)."""
    global _cached_settings
    _cached_settings = None
