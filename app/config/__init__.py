"""Uygulama yapılandırması.

Tüm ayarlar `.env` / ortam değişkenlerinden okunur ve tip güvenli dataclass'lara
dönüştürülür::

    from app.config import get_settings

    settings = get_settings()
    print(settings.camera.width, settings.detection.confidence)

`.env` yoksa uygulama çökmez; güvenli varsayılanlar kullanılır.
"""

from __future__ import annotations

from .settings import (
    PROJECT_ROOT,
    CameraSettings,
    ConfigurationError,
    DetectionSettings,
    LogSettings,
    Settings,
    TTSSettings,
    UISettings,
    VLMSettings,
    get_settings,
    reset_settings_cache,
)

__all__ = [
    "PROJECT_ROOT",
    "CameraSettings",
    "ConfigurationError",
    "DetectionSettings",
    "LogSettings",
    "Settings",
    "TTSSettings",
    "UISettings",
    "VLMSettings",
    "get_settings",
    "reset_settings_cache",
]
