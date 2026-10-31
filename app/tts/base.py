"""Sesli geri bildirim motoru sözleşmesi (M3).

Arayüz bilinçli olarak küçük tutulur: yeni bir motor eklemek tek dosya işidir.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class SpeechError(RuntimeError):
    """Seslendirme başarısız."""


class SpeechUnavailableError(SpeechError):
    """Motor kullanılamıyor (eksik paket, eksik ses aygıtı, henüz uygulanmadı)."""


@runtime_checkable
class SpeechEngine(Protocol):
    """Tüm TTS motorlarının uygulaması gereken arayüz."""

    @property
    def name(self) -> str: ...

    def speak(self, text: str) -> None:
        """Metni seslendirir. Çağrı bloklayabilir; işçi thread'inden çağrılmalı."""
        ...

    def stop(self) -> None:
        """Sürmekte olan seslendirmeyi mümkün olan en kısa sürede keser."""
        ...

    def is_busy(self) -> bool: ...

    def close(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        ...


def normalize_text(text: str) -> str:
    """TTS öncesi hafif metin temizliği (M3'te genişletilecek)."""
    return " ".join(text.split())
