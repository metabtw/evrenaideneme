"""Vision LLM istemci sözleşmesi (M2'de doldurulacak).

Bu dosya bilinçli olarak yalnızca arayüz + hata tiplerini içerir. Gerçek
Ollama / OpenAI uyumlu istemciler M2 fazında eklenecek; üst katmanlar bugünden
bu sözleşmeye göre yazılabilir.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.config import VLMSettings


class VLMError(RuntimeError):
    """Vision LLM isteği başarısız."""


class VLMUnavailableError(VLMError):
    """İstemci henüz uygulanmadı veya model/servis erişilemiyor."""


@runtime_checkable
class VLMClient(Protocol):
    """Tüm sağlayıcıların uygulaması gereken arayüz."""

    @property
    def name(self) -> str:
        """Görünen sağlayıcı/model adı (log ve HUD için)."""
        ...

    def describe(self, frame_jpeg: bytes, prompt: str) -> str:
        """JPEG kare + Türkçe prompt -> Türkçe yanıt.

        Raises:
            VLMUnavailableError: Sağlayıcıya ulaşılamıyor.
            VLMError: İstek başarısız veya yanıt boş.
        """
        ...

    def close(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        ...


class UnavailableVLMClient:
    """M2 öncesi yer tutucu: her istekte anlaşılır hata verir."""

    def __init__(self, settings: VLMSettings | None = None) -> None:
        self.settings = settings or VLMSettings()

    @property
    def name(self) -> str:
        return f"{self.settings.provider}:{self.settings.model} (hazır değil)"

    def describe(self, frame_jpeg: bytes, prompt: str) -> str:
        raise VLMUnavailableError(
            "Vision LLM istemcisi henüz uygulanmadı (M2 fazı). "
            "Bu sürüm yalnızca canlı nesne tespiti yapar."
        )

    def close(self) -> None:
        return None


def create_vlm_client(settings: VLMSettings | None = None) -> VLMClient:
    """Sağlayıcıya göre istemci döndürür.

    M2'de ``OllamaClient`` / ``OpenAICompatibleClient`` ile değiştirilecek.
    """
    return UnavailableVLMClient(settings)
