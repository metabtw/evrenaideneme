"""Vision LLM istemcileri (M2).

İki sağlayıcı, tek arayüz (:class:`VLMClient`):

* :class:`OllamaClient` — yerel, tamamen çevrimdışı (varsayılan).
* :class:`OpenAICompatibleClient` — bulut veya uzak sunucu (opsiyonel).

Tasarım kuralları:
    * HTTP oturumu enjekte edilebilir (testlerde sahte sunucu/oturum).
    * Yalnızca ``describe(frame_jpeg, prompt)`` bilinir; çağıran taraf hangi
      sağlayıcının kullanıldığını bilmek zorunda değildir.
    * Hatalar Türkçe ve eyleme dönüktür: "ollama pull llava:7b" gibi.
    * Ana döngüyü bloklamamak çağıranın sorumluluğudur (bkz. ``analysis/worker``).
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from app.config import VLMSettings
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Geçici hatalarda deneme sayısı (ilk istek + tekrar).
DEFAULT_RETRIES = 2
#: Üstel geri çekilme tabanı (saniye): 0.5, 1.0 …
BACKOFF_BASE = 0.5


class VLMError(RuntimeError):
    """Vision LLM isteği başarısız."""


class VLMUnavailableError(VLMError):
    """Sağlayıcıya ulaşılamıyor (servis kapalı, ağ yok, model eksik)."""


class VLMTimeoutError(VLMUnavailableError):
    """İstek zaman aşımına uğradı."""


class VLMResponseError(VLMError):
    """Yanıt beklenen biçimde değil veya boş."""


@runtime_checkable
class VLMClient(Protocol):
    """Tüm sağlayıcıların uygulaması gereken arayüz."""

    @property
    def name(self) -> str:
        """Görünen sağlayıcı/model adı (log ve HUD için)."""
        ...

    def describe(self, frame_jpeg: bytes, prompt: str) -> str:
        """JPEG kare + Türkçe prompt -> Türkçe yanıt."""
        ...

    def close(self) -> None:
        """Kaynakları serbest bırakır (idempotent)."""
        ...


def _requests() -> Any:
    try:
        import requests  # noqa: PLC0415 - ağır/isteğe bağlı bağımlılık
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise VLMUnavailableError(
            "requests paketi bulunamadı. Kurulum: pip install -r requirements.txt"
        ) from exc
    return requests


def _b64(frame_jpeg: bytes) -> str:
    """Baytları base64 metne çevirir (zaten base64 ise dokunmaz)."""
    if not frame_jpeg:
        raise VLMResponseError("Görüntü verisi boş; kare kodlanamamış olabilir.")
    try:
        return frame_jpeg.decode("ascii")
    except UnicodeDecodeError:
        return base64.b64encode(frame_jpeg).decode("ascii")


class _HttpVLMClient:
    """Ortak HTTP davranışı: oturum, zaman aşımı, yeniden deneme, hata eşleme."""

    def __init__(
        self,
        settings: VLMSettings,
        *,
        session: Any | None = None,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._session = session
        self._retries = max(0, retries)
        self._sleep = sleep

    # -- alt sınıflar uygular ---------------------------------------------

    def _request_body(self, prompt: str, image_b64: str) -> dict[str, Any]:
        raise NotImplementedError

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _parse(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    # -- ortak -------------------------------------------------------------

    @property
    def session(self) -> Any:
        if self._session is None:
            requests = _requests()
            self._session = requests.Session()
        return self._session

    def describe(self, frame_jpeg: bytes, prompt: str) -> str:
        if not prompt.strip():
            raise VLMResponseError("Prompt boş olamaz.")

        body = self._request_body(prompt, _b64(frame_jpeg))
        payload = self._post(body)
        answer = self._parse(payload).strip()
        if not answer:
            raise VLMResponseError(
                "Model boş yanıt döndürdü. Görüntü çok küçük/karanlık olabilir; tekrar deneyin."
            )
        return answer

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        requests = _requests()
        endpoint = self._endpoint()
        last_error: Exception | None = None

        for attempt in range(self._retries + 1):
            try:
                response = self.session.post(
                    endpoint,
                    data=json.dumps(body),
                    headers=self._headers(),
                    timeout=self.settings.timeout_seconds,
                )
            except requests.Timeout as exc:
                last_error = VLMTimeoutError(
                    f"{self.name}: istek {self.settings.timeout_seconds:.0f} sn içinde "
                    "yanıtlanmadı. Daha küçük bir model deneyin veya "
                    "VLM_TIMEOUT_SECONDS değerini artırın."
                )
                last_error.__cause__ = exc
            except requests.RequestException as exc:
                last_error = VLMUnavailableError(
                    f"{self.name}: sunucuya ulaşılamadı ({endpoint}). "
                    f"Sağlayıcının çalıştığını kontrol edin. Ayrıntı: {exc}"
                )
                last_error.__cause__ = exc
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise VLMResponseError(
                            f"{self.name}: yanıt JSON değil. Ayrıntı: {exc}"
                        ) from exc

                if response.status_code < 500:
                    raise self._client_error(response)

                last_error = VLMUnavailableError(
                    f"{self.name}: sunucu hatası (HTTP {response.status_code}). "
                    "Sağlayıcı günlüklerine bakın."
                )

            if attempt < self._retries:
                delay = BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "%s başarısız (%s/%s), %.1f sn sonra tekrar: %s",
                    self.name,
                    attempt + 1,
                    self._retries + 1,
                    delay,
                    last_error,
                )
                self._sleep(delay)

        assert last_error is not None
        raise last_error

    def _client_error(self, response: Any) -> VLMError:
        detail = _short_body(response)
        return VLMError(f"{self.name}: istek reddedildi (HTTP {response.status_code}). {detail}")

    def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - kapanış hatası önemsiz
                logger.debug("HTTP oturumu kapatılırken hata", exc_info=True)


class OllamaClient(_HttpVLMClient):
    """Yerel Ollama sunucusu (``POST /api/generate``)."""

    def __init__(
        self,
        settings: VLMSettings | None = None,
        *,
        session: Any | None = None,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(settings or VLMSettings(), session=session, retries=retries, sleep=sleep)

    @property
    def name(self) -> str:
        return f"ollama:{self.settings.model}"

    def _endpoint(self) -> str:
        return f"{self.settings.host}/api/generate"

    def _request_body(self, prompt: str, image_b64: str) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }

    def _parse(self, payload: dict[str, Any]) -> str:
        answer = payload.get("response")
        if not isinstance(answer, str):
            raise VLMResponseError(
                "Ollama yanıtında 'response' alanı yok. Model görsel destekli mi? "
                f"Öneri: ollama pull {self.settings.model}"
            )
        return answer

    def _client_error(self, response: Any) -> VLMError:
        if response.status_code == 404:
            return VLMUnavailableError(
                f"Ollama modeli bulunamadı: {self.settings.model}. "
                f"Kurulum: ollama pull {self.settings.model}"
            )
        return VLMError(
            f"{self.name}: istek reddedildi (HTTP {response.status_code}). {_short_body(response)}"
        )

    # -- sağlık kontrolü ---------------------------------------------------

    def available_models(self) -> list[str]:
        """Sunucudaki model adlarını döndürür; sunucu kapalıysa anlaşılır hata."""
        requests = _requests()
        try:
            response = self.session.get(
                f"{self.settings.host}/api/tags", timeout=min(10.0, self.settings.timeout_seconds)
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise VLMUnavailableError(
                f"Ollama'ya ulaşılamadı ({self.settings.host}). "
                "Kurulum: https://ollama.com/download — ardından `ollama serve`."
            ) from exc
        except ValueError as exc:
            raise VLMResponseError(
                f"Ollama model listesi okunamadı ({self.settings.host})."
            ) from exc

        models = payload.get("models") or []
        return [str(item.get("name", "")) for item in models if item.get("name")]

    def health_check(self) -> tuple[bool, str]:
        """``(hazır_mı, mesaj)`` döndürür; uygulama başlangıcında kullanılır."""
        try:
            models = self.available_models()
        except VLMError as exc:
            return False, str(exc)

        wanted = self.settings.model
        if _model_matches(wanted, models):
            return True, f"{wanted} hazır ({len(models)} model kurulu)."
        return False, (
            f"'{wanted}' modeli kurulu değil. Kurulum: ollama pull {wanted}"
            + (f" — kurulu modeller: {', '.join(models)}" if models else "")
        )


class OpenAICompatibleClient(_HttpVLMClient):
    """OpenAI uyumlu ``/chat/completions`` (bulut veya uzak sunucu)."""

    def __init__(
        self,
        settings: VLMSettings | None = None,
        *,
        session: Any | None = None,
        retries: int = DEFAULT_RETRIES,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(settings or VLMSettings(), session=session, retries=retries, sleep=sleep)
        if not self.settings.api_base:
            raise VLMUnavailableError(
                "VLM_PROVIDER=openai-compatible için VLM_API_BASE zorunlu "
                "(örn. https://api.openai.com/v1)."
            )
        if not self.settings.api_key:
            raise VLMUnavailableError(
                "VLM_PROVIDER=openai-compatible için VLM_API_KEY zorunlu. "
                "Anahtarı yalnızca yerel .env dosyanıza yazın."
            )

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def _model(self) -> str:
        return self.settings.api_model or self.settings.model

    def _endpoint(self) -> str:
        return f"{self.settings.api_base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.api_key}",
        }

    def _request_body(self, prompt: str, image_b64: str) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "stream": False,
        }

    def _parse(self, payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMResponseError(
                "Sağlayıcı yanıtı beklenen biçimde değil (choices[0].message.content)."
            ) from exc

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            )
        raise VLMResponseError("Model yanıtı metin değil.")


class UnavailableVLMClient:
    """Yapılandırma geçersizken kullanılan güvenli yer tutucu."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    @property
    def name(self) -> str:
        return "vlm:yok"

    def describe(self, frame_jpeg: bytes, prompt: str) -> str:
        raise VLMUnavailableError(self.reason)

    def close(self) -> None:
        return None


def create_vlm_client(settings: VLMSettings | None = None, **kwargs: Any) -> VLMClient:
    """Ayardaki sağlayıcıya göre istemci oluşturur.

    Yapılandırma eksikse uygulamayı düşürmek yerine her istekte anlaşılır hata
    veren bir yer tutucu döner; böylece M1 canlı tespit çalışmaya devam eder.
    """
    settings = settings or VLMSettings()

    try:
        if settings.provider == "ollama":
            return OllamaClient(settings, **kwargs)
        if settings.provider == "openai-compatible":
            return OpenAICompatibleClient(settings, **kwargs)
        return UnavailableVLMClient(f"Bilinmeyen VLM sağlayıcısı: {settings.provider!r}")
    except VLMError as exc:
        logger.warning("VLM istemcisi oluşturulamadı: %s", exc)
        return UnavailableVLMClient(str(exc))


def _model_matches(wanted: str, available: list[str]) -> bool:
    """``llava`` -> ``llava:7b`` gibi etiketsiz istekleri de eşleştirir."""
    if wanted in available:
        return True
    if ":" in wanted:
        return False
    return any(name.split(":")[0] == wanted for name in available)


def _short_body(response: Any, limit: int = 200) -> str:
    try:
        text = response.text or ""
    except Exception:  # noqa: BLE001 - gövde okunamıyorsa ayrıntı yok
        return ""
    text = " ".join(text.split())
    return text[:limit]
