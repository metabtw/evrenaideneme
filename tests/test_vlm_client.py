"""VLM istemci testleri: sahte HTTP oturumu ile ağa çıkmadan sözleşme doğrulaması."""

from __future__ import annotations

import json

import pytest
from app.analysis import (
    OllamaClient,
    OpenAICompatibleClient,
    VLMError,
    VLMResponseError,
    VLMTimeoutError,
    VLMUnavailableError,
)
from app.config import VLMSettings


class FakeResponse:
    def __init__(self, *, status: int = 200, payload: object = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text
        self._raw = json.dumps(payload) if payload is not None else text

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("JSON değil")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Sırayla önceden tanımlı yanıtları (veya istisnaları) döndürür."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, object]] = []
        self.closed = False

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        outcome = self._outcomes.pop(0) if self._outcomes else FakeResponse()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    def get(self, url: str, **_kwargs: object) -> FakeResponse:
        outcome = self._outcomes.pop(0) if self._outcomes else FakeResponse()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    def close(self) -> None:
        self.closed = True


def _requests_module():
    import requests

    return requests


def _settings(**overrides: object) -> VLMSettings:
    base = VLMSettings(provider="ollama", model="llava:7b", timeout_seconds=30)
    return VLMSettings(**{**_as_dict(base), **overrides})


def _as_dict(settings: VLMSettings) -> dict[str, object]:
    return {
        "provider": settings.provider,
        "host": settings.host,
        "model": settings.model,
        "timeout_seconds": settings.timeout_seconds,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "api_base": settings.api_base,
        "api_key": settings.api_key,
        "api_model": settings.api_model,
    }


def _sleeps() -> list[float]:
    calls: list[float] = []
    return calls


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def test_ollama_builds_expected_request() -> None:
    session = FakeSession(FakeResponse(payload={"response": "Masada bir fincan var."}))
    client = OllamaClient(_settings(), session=session, sleep=lambda _s: None)

    answer = client.describe(b"\xff\xd8jpegbytes", "Sahneyi açıkla")

    assert answer == "Masada bir fincan var."
    request = session.requests[0]
    assert request["url"] == "http://localhost:11434/api/generate"
    body = json.loads(str(request["data"]))
    assert body["model"] == "llava:7b"
    assert body["stream"] is False
    assert body["prompt"] == "Sahneyi açıkla"
    assert body["options"] == {"temperature": 0.2, "num_predict": 256}
    # Bayt verisi base64'e çevrilmeli (JSON'a yazılabilir olmalı).
    assert isinstance(body["images"][0], str)
    assert body["images"][0].isascii()


def test_ollama_accepts_already_base64_input() -> None:
    session = FakeSession(FakeResponse(payload={"response": "tamam"}))
    client = OllamaClient(_settings(), session=session, sleep=lambda _s: None)

    client.describe(b"YWJj", "açıkla")

    body = json.loads(str(session.requests[0]["data"]))
    assert body["images"] == ["YWJj"]


def test_missing_model_gives_install_hint() -> None:
    session = FakeSession(FakeResponse(status=404, payload={"error": "model not found"}))
    client = OllamaClient(_settings(), session=session, sleep=lambda _s: None)

    with pytest.raises(VLMUnavailableError, match="ollama pull llava:7b"):
        client.describe(b"x", "açıkla")


def test_connection_error_is_translated() -> None:
    requests = _requests_module()
    session = FakeSession(requests.ConnectionError("bağlantı reddedildi"))
    client = OllamaClient(_settings(), session=session, retries=0, sleep=lambda _s: None)

    with pytest.raises(VLMUnavailableError, match="ulaşılamadı"):
        client.describe(b"x", "açıkla")


def test_timeout_retries_then_succeeds() -> None:
    requests = _requests_module()
    sleeps = _sleeps()
    session = FakeSession(
        requests.Timeout("yavaş"),
        FakeResponse(payload={"response": "ikinci denemede geldi"}),
    )
    client = OllamaClient(_settings(), session=session, retries=1, sleep=sleeps.append)

    assert client.describe(b"x", "açıkla") == "ikinci denemede geldi"
    assert sleeps == [0.5]
    assert len(session.requests) == 2


def test_timeout_exhausted_raises_timeout_error() -> None:
    requests = _requests_module()
    session = FakeSession(requests.Timeout("yavaş"), requests.Timeout("yavaş"))
    client = OllamaClient(_settings(), session=session, retries=1, sleep=lambda _s: None)

    with pytest.raises(VLMTimeoutError, match="30 sn"):
        client.describe(b"x", "açıkla")


def test_server_error_retries_with_backoff() -> None:
    sleeps = _sleeps()
    session = FakeSession(
        FakeResponse(status=500, text="boom"),
        FakeResponse(status=503, text="busy"),
        FakeResponse(payload={"response": "sonunda"}),
    )
    client = OllamaClient(_settings(), session=session, retries=2, sleep=sleeps.append)

    assert client.describe(b"x", "açıkla") == "sonunda"
    assert sleeps == [0.5, 1.0]


def test_empty_response_is_rejected() -> None:
    session = FakeSession(FakeResponse(payload={"response": "   "}))
    client = OllamaClient(_settings(), session=session, sleep=lambda _s: None)

    with pytest.raises(VLMResponseError, match="boş yanıt"):
        client.describe(b"x", "açıkla")


def test_response_without_response_field_mentions_vision_support() -> None:
    session = FakeSession(FakeResponse(payload={"done": True}))
    client = OllamaClient(_settings(), session=session, sleep=lambda _s: None)

    with pytest.raises(VLMResponseError, match="görsel destekli mi"):
        client.describe(b"x", "açıkla")


def test_empty_prompt_and_empty_image_are_rejected() -> None:
    client = OllamaClient(_settings(), session=FakeSession(), sleep=lambda _s: None)

    with pytest.raises(VLMResponseError, match="Prompt boş"):
        client.describe(b"x", "   ")
    with pytest.raises(VLMResponseError, match="Görüntü verisi boş"):
        client.describe(b"", "açıkla")


def test_non_json_success_body_is_reported() -> None:
    session = FakeSession(FakeResponse(status=200, payload=None, text="<html>proxy</html>"))
    client = OllamaClient(_settings(), session=session, sleep=lambda _s: None)

    with pytest.raises(VLMResponseError, match="JSON değil"):
        client.describe(b"x", "açıkla")


def test_close_is_idempotent() -> None:
    session = FakeSession()
    client = OllamaClient(_settings(), session=session)

    client.close()
    client.close()

    assert session.closed is True


# ---------------------------------------------------------------------------
# Sağlık kontrolü
# ---------------------------------------------------------------------------


def test_health_check_reports_missing_model_with_installed_list() -> None:
    session = FakeSession(
        FakeResponse(payload={"models": [{"name": "llama3.1:8b"}, {"name": "llava:13b"}]})
    )
    client = OllamaClient(_settings(), session=session)

    ready, message = client.health_check()

    assert ready is False
    assert "ollama pull llava:7b" in message
    assert "llama3.1:8b" in message


def test_health_check_accepts_untagged_model_name() -> None:
    session = FakeSession(FakeResponse(payload={"models": [{"name": "llava:7b"}]}))
    client = OllamaClient(_settings(model="llava"), session=session)

    ready, message = client.health_check()

    assert ready is True
    assert "hazır" in message


def test_health_check_when_server_down() -> None:
    requests = _requests_module()
    session = FakeSession(requests.ConnectionError("yok"))
    client = OllamaClient(_settings(), session=session)

    ready, message = client.health_check()

    assert ready is False
    assert "ulaşılamadı" in message


# ---------------------------------------------------------------------------
# OpenAI uyumlu
# ---------------------------------------------------------------------------


def _openai_settings(**overrides: object) -> VLMSettings:
    return _settings(
        provider="openai-compatible",
        api_base="https://api.example.com/v1",
        api_key="test-anahtar",
        api_model="gpt-4o-mini",
        **overrides,
    )


def test_openai_client_requires_configuration() -> None:
    with pytest.raises(VLMUnavailableError, match="VLM_API_BASE"):
        OpenAICompatibleClient(_settings(provider="openai-compatible"), session=FakeSession())

    with pytest.raises(VLMUnavailableError, match="VLM_API_KEY"):
        OpenAICompatibleClient(
            _settings(provider="openai-compatible", api_base="https://x/v1"),
            session=FakeSession(),
        )


def test_openai_request_shape_and_parsing() -> None:
    session = FakeSession(
        FakeResponse(payload={"choices": [{"message": {"content": "Kısa Türkçe yanıt."}}]})
    )
    client = OpenAICompatibleClient(_openai_settings(), session=session, sleep=lambda _s: None)

    answer = client.describe(b"jpegbytes", "Sahneyi açıkla")

    assert answer == "Kısa Türkçe yanıt."
    request = session.requests[0]
    assert request["url"] == "https://api.example.com/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-anahtar"
    body = json.loads(str(request["data"]))
    assert body["model"] == "gpt-4o-mini"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Sahneyi açıkla"}
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_list_content_is_joined() -> None:
    session = FakeSession(
        FakeResponse(
            payload={"choices": [{"message": {"content": [{"type": "text", "text": "iki"}]}}]}
        )
    )
    client = OpenAICompatibleClient(_openai_settings(), session=session, sleep=lambda _s: None)

    assert client.describe(b"x", "açıkla") == "iki"


def test_openai_malformed_payload() -> None:
    session = FakeSession(FakeResponse(payload={"choices": []}))
    client = OpenAICompatibleClient(_openai_settings(), session=session, sleep=lambda _s: None)

    with pytest.raises(VLMResponseError, match="beklenen biçimde"):
        client.describe(b"x", "açıkla")


def test_openai_401_is_not_retried() -> None:
    session = FakeSession(FakeResponse(status=401, text="invalid api key"))
    client = OpenAICompatibleClient(_openai_settings(), session=session, sleep=lambda _s: None)

    with pytest.raises(VLMError, match="HTTP 401") as excinfo:
        client.describe(b"x", "açıkla")

    assert isinstance(excinfo.value, VLMError)
    assert len(session.requests) == 1
