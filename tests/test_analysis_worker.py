"""AnalysisWorker testleri: tek uçuş, latest-frame-wins, hata dayanıklılığı."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from app.analysis import AnalysisMode, AnalysisWorker, VLMUnavailableError
from app.analysis.vlm_client import VLMClient


class _Frame:
    """Tek kanallı kare: hangi karenin analiz edildiğini ayırt etmek için."""

    @staticmethod
    def make(value: int) -> np.ndarray:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[0, 0, 0] = value
        return frame


class FakeClient:
    """``describe`` çağrılarını kaydeden sahte istemci."""

    def __init__(self, answer: str = "yanıt", *, delay: float = 0.0) -> None:
        self.answer = answer
        self.delay = delay
        self.prompts: list[str] = []
        self.closed = False
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def name(self) -> str:
        return "fake:vlm"

    def describe(self, frame_jpeg: bytes, prompt: str) -> str:
        self.prompts.append(prompt)
        self.started.set()
        if self.delay:
            assert self.release.wait(timeout=2.0)
        return self.answer

    def close(self) -> None:
        self.closed = True


class FailingClient(FakeClient):
    def describe(self, frame_jpeg: bytes, prompt: str) -> str:
        raise VLMUnavailableError("model yok")


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_successful_analysis_calls_callback() -> None:
    results: list[object] = []
    client = FakeClient("Masada bir fincan var.")

    with AnalysisWorker(client, model="llava:7b", on_result=results.append) as worker:
        worker.submit(_Frame.make(1), mode=AnalysisMode.DESCRIBE)
        result = worker.wait_for_result(timeout=3.0)

    assert result is not None
    assert result.answer == "Masada bir fincan var."
    assert result.mode is AnalysisMode.DESCRIBE
    assert result.elapsed_ms >= 0
    assert results and results[0] is result
    assert client.closed is True


def test_prompt_uses_mode_and_model_hint() -> None:
    client = FakeClient()
    with AnalysisWorker(client, model="qwen2.5vl:7b") as worker:
        worker.submit(_Frame.make(1), mode=AnalysisMode.OCR)
        worker.wait_for_result(timeout=3.0)

    assert "yazıyı" in client.prompts[0]
    assert "Türkçe karakterleri koru" in client.prompts[0]


def test_count_mode_requires_target() -> None:
    with AnalysisWorker(FakeClient()) as worker, pytest.raises(ValueError, match="hedef nesne"):
        worker.submit(_Frame.make(1), mode=AnalysisMode.COUNT)


def test_empty_frame_rejected() -> None:
    with AnalysisWorker(FakeClient()) as worker, pytest.raises(ValueError, match="Boş kare"):
        worker.submit(np.zeros((0, 0, 3), dtype=np.uint8))


def test_single_flight_replaces_pending_request() -> None:
    client = FakeClient(delay=0.5)
    with AnalysisWorker(client) as worker:
        worker.submit(_Frame.make(1))
        assert client.started.wait(timeout=2.0)

        # İstek sürerken iki tetikleme: kuyrukta yalnızca sonuncusu kalmalı.
        worker.submit(_Frame.make(2))
        worker.submit(_Frame.make(3))
        assert worker.is_busy is True
        assert worker.stats["replaced"] == 1

        client.release.set()
        assert _wait(lambda: worker.stats["completed"] >= 2)

        assert len(client.prompts) == 2
        assert worker.stats["replaced"] == 1


def test_cancel_drops_pending_request() -> None:
    client = FakeClient(delay=0.5)
    with AnalysisWorker(client) as worker:
        worker.submit(_Frame.make(1))
        assert client.started.wait(timeout=2.0)
        worker.submit(_Frame.make(2))
        worker.cancel()
        assert worker.stats["pending"] == 0

        client.release.set()
        assert _wait(lambda: worker.stats["completed"] >= 1)
        time.sleep(0.1)
        assert len(client.prompts) == 1


def test_error_callback_and_worker_survives() -> None:
    errors: list[str] = []
    client = FailingClient()

    with AnalysisWorker(client, on_error=errors.append) as worker:
        worker.submit(_Frame.make(1))
        assert _wait(lambda: bool(errors))

        assert "model yok" in errors[0]
        assert worker.is_running is True
        assert "model yok" in worker.last_error

        # Hata sonrası yeni istekler işlenmeye devam eder.
        client.answer = "toparladı"
        worker._client = FakeClient("toparladı")  # noqa: SLF001 - test kancası
        worker.submit(_Frame.make(2))
        assert _wait(lambda: worker.stats["completed"] >= 1)


def test_callback_exception_does_not_kill_worker() -> None:
    def boom(_result: object) -> None:
        raise RuntimeError("callback patladı")

    with AnalysisWorker(FakeClient(), on_result=boom) as worker:
        worker.submit(_Frame.make(1))
        assert _wait(lambda: worker.stats["completed"] >= 1)
        assert worker.is_running is True


def test_stop_marks_not_running_and_is_idempotent() -> None:
    worker = AnalysisWorker(FakeClient()).start()
    assert worker.is_running is True
    worker.stop()
    assert worker.is_running is False
    worker.stop()


def test_worker_implements_vlm_protocol_shape() -> None:
    """Worker istemciyi içeride tutar; dışarıya sonuç verir (arayüz ayrımı)."""
    client: VLMClient = FakeClient()
    assert hasattr(client, "describe")
    with AnalysisWorker(client) as worker:
        assert worker.latest_result() is None
