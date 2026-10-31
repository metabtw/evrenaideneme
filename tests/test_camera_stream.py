"""CameraStream testleri: sahte VideoCapture ile donanımsız doğrulama."""

from __future__ import annotations

import time

import numpy as np
import pytest
from app.camera import CameraError, CameraStream
from app.config import CameraSettings


class FakeCapture:
    """Tek karelik sahte webcam."""

    def __init__(self, *, opened: bool = True) -> None:
        self._opened = opened
        self.released = False
        self.props: dict[int, float] = {}
        self.reads = 0

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV adı
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        self.reads += 1
        frame = np.full((120, 160, 3), self.reads % 255, dtype=np.uint8)
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def release(self) -> None:
        self.released = True


class FailingCapture(FakeCapture):
    def read(self) -> tuple[bool, None]:
        self.reads += 1
        return False, None


def _settings(**overrides: object) -> CameraSettings:
    base = CameraSettings(index=0, width=160, height=120, target_fps=200)
    return CameraSettings(**{**_as_dict(base), **overrides})


def _as_dict(settings: CameraSettings) -> dict[str, object]:
    return {
        "index": settings.index,
        "width": settings.width,
        "height": settings.height,
        "target_fps": settings.target_fps,
        "backend": settings.backend,
        "reconnect_attempts": settings.reconnect_attempts,
        "reconnect_delay": settings.reconnect_delay,
    }


def test_reads_frames_and_stops_cleanly() -> None:
    capture = FakeCapture()
    stream = CameraStream(_settings(), capture_factory=lambda _i, _b: capture)

    with stream:
        frame = stream.wait_for_frame(timeout=2.0)
        assert frame is not None
        assert frame.shape == (120, 160, 3)
        assert stream.is_running
        time.sleep(0.05)
        assert stream.stats.frames_read > 0

    assert capture.released is True
    assert stream.is_running is False
    assert stream.read() is not None  # son kare bellekte kalır


def test_read_returns_copy_by_default() -> None:
    capture = FakeCapture()
    stream = CameraStream(_settings(), capture_factory=lambda _i, _b: capture)

    with stream:
        first = stream.wait_for_frame(timeout=2.0)
        assert first is not None
        first[:] = 0
        second = stream.read()
        assert second is not None
        assert second.any()  # kopya olduğu için mutasyon kalıcı değil


def test_open_failure_raises_camera_error() -> None:
    capture = FakeCapture(opened=False)
    stream = CameraStream(_settings(), capture_factory=lambda _i, _b: capture)

    with pytest.raises(CameraError, match="Kamera açılamadı"):
        stream.start()
    assert capture.released is True


def test_reconnects_after_read_failure() -> None:
    failing = FailingCapture()
    healthy = FakeCapture()
    captures = [failing, healthy]

    def factory(_index: int, _backend: int | None) -> FakeCapture:
        return captures.pop(0) if captures else healthy

    stream = CameraStream(
        _settings(reconnect_attempts=1, reconnect_delay=0.0),
        capture_factory=factory,
    )

    with stream:
        frame = stream.wait_for_frame(timeout=2.0)
        assert frame is not None
        assert stream.stats.reconnects == 1
        assert failing.released is True


def test_read_latest_only_reports_new_frames() -> None:
    capture = FakeCapture()
    stream = CameraStream(_settings(target_fps=60), capture_factory=lambda _i, _b: capture)

    with stream:
        assert stream.wait_for_frame(timeout=2.0) is not None
        sequence, frame = stream.read_latest(-1, copy=False)
        assert frame is not None
        assert sequence >= 1

        # Aynı sıra numarası verilirse yeni kare dönmez.
        same_sequence, repeated = stream.read_latest(sequence, copy=False)
        assert same_sequence == sequence
        assert repeated is None

        time.sleep(0.1)
        newer_sequence, newer_frame = stream.read_latest(sequence, copy=False)
        assert newer_frame is not None
        assert newer_sequence > sequence


def test_list_cameras_uses_factory() -> None:
    def factory(index: int, _backend: int | None) -> FakeCapture:
        return FakeCapture(opened=index in (0, 2))

    stream = CameraStream(_settings(), capture_factory=factory)
    assert stream.list_cameras(max_index=2) == [0, 2]
