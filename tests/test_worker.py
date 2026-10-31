"""DetectionWorker testleri: latest-frame-wins ve hata dayanıklılığı."""

from __future__ import annotations

import threading
import time

import numpy as np
from app.detection import Detection, DetectionWorker


def _frame(value: int) -> np.ndarray:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[0, 0, 0] = value
    return frame


class EchoDetector:
    def __init__(self) -> None:
        self.seen: list[int] = []

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.seen.append(int(frame[0, 0, 0]))
        return [Detection(0, "kişi", 0.9, (1, 1, 5, 5))]


class BlockingDetector(EchoDetector):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.seen.append(int(frame[0, 0, 0]))
        self.started.set()
        assert self.release.wait(timeout=2.0)
        return []


class BrokenDetector:
    def detect(self, _frame: np.ndarray) -> list[Detection]:
        raise RuntimeError("model çöktü")


def test_processes_frames() -> None:
    detector = EchoDetector()
    with DetectionWorker(detector) as worker:
        worker.submit(_frame(1))
        worker.wait_for_result(timeout=2.0)
        worker.submit(_frame(2))

        deadline = time.monotonic() + 2.0
        while worker.stats.processed < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

    stats = worker.stats
    assert stats.processed == 2
    assert detector.seen == [1, 2]
    assert stats.avg_inference_ms > 0
    assert worker.latest_result() is not None


def test_latest_frame_wins_and_drops_old_ones() -> None:
    detector = BlockingDetector()
    with DetectionWorker(detector, copy_frames=True) as worker:
        worker.submit(_frame(1))
        assert detector.started.wait(timeout=2.0)

        worker.submit(_frame(2))
        worker.submit(_frame(3))
        detector.release.set()

        deadline = time.monotonic() + 2.0
        while worker.stats.processed < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

        result = worker.latest_result()

    assert detector.seen == [1, 3]
    assert worker.stats.dropped == 1
    assert result is not None
    assert result.frame_id == 3


def test_submit_rejects_empty_frame() -> None:
    with DetectionWorker(EchoDetector()) as worker:
        try:
            worker.submit(np.zeros((0, 0, 3), dtype=np.uint8))
        except ValueError as exc:
            assert "Boş kare" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("ValueError bekleniyordu")


def test_worker_survives_detector_error() -> None:
    with DetectionWorker(BrokenDetector()) as worker:  # type: ignore[arg-type]
        worker.submit(_frame(1))
        deadline = time.monotonic() + 2.0
        while not worker.stats.last_error and time.monotonic() < deadline:
            time.sleep(0.01)

        assert "model çöktü" in worker.stats.last_error
        assert worker.is_running is True

        # Hata sonrası işçi yeni kareleri kabul etmeye devam eder.
        worker.submit(_frame(2))

    assert worker.is_running is False
