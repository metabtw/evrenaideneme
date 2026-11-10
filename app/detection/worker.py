"""Tespit işçisi: kamera akışını bloklamadan arka planda YOLO çalıştırır.

Strateji **latest-frame-wins**: işçi meşgulken yeni kareler kuyruğa yığılmaz;
yalnızca en son kare saklanır. Böylece analiz hiçbir zaman bayat karelerle
geride kalmaz ve ana döngü asla beklemez.

    worker = DetectionWorker(detector).start()
    worker.submit(frame)                    # kopya alınır, hemen döner
    result = worker.latest_result()         # None veya son sonuç
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.detection.detector import Detection, ObjectDetector
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Bir kareye ait tespit sonucu."""

    frame_id: int
    detections: tuple[Detection, ...]
    inference_ms: float
    timestamp: float

    @property
    def is_empty(self) -> bool:
        return not self.detections

    def labels(self) -> list[str]:
        return [detection.label for detection in self.detections]


@dataclass(frozen=True, slots=True)
class WorkerStats:
    """İşçi sayacı (log/HUD için)."""

    processed: int = 0
    dropped: int = 0
    avg_inference_ms: float = 0.0
    last_error: str = ""
    running: bool = False


class DetectionWorker:
    """Tek üretici / tek tüketici tespit thread'i."""

    def __init__(
        self,
        detector: ObjectDetector,
        *,
        tagger: Any | None = None,
        name: str = "evrenaidd-detection",
        copy_frames: bool = True,
    ) -> None:
        self._detector = detector
        self._tagger = tagger
        self._name = name
        self._copy_frames = copy_frames

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._pending: tuple[int, np.ndarray] | None = None
        self._latest: DetectionResult | None = None
        self._next_frame_id = 0

        self._processed = 0
        self._dropped = 0
        self._total_inference_ms = 0.0
        self._last_error = ""

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> DetectionWorker:
        if self.is_running:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.debug("Tespit işçisi başladı (%s)", self._name)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """İşçiyi durdurur; işlenmekte olan kare varsa bitmesini bekler."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Tespit işçisi %.1f sn içinde durmadı", timeout)
        self._thread = None

    def __enter__(self) -> DetectionWorker:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- üretici tarafı ----------------------------------------------------

    def submit(self, frame: np.ndarray) -> int:
        """Kareyi işçiye bırakır ve hemen döner.

        İşçi meşgulse önceki bekleyen kare düşürülür (latest-frame-wins).

        Returns:
            Bu kareye atanan artan kimlik numarası.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Boş kare gönderilemez")

        payload = frame.copy() if self._copy_frames else frame
        with self._condition:
            if self._pending is not None:
                self._dropped += 1
            self._next_frame_id += 1
            frame_id = self._next_frame_id
            self._pending = (frame_id, payload)
            self._condition.notify()
        return frame_id

    # -- tüketici tarafı ---------------------------------------------------

    def latest_result(self) -> DetectionResult | None:
        with self._condition:
            return self._latest

    def wait_for_result(self, timeout: float = 5.0) -> DetectionResult | None:
        """Bir sonraki sonucu (en fazla ``timeout`` sn) bekler."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._latest
                self._condition.wait(timeout=remaining)
                if self._latest is not None:
                    return self._latest

    @property
    def stats(self) -> WorkerStats:
        with self._condition:
            average = self._total_inference_ms / self._processed if self._processed else 0.0
            return WorkerStats(
                processed=self._processed,
                dropped=self._dropped,
                avg_inference_ms=average,
                last_error=self._last_error,
                running=self.is_running,
            )

    # -- thread gövdesi ----------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while self._pending is None and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.2)
                if self._stop_event.is_set():
                    return
                assert self._pending is not None
                frame_id, frame = self._pending
                self._pending = None

            started = time.perf_counter()
            try:
                detections = self._detector.detect(frame)
                if self._tagger is not None:
                    detections = self._tagger.relabel(frame, detections)
            except Exception as exc:  # noqa: BLE001 - işçi asla ölmemeli
                with self._condition:
                    self._last_error = str(exc)
                    self._condition.notify_all()
                logger.exception("Tespit başarısız (kare #%s): %s", frame_id, exc)
                continue

            inference_ms = (time.perf_counter() - started) * 1000.0
            result = DetectionResult(
                frame_id=frame_id,
                detections=tuple(detections),
                inference_ms=inference_ms,
                timestamp=time.time(),
            )
            with self._condition:
                self._latest = result
                self._processed += 1
                self._total_inference_ms += inference_ms
                self._last_error = ""
                self._condition.notify_all()

        logger.debug("Tespit işçisi durdu (%s)", self._name)
