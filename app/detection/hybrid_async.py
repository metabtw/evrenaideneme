"""CLIP etiketlemeyi ayrı thread'de çalıştıran sarmalayıcı.

Neden?
    Ölçüm: CLIP bölge taraması CPU'da ~470 ms. Seyrek çalıştırınca (1.4 sn'de
    bir) medyan 50 ms'ye inse de o karede **500 ms donma** oluyordu
    (p95 = 512 ms). Görme engelli kullanıcı için kare atlaması kabul edilemez.

    Çözüm: CLIP kendi thread'inde çalışır. ``relabel`` **hiç beklemez**;
    elindeki önbelleği uygular ve arka plana taze istek bırakır. Bu, projedeki
    diğer ağır işlerle (``DetectionWorker``, ``AnalysisWorker``) aynı desendir.

Akış::

    relabel(frame, detections) -> hemen önbellek etiketleri (0 ms)
                                   │
                                   └─► ClipWorker: sırası geldiyse CLIP çalıştır
                                                    (aynı anda en fazla 1 istek)
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from app.detection.detector import Detection
from app.detection.hybrid import HybridTagger
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AsyncHybridTagger:
    """``HybridTagger``'ı arka plan thread'i ile saran, bloklamayan sarmalayıcı.

    Arayüz ``HybridTagger.relabel`` ile aynıdır; ``DetectionWorker`` değişmeden
    kullanabilir (duck typing).
    """

    def __init__(
        self,
        tagger: HybridTagger,
        *,
        min_interval: float | None = None,
        name: str = "evrenaidd-clip",
    ) -> None:
        self._tagger = tagger
        self._interval = min_interval if min_interval is not None else tagger.config.min_interval
        self._name = name

        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._pending: tuple[np.ndarray, list[Detection]] | None = None
        self._in_flight = False
        self._last_run = float("-inf")
        #: Son CLIP sonucu (kare + etiketler aynı anda).
        self._labels: list[Detection] = []

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> AsyncHybridTagger:
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.debug("CLIP thread'i başladı (%s)", self._name)
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        with self._wakeup:
            self._wakeup.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("CLIP thread'i %.1f sn içinde durmadı", timeout)
        self._thread = None

    def close(self) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- HybridTagger uyumlu arayüz ---------------------------------------

    @property
    def stats(self) -> Any:
        return self._tagger.stats

    @property
    def catalog(self) -> Any:
        return self._tagger.catalog

    @property
    def classifier(self) -> Any:
        """Alttaki CLIP sınıflandırıcısı (ön yükleme/teşhis için)."""
        return self._tagger.classifier

    @property
    def interval(self) -> float:
        return self._interval

    def relabel(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """**Bloklamaz**: önbellek etiketlerini uygular, arka plana istek bırakır."""
        now = time.monotonic()
        due = (now - self._last_run) >= self._interval and not self._in_flight

        if due:
            with self._wakeup:
                if not self._in_flight:
                    # Kare kopyası: ana thread çizim yaparken değişmesin.
                    self._pending = (frame.copy(), list(detections))
                    self._last_run = now
                    self._in_flight = True
                    self._wakeup.notify()

        with self._lock:
            cached = self._labels

        if not detections:
            # YOLO boş: son CLIP bulgularını kullan (bayat ama titremesiz).
            return list(cached)

        if not cached:
            return detections

        applied: list[Detection] = []
        for detection in detections:
            match = _match_label(detection.box, cached)
            applied.append(match if match is not None else detection)
        return applied

    # -- thread gövdesi ----------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._wakeup:
                while self._pending is None and not self._stop_event.is_set():
                    self._wakeup.wait(timeout=0.2)
                if self._stop_event.is_set():
                    return
                pending, self._pending = self._pending, None

            if pending is None:
                continue

            frame, detections = pending
            started = time.perf_counter()
            try:
                result = self._tagger.relabel(frame, detections)
            except Exception:  # noqa: BLE001 - arka plan işi akışı düşürmemeli
                logger.exception("CLIP etiketleme hatası (arka plan)")
                with self._lock:
                    self._in_flight = False
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._labels = list(result)
                self._in_flight = False

            # Ölçülen süreye göre aralığı uyarla (görev döngüsü ≤ ~1/3).
            budget = self._tagger.config.budget_ms
            base = max(0.0, self._tagger.config.min_interval)
            if elapsed_ms > budget:
                self._interval = max(base, elapsed_ms / 1000.0 * 3.0)
            else:
                self._interval = base


def _match_label(
    box: tuple[int, int, int, int],
    cached: list[Detection],
    *,
    iou_threshold: float = 0.5,
) -> Detection | None:
    from app.detection.hybrid import _best_label_match

    return _best_label_match(box, cached, iou_threshold=iou_threshold)
