"""AsyncHybridTagger testleri: relabel bloklamamalı, arka plan tazelemeli."""

from __future__ import annotations

import time

import numpy as np
from app.catalog import ClassCatalog
from app.detection.clip import TaggedDetection
from app.detection.detector import Detection
from app.detection.hybrid import HybridConfig, HybridTagger
from app.detection.hybrid_async import AsyncHybridTagger

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
CATALOG = ClassCatalog.from_names(["su ısıtıcısı", "kitap", "fincan"])


class SlowClassifier:
    """CLIP gibi yavaş davranan sahte sınıflandırıcı."""

    def __init__(self, catalog: ClassCatalog, *, delay: float = 0.25) -> None:
        self.catalog = catalog
        self.delay = delay
        self.calls = 0

    def tag(self, frame, boxes, *, top_k: int = 2):  # noqa: ANN001
        self.calls += 1
        time.sleep(self.delay)
        return [
            TaggedDetection(box=box, label="su ısıtıcısı", confidence=0.9, index=0) for box in boxes
        ]


class SlowRegion:
    def __init__(self, *, delay: float = 0.2) -> None:
        self.delay = delay
        self.last_ms = delay * 1000
        self.calls = 0

    def detect(self, frame, *, top_k: int = 1):  # noqa: ANN001
        self.calls += 1
        time.sleep(self.delay)
        return [Detection(0, "su ısıtıcısı", 0.95, (0, 0, 320, 240))]


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make(delay: float = 0.25, *, region: bool = False) -> AsyncHybridTagger:
    classifier = SlowClassifier(CATALOG, delay=delay)
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(min_interval=0.05, budget_ms=400.0),
        region_detector=SlowRegion() if region else None,  # type: ignore[arg-type]
    )
    return AsyncHybridTagger(tagger, min_interval=0.05).start()


def test_relabel_never_blocks() -> None:
    """CLIP 250 ms sürse de her relabel çağrısı anında dönmeli."""
    tagger = _make(delay=0.25)
    detection = Detection(0, "vazo", 0.8, (10, 10, 100, 100))

    durations = []
    for _ in range(20):
        started = time.perf_counter()
        tagger.relabel(FRAME, [detection])
        durations.append((time.perf_counter() - started) * 1000)

    tagger.stop()
    assert max(durations) < 100, f"en yavaş çağrı {max(durations):.0f} ms"


def test_background_thread_refreshes_labels() -> None:
    tagger = _make(delay=0.05)
    detection = Detection(0, "vazo", 0.8, (10, 10, 100, 100))

    # İlk çağrılar önbellek boş: YOLO etiketi döner.
    first = tagger.relabel(FRAME, [detection])
    assert first[0].label == "vazo"

    # Arka plan bitince etiket tazelenir.
    assert _wait(lambda: tagger._labels != [], timeout=3.0)
    time.sleep(0.6)
    refreshed = tagger.relabel(FRAME, [detection])

    tagger.stop()
    assert refreshed[0].label == "su ısıtıcısı"


def test_region_result_used_when_yolo_empty() -> None:
    tagger = _make(delay=0.05, region=True)

    tagger.relabel(FRAME, [])
    assert _wait(lambda: bool(tagger._labels), timeout=3.0)

    result = tagger.relabel(FRAME, [])
    tagger.stop()
    assert result
    assert result[0].label == "su ısıtıcısı"


def test_stop_is_idempotent_and_joins() -> None:
    tagger = _make(delay=0.05)
    tagger.relabel(FRAME, [Detection(0, "vazo", 0.8, (0, 0, 10, 10))])

    tagger.stop()
    assert tagger.is_running is False
    tagger.stop()
    tagger.close()


def test_interval_adapts_to_slow_clip() -> None:
    """Bütçeyi (400 ms) aşan CLIP, aralığı 3× süreye çıkarır."""
    tagger = _make(delay=0.45)
    tagger.relabel(FRAME, [Detection(0, "vazo", 0.8, (0, 0, 100, 100))])

    assert _wait(lambda: tagger.interval > 1.0, timeout=3.0)
    tagger.stop()


def test_catalog_and_stats_exposed() -> None:
    tagger = _make(delay=0.01)
    assert tagger.catalog.labels == ["su ısıtıcısı", "kitap", "fincan"]
    assert tagger.stats is not None
    tagger.stop()
