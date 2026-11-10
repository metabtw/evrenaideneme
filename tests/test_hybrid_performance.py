"""Hibrit etiketleyici performans testleri: CLIP seyrek çalışmalı, akış durmamalı."""

from __future__ import annotations

import numpy as np
from app.catalog import ClassCatalog
from app.detection.clip import TaggedDetection
from app.detection.detector import Detection
from app.detection.hybrid import HybridConfig, HybridTagger

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
CATALOG = ClassCatalog.from_names(["su ısıtıcısı", "kitap", "fincan"])


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountingClassifier:
    """Kaç kez çağrıldığını sayan sahte CLIP."""

    def __init__(self, catalog: ClassCatalog, results: list[TaggedDetection]) -> None:
        self.catalog = catalog
        self.results = results
        self.calls = 0

    def tag(self, frame, boxes, *, top_k: int = 2):  # noqa: ANN001
        self.calls += 1
        return [item for item in self.results if item.box in boxes]


class CountingRegionDetector:
    """Kaç kez tarandığını sayan sahte bölge dedektörü."""

    def __init__(self, results: list[Detection], *, ms: float = 470.0) -> None:
        self.results = results
        self.calls = 0
        self.last_ms = ms

    def detect(self, frame, *, top_k: int = 1):  # noqa: ANN001
        self.calls += 1
        return list(self.results)


def _yolo(label: str = "vazo", box=(10, 10, 100, 100)) -> Detection:
    return Detection(class_id=0, label=label, confidence=0.8, box=box)


def _clip(label: str, box=(10, 10, 100, 100), confidence: float = 0.9):
    return TaggedDetection(box=box, label=label, confidence=confidence, index=0)


def test_clip_runs_once_per_interval_not_every_frame() -> None:
    """30 kare / 1 saniye: CLIP en fazla interval kadar çalışmalı."""
    clock = FakeClock()
    classifier = CountingClassifier(CATALOG, [_clip("fincan")])
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(min_interval=1.0),
        clock=clock,
    )

    for _ in range(30):
        tagger.relabel(FRAME, [_yolo()])
        clock.advance(1 / 30)

    # 1 saniyelik pencerede 1 kez (+başlangıç) çalışır; 30 kez değil.
    assert classifier.calls <= 2, f"CLIP {classifier.calls} kez çalıştı"
    assert tagger.stats.skipped >= 28


def test_region_scan_throttled_when_yolo_finds_nothing() -> None:
    """Asıl kasma sebebi: YOLO boşken bölge taraması her karede çalışıyordu."""
    clock = FakeClock()
    classifier = CountingClassifier(CATALOG, [])
    region = CountingRegionDetector([Detection(0, "su ısıtıcısı", 1.0, (0, 0, 320, 240))])
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(min_interval=1.0),
        region_detector=region,  # type: ignore[arg-type]
        clock=clock,
    )

    for _ in range(60):  # 2 saniyelik akış, YOLO hep boş
        tagger.relabel(FRAME, [])
        clock.advance(1 / 30)

    assert region.calls <= 3, f"bölge taraması {region.calls} kez çalıştı"


def test_cached_result_used_between_scans_avoids_flicker() -> None:
    clock = FakeClock()
    classifier = CountingClassifier(CATALOG, [])
    found = [Detection(0, "su ısıtıcısı", 0.95, (0, 0, 320, 240))]
    region = CountingRegionDetector(found)
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(min_interval=1.0),
        region_detector=region,  # type: ignore[arg-type]
        clock=clock,
    )

    first = tagger.relabel(FRAME, [])
    clock.advance(0.1)
    second = tagger.relabel(FRAME, [])

    assert region.calls == 1
    assert first and second  # titreme yok: önbellek kullanıldı
    assert second[0].label == "su ısıtıcısı"


def test_adaptive_interval_grows_for_slow_clip() -> None:
    clock = FakeClock()
    classifier = CountingClassifier(CATALOG, [])
    region = CountingRegionDetector([Detection(0, "kitap", 0.9, (0, 0, 320, 240))], ms=470.0)
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(min_interval=1.0, budget_ms=400.0),
        region_detector=region,  # type: ignore[arg-type]
        clock=clock,
    )

    tagger.relabel(FRAME, [])

    # 470 ms > 400 ms bütçe -> aralık 3× süreye çıkar (~1.41 sn).
    assert tagger.interval > 1.3
    assert tagger.stats.last_clip_ms == 470.0


def test_fast_clip_keeps_min_interval() -> None:
    clock = FakeClock()
    classifier = CountingClassifier(CATALOG, [])
    region = CountingRegionDetector([], ms=50.0)
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(min_interval=1.0, budget_ms=400.0),
        region_detector=region,  # type: ignore[arg-type]
        clock=clock,
    )

    tagger.relabel(FRAME, [])

    assert tagger.interval == 1.0


def test_cached_labels_applied_to_new_boxes() -> None:
    """Kutular her karede değişse de etiket önbellekten gelir (CLIP tekrar çalışmaz)."""
    clock = FakeClock()
    classifier = CountingClassifier(CATALOG, [_clip("fincan", box=(10, 10, 100, 100))])
    tagger = HybridTagger(classifier, HybridConfig(min_interval=1.0), clock=clock)  # type: ignore[arg-type]

    tagger.relabel(FRAME, [_yolo(box=(10, 10, 100, 100))])
    clock.advance(0.2)
    # Aynı kutu (küçük kayma): önbellekten etiket uygulanır.
    result = tagger.relabel(FRAME, [_yolo(box=(12, 12, 101, 101))])

    assert classifier.calls == 1
    assert result[0].label == "fincan"


def test_zero_interval_keeps_old_behaviour() -> None:
    clock = FakeClock()
    classifier = CountingClassifier(CATALOG, [_clip("fincan")])
    tagger = HybridTagger(classifier, HybridConfig(min_interval=0.0), clock=clock)  # type: ignore[arg-type]

    for _ in range(5):
        tagger.relabel(FRAME, [_yolo()])
        clock.advance(0.01)

    assert classifier.calls == 5
