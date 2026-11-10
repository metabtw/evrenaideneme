"""Bölge taramalı CLIP testleri (model gerektirmez)."""

from __future__ import annotations

import numpy as np
import pytest
from app.catalog import ClassCatalog
from app.detection.clip import TaggedDetection
from app.detection.clip_region import (
    ClipRegionDetector,
    dedupe_detections,
    propose_regions,
)
from app.detection.detector import Detection


def test_propose_regions_covers_frame_and_center() -> None:
    regions = propose_regions(1280, 720)
    names = {region.name for region in regions}

    assert "tam" in names
    assert "merkez" in names
    assert len(regions) >= 8
    # Tüm bölgeler geçerli ve minimum boyutta
    for region in regions:
        x1, y1, x2, y2 = region.box
        assert x2 > x1 and y2 > y1
        assert x2 - x1 >= 32 and y2 - y1 >= 32


def test_propose_regions_handles_tiny_frames() -> None:
    assert propose_regions(10, 10) == []
    assert propose_regions(0, 0) == []


def test_dedupe_keeps_highest_confidence_per_overlap() -> None:
    a = Detection(0, "kitap", 0.9, (0, 0, 100, 100))
    b = Detection(0, "kitap", 0.5, (5, 5, 100, 100))  # neredeyse aynı
    c = Detection(0, "kitap", 0.8, (500, 500, 600, 600))  # ayrı bölge

    kept = dedupe_detections([a, b, c])

    assert len(kept) == 2
    assert kept[0].confidence == pytest.approx(0.9)
    assert all(item.confidence for item in kept)


def test_dedupe_keeps_different_labels() -> None:
    a = Detection(0, "kitap", 0.9, (0, 0, 100, 100))
    b = Detection(1, "fincan", 0.8, (0, 0, 100, 100))

    assert len(dedupe_detections([a, b])) == 2


class FakeClassifier:
    def __init__(self, catalog: ClassCatalog, results: list[TaggedDetection]) -> None:
        self.catalog = catalog
        self.results = results

    def tag(self, frame, boxes, *, top_k: int = 2):  # noqa: ANN001
        return [item for item in self.results if item.box in boxes]


def test_region_detector_returns_confident_boxes() -> None:
    catalog = ClassCatalog.from_names(["su ısıtıcısı", "kitap"])
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    full = (0, 0, 320, 240)
    classifier = FakeClassifier(
        catalog,
        [
            TaggedDetection(box=full, label="su ısıtıcısı", confidence=0.95, index=0),
            TaggedDetection(box=(0, 0, 160, 120), label="kitap", confidence=0.10, index=1),
        ],
    )
    detector = ClipRegionDetector(classifier, min_confidence=0.30)  # type: ignore[arg-type]

    results = detector.detect(frame)

    assert len(results) == 1
    assert results[0].label == "su ısıtıcısı"
    assert results[0].confidence == pytest.approx(0.95)


def test_region_detector_handles_empty_frame() -> None:
    catalog = ClassCatalog.from_names(["kitap"])
    detector = ClipRegionDetector(FakeClassifier(catalog, []))  # type: ignore[arg-type]
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_catalog_exposed() -> None:
    catalog = ClassCatalog.from_names(["kitap", "fincan"])
    detector = ClipRegionDetector(FakeClassifier(catalog, []))  # type: ignore[arg-type]
    assert detector.catalog.labels == ["kitap", "fincan"]
