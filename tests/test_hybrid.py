"""Hibrit etiketleyici testleri (sahte CLIP ile, model indirmeden)."""

from __future__ import annotations

import numpy as np
import pytest
from app.catalog import ClassCatalog
from app.detection.clip import TaggedDetection
from app.detection.detector import Detection
from app.detection.hybrid import (
    HybridConfig,
    HybridTagger,
    create_hybrid_tagger,
)

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


class FakeClassifier:
    """``ClipClassifier`` yerine geçen sahte sınıflandırıcı."""

    def __init__(self, catalog: ClassCatalog, results: list[TaggedDetection]) -> None:
        self.catalog = catalog
        self._results = results
        self.calls = 0

    def tag(self, frame, boxes, *, top_k: int = 2):  # noqa: ANN001
        self.calls += 1
        return [item for item in self._results if item.box in boxes]


def _yolo(label: str, box=(10, 10, 100, 100), class_id: int = 0) -> Detection:
    return Detection(class_id=class_id, label=label, confidence=0.8, box=box)


def _clip(label: str, box=(10, 10, 100, 100), confidence: float = 0.9, index: int = 0):
    return TaggedDetection(box=box, label=label, confidence=confidence, index=index)


CATALOG = ClassCatalog.from_names(["su ısıtıcısı", "kitap", "fincan"])


def test_clip_replaces_label_when_yolo_label_not_in_catalog() -> None:
    """YOLO 'vazo' diyor ama katalogda vazo yok; CLIP 'su ısıtıcısı' diyor."""
    classifier = FakeClassifier(CATALOG, [_clip("su ısıtıcısı", confidence=0.95)])
    tagger = HybridTagger(classifier)  # type: ignore[arg-type]

    result = tagger.relabel(FRAME, [_yolo("vazo")])

    assert result[0].label == "su ısıtıcısı"
    assert result[0].confidence == pytest.approx(0.95)
    assert tagger.stats.clip_used == 1


def test_clip_wins_when_clearly_more_confident() -> None:
    """İkisi de katalogda ama CLIP belirgin biçimde daha güvenli -> CLIP kazanır.

    Gerçek vaka: fincan görselinde YOLO "vazo" (0.85), CLIP "fincan" (0.76).
    """
    classifier = FakeClassifier(CATALOG, [_clip("fincan", confidence=0.99)])
    tagger = HybridTagger(classifier)  # type: ignore[arg-type]

    result = tagger.relabel(FRAME, [_yolo("kitap")])

    assert result[0].label == "fincan"
    assert tagger.stats.clip_used == 1


def test_yolo_kept_when_clip_not_more_confident() -> None:
    """CLIP daha az güvenliyse YOLO etiketi korunur."""
    classifier = FakeClassifier(CATALOG, [_clip("fincan", confidence=0.60)])
    tagger = HybridTagger(classifier)  # type: ignore[arg-type]

    result = tagger.relabel(FRAME, [_yolo("kitap")])  # YOLO güveni 0.8

    assert result[0].label == "kitap"
    assert tagger.stats.yolo_kept == 1


def test_clip_margin_can_require_larger_advantage() -> None:
    classifier = FakeClassifier(CATALOG, [_clip("fincan", confidence=0.85)])
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(clip_margin=0.2),  # 0.85 < 0.8 + 0.2
    )

    result = tagger.relabel(FRAME, [_yolo("kitap")])

    assert result[0].label == "kitap"


def test_low_clip_confidence_keeps_yolo_label() -> None:
    classifier = FakeClassifier(CATALOG, [_clip("kitap", confidence=0.10)])
    tagger = HybridTagger(classifier, HybridConfig(clip_override=0.45))  # type: ignore[arg-type]

    result = tagger.relabel(FRAME, [_yolo("vazo")])

    assert result[0].label == "vazo"  # eşik altı: YOLO korunur


def test_keep_yolo_when_in_catalog_can_be_disabled() -> None:
    classifier = FakeClassifier(CATALOG, [_clip("fincan", confidence=0.9)])
    tagger = HybridTagger(
        classifier,  # type: ignore[arg-type]
        HybridConfig(keep_yolo_when_in_catalog=False),
    )

    result = tagger.relabel(FRAME, [_yolo("kitap")])

    assert result[0].label == "fincan"


def test_missing_clip_result_keeps_yolo() -> None:
    classifier = FakeClassifier(CATALOG, [])
    tagger = HybridTagger(classifier)  # type: ignore[arg-type]

    result = tagger.relabel(FRAME, [_yolo("sandalye")])

    assert result[0].label == "sandalye"


def test_empty_input_and_disabled() -> None:
    classifier = FakeClassifier(CATALOG, [])
    tagger = HybridTagger(classifier)  # type: ignore[arg-type]

    assert tagger.relabel(FRAME, []) == []
    assert classifier.calls == 0

    disabled = HybridTagger(
        FakeClassifier(CATALOG, [_clip("kitap")]),  # type: ignore[arg-type]
        HybridConfig(enabled=False),
    )
    detections = [_yolo("vazo")]
    assert disabled.relabel(FRAME, detections) is detections


def test_sorting_by_confidence_desc() -> None:
    classifier = FakeClassifier(
        CATALOG,
        [
            _clip("kitap", box=(0, 0, 50, 50), confidence=0.5),
            _clip("fincan", box=(60, 60, 120, 120), confidence=0.95),
        ],
    )
    tagger = HybridTagger(classifier)  # type: ignore[arg-type]

    result = tagger.relabel(
        FRAME,
        [_yolo("vazo", (0, 0, 50, 50)), _yolo("vazo", (60, 60, 120, 120))],
    )

    assert [d.label for d in result] == ["fincan", "kitap"]


def test_create_tagger_returns_none_without_catalog() -> None:
    assert create_hybrid_tagger(None) is None


def test_create_tagger_returns_none_for_missing_file() -> None:
    assert create_hybrid_tagger("yok/olan/dosya.yaml") is None


def test_create_tagger_loads_catalog(tmp_path) -> None:  # noqa: ANN001
    file = tmp_path / "classes.yaml"
    file.write_text("names:\n  0: kitap\n  1: fincan\n", encoding="utf-8")

    tagger = create_hybrid_tagger(str(file))

    assert tagger is not None
    assert tagger.catalog.labels == ["kitap", "fincan"]
