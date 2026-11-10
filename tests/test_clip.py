"""CLIP sınıflandırıcı testleri (model gerektirmeyen kısımlar).

Gerçek CLIP ağırlıkları indirilmeden de sözleşme doğrulanabilir: katalog
doğrulaması, boş girdi, eşik filtreleme mantığı. Model yükleme ayrı test edilir
(``pytest.mark.skipif``).
"""

from __future__ import annotations

import numpy as np
import pytest
from app.catalog import ClassCatalog
from app.detection.clip import ClassifierError, ClipClassifier, boxes_from_detections
from app.detection.detector import Detection


def _clip_available() -> bool:
    """Yalnızca bağımlılık **ve** önbellekte ağırlık varsa test çalışır.

    Aksi halde CI'da test ağdan yüzlerce MB indirmeye kalkar.
    """
    try:
        import open_clip  # noqa: F401, PLC0415
        import torch  # noqa: F401, PLC0415
    except ImportError:
        return False

    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    if os.environ.get("EVRENAID_CLIP_TEST") == "1":
        return True
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    return any(cache.glob("models--laion--CLIP-ViT-B-32*"))


def test_empty_catalog_raises_on_load() -> None:
    classifier = ClipClassifier(ClassCatalog())
    with pytest.raises(ClassifierError, match="Katalog boş"):
        classifier.load()


def test_tag_without_boxes_returns_empty() -> None:
    classifier = ClipClassifier(ClassCatalog.from_names(["kitap"]))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    assert classifier.tag(frame, []) == []
    assert classifier.is_loaded is False  # boş kutu modeli yüklemez


def test_tiny_boxes_are_skipped_before_inference() -> None:
    """8×8'den küçük kutular CLIP'e verilmez (anlamsız kırpma)."""
    classifier = ClipClassifier(ClassCatalog.from_names(["kitap"]))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    # Bağımlılık yoksa load() hata verir; varsa kırpma filtresi boş liste döner.
    try:
        results = classifier.tag(frame, [(0, 0, 5, 5)])
    except ClassifierError:
        return
    assert results == []


def test_boxes_from_detections_drops_labels() -> None:
    detections = [
        Detection(0, "kişi", 0.9, (10, 20, 30, 40)),
        Detection(63, "laptop", 0.8, (50, 60, 90, 100)),
    ]

    boxes = boxes_from_detections(detections)

    assert boxes == [(10, 20, 30, 40), (50, 60, 90, 100)]


def test_catalog_prompts_are_exposed() -> None:
    catalog = ClassCatalog.from_names(["su ısıtıcısı", "kitap"])
    classifier = ClipClassifier(catalog)

    assert classifier.catalog.labels == ["su ısıtıcısı", "kitap"]
    assert len(classifier.catalog.prompts) == 2


def test_close_is_idempotent() -> None:
    classifier = ClipClassifier(ClassCatalog.from_names(["kitap"]))
    classifier.close()
    classifier.close()
    assert classifier.is_loaded is False


@pytest.mark.skipif(not _clip_available(), reason="CLIP bağımlılığı yok")
def test_real_clip_tags_a_block() -> None:  # pragma: no cover - ağırlık ister
    """Model varsa: kutuyu en az bir katalog sınıfıyla etiketler."""
    classifier = ClipClassifier(ClassCatalog.from_names(["kitap", "fincan", "su ısıtıcısı"]))
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    frame[60:180, 80:240] = (30, 30, 140)

    results = classifier.tag(frame, [(80, 60, 240, 180)])

    assert results
    assert results[0].label in classifier.catalog.labels
    assert 0.0 <= results[0].confidence <= 1.0
