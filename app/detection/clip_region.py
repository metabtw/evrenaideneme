"""Bölge taramalı CLIP algılayıcı: kutu üretici olarak CLIP.

Neden gerekli?
    Hibrit testinde (``tools/bench_hybrid.py``) ortaya çıktı: YOLO, COCO'da
    olmayan nesneler için **hiç kutu üretmiyor**. Kutu yoksa CLIP'in
    sınıflandıracağı kırpım da yok → hibrit 2/6'da kalıyor.

    CLIP tek başına konum vermez, ama **bölge önerileri** üretilebilir:
    tüm kare, dört çeyrek, merkez kırpım, üçte birler. Her bölge katalog
    sınıflarına göre puanlanır; eşiği geçen en iyi bölgeler kutu olarak döner.

Bu, "zavallı adamın açık-kelimeli dedektörü"dür: eğitim yok, indirme yok,
sadece CLIP + geometri. Sınırları dürüstçe:

    * Birden çok küçük nesne aynı karede ise zayıf (bölge ızgarası kaba).
    * Kutu sınırları yaklaşıktır (gerçek nesne sınırı değil, bölge sınırı).
    * Sahne başına ~10 CLIP kırpımı maliyeti var (CPU'da ~150-300 ms).

Bu yüzden üretimde **YOLO öncelikli**, CLIP bölge taraması yalnızca YOLO boş
döndüğünde veya periyodik olarak (her N kare) devreye girer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.catalog import ClassCatalog
from app.detection.clip import ClassifierError, ClipClassifier
from app.detection.detector import Detection
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Region:
    """Aday bölge (piksel)."""

    box: tuple[int, int, int, int]
    name: str


def propose_regions(width: int, height: int, *, include_thirds: bool = True) -> list[Region]:
    """Kare için aday bölgeler üretir (merkez ağırlıklı).

    Sıra önemlidir: büyükten küçüğe. Aynı nesne birden fazla bölgede
    yakalanırsa en büyük skorlu bölge kazanır (``dedupe``).
    """
    if width <= 0 or height <= 0:
        return []

    regions: list[Region] = [
        Region((0, 0, width, height), "tam"),
        Region(
            (int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)),
            "ic",
        ),
        Region(
            (int(width * 0.2), int(height * 0.15), int(width * 0.8), int(height * 0.85)),
            "merkez",
        ),
    ]

    # Dört çeyrek ve alt/üst yarımlar (yakın nesneler genelde altta/ortada).
    quarters = [
        (0, 0, width // 2, height // 2, "sol-ust"),
        (width // 2, 0, width, height // 2, "sag-ust"),
        (0, height // 2, width // 2, height, "sol-alt"),
        (width // 2, height // 2, width, height, "sag-alt"),
        (0, height // 3, width, height, "alt-2/3"),
    ]
    regions.extend(Region(box[:4], box[4]) for box in quarters)

    if include_thirds:
        third = width // 3
        regions.extend(
            [
                Region((0, 0, third * 2, height), "sol-2/3"),
                Region((third, 0, width, height), "sag-2/3"),
            ]
        )

    return [
        region
        for region in regions
        if region.box[2] - region.box[0] >= 32 and region.box[3] - region.box[1] >= 32
    ]


def dedupe_detections(
    detections: list[Detection], *, iou_threshold: float = 0.6
) -> list[Detection]:
    """Aynı sınıftan örtüşen kutuları tekilleştirir (greedy NMS)."""
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if any(
            existing.label == candidate.label and _iou(existing.box, candidate.box) > iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ClipRegionDetector:
    """Bölge taramasıyla kutu + etiket üreten CLIP dedektörü."""

    def __init__(
        self,
        classifier: ClipClassifier,
        *,
        min_confidence: float = 0.30,
        include_thirds: bool = True,
    ) -> None:
        self.classifier = classifier
        self.min_confidence = min_confidence
        self.include_thirds = include_thirds
        self._last_ms = 0.0

    @property
    def catalog(self) -> ClassCatalog:
        return self.classifier.catalog

    @property
    def last_ms(self) -> float:
        """Son taramanın süresi (ms)."""
        return self._last_ms

    def detect(self, frame: np.ndarray, *, top_k: int = 1) -> list[Detection]:
        """Karedeki en olası nesneleri bölge taramasıyla bulur."""
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]
        regions = propose_regions(width, height, include_thirds=self.include_thirds)
        if not regions:
            return []

        started = time.perf_counter()
        try:
            tagged = self.classifier.tag(frame, [region.box for region in regions])
        except ClassifierError as exc:
            logger.warning("CLIP bölge taraması başarısız: %s", exc)
            return []

        candidates = [
            Detection(
                class_id=item.index,
                label=item.label,
                confidence=item.confidence,
                box=item.box,
            )
            for item in tagged
            if item.confidence >= self.min_confidence
        ]
        self._last_ms = (time.perf_counter() - started) * 1000.0

        # Aynı sınıftan örtüşen bölgeleri tekilleştir, sonra en iyileri al.
        merged = dedupe_detections(candidates, iou_threshold=0.5)
        merged.sort(key=lambda item: item.confidence, reverse=True)
        return merged[: max(1, top_k * 3)]
