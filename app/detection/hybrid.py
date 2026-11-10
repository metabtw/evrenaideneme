"""Hibrit etiketleyici: YOLO kutuları + CLIP sınıf kararı.

Ölçüme dayalı tasarım kararı (``tools/bench_clip_strategies.py``, 6 görsel):

    | Strateji                        | Doğru |
    |---------------------------------|-------|
    | YOLOv8n (COCO)                  |  0/6  |  <- kettle yok, kitap kaçıyor
    | CLIP tek istem, 16 sınıf        |  4/6  |
    | CLIP tek istem, vazosuz 15      |  5/6  |
    | CLIP, yalnız gerekli 6 sınıf    |  6/6  |

Yorum: CLIP **çok iyi** ama katalogda birbirine benzeyen sınıflar (vazo/fincan/
kadeh) birbirini çalıyor. YOLO ise büyük nesnelerin **kutusunu** ve COCO
sınıfını iyi buluyor (laptop, klavye, sandalye).

Bu yüzden ikisini birleştiriyoruz:

    1. YOLO kutu + COCO etiketi üretir (kutu kalitesi buradan).
    2. CLIP her kutuyu katalog sınıflarıyla puanlar.
    3. **Karar kuralı:**
       * CLIP güveni yüksekse (>= ``clip_override``) ve CLIP etiketi katalogda
         varsa → CLIP kazanır (kettle gibi COCO'da olmayan sınıflar buradan).
       * Aksi halde YOLO'nun COCO etiketi korunur (laptop/klavye gibi zaten
         doğru olan sınıflar bozulmaz).

Böylece "COCO'da olmayanı CLIP ekler, COCO'da doğru olanı YOLO korur."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.catalog import ClassCatalog
from app.detection.clip import ClassifierError, ClipClassifier
from app.detection.clip_region import ClipRegionDetector
from app.detection.detector import Detection
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Katalogda olmayan YOLO etiketlerini CLIP ile değiştirmek için gereken güven.
DEFAULT_CLIP_OVERRIDE = 0.45


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Hibrit etiketleme ayarları."""

    enabled: bool = True
    clip_override: float = DEFAULT_CLIP_OVERRIDE
    #: YOLO etiketi katalogda olsa bile CLIP bu kadar daha güvenliyse CLIP kazanır.
    #: Ölçüm: fincan görselinde YOLO "vazo" (0.85), CLIP "fincan" (0.76) —
    #: bu durumda YOLO'nun korunması yanlış sonuç veriyordu.
    clip_margin: float = 0.0
    keep_yolo_when_in_catalog: bool = True


@dataclass(frozen=True, slots=True)
class HybridStats:
    """Hibrit etiketleyicinin sayaçları."""

    frames: int = 0
    clip_used: int = 0
    yolo_kept: int = 0
    clip_failed: int = 0
    region_scans: int = 0


class HybridTagger:
    """YOLO tespitlerini CLIP ile iyileştirir; YOLO boşsa bölge taraması yapar."""

    def __init__(
        self,
        classifier: ClipClassifier,
        config: HybridConfig | None = None,
        *,
        region_detector: ClipRegionDetector | None = None,
    ) -> None:
        self.classifier = classifier
        self.config = config or HybridConfig()
        self.region_detector = region_detector
        self._stats = HybridStats()
        self._catalog_labels = {entry.tr for entry in classifier.catalog}

    @property
    def stats(self) -> HybridStats:
        return self._stats

    @property
    def catalog(self) -> ClassCatalog:
        return self.classifier.catalog

    def _region_fallback(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """YOLO boş döndüğünde bölge taramasıyla kutu üretir (pahalı yol)."""
        assert self.region_detector is not None
        found = self.region_detector.detect(frame)
        models = {
            "frames": self._stats.frames + 1,
            "clip_used": self._stats.clip_used + len(found),
            "yolo_kept": self._stats.yolo_kept,
            "clip_failed": self._stats.clip_failed,
            "region_scans": self._stats.region_scans + 1,
        }
        self._stats = HybridStats(**models)  # type: ignore[arg-type]
        if found:
            logger.debug(
                "YOLO boş; bölge taraması %d nesne buldu (%.0f ms)",
                len(found),
                self.region_detector.last_ms,
            )
        return found or detections

    def relabel(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Tespitleri hibrit kuralla yeniden etiketler.

        Args:
            frame: BGR kare.
            detections: YOLO tespitleri (kutu + COCO etiketi).

        Returns:
            Yeni :class:`Detection` listesi (girdi değiştirilmez).
        """
        if not detections or not self.config.enabled:
            # YOLO hiç kutu bulamadı: bölge taramasına düş (kettle/kitap senaryosu).
            if self.config.enabled and self.region_detector is not None and not detections:
                return self._region_fallback(frame, detections)
            return detections

        frames = self._stats.frames + 1
        try:
            tagged = self.classifier.tag(frame, [d.box for d in detections])
        except ClassifierError as exc:
            logger.warning("CLIP etiketleme başarısız, YOLO etiketleri korunuyor: %s", exc)
            self._stats = _replace(
                self._stats, frames=frames, clip_failed=self._stats.clip_failed + 1
            )
            return detections

        # Kutu -> CLIP sonucu
        by_box = {item.box: item for item in tagged}

        result: list[Detection] = []
        clip_used = self._stats.clip_used
        yolo_kept = self._stats.yolo_kept

        for detection in detections:
            clip = by_box.get(detection.box)
            yolo_in_catalog = detection.label in self._catalog_labels
            use_clip = False

            if clip is not None and clip.confidence >= self.config.clip_override:
                if not yolo_in_catalog:
                    # Katalogda olmayan YOLO etiketi: CLIP kararına bırak.
                    use_clip = True
                elif not self.config.keep_yolo_when_in_catalog:
                    use_clip = True
                elif clip.label != detection.label:
                    # İkisi de katalogda ama farklı sınıf söylüyor: hangisi daha
                    # güvenli? CLIP belirgin biçimde öndeyse onu kullan.
                    use_clip = clip.confidence >= detection.confidence + self.config.clip_margin

            if use_clip:
                clip_used += 1
                result.append(
                    Detection(
                        class_id=clip.index,
                        label=clip.label,
                        confidence=clip.confidence,
                        box=detection.box,
                    )
                )
            else:
                yolo_kept += 1
                result.append(detection)

        self._stats = HybridStats(
            frames=frames,
            clip_used=clip_used,
            yolo_kept=yolo_kept,
            clip_failed=self._stats.clip_failed,
            region_scans=self._stats.region_scans,
        )
        result.sort(key=lambda item: item.confidence, reverse=True)
        return result


def _replace(stats: HybridStats, **changes: object) -> HybridStats:
    data = {
        "frames": stats.frames,
        "clip_used": stats.clip_used,
        "yolo_kept": stats.yolo_kept,
        "clip_failed": stats.clip_failed,
        "region_scans": stats.region_scans,
    }
    data.update(changes)
    return HybridStats(**data)  # type: ignore[arg-type]


def create_hybrid_tagger(
    catalog_path: str | None = None,
    *,
    clip_override: float = DEFAULT_CLIP_OVERRIDE,
    min_confidence: float = 0.25,
    region_fallback: bool = True,
) -> HybridTagger | None:
    """Katalog varsa hibrit etiketleyici kurar, yoksa ``None``.

    Hata durumunda uygulama düşmez: CLIP kurulamazsa YOLO tek başına çalışır.
    """
    if not catalog_path:
        return None
    try:
        catalog = ClassCatalog.load(catalog_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Katalog okunamadı (%s): %s", catalog_path, exc)
        return None

    classifier = ClipClassifier(catalog, min_confidence=min_confidence)
    region_detector = (
        ClipRegionDetector(classifier, min_confidence=max(0.30, min_confidence))
        if region_fallback
        else None
    )
    logger.info(
        "Hibrit etiketleyici hazır: %d sınıf (%s), bölge taraması=%s",
        len(catalog),
        catalog_path,
        "açık" if region_fallback else "kapalı",
    )
    return HybridTagger(
        classifier,
        HybridConfig(clip_override=clip_override),
        region_detector=region_detector,
    )
