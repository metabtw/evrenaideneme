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

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    #: CLIP'in çalışma aralığı (saniye). Ölçüm: bölge taraması 470 ms sürüyor;
    #: her karede çalıştırılırsa akış 2 FPS'e düşer. Bu yüzden CLIP seyrek
    #: çalışır, sonuç önbelleğe alınır ve YOLO her karede akmaya devam eder.
    min_interval: float = 1.0
    #: Ölçülen CLIP süresi bu bütçeyi aşarsa aralık otomatik büyür.
    budget_ms: float = 400.0


@dataclass(frozen=True, slots=True)
class HybridStats:
    """Hibrit etiketleyicinin sayaçları."""

    frames: int = 0
    clip_used: int = 0
    yolo_kept: int = 0
    clip_failed: int = 0
    region_scans: int = 0
    skipped: int = 0
    last_clip_ms: float = 0.0
    interval: float = 0.0


class HybridTagger:
    """YOLO tespitlerini CLIP ile iyileştirir; YOLO boşsa bölge taraması yapar.

    **Performans kuralı:** CLIP pahalıdır (bölge taraması ~470 ms, CPU). Her
    karede çalıştırılırsa akış 2 FPS'e düşer (ölçüldü). Bu yüzden CLIP kendi
    temposunda çalışır (``min_interval``), sonuçlar önbelleğe alınır ve
    YOLO her karede akmaya devam eder. Aralık, ölçülen CLIP süresine göre
    otomatik büyür (görev döngüsü ≤ ~1/3).
    """

    def __init__(
        self,
        classifier: ClipClassifier,
        config: HybridConfig | None = None,
        *,
        region_detector: ClipRegionDetector | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.classifier = classifier
        self.config = config or HybridConfig()
        self.region_detector = region_detector
        self._clock = clock or time.monotonic
        self._stats = HybridStats()
        self._catalog_labels = {entry.tr for entry in classifier.catalog}

        self._last_run = float("-inf")
        self._interval = max(0.0, self.config.min_interval)
        #: Son CLIP turunda üretilen tespitler (bölge taraması için).
        self._cache: list[Detection] = []
        #: Son CLIP turunda kutulara atanan etiketler (hızlı yol için).
        self._label_cache: list[Detection] = []

    @property
    def stats(self) -> HybridStats:
        return self._stats

    @property
    def catalog(self) -> ClassCatalog:
        return self.classifier.catalog

    @property
    def interval(self) -> float:
        """CLIP'in etkin çalışma aralığı (saniye)."""
        return self._interval

    def _should_run(self, now: float) -> bool:
        return (now - self._last_run) >= self._interval

    def _note_duration(self, elapsed_ms: float) -> None:
        """Görev döngüsünü sınırlamak için aralığı uyarlar."""
        base = max(0.0, self.config.min_interval)
        # CLIP süresi bütçeyi aşıyorsa aralığı 3× süreye çıkar (döngü ≤ ~%33).
        if elapsed_ms > self.config.budget_ms:
            self._interval = max(base, elapsed_ms / 1000.0 * 3.0)
        else:
            self._interval = base

    def _apply_cached_labels(self, detections: list[Detection]) -> list[Detection]:
        """Önbellekteki etiketleri güncel kutulara IoU ile uygular."""
        if not self._label_cache:
            return detections

        result: list[Detection] = []
        for detection in detections:
            match = _best_label_match(detection.box, self._label_cache)
            result.append(match if match is not None else detection)
        return result

    def _region_fallback(
        self, frame: np.ndarray, detections: list[Detection], now: float
    ) -> list[Detection]:
        """YOLO boş döndüğünde bölge taramasıyla kutu üretir (pahalı yol)."""
        assert self.region_detector is not None
        if not self._should_run(now):
            # Henüz sıra gelmedi: son sonucu tekrar kullan (titremeyi önler).
            return list(self._cache) if self._cache else detections

        found = self.region_detector.detect(frame)
        elapsed_ms = self.region_detector.last_ms
        self._last_run = now
        self._note_duration(elapsed_ms)
        self._cache = found
        self._label_cache = found

        self._stats = HybridStats(
            frames=self._stats.frames + 1,
            clip_used=self._stats.clip_used + len(found),
            yolo_kept=self._stats.yolo_kept,
            clip_failed=self._stats.clip_failed,
            region_scans=self._stats.region_scans + 1,
            skipped=self._stats.skipped,
            last_clip_ms=elapsed_ms,
            interval=self._interval,
        )
        if found:
            logger.debug(
                "YOLO boş; bölge taraması %d nesne buldu (%.0f ms, aralık %.1f sn)",
                len(found),
                elapsed_ms,
                self._interval,
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
        if not self.config.enabled:
            return detections

        now = self._clock()

        if not detections:
            # YOLO hiç kutu bulamadı: bölge taramasına düş (kettle/kitap senaryosu).
            if self.region_detector is not None:
                return self._region_fallback(frame, detections, now)
            return detections

        if not self._should_run(now):
            # CLIP sırası gelmedi: önbellekteki etiketleri uygula (hızlı yol).
            self._stats = _replace(
                self._stats, frames=self._stats.frames + 1, skipped=self._stats.skipped + 1
            )
            return self._apply_cached_labels(detections)

        started = time.perf_counter()
        frames = self._stats.frames + 1
        try:
            tagged = self.classifier.tag(frame, [d.box for d in detections])
        except ClassifierError as exc:
            logger.warning("CLIP etiketleme başarısız, YOLO etiketleri korunuyor: %s", exc)
            self._stats = _replace(
                self._stats, frames=frames, clip_failed=self._stats.clip_failed + 1
            )
            return detections

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._last_run = now
        self._note_duration(elapsed_ms)

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
            skipped=self._stats.skipped,
            last_clip_ms=elapsed_ms,
            interval=self._interval,
        )
        # Etiket önbelleği: CLIP kararlarını sonraki karelerde yeniden kullan.
        self._label_cache = result
        self._cache = result
        result = list(result)
        result.sort(key=lambda item: item.confidence, reverse=True)
        return result


def _replace(stats: HybridStats, **changes: object) -> HybridStats:
    data = {
        "frames": stats.frames,
        "clip_used": stats.clip_used,
        "yolo_kept": stats.yolo_kept,
        "clip_failed": stats.clip_failed,
        "region_scans": stats.region_scans,
        "skipped": stats.skipped,
        "last_clip_ms": stats.last_clip_ms,
        "interval": stats.interval,
    }
    data.update(changes)
    return HybridStats(**data)  # type: ignore[arg-type]


def _best_label_match(
    box: tuple[int, int, int, int],
    cache: list[Detection],
    *,
    iou_threshold: float = 0.5,
) -> Detection | None:
    """Önbellekten en iyi örtüşen etiketi bulur (güncel kutuyla)."""
    best: Detection | None = None
    best_iou = 0.0
    for cached in cache:
        overlap = _iou(box, cached.box)
        if overlap >= iou_threshold and overlap > best_iou:
            best = cached
            best_iou = overlap
    if best is None:
        return None
    return Detection(
        class_id=best.class_id,
        label=best.label,
        confidence=best.confidence,
        box=box,
    )


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


def create_hybrid_tagger(
    catalog_path: str | None = None,
    *,
    clip_override: float = DEFAULT_CLIP_OVERRIDE,
    min_confidence: float = 0.25,
    region_fallback: bool = True,
    min_interval: float = 1.0,
    background: bool = True,
) -> Any:
    """Katalog varsa hibrit etiketleyici kurar, yoksa ``None``.

    Args:
        background: ``True`` (varsayılan) ise CLIP kendi thread'inde çalışır ve
            ``relabel`` **hiç beklemez**. Ölçüm: senkron modda p95 512 ms
            donma vardı; thread'e alınca ana döngü ~50 ms'de kalıyor.

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
    tagger = HybridTagger(
        classifier,
        HybridConfig(clip_override=clip_override, min_interval=min_interval),
        region_detector=region_detector,
    )
    logger.info(
        "Hibrit etiketleyici hazır: %d sınıf (%s), bölge taraması=%s, arka plan=%s",
        len(catalog),
        catalog_path,
        "açık" if region_fallback else "kapalı",
        "açık" if background else "kapalı",
    )
    if not background:
        return tagger

    from app.detection.hybrid_async import AsyncHybridTagger  # noqa: PLC0415

    return AsyncHybridTagger(tagger, min_interval=min_interval).start()
