"""YOLOv8 sarmalayıcı.

Sorumluluklar:
    * Modeli **tembel** ve **tek seferlik** yüklemek (thread-güvenli).
    * ``detect(frame)`` ile saf veri döndürmek (çizim yapmaz).
    * Ultralytics'e bağımlılığı bu dosyada izole etmek; testlerde sahte model
      enjekte edilebilsin diye ``model_factory`` parametresi sunmak.

Örnek::

    detector = ObjectDetector(settings.detection)
    detections = detector.detect(frame)
    for d in detections:
        print(d.label, d.confidence, d.box)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from app.config import DetectionSettings
from app.detection.labels import resolve_label
from app.utils.logging import get_logger

logger = get_logger(__name__)

ModelFactory = Callable[[str], Any]


class DetectorError(RuntimeError):
    """Model yüklenemedi veya tespit çalıştırılamadı."""


@dataclass(frozen=True, slots=True)
class Detection:
    """Tek bir tespit sonucu (çizimden bağımsız saf veri)."""

    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (piksel)

    @property
    def width(self) -> int:
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(0, self.box[3] - self.box[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "box": list(self.box),
            "center": list(self.center),
        }


class _BoxesLike(Protocol):
    xyxy: Any
    cls: Any
    conf: Any


class _ResultLike(Protocol):
    boxes: _BoxesLike
    names: dict[int, str]


class ObjectDetector:
    """Ultralytics YOLOv8 modelini saran tespit edici."""

    def __init__(
        self,
        settings: DetectionSettings | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.settings = settings or DetectionSettings()
        self._model_factory = model_factory or _default_model_factory
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._load_ms: float = 0.0

    # -- model -------------------------------------------------------------

    def load(self) -> None:
        """Modeli (gerekiyorsa) yükler. Birden fazla thread aynı anda çağırsa da
        model yalnızca bir kez yüklenir."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # çift kontrollü kilitleme
                return

            started = time.perf_counter()
            try:
                model = self._model_factory(self.settings.model_path)
                setter = getattr(model, "to", None)
                if setter is not None and self.settings.device_or_none:
                    setter(self.settings.device_or_none)
            except Exception as exc:  # noqa: BLE001 - ultralytics çok çeşitli hata verir
                raise DetectorError(
                    f"YOLO modeli yüklenemedi ({self.settings.model_path}). İlk çalıştırmada ağırlık "
                    f"indirilir; internet bağlantısını kontrol edin. Ayrıntı: {exc}"
                ) from exc

            self._model = model
            self._load_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "YOLO modeli yüklendi: %s (%.0f ms, cihaz=%s)",
                self.settings.model_path,
                self._load_ms,
                self.settings.device,
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_ms(self) -> float:
        return self._load_ms

    # -- tespit ------------------------------------------------------------

    def detect(self, frame: np.ndarray, *, class_agnostic: bool = False) -> list[Detection]:
        """Bir BGR karesinde nesne tespiti yapar.

        Args:
            frame: ``(H, W, 3)`` uint8 BGR kare.
            class_agnostic: ``True`` ise modelin sınıf etiketleri **yok sayılır**
                ve tüm kutular ``nesne`` etiketiyle döner. CLIP ile yeniden
                etiketleme (bkz. :class:`app.detection.clip.ClipClassifier`)
                için kullanılır: kutu kalitesi aynı kalır, sınıf kararı CLIP'e
                bırakılır.

        Returns:
            Güven eşiğini geçen tespitler; güvenli biçimde azalan sırada.

        Raises:
            DetectorError: Model yüklenemezse veya çıkarım başarısız olursa.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise DetectorError(f"Beklenen BGR kare (H, W, 3); alınan şekil: {frame.shape}")

        self.load()
        assert self._model is not None  # load() garanti eder

        try:
            results = self._model.predict(
                frame,
                conf=self.settings.confidence,
                iou=self.settings.iou,
                imgsz=self.settings.image_size,
                device=self.settings.device_or_none,
                max_det=self.settings.max_detections,
                classes=None if class_agnostic else self.settings.classes_or_none,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"Nesne tespiti başarısız: {exc}") from exc

        if not results:
            return []
        return self._parse_result(results[0], class_agnostic=class_agnostic)

    def _parse_result(
        self, result: _ResultLike, *, class_agnostic: bool = False
    ) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        raw_names = getattr(result, "names", {}) or {}
        names: dict[int, str] = (
            dict(enumerate(raw_names)) if isinstance(raw_names, (list, tuple)) else dict(raw_names)
        )
        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
        class_ids = _to_numpy(boxes.cls).astype(int).reshape(-1)
        confidences = _to_numpy(boxes.conf).astype(float).reshape(-1)

        detections: list[Detection] = []
        for coords, class_id, confidence in zip(xyxy, class_ids, confidences, strict=False):
            if confidence < self.settings.confidence:
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in coords)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    class_id=int(class_id),
                    label=(
                        "nesne"
                        if class_agnostic
                        else resolve_label(
                            int(class_id), names.get(int(class_id), f"sinif-{class_id}")
                        )
                    ),
                    confidence=float(confidence),
                    box=(x1, y1, x2, y2),
                )
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections


def _to_numpy(value: Any) -> np.ndarray:
    """Torch tensor / liste / ndarray -> ndarray (CPU)."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _default_model_factory(model_path: str) -> Any:
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - ağır bağımlılık
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise DetectorError(
            "ultralytics paketi bulunamadı. Kurulum: pip install -r requirements.txt"
        ) from exc
    return YOLO(model_path)
