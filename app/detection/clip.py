"""CLIP tabanlı açık kelime dağarcıklı sınıflandırıcı.

Sorun: COCO modeli yalnızca 80 sınıf bilir; ``su ısıtıcısı`` COCO'da yoktur,
``kitap`` ise dik/yatık duruşlarda kaçırılır (ölçüldü: 0 tespit).

Çözüm: YOLO'nun sınıf bilgisine güvenmek yerine, kutu içindeki görüntüyü
CLIP ile **kullanıcının verdiği sınıf listesine** göre sınıflandırmak. Böylece:

    * Görsel toplamaya/eğitime gerek yok (kullanıcı sadece ad listesi yazar).
    * COCO'da olmayan sınıflar tanınır (kettle, ataş, hesap makinesi …).
    * Türkçe adlar İngilizce istemlere çevrilir (CLIP İngilizce eğitildi).

Mimari (iki aşamalı algılama):
    1. **YOLO** kutuları bulur (sınıfı umursamadan, class-agnostic).
    2. **CLIP** her kutuyu katalogdaki sınıflara göre puanlar.

Bu, "nesne nerede?" sorusunu YOLO'ya, "ne?" sorusunu CLIP'e verir. İkisi de
sabit maliyetlidir; eğitim gerekmez.

Dürüst sınır: CLIP küçük/örtüşen nesnelerde ve ince ayrımlarda (bardak vs
fincan) zayıftır. Kutu kalitesi YOLO'ya bağlıdır.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.catalog import ClassCatalog
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Desteklenen CLIP mimarileri (ilk yükleme ağırlık indirir).
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class ClassifierError(RuntimeError):
    """CLIP modeli yüklenemedi veya sınıflandırma başarısız."""


@dataclass(frozen=True, slots=True)
class TaggedDetection:
    """CLIP ile etiketlenmiş tespit."""

    box: tuple[int, int, int, int]
    label: str
    confidence: float
    index: int
    alternatives: tuple[tuple[str, float], ...] = ()

    @property
    def center(self) -> tuple[int, int]:
        return ((self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2)


class ClipClassifier:
    """Kutuları, katalog sınıflarına göre CLIP ile etiketler.

    Örnek::

        catalog = ClassCatalog.from_names(["su ısıtıcısı", "kitap", "fincan"])
        clf = ClipClassifier(catalog)
        clf.load()                                    # ilk seferde indirir
        tagged = clf.tag(frame, boxes)                # [(x1,y1,x2,y2), ...]
    """

    def __init__(
        self,
        catalog: ClassCatalog | None = None,
        *,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
        min_confidence: float = 0.25,
        batch_size: int = 16,
    ) -> None:
        self.catalog = catalog or ClassCatalog()
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = device
        self.min_confidence = min_confidence
        self.batch_size = max(1, batch_size)

        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._text_features: Any = None
        self._lock = threading.Lock()
        self._load_ms = 0.0
        self._tag_ms = 0.0

    # -- model -------------------------------------------------------------

    def load(self) -> None:
        """Modeli ve metin özelliklerini (bir kez) hazırlar."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.catalog.entries:
                raise ClassifierError(
                    "Katalog boş. En az bir sınıf tanımlayın "
                    "(app/catalog.py veya datasets/classes.yaml)."
                )

            started = time.perf_counter()
            try:
                import open_clip  # noqa: PLC0415 - ağır bağımlılık
                import torch  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover
                raise ClassifierError(
                    "CLIP bağımlılıkları eksik. Kurulum: pip install open-clip-torch"
                ) from exc

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name, pretrained=self.pretrained
                )
                tokenizer = open_clip.get_tokenizer(self.model_name)
            except Exception as exc:  # noqa: BLE001 - indirme/ağ hataları
                raise ClassifierError(
                    f"CLIP modeli yüklenemedi ({self.model_name}/{self.pretrained}): {exc}"
                ) from exc

            model.eval()
            try:
                model = model.to(device)
            except Exception:  # noqa: BLE001 - CPU'da kalır
                device = "cpu"

            with torch.no_grad():
                tokens = tokenizer(self.catalog.prompts).to(device)
                features = model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._text_features = features
            self._device = device
            self._load_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "CLIP hazır: %s/%s, %s sınıf, cihaz=%s (%.0f ms)",
                self.model_name,
                self.pretrained,
                len(self.catalog),
                device,
                self._load_ms,
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_ms(self) -> float:
        return self._load_ms

    @property
    def tag_ms(self) -> float:
        """Son sınıflandırmanın süresi (ms)."""
        return self._tag_ms

    # -- sınıflandırma -----------------------------------------------------

    def tag(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        *,
        top_k: int = 2,
    ) -> list[TaggedDetection]:
        """Kutuları katalog sınıflarıyla etiketler.

        Args:
            frame: BGR kare.
            boxes: ``(x1, y1, x2, y2)`` piksel kutuları.
            top_k: Her kutu için döndürülecek alternatif sayısı.

        Returns:
            ``min_confidence`` ve sınıfın kendi eşiğini geçen etiketler.
        """
        if not boxes:
            return []
        self.load()

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        started = time.perf_counter()
        height, width = frame.shape[:2]
        crops = []
        kept: list[tuple[int, int, int, int]] = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, min(int(x1), width - 1))
            y1 = max(0, min(int(y1), height - 1))
            x2 = max(0, min(int(x2), width))
            y2 = max(0, min(int(y2), height))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = frame[y1:y2, x1:x2]
            # OpenCV BGR -> RGB PIL
            crops.append(Image.fromarray(crop[:, :, ::-1]))
            kept.append((x1, y1, x2, y2))

        if not crops:
            return []

        results: list[TaggedDetection] = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch = crops[start : start + self.batch_size]
                batch_boxes = kept[start : start + self.batch_size]
                tensor = torch.stack([self._preprocess(image) for image in batch]).to(self._device)
                image_features = self._model.encode_image(tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                probs = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)

                for row, box in zip(probs, batch_boxes, strict=False):
                    order = torch.argsort(row, descending=True)[: max(1, top_k)]
                    candidates = [
                        (self.catalog.entries[int(i)].tr, float(row[int(i)])) for i in order
                    ]
                    best_label, best_score = candidates[0]
                    entry = next(
                        (item for item in self.catalog.entries if item.tr == best_label),
                        None,
                    )
                    threshold = max(
                        self.min_confidence,
                        entry.min_confidence if entry else self.min_confidence,
                    )
                    if best_score < threshold:
                        continue
                    results.append(
                        TaggedDetection(
                            box=box,
                            label=best_label,
                            confidence=best_score,
                            index=entry.index if entry else -1,
                            alternatives=tuple(candidates[1:]),
                        )
                    )

        self._tag_ms = (time.perf_counter() - started) * 1000.0
        return results

    def close(self) -> None:
        self._model = None
        self._text_features = None
        self._preprocess = None
        self._tokenizer = None


def boxes_from_detections(detections: Any) -> list[tuple[int, int, int, int]]:
    """``Detection`` listesinden yalnızca kutuları çıkarır (sınıf adı önemsiz)."""
    return [tuple(detection.box) for detection in detections]
