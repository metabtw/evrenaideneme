"""Tespit kutularını kareye çizen saf görselleştirme fonksiyonları.

Bu modül yalnızca çizim yapar; tespit hesaplamaz (bkz. :mod:`app.detection.detector`).
OpenCV içe aktarımı çağrı anındadır, böylece modül başsız ortamlarda da
içe aktarılabilir.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.detection.detector import Detection
from app.utils.text import to_ascii

#: Yüksek kontrastlı BGR paleti; sınıf kimliğine göre döngüsel seçilir.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 200, 0),
    (255, 120, 0),
    (0, 180, 255),
    (255, 0, 180),
    (0, 255, 255),
    (255, 80, 80),
    (180, 255, 0),
    (255, 200, 0),
)


def _color_for(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    detections: Iterable[Detection],
    *,
    show_confidence: bool = True,
    show_labels: bool = True,
    thickness: int | None = None,
    font_scale: float | None = None,
) -> np.ndarray:
    """Tespitleri kare üzerine çizer ve aynı kareyi döndürür.

    Kutu ve etiket arka planı, kare kenarlarından taşmayacak şekilde kırpılır.
    Türkçe karakterler OpenCV'de bozuk görüneceği için etiketler ASCII'ye çevrilir.
    """
    try:
        import cv2  # noqa: PLC0415 - ağır bağımlılık
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise RuntimeError("OpenCV bulunamadı. Kurulum: pip install -r requirements.txt") from exc

    height, width = frame.shape[:2]
    if thickness is None:
        thickness = max(1, round(min(width, height) / 400))
    if font_scale is None:
        font_scale = max(0.4, min(width, height) / 1280)

    for detection in detections:
        color = _color_for(detection.class_id)
        x1, y1, x2, y2 = detection.box
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width - 1))
        y2 = max(1, min(y2, height - 1))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if not (show_labels or show_confidence):
            continue

        text_parts: list[str] = []
        if show_labels:
            text_parts.append(to_ascii(detection.label))
        if show_confidence:
            text_parts.append(f"{detection.confidence:.0%}")
        text = " ".join(text_parts)

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        label_bottom = y1 - 4
        if label_bottom - text_height - baseline < 0:
            # Kutunun üstünde yer yoksa etiketi kutunun içine yaz.
            label_bottom = y1 + text_height + baseline + 4

        box_top = max(0, label_bottom - text_height - baseline)
        box_right = min(width - 1, x1 + text_width + 4)
        cv2.rectangle(frame, (x1, box_top), (box_right, label_bottom), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 2, label_bottom - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return frame
