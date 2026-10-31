"""Kare kodlama yardımcıları (M2'de Vision LLM'e göndermek için hazır).

OpenCV içe aktarımı fonksiyon içinde yapılır; böylece bu modül OpenCV
kurulmadan da (ör. config testlerinde) içe aktarılabilir.
"""

from __future__ import annotations

import base64

import numpy as np


class FrameEncodingError(RuntimeError):
    """Kare JPEG'e çevrilemedi."""


def _cv2():
    try:
        import cv2  # noqa: PLC0415 - isteğe bağlı ağır bağımlılık
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise FrameEncodingError(
            "OpenCV bulunamadı. Kurulum: pip install -r requirements.txt"
        ) from exc
    return cv2


def resize_max_side(frame: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Uzun kenarı ``max_side`` pikseli geçmeyecek şekilde küçültür.

    Küçültme gerekmiyorsa aynı diziyi döndürür (kopya maliyeti yok).
    """
    if max_side <= 0:
        raise ValueError("max_side pozitif olmalı")
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame

    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    cv2 = _cv2()
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def encode_jpeg(
    frame: np.ndarray,
    quality: int = 85,
    *,
    max_side: int | None = 1024,
) -> bytes:
    """BGR kareyi JPEG baytlarına çevirir."""
    if not 1 <= quality <= 100:
        raise ValueError("quality 1-100 aralığında olmalı")
    if max_side:
        frame = resize_max_side(frame, max_side)

    cv2 = _cv2()
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise FrameEncodingError("Kare JPEG olarak kodlanamadı")
    return buffer.tobytes()


def encode_jpeg_base64(
    frame: np.ndarray,
    quality: int = 85,
    *,
    max_side: int | None = 1024,
) -> str:
    """Kareyi Ollama'nın beklediği base64 JPEG dizesine çevirir."""
    payload = encode_jpeg(frame, quality=quality, max_side=max_side)
    return base64.b64encode(payload).decode("ascii")
