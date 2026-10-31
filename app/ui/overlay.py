"""HUD / overlay: FPS, durum rozeti, tespit sayacı ve mesaj paneli."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.state import AppStatus
from app.utils.text import to_ascii, wrap_text

_STATUS_COLORS: dict[AppStatus, tuple[int, int, int]] = {
    AppStatus.IDLE: (150, 150, 150),
    AppStatus.CAPTURING: (0, 200, 0),
    AppStatus.ANALYZING: (0, 170, 255),
    AppStatus.SPEAKING: (255, 0, 200),
    AppStatus.ERROR: (0, 0, 255),
}

_FONT = None  # cv2.FONT_HERSHEY_SIMPLEX, çağrı anında doldurulur


def _cv2():
    global _FONT
    try:
        import cv2  # noqa: PLC0415 - ağır bağımlılık
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise RuntimeError("OpenCV bulunamadı. Kurulum: pip install -r requirements.txt") from exc
    if _FONT is None:
        _FONT = cv2.FONT_HERSHEY_SIMPLEX
    return cv2


def draw_hud(
    frame: np.ndarray,
    *,
    fps: float = 0.0,
    status: AppStatus = AppStatus.IDLE,
    detection_count: int = 0,
    inference_ms: float = 0.0,
    message: str = "",
    show_fps: bool = True,
) -> np.ndarray:
    """Kareye durum bilgisini çizer ve aynı kareyi döndürür.

    Args:
        frame: BGR kare (yerinde değiştirilir).
        fps: Ölçülen akış hızı.
        status: Durum makinesi değeri; rozet rengi buradan gelir.
        detection_count: Son karedeki tespit sayısı.
        inference_ms: Son çıkarım süresi (ms).
        message: Alt panelde gösterilecek metin (ör. VLM yanıtı, hata).
        show_fps: FPS göstergesini çiz.
    """
    cv2 = _cv2()
    height, width = frame.shape[:2]
    font = _FONT
    scale = max(0.45, min(width, height) / 1400.0)
    thickness = max(1, round(min(width, height) / 500))

    # --- sol üst: FPS + tespit özeti -------------------------------------
    top_lines: list[tuple[str, tuple[int, int, int]]] = []
    if show_fps:
        top_lines.append((f"FPS {fps:5.1f}", (255, 255, 255)))
    top_lines.append((f"Nesne: {detection_count}", (200, 255, 200)))
    if inference_ms > 0:
        top_lines.append((f"Tespit: {inference_ms:4.0f} ms", (200, 220, 255)))

    y = int(24 * scale + 8)
    for text, color in top_lines:
        cv2.putText(frame, text, (12, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), font, scale, color, thickness, cv2.LINE_AA)
        y += int(26 * scale + 6)

    # --- sağ üst: durum rozeti -------------------------------------------
    _draw_badge(
        frame,
        status.ascii,
        color=_STATUS_COLORS.get(status, (150, 150, 150)),
        font_scale=scale,
        thickness=thickness,
    )

    # --- alt panel: serbest metin ----------------------------------------
    if message:
        _draw_message_panel(frame, message, font_scale=scale, thickness=thickness)

    return frame


def _draw_badge(
    frame: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    text = to_ascii(text)

    (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    padding_x = int(10 * font_scale + 4)
    padding_y = int(6 * font_scale + 2)
    box_width = text_width + 2 * padding_x
    box_height = text_height + baseline + 2 * padding_y
    x2 = width - 1
    y1 = 10
    x1 = max(0, x2 - box_width)

    cv2.rectangle(frame, (x1, y1), (x2, y1 + box_height), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + padding_x, y1 + padding_y + text_height),
        _FONT,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _draw_message_panel(
    frame: np.ndarray,
    message: str,
    *,
    font_scale: float,
    thickness: int,
) -> None:
    cv2 = _cv2()
    height, width = frame.shape[:2]

    char_width = max(12, int(11 * font_scale * 1.05))
    max_chars = max(20, width // char_width)
    lines = wrap_text(to_ascii(message), max_chars)[:4]

    line_height = int(22 * font_scale + 6)
    padding = int(10 * font_scale + 6)
    panel_height = line_height * len(lines) + 2 * padding
    panel_top = max(0, height - panel_height - 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, panel_top), (width - 8, height - 8), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = panel_top + padding + int(line_height * 0.7)
    for line in lines:
        cv2.putText(
            frame,
            line,
            (8 + padding, y),
            _FONT,
            font_scale,
            (235, 235, 235),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def latest_labels(labels: Iterable[str], limit: int = 5) -> str:
    """Etiketleri tek satırda özetler (``kişi, araba, ...``)."""
    items = list(dict.fromkeys(labels))
    if not items:
        return ""
    head = ", ".join(items[:limit])
    if len(items) > limit:
        head += f" +{len(items) - limit}"
    return head
