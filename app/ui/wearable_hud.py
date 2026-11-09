"""Giyilebilir HUD: gözlük ekranı için minimal, yüksek kontrastlı gösterge.

Tasarım kuralları (gözlük bağlamı):
    * Küçük, yarı saydam ekran → az bilgi, büyük yazı, yüksek kontrast.
    * Kullanıcı yürürken bakar → kritik uyarı kırmızı ve kalın.
    * Mod rozeti her zaman görünür: ``OTOPILOT`` / ``DONDURULDU``.
"""

from __future__ import annotations

import numpy as np

from app.modes.deep import DeepAnalysisState
from app.spatial import ProximityLevel, SpatialReport
from app.utils.text import to_ascii
from app.wearable import ModeName

_LEVEL_COLORS = {
    ProximityLevel.SAFE: (140, 140, 140),
    ProximityLevel.NEAR: (0, 190, 255),
    ProximityLevel.CRITICAL: (0, 0, 255),
}


def _cv2():
    try:
        import cv2  # noqa: PLC0415 - ağır bağımlılık
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV bulunamadı. Kurulum: pip install -r requirements.txt") from exc
    return cv2


def draw_wearable_hud(
    frame: np.ndarray,
    *,
    mode: ModeName,
    report: SpatialReport | None = None,
    fps: float = 0.0,
    tts_busy: bool = False,
    tts_text: str = "",
    deep_state: DeepAnalysisState | None = None,
    frozen: bool = False,
) -> np.ndarray:
    """Gözlük görünümünü çizer ve aynı kareyi döndürür (yerinde)."""
    cv2 = _cv2()
    height, width = frame.shape[:2]

    scale = max(0.5, min(width, height) / 900.0)
    thickness = max(1, round(min(width, height) / 450))

    # --- üst sol: mod rozeti + FPS ---------------------------------------
    is_critical = report.is_critical if report is not None else False
    badge_text = "DONDURULDU" if frozen or mode is ModeName.DEEP else "OTOPILOT"
    badge_color = (0, 170, 255) if (frozen or mode is ModeName.DEEP) else (0, 190, 0)

    _pill(frame, badge_text, 10, 10, badge_color, scale * 1.05, thickness)

    fps_text = to_ascii(f"FPS {fps:4.1f}")
    _text(frame, fps_text, 14, int(56 * scale + 10), scale, thickness, (230, 230, 230))

    # --- alt orta: ses durumu --------------------------------------------
    if deep_state is not None and deep_state.active:
        _text(
            frame,
            to_ascii("ANALIZ EDILIYOR..."),
            int(width * 0.30),
            int(height * 0.55),
            scale * 1.1,
            thickness + 1,
            (0, 220, 255),
        )
    elif tts_busy and tts_text:
        _caption(frame, tts_text, scale, thickness)

    # --- orta sağ: yakınlık göstergesi -----------------------------------
    if report is not None and report.nearest is not None:
        nearest = report.nearest
        color = _LEVEL_COLORS[nearest.level]
        bar_x2 = width - 12
        bar_x1 = int(width * 0.62)
        bar_y = int(height * 0.12)
        bar_h = int(16 * scale + 6)

        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (40, 40, 40), -1)
        filled = int((bar_x2 - bar_x1) * min(1.0, nearest.vertical_ratio))
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + filled, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (220, 220, 220), thickness)

        _text(
            frame,
            to_ascii(f"{nearest.label} %{nearest.vertical_ratio * 100:.0f}"),
            bar_x1,
            bar_y + bar_h + int(24 * scale + 6),
            scale,
            thickness,
            color,
        )

        if is_critical:
            # Kritik uyarı: kutu kalın kırmızı + üst şerit.
            x1, y1, x2, y2 = nearest.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness * 3)
            cv2.rectangle(frame, (0, 0), (width - 1, int(8 * scale + 4)), (0, 0, 255), -1)

    return frame


def _pill(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    cv2 = _cv2()
    (text_width, text_height), baseline = cv2.getTextSize(
        to_ascii(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    pad_x, pad_y = int(12 * font_scale), int(7 * font_scale)
    cv2.rectangle(
        frame,
        (x, y),
        (x + text_width + 2 * pad_x, y + text_height + baseline + 2 * pad_y),
        color,
        -1,
    )
    cv2.putText(
        frame,
        to_ascii(text),
        (x + pad_x, y + pad_y + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
    color: tuple[int, int, int],
) -> None:
    cv2 = _cv2()
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _caption(
    frame: np.ndarray,
    text: str,
    font_scale: float,
    thickness: int,
) -> None:
    """Konuşulan metni alt şeritte gösterir (2 satır, kırpılmış)."""
    cv2 = _cv2()
    height, width = frame.shape[:2]
    chars = max(24, int(width / (11 * font_scale)))
    clean = to_ascii(" ".join(text.split()))
    lines = [clean[i : i + chars] for i in range(0, min(len(clean), chars * 2), chars)][:2]

    panel_h = int((24 * font_scale + 8) * len(lines) + 12)
    top = max(0, height - panel_h - 8)
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, top), (width - 6, height - 6), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = top + int(22 * font_scale + 6)
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )
        y += int(24 * font_scale + 8)
