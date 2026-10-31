"""Çizim ve HUD testleri (OpenCV gerekir; yoksa atlanır)."""

from __future__ import annotations

import numpy as np
import pytest
from app.detection import Detection
from app.state import AppStatus
from app.ui import latest_labels

cv2 = pytest.importorskip("cv2")

from app.detection.draw import draw_detections  # noqa: E402
from app.ui.overlay import draw_hud  # noqa: E402


def _frame(width: int = 320, height: int = 240) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_draw_detections_changes_frame_in_place() -> None:
    frame = _frame()
    detections = [
        Detection(0, "kişi", 0.92, (10, 20, 120, 200)),
        Detection(2, "araba", 0.55, (150, 60, 300, 180)),
    ]

    returned = draw_detections(frame, detections)

    assert returned is frame
    assert frame.shape == (240, 320, 3)
    assert cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 0


def test_draw_detections_clips_out_of_bounds_boxes() -> None:
    frame = _frame(80, 60)
    detections = [Detection(7, "kamyon", 0.4, (-40, -30, 500, 500))]

    draw_detections(frame, detections)  # istisna fırlatmamalı

    assert frame.any()


def test_draw_detections_noop_on_empty_list() -> None:
    frame = _frame()
    draw_detections(frame, [])
    assert not frame.any()


@pytest.mark.parametrize("status", list(AppStatus))
def test_draw_hud_all_statuses(status: AppStatus) -> None:
    frame = _frame(640, 360)

    draw_hud(
        frame,
        fps=29.7,
        status=status,
        detection_count=3,
        inference_ms=42.0,
        message="Masada bir dizüstü bilgisayar ve bir fincan görünüyor.",
    )

    assert frame.any()


def test_draw_hud_tiny_frame_does_not_crash() -> None:
    frame = _frame(32, 32)
    draw_hud(frame, fps=1.0, status=AppStatus.IDLE, message="kısa")
    assert frame.shape == (32, 32, 3)


def test_latest_labels_deduplicates_and_truncates() -> None:
    assert latest_labels([]) == ""
    assert latest_labels(["kişi", "kişi", "araba"]) == "kişi, araba"
    assert latest_labels(["a", "b", "c"], limit=2) == "a, b +1"
