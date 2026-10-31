"""CLI iskeleti testleri (kamera/model gerektirmez)."""

from __future__ import annotations

import pytest
from app.config import Settings
from app.main import _apply_overrides, build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.no_gui is False
    assert args.list_cameras is False
    assert args.duration == 0.0
    assert args.camera_index is None


def test_overrides_do_not_mutate_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_CAMERA_INDEX", "0")
    original = Settings.load(environ={})
    args = build_parser().parse_args(
        ["--no-gui", "--camera-index", "2", "--confidence", "0.6", "--model", "yolov8s.pt"]
    )

    updated = _apply_overrides(original, args)

    assert original.camera.index == 0
    assert original.detection.confidence != 0.6
    assert updated.camera.index == 2
    assert updated.detection.confidence == pytest.approx(0.6)
    assert updated.detection.model_path == "yolov8s.pt"


def test_invalid_confidence_exits() -> None:
    args = build_parser().parse_args(["--confidence", "2"])
    with pytest.raises(SystemExit):
        _apply_overrides(Settings.load(environ={}), args)
