"""SpatialAnalyzer testleri: %60 eşiği, bölge, debounce ve cooldown."""

from __future__ import annotations

from app.config import SpatialSettings
from app.detection import Detection
from app.spatial import ProximityLevel, SpatialAnalyzer, Zone


def _box(class_id: int = 0, label: str = "kişi", box: tuple[int, int, int, int] = (0, 0, 10, 10)):
    return Detection(class_id=class_id, label=label, confidence=0.9, box=box)


FRAME = (720, 1280, 3)  # 1280x720


def test_critical_when_box_covers_60_percent_of_height() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=1, hazard_only=False))
    # 720 * 0.60 = 432 px yükseklik
    detection = _box(box=(100, 100, 500, 532))

    report = analyzer.analyze([detection], FRAME)

    assert report.nearest is not None
    assert report.nearest.level is ProximityLevel.CRITICAL
    assert report.is_critical is True
    assert report.nearest.vertical_ratio >= 0.60


def test_just_below_threshold_is_near_not_critical() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=1, hazard_only=False))
    # 400/720 = %55.5 -> near (>=0.35), kritik değil
    report = analyzer.analyze([_box(box=(100, 100, 500, 500))], FRAME)

    assert report.nearest is not None
    assert report.nearest.level is ProximityLevel.NEAR
    assert report.is_critical is False


def test_small_far_object_is_ignored() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=1, hazard_only=False))
    # 100/720 = %13.9
    report = analyzer.analyze([_box(box=(10, 10, 120, 110))], FRAME)

    assert report.alerts == ()
    assert report.nearest is None


def test_zone_detection_left_center_right() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=1, hazard_only=False))
    left = _box(box=(0, 100, 300, 700))  # merkez x=150 < 426
    center = _box(box=(450, 100, 800, 700))  # merkez x=625
    right = _box(box=(1000, 100, 1280, 700))  # merkez x=1140 > 853

    assert analyzer.analyze([left], FRAME).nearest.zone is Zone.LEFT
    assert analyzer.analyze([center], FRAME).nearest.zone is Zone.CENTER
    assert analyzer.analyze([right], FRAME).nearest.zone is Zone.RIGHT


def test_confirm_frames_debounce() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=3, hazard_only=False))
    detection = [_box(box=(100, 100, 500, 600))]

    first = analyzer.analyze(detection, FRAME, now=100.0)
    second = analyzer.analyze(detection, FRAME, now=100.1)
    third = analyzer.analyze(detection, FRAME, now=100.2)

    assert first.announcements == ()
    assert second.announcements == ()
    assert len(third.announcements) == 1  # 3. karede onaylandı


def test_streak_resets_when_object_disappears() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=2, hazard_only=False))
    detection = [_box(box=(100, 100, 500, 600))]

    analyzer.analyze(detection, FRAME, now=1.0)
    analyzer.analyze([], FRAME, now=1.1)  # nesne kayboldu
    report = analyzer.analyze(detection, FRAME, now=1.2)

    assert report.announcements == ()  # sayaç sıfırlandı


def test_cooldown_prevents_repeat_announcements() -> None:
    analyzer = SpatialAnalyzer(
        SpatialSettings(confirm_frames=1, announce_cooldown=5.0, hazard_only=False)
    )
    detection = [_box(box=(100, 100, 500, 600))]

    assert len(analyzer.analyze(detection, FRAME, now=10.0).announcements) == 1
    assert analyzer.analyze(detection, FRAME, now=12.0).announcements == ()
    assert len(analyzer.analyze(detection, FRAME, now=16.0).announcements) == 1


def test_hazard_whitelist_filters_announcements() -> None:
    analyzer = SpatialAnalyzer(
        SpatialSettings(confirm_frames=1, hazard_only=True, hazard_classes=("araba",))
    )
    person = [_box(0, "kişi", (100, 100, 500, 600))]
    car = [_box(2, "araba", (100, 100, 500, 600))]

    assert analyzer.analyze(person, FRAME, now=1.0).announcements == ()
    assert len(analyzer.analyze(car, FRAME, now=1.1).announcements) == 1


def test_clipping_boxes_outside_frame() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=1, hazard_only=False))
    report = analyzer.analyze([_box(box=(-500, -500, 5000, 5000))], FRAME)

    assert report.nearest is not None
    assert report.nearest.level is ProximityLevel.CRITICAL
    assert report.nearest.vertical_ratio <= 1.0


def test_empty_detections_and_invalid_shape() -> None:
    analyzer = SpatialAnalyzer()

    assert analyzer.analyze([], FRAME).alerts == ()
    try:
        analyzer.analyze([], (720,))
    except ValueError as exc:
        assert "şekli" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ValueError bekleniyordu")


def test_message_is_turkish_and_actionable() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=1, hazard_only=False))
    report = analyzer.analyze([_box(0, "sandalye", (100, 100, 500, 600))], FRAME)

    message = report.nearest.message
    assert "sandalye" in message
    assert "yakın" in message or "yaklaşıyor" in message


def test_reset_clears_state() -> None:
    analyzer = SpatialAnalyzer(SpatialSettings(confirm_frames=1, hazard_only=False))
    detection = [_box(box=(100, 100, 500, 600))]
    analyzer.analyze(detection, FRAME, now=1.0)

    analyzer.reset()
    report = analyzer.analyze(detection, FRAME, now=1.1)

    assert len(report.announcements) == 1  # cooldown sıfırlandı
