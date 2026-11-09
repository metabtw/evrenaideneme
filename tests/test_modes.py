"""Çift mod testleri: otopilot uyarıları, derin analiz, STT ayrıştırma."""

from __future__ import annotations

from datetime import date

import pytest
from app.analysis import AnalysisMode, AnalysisResult
from app.config import SpatialSettings
from app.detection import Detection
from app.modes import AutopilotMode, DeepAnalysisMode, parse_expiry
from app.modes.autopilot import BANKNOTE_LABELS

FRAME = (720, 1280, 3)


class RecordingTTS:
    """``TextToSpeech`` yerine geçen, çağrıları kaydeden sahte servis."""

    def __init__(self) -> None:
        self.normal: list[str] = []
        self.critical: list[str] = []
        self.background: list[str] = []
        self.enabled = True
        self.is_busy = False
        self.current_text = ""

    def say(self, text, *, priority=None, interrupt=False) -> bool:  # noqa: ANN001
        bucket = {
            "CRITICAL": self.critical,
            "BACKGROUND": self.background,
        }.get(getattr(priority, "name", ""), self.normal)
        bucket.append(text)
        return True

    def say_critical(self, text: str) -> bool:
        self.critical.append(text)
        return True

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled


def _det(label: str = "sandalye", box=(100, 100, 500, 600), class_id: int = 56):
    return Detection(class_id=class_id, label=label, confidence=0.9, box=box)


# ---------------------------------------------------------------------------
# Otopilot
# ---------------------------------------------------------------------------


def test_autopilot_announces_critical_obstacle_via_tts() -> None:
    tts = RecordingTTS()
    mode = AutopilotMode(
        tts,
        settings=SpatialSettings(confirm_frames=1, hazard_only=False),  # type: ignore[arg-type]
    )

    mode.process([_det()], FRAME, now=1.0)

    assert len(tts.critical) == 1
    assert "sandalye" in tts.critical[0]
    assert mode.stats.critical_alerts == 1


def test_autopilot_near_obstacle_uses_normal_priority() -> None:
    tts = RecordingTTS()
    mode = AutopilotMode(
        tts,
        settings=SpatialSettings(confirm_frames=1, hazard_only=False),  # type: ignore[arg-type]
    )

    mode.process([_det(box=(100, 100, 500, 500))], FRAME, now=1.0)

    assert tts.normal and not tts.critical
    assert mode.stats.near_alerts == 1


def test_autopilot_without_tts_does_not_crash() -> None:
    mode = AutopilotMode(settings=SpatialSettings(confirm_frames=1, hazard_only=False))
    report = mode.process([_det()], FRAME, now=1.0)

    assert report.is_critical is True
    assert mode.stats.critical_alerts == 1


def test_autopilot_banknote_summing() -> None:
    mode = AutopilotMode(settings=SpatialSettings(confirm_frames=1))
    found = [
        Detection(0, "100 lira", 0.9, (0, 0, 100, 100)),
        Detection(0, "50 lira", 0.8, (0, 0, 100, 100)),
    ]

    assert mode._sum_banknotes(found) == 150  # noqa: SLF001
    assert mode._sum_banknotes([Detection(0, "banknot", 0.9, (0, 0, 10, 10))]) is None


def test_autopilot_reports_banknote_total() -> None:
    tts = RecordingTTS()
    mode = AutopilotMode(tts, settings=SpatialSettings(confirm_frames=1))  # type: ignore[arg-type]
    notes = [Detection(0, "200 lira", 0.9, (0, 0, 100, 100))]

    mode.process(notes, FRAME, now=1.0)

    assert any("200" in text for text in tts.normal)
    assert mode.stats.banknote_reports == 1


def test_banknote_cooldown_avoids_repetition() -> None:
    tts = RecordingTTS()
    mode = AutopilotMode(
        tts,
        settings=SpatialSettings(confirm_frames=1),  # type: ignore[arg-type]
        banknote_cooldown=5.0,
    )
    notes = [Detection(0, "100 lira", 0.9, (0, 0, 100, 100))]

    mode.process(notes, FRAME, now=10.0)
    mode.process(notes, FRAME, now=11.0)

    assert mode.stats.banknote_reports == 1


def test_autopilot_announce_labels_mode() -> None:
    tts = RecordingTTS()
    mode = AutopilotMode(
        tts,
        settings=SpatialSettings(confirm_frames=1, hazard_only=False),  # type: ignore[arg-type]
        announce_labels=True,
    )
    # Küçük (güvenli) nesneler: uyarı yok, ama anons var.
    small = [
        Detection(56, "sandalye", 0.9, (0, 0, 40, 40)),
        Detection(67, "cep telefonu", 0.8, (0, 0, 40, 40)),
        Detection(56, "sandalye", 0.7, (0, 0, 40, 40)),
    ]

    mode.process(small, FRAME, now=1.0)

    assert tts.background, "etiket anonsu beklenir"
    assert "sandalye" in tts.background[0]
    assert tts.background[0].count("sandalye") == 1  # tekrarlar ayıklandı


def test_autopilot_labels_announcement_survives_many_detections() -> None:
    tts = RecordingTTS()
    mode = AutopilotMode(
        tts,
        settings=SpatialSettings(confirm_frames=1, hazard_only=False),  # type: ignore[arg-type]
        announce_labels=True,
    )
    many = [Detection(i, f"nesne-{i}", 0.5, (0, 0, 20, 20)) for i in range(10)]

    mode.process(many, FRAME, now=1.0)  # KeyError regresyon testi

    assert tts.background


def test_banknote_labels_cover_common_denominations() -> None:
    assert "100 lira" in BANKNOTE_LABELS
    assert "200 lira" in BANKNOTE_LABELS


# ---------------------------------------------------------------------------
# STT (son tüketim tarihi) ayrıştırma
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12.03.2027", date(2027, 3, 12)),
        ("Son tüketim tarihi: 12/03/2027", date(2027, 3, 12)),
        ("STT 12-03-2027", date(2027, 3, 12)),
        ("12.03.27", date(2027, 3, 12)),
        ("12 Mart 2027", date(2027, 3, 12)),
        ("Son kullanma: 5 Ağustos 2028", date(2028, 8, 5)),
    ],
)
def test_parse_expiry_formats(text: str, expected: date) -> None:
    info = parse_expiry(text, today=date(2026, 10, 31))
    assert info.parsed == expected


def test_parse_expiry_marks_expired_product() -> None:
    info = parse_expiry("01.01.2024", today=date(2026, 10, 31))

    assert info.expired is True
    assert info.days_left is not None and info.days_left < 0
    assert "dolmuş" in info.spoken


def test_parse_expiry_reports_days_left_and_spoken_date() -> None:
    info = parse_expiry("02.11.2026", today=date(2026, 10, 31))

    assert info.expired is False
    assert info.days_left == 2
    assert "Kasım" in info.spoken
    assert "2 gün" in info.spoken


def test_parse_expiry_returns_none_without_date() -> None:
    info = parse_expiry("Bu bir süt kartonu.", today=date(2026, 10, 31))

    assert info.parsed is None
    assert info.spoken == ""


def test_parse_expiry_rejects_impossible_dates() -> None:
    info = parse_expiry("31.02.2027", today=date(2026, 10, 31))
    assert info.parsed is None


# ---------------------------------------------------------------------------
# Derin analiz modu
# ---------------------------------------------------------------------------


def _result(answer: str, mode: AnalysisMode = AnalysisMode.EXPIRY) -> AnalysisResult:
    return AnalysisResult(answer=answer, mode=mode, elapsed_ms=120.0, frame_id=1, timestamp=0.0)


def test_deep_mode_speaks_expiry_date() -> None:
    tts = RecordingTTS()
    mode = DeepAnalysisMode(tts=tts)  # type: ignore[arg-type]

    mode.begin()
    assert mode.is_active is True
    message = mode.complete(_result("12.03.2027"), mode=AnalysisMode.EXPIRY)

    assert "Son tüketim tarihi 12 Mart 2027" in message
    assert tts.normal and "Mart" in tts.normal[-1]
    assert mode.is_active is False


def test_deep_mode_falls_back_to_raw_text_when_no_date() -> None:
    tts = RecordingTTS()
    mode = DeepAnalysisMode(tts=tts)  # type: ignore[arg-type]

    mode.begin()
    message = mode.complete(_result("Ambalajda tarih görünmüyor."), mode=AnalysisMode.EXPIRY)

    assert "tarih" in message.lower()
    assert mode.is_active is False


def test_deep_mode_describe_mode_speaks_answer() -> None:
    tts = RecordingTTS()
    mode = DeepAnalysisMode(tts=tts)  # type: ignore[arg-type]

    mode.begin()
    message = mode.complete(_result("Masada bir fincan var.", AnalysisMode.DESCRIBE))

    assert "fincan" in message
    assert tts.normal


def test_deep_mode_fail_speaks_and_returns_to_live() -> None:
    tts = RecordingTTS()
    mode = DeepAnalysisMode(tts=tts)  # type: ignore[arg-type]

    mode.begin()
    spoken = mode.fail("Ollama'ya ulaşılamadı.")

    assert "Analiz yapılamadı" in spoken
    assert mode.is_active is False


def test_deep_mode_without_tts_is_safe() -> None:
    mode = DeepAnalysisMode()
    mode.begin()
    message = mode.complete(_result("01.01.2030"), mode=AnalysisMode.EXPIRY)

    assert "2030" in message
    assert mode.is_active is False
