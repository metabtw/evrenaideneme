"""WearableAssistant testleri: mod geçişi, kare dondurma, parça bağlama."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from app.analysis import AnalysisMode, AnalysisResult
from app.config import DetectionSettings, Settings, SpatialSettings
from app.detection import Detection
from app.modes import AutopilotMode, DeepAnalysisMode
from app.wearable import ModeName, WearableAssistant

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class SimpleResult:
    """``DetectionResult`` yerine geçen minimal sonuç."""

    detections: tuple[Detection, ...]
    inference_ms: float
    frame_id: int = 1
    timestamp: float = 0.0


class FakeDetectionWorker:
    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.submitted: list[np.ndarray] = []
        self._detections = detections or []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True

    def submit(self, frame: np.ndarray) -> int:
        self.submitted.append(frame)
        return len(self.submitted)

    def latest_result(self):
        if not self._detections:
            return None
        return SimpleResult(tuple(self._detections), 5.0)


class FakeAnalysisWorker:
    def __init__(self) -> None:
        self.submitted: list[tuple[np.ndarray, AnalysisMode, str | None]] = []
        self.is_busy = False
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True

    def submit(self, frame, *, mode=AnalysisMode.DESCRIBE, target=None) -> int:
        if mode is AnalysisMode.COUNT and not target:
            raise ValueError("COUNT modu için hedef nesne adı gerekli")
        self.submitted.append((frame, mode, target))
        self.is_busy = True
        return len(self.submitted)

    def set_callbacks(self, *, on_result=None, on_error=None) -> None:
        self.on_result = on_result
        self.on_error = on_error


class RecordingTTS:
    def __init__(self) -> None:
        self.normal: list[str] = []
        self.critical: list[str] = []
        self.background: list[str] = []
        self.is_busy = False
        self.current_text = ""
        self.enabled = True

    def say(self, text, *, priority=None, interrupt=False) -> bool:  # noqa: ANN001
        self.normal.append(text)
        return True

    def say_critical(self, text: str) -> bool:
        self.critical.append(text)
        return True

    def close(self) -> None:
        return None


def _settings(**overrides) -> Settings:
    return replace(
        Settings(
            detection=DetectionSettings(),
            spatial=SpatialSettings(confirm_frames=1, hazard_only=False),
        ),
        **overrides,
    )


def _assistant(**kwargs) -> WearableAssistant:
    tts = kwargs.pop("tts", RecordingTTS())
    detection_worker = kwargs.pop("detection_worker", FakeDetectionWorker())
    analysis_worker = kwargs.pop("analysis_worker", FakeAnalysisWorker())
    settings = kwargs.pop("settings", _settings())

    assistant = WearableAssistant(
        settings,
        detection_worker=detection_worker,
        analysis_worker=analysis_worker,
        tts=tts,
        autopilot=AutopilotMode(tts, settings=settings.spatial),  # type: ignore[arg-type]
        deep=DeepAnalysisMode(tts=tts),  # type: ignore[arg-type]
    )
    return assistant


def test_starts_in_autopilot_mode() -> None:
    assistant = _assistant()
    assert assistant.mode is ModeName.AUTOPILOT
    assert assistant.is_frozen is False


def test_handle_frame_runs_autopilot_and_returns_report() -> None:
    detection_worker = FakeDetectionWorker([Detection(56, "sandalye", 0.9, (100, 100, 500, 600))])
    assistant = _assistant(detection_worker=detection_worker)

    report = assistant.handle_frame(FRAME)

    assert report is not None
    assert report.is_critical is True
    assert detection_worker.submitted  # kare işçiye verildi


def test_deep_request_freezes_frame_and_switches_mode() -> None:
    analysis_worker = FakeAnalysisWorker()
    assistant = _assistant(analysis_worker=analysis_worker)
    frame = FRAME.copy()

    accepted = assistant.request_deep_analysis(frame, mode=AnalysisMode.EXPIRY)

    assert accepted is True
    assert assistant.mode is ModeName.DEEP
    assert assistant.is_frozen is True
    assert assistant.frozen_frame is not None
    assert assistant.frozen_frame is not frame  # kopya alındı
    assert analysis_worker.submitted[0][1] is AnalysisMode.EXPIRY


def test_handle_frame_in_deep_mode_does_not_submit_detection() -> None:
    detection_worker = FakeDetectionWorker()
    analysis_worker = FakeAnalysisWorker()
    assistant = _assistant(detection_worker=detection_worker, analysis_worker=analysis_worker)
    assistant.request_deep_analysis(FRAME)
    detection_worker.submitted.clear()

    report = assistant.handle_frame(FRAME)

    assert report is None
    assert detection_worker.submitted == []


def test_busy_analysis_rejects_new_request() -> None:
    analysis_worker = FakeAnalysisWorker()
    analysis_worker.is_busy = True
    assistant = _assistant(analysis_worker=analysis_worker)

    assert assistant.request_deep_analysis(FRAME) is False
    assert assistant.mode is ModeName.AUTOPILOT


def test_request_without_frame_is_rejected() -> None:
    assistant = _assistant()
    assert assistant.request_deep_analysis(None) is False


def test_result_callback_speaks_and_resumes_autopilot() -> None:
    tts = RecordingTTS()
    assistant = _assistant(tts=tts)
    assistant.request_deep_analysis(FRAME, mode=AnalysisMode.EXPIRY)

    result = AnalysisResult(
        answer="12.03.2027", mode=AnalysisMode.EXPIRY, elapsed_ms=50.0, frame_id=1, timestamp=0.0
    )
    message = assistant.on_analysis_result(result)

    assert "Mart" in message
    assert assistant.mode is ModeName.AUTOPILOT
    assert assistant.is_frozen is False
    assert tts.normal


def test_error_callback_returns_to_autopilot() -> None:
    assistant = _assistant()
    assistant.request_deep_analysis(FRAME)

    assistant.on_analysis_error("Ollama kapalı")

    assert assistant.mode is ModeName.AUTOPILOT


def test_resume_forces_autopilot() -> None:
    assistant = _assistant()
    assistant.request_deep_analysis(FRAME)
    assistant.resume()

    assert assistant.mode is ModeName.AUTOPILOT


def test_deep_timeout_returns_to_autopilot() -> None:
    assistant = _assistant()
    assistant.deep_timeout = 0.0
    assistant.request_deep_analysis(FRAME)

    assistant.handle_frame(FRAME)

    assert assistant.mode is ModeName.AUTOPILOT


def test_close_stops_all_parts() -> None:
    detection_worker = FakeDetectionWorker()
    analysis_worker = FakeAnalysisWorker()
    assistant = _assistant(detection_worker=detection_worker, analysis_worker=analysis_worker)
    assistant.start()
    assistant.close()

    assert detection_worker.stopped is True
    assert analysis_worker.stopped is True


def test_stats_snapshot() -> None:
    assistant = _assistant()
    assistant.start()
    assistant.handle_frame(FRAME)

    stats = assistant.stats

    assert stats["mode"] == "autopilot"
    assert stats["frames"] >= 1
    assert stats["deep_requests"] == 0
