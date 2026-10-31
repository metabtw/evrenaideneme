"""ObjectDetector testleri: sahte model ile ultralytics'e ihtiyaç duymadan."""

from __future__ import annotations

import numpy as np
import pytest
from app.config import DetectionSettings
from app.detection import Detection, DetectorError, ObjectDetector


class _Boxes:
    def __init__(self, xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf

    def __len__(self) -> int:
        return len(self.conf)


class _Result:
    names = {0: "person", 2: "car", 7: "truck"}

    def __init__(self) -> None:
        self.boxes = _Boxes(
            xyxy=np.array(
                [[10.4, 20.6, 50.2, 80.9], [5.0, 5.0, 20.0, 20.0], [0.0, 0.0, 4.0, 4.0]],
                dtype=np.float32,
            ),
            cls=np.array([0, 2, 7]),
            conf=np.array([0.90, 0.40, 0.10]),
        )


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, _frame: np.ndarray, **_kwargs: object) -> list[_Result]:
        self.calls += 1
        return [_Result()]


def _frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_detect_returns_turkish_labels_and_filters_low_confidence() -> None:
    detector = ObjectDetector(
        DetectionSettings(confidence=0.35), model_factory=lambda _path: FakeModel()
    )

    detections = detector.detect(_frame())

    assert [d.label for d in detections] == ["kişi", "araba"]
    assert [d.class_id for d in detections] == [0, 2]
    assert detections[0].box == (10, 21, 50, 81)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detector.is_loaded is True


def test_model_is_loaded_only_once() -> None:
    created: list[FakeModel] = []

    def factory(_path: str) -> FakeModel:
        model = FakeModel()
        created.append(model)
        return model

    detector = ObjectDetector(DetectionSettings(), model_factory=factory)
    detector.detect(_frame())
    detector.detect(_frame())

    assert len(created) == 1
    assert created[0].calls == 2


def test_empty_and_invalid_frames() -> None:
    detector = ObjectDetector(DetectionSettings(), model_factory=lambda _p: FakeModel())

    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []
    assert detector.is_loaded is False

    with pytest.raises(DetectorError, match="BGR"):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))


def test_model_load_failure_is_wrapped() -> None:
    def factory(_path: str) -> FakeModel:
        raise RuntimeError("ağırlık indirilemedi")

    detector = ObjectDetector(DetectionSettings(), model_factory=factory)

    with pytest.raises(DetectorError, match="YOLO modeli yüklenemedi"):
        detector.detect(_frame())


def test_detection_geometry_helpers() -> None:
    detection = Detection(class_id=0, label="kişi", confidence=0.8, box=(10, 20, 40, 80))

    assert detection.width == 30
    assert detection.height == 60
    assert detection.area == 1800
    assert detection.center == (25, 50)
    assert detection.as_dict()["box"] == [10, 20, 40, 80]
