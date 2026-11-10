"""Hibrit modu (YOLO + CLIP bölge yedeği) gerçek görsellerde ölçer."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from app.detection.hybrid import create_hybrid_tagger
from app.config import DetectionSettings
from app.detection import ObjectDetector

EXPECTED = {
    "sup_kettle": "su ısıtıcısı",
    "kitap_dik": "kitap",
    "kitap_yatik": "kitap",
    "sarap_kadehi": "şarap kadehi",
    "fincan": "fincan",
    "laptop": "dizüstü bilgisayar",
}


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_images")
    tagger = create_hybrid_tagger(
        "datasets/classes.yaml", clip_override=0.45, min_confidence=0.15
    )
    assert tagger is not None
    detector = ObjectDetector(DetectionSettings(confidence=0.20))

    yolo_hits = 0
    hybrid_hits = 0
    total = 0
    for name, expected in EXPECTED.items():
        frame = cv2.imread(str(folder / f"{name}.jpg"))
        if frame is None:
            continue
        yolo = detector.detect(frame)
        hybrid = tagger.relabel(frame, yolo)

        yolo_labels = [d.label for d in yolo]
        hybrid_labels = [d.label for d in hybrid]
        yolo_ok = expected in yolo_labels
        hybrid_ok = expected in hybrid_labels
        yolo_hits += 1 if yolo_ok else 0
        hybrid_hits += 1 if hybrid_ok else 0
        total += 1

        print(f"  {name:14s} beklenen={expected}")
        print(f"      YOLO  {'OK ' if yolo_ok else ' X '}: {', '.join(yolo_labels[:3]) or '(yok)'}")
        print(f"      HIBRIT{'OK ' if hybrid_ok else ' X '}: {', '.join(hybrid_labels[:3]) or '(yok)'}")

    print(f"\nYOLO  : {yolo_hits}/{total}")
    print(f"HIBRIT: {hybrid_hits}/{total}")
    print(f"istatistik: {tagger.stats}")


if __name__ == "__main__":
    main()
