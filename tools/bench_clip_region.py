"""Bölge taramalı CLIP'i gerçek görsellerde ölçer."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from app.catalog import ClassCatalog
from app.detection.clip import ClipClassifier
from app.detection.clip_region import ClipRegionDetector, propose_regions

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
    catalog = ClassCatalog.load("datasets/classes.yaml")
    classifier = ClipClassifier(catalog, min_confidence=0.10)
    detector = ClipRegionDetector(classifier, min_confidence=0.30)

    hits = 0
    total = 0
    for name, expected in EXPECTED.items():
        frame = cv2.imread(str(folder / f"{name}.jpg"))
        if frame is None:
            continue
        results = detector.detect(frame, top_k=3)
        labels = [item.label for item in results]
        ok = expected in labels
        hits += 1 if ok else 0
        total += 1

        top = ", ".join(f"{item.label}:{item.confidence:.2f}" for item in results[:3])
        print(f"  {'OK ' if ok else '  X'} {name:14s} ({detector.last_ms:5.0f} ms) {top or '(yok)'}")

    print(f"\nbölge taraması doğru: {hits}/{total}")
    frame = cv2.imread(str(folder / "sup_kettle.jpg"))
    if frame is not None:
        print(f"bölge sayısı: {len(propose_regions(frame.shape[1], frame.shape[0]))}")


if __name__ == "__main__":
    main()
