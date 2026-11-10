"""CLIP'i şikayet edilen görsellerde sınar: kettle, kitap (dik/yatık), kadeh."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import cv2  # noqa: E402

from app.catalog import ClassCatalog  # noqa: E402
from app.detection.clip import ClipClassifier  # noqa: E402

EXPECTED = {
    "sup_kettle": "su ısıtıcısı",
    "kitap_dik": "kitap",
    "kitap_yatik": "kitap",
    "sarap_kadehi": "şarap kadehi",
    "fincan": "fincan",
    "laptop": "dizüstü bilgisayar",
}

CATALOG = ClassCatalog.from_names(
    [
        "su ısıtıcısı",
        "kitap",
        "fincan",
        "dizüstü bilgisayar",
        "şarap kadehi",
        "şişe",
        "kase",
        "tencere",
        "sürahi",
        "termos",
        "vazo",
        "saksı bitkisi",
        "kalem",
        "telefon",
        "klavye",
        "monitör",
    ]
)


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "bench_images"
    classifier = ClipClassifier(CATALOG, min_confidence=0.10)
    classifier.load()

    hits = 0
    total = 0
    for name, expected in EXPECTED.items():
        path = f"{folder}/{name}.jpg"
        frame = cv2.imread(path)
        if frame is None:
            print(f"  ?  {name}: bulunamadı")
            continue
        height, width = frame.shape[:2]
        # Tüm kareyi tek kutu kabul et (YOLO'nun bulamadığı durum senaryosu).
        results = classifier.tag(frame, [(0, 0, width, height)], top_k=3)
        total += 1
        if results:
            best = results[0]
            ok = best.label == expected
            hits += 1 if ok else 0
            alts = ", ".join(f"{lbl}:{sc:.2f}" for lbl, sc in best.alternatives)
            print(
                f"  {'OK ' if ok else '  X'} {name:14s} -> {best.label:20s} "
                f"({best.confidence:.2f})  alt: {alts}"
            )
        else:
            print(f"  X  {name:14s} -> eşik altı (etiket yok)")
        print(f"      beklenen: {expected}")

    print(f"\ndoğru: {hits}/{total}")


if __name__ == "__main__":
    main()
