"""Model karşılaştırma: eski (COCO) ve yeni (özel) model aynı görsellerde.

Neden? "Düzeldi mi?" sorusu ölçümle yanıtlanmalı. Bu araç aynı görsel
klasörünü iki modelle çalıştırır ve sınıf bazında ne bulduğunu yan yana yazar.

    python tools/compare_models.py --images bench_images \
        --models yolov8n.pt,models/evrenaidd-v1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED = {
    "kitap_dik": "kitap",
    "kitap_yatik": "kitap",
    "sup_kettle": "su ısıtıcısı",
    "fincan": "fincan",
    "sarap_kadehi": "şarap kadehi",
    "laptop": "dizüstü bilgisayar",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Model karşılaştırma")
    parser.add_argument("--images", default="bench_images")
    parser.add_argument("--models", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    folder = Path(args.images)
    files = sorted(folder.glob("*.jpg")) or sorted(folder.glob("*.png"))
    if not files:
        print(f"HATA: görsel yok: {folder}", file=sys.stderr)
        return 1

    for model_path in [m.strip() for m in args.models.split(",") if m.strip()]:
        if not Path(model_path).exists() and not model_path.startswith("yolo"):
            print(f"UYARI: {model_path} yok, atlanıyor")
            continue
        model = YOLO(model_path)
        print(f"\n=== {model_path} ===")
        hits = 0
        for path in files:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            result = model.predict(
                frame, conf=args.conf, imgsz=args.imgsz, verbose=False, device="cpu"
            )[0]
            names = result.names
            found = [
                f"{names[int(box.cls.item())]}:{float(box.conf.item()):.2f}"
                for box in result.boxes
            ]
            expected = EXPECTED.get(path.stem)
            correct = expected and any(
                expected.lower() in item.lower() for item in found
            )
            hits += 1 if correct else 0
            mark = "OK " if correct else "  X"
            print(f"  {mark} {path.stem:16s} {', '.join(found[:3]) or '(tespit yok)'}")
        print(f"  doğru: {hits}/{len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
