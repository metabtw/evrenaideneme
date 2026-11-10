"""Özel YOLO modeli eğitimi (transfer learning).

Neden gerekli?
    COCO modeli ``su ısıtıcısı`` sınıfını bilmez (COCO'da yok) ve ``kitap``
    sınıfını dik/yatık duruşlarda kaçırır. Ölçüm: ``yolov8n`` kitap görselinde
    hiç tespit üretmedi, ``yolov8m`` (10× yavaş) de üretmedi. Kapasite değil,
    **sınıf kapsamı ve alan farkı** sorunu; çözümü kendi verisiyle eğitmektir.

Kullanım::

    python tools/train.py                                # varsayılan ayarlar
    python tools/train.py --epochs 60 --model yolov8s.pt
    python tools/train.py --resume                       # yarıda kalanı sürdür

Çıktı: ``models/evrenaidd-v1.pt`` + ``runs/train/...`` altında metrikler.
Eğitim sonrası ``.env`` içinde ``APP_MODEL_PATH=models/evrenaidd-v1.pt``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="EvrenAid YOLO eğitimi")
    parser.add_argument("--data", default="datasets/evrenaidd/dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Başlangıç ağırlığı")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="0 | cpu | mps (boş = otomatik)")
    parser.add_argument("--patience", type=int, default=15, help="Erken durma sabrı")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="evrenaidd-v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        print(
            f"HATA: veri seti bulunamadı: {data_path}\n"
            "Önce: python tools/build_dataset.py --per-class 150",
            file=sys.stderr,
        )
        return 1

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"HATA: {exc}. Kurulum: pip install -r requirements.txt", file=sys.stderr)
        return 1

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"cihaz: {device} (cuda={torch.cuda.is_available()})")
    if device == "cpu":
        print(
            "UYARI: GPU yok; eğitim saatler sürebilir. "
            "Hızlandırmak için CUDA'lı torch kurun."
        )

    # Veri seti özeti (kullanıcı ne eğittiğini görsün)
    summary_file = data_path.parent / "summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        print(
            f"veri: {summary.get('images_ok', '?')} görsel "
            f"(train={summary.get('train', '?')}, val={summary.get('val', '?')}), "
            f"{len(summary.get('classes', []))} sınıf"
        )

    model = YOLO(args.model)
    started = time.perf_counter()

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        workers=args.workers,
        project="runs/train",
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        seed=42,
        # Renk/ölçek artırma: ev içi ışık ve mesafe çeşitliliği için.
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        mosaic=1.0,
        # Küçük nesneler (kalem, ataş) için ölçek kaybını azalt.
        close_mosaic=10,
        plots=True,
        verbose=True,
    )

    elapsed = time.perf_counter() - started
    print(f"\neğitim süresi: {elapsed / 60:.1f} dakika")

    save_dir = Path(getattr(results, "save_dir", f"runs/train/{args.name}"))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"HATA: en iyi ağırlık bulunamadı: {best}", file=sys.stderr)
        return 1

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / f"{args.name}.pt"
    shutil.copyfile(best, target)

    # --- doğrulama metrikleri --------------------------------------------
    metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=device)
    report = {
        "model": str(target),
        "classes": list(getattr(metrics.box, "ap_class_index", [])),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "train_minutes": round(elapsed / 60, 1),
    }
    report_path = save_dir / "evrenaidd_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nmodel: {target}")
    print(f"mAP50   : {report['mAP50']:.3f}")
    print(f"mAP50-95: {report['mAP50-95']:.3f}")
    print(f"precision: {report['precision']:.3f}  recall: {report['recall']:.3f}")
    print(f"rapor: {report_path}")
    print(f"\nKullanmak için .env: APP_MODEL_PATH={target.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
