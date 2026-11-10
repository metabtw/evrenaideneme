"""Open Images V7'den seçili sınıfları indirip YOLO formatına çevirir.

Neden bu araç?
    COCO'da "su ısıtıcısı", "kartonpiyer" gibi sınıflar **yok**; bu yüzden
    hazır modeller o nesnelere en yakın şeyi söyler. Open Images V7'de 600
    sınıf kutu anotasyonlu olarak var (kettle, book, laptop, mug …).

Kaynaklar (doğrulandı, 2026-11):
    * class-descriptions-boxable.csv  (~11 MB)
    * validation-annotations-bbox.csv (~17 MB, 204k kutu)
    * validation-images-with-rotation.csv (~15 MB, ImageID -> OriginalURL)

Görseller orijinal kaynaklarından (Flickr vb.) indirilir; bir kısmı kalkmış
olabilir, bu yüzden başarısız indirmeler sayılır ve atlanır.

Kullanım::

    python tools/build_dataset.py --per-class 150
    python tools/build_dataset.py --classes "kitap,su ısıtıcısı,fincan" --per-class 80
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

H = {"User-Agent": "Mozilla/5.0 (EvrenAid dataset tool)"}
BASE = "https://storage.googleapis.com/openimages/2018_04"
BBOX_URL = f"{BASE}/validation/validation-annotations-bbox.csv"
IMAGES_URL = f"{BASE}/validation/validation-images-with-rotation.csv"
CLASS_URL = f"{BASE}/class-descriptions-boxable.csv"
CACHE = Path("datasets/.cache")


def download_csv(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / name
    if target.is_file() and target.stat().st_size > 1000:
        print(f"  önbellek: {name}")
        return target
    print(f"  indiriliyor: {name} …")
    with requests.get(url, headers=H, timeout=300, stream=True) as response:
        response.raise_for_status()
        with target.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    print(f"  kaydedildi: {target.stat().st_size / 1e6:.1f} MB")
    return target


def load_class_map(path: Path) -> dict[str, str]:
    """Türkçe ad -> Open Images mid (tools/map_openimages.py çıktısı)."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mapping[row["turkish"]] = row["mid"]
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Images -> YOLO veri seti")
    parser.add_argument("--classes", help="Virgülle ayrılmış Türkçe sınıf adları")
    parser.add_argument(
        "--classes-file",
        default="datasets/oid_classes.csv",
        help="tools/map_openimages.py çıktısı",
    )
    parser.add_argument("--per-class", type=int, default=120, help="Sınıf başına görsel")
    parser.add_argument("--out", default="datasets/evrenaidd", help="Çıktı dizini")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-cache", action="store_true", help="Ham görselleri de sakla"
    )
    args = parser.parse_args()

    try:
        import cv2  # noqa: PLC0415 - görsel doğrulama için
    except ImportError:
        print("OpenCV gerekli: pip install -r requirements.txt", file=sys.stderr)
        return 1

    random.seed(args.seed)
    class_map = load_class_map(Path(args.classes_file))
    if args.classes:
        wanted = [name.strip() for name in args.classes.split(",") if name.strip()]
    else:
        wanted = list(class_map)

    unknown = [name for name in wanted if name not in class_map]
    if unknown:
        print(f"UYARI: eşlemesi olmayan sınıflar atlandı: {', '.join(unknown)}")

    names = [name for name in wanted if name in class_map]
    if not names:
        print("HATA: indirilecek sınıf yok.", file=sys.stderr)
        return 1
    mid_to_index = {class_map[name]: index for index, name in enumerate(names)}
    print(f"{len(names)} sınıf, sınıf başına {args.per_class} görsel")

    print("anotasyonlar:")
    bbox_path = download_csv(BBOX_URL, "validation-annotations-bbox.csv")
    images_path = download_csv(IMAGES_URL, "validation-images-with-rotation.csv")

    # --- URL haritası ----------------------------------------------------
    urls: dict[str, str] = {}
    with images_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            url = (row.get("OriginalURL") or "").strip()
            if url:
                urls[row["ImageID"]] = url
    print(f"  {len(urls)} görsel URL'si")

    # --- anotasyonları sınıf bazında topla -------------------------------
    per_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    with bbox_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            index = mid_to_index.get(row["LabelName"])
            if index is None:
                continue
            if row["ImageID"] not in urls:
                continue
            if float(row.get("Confidence", 1)) < 0.6:
                continue
            per_class[index].append(row)

    selected: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, rows in per_class.items():
        by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_image[str(row["ImageID"])].append(row)
        image_ids = list(by_image)
        random.shuffle(image_ids)
        chosen = image_ids[: args.per_class]
        for image_id in chosen:
            selected[image_id].extend(by_image[image_id])
        print(f"  {names[index]:22s} {len(chosen):4d} görsel")

    # --- görselleri indir -------------------------------------------------
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_ids = list(selected)
    random.shuffle(image_ids)
    val_count = int(len(image_ids) * args.val_ratio)
    val_ids = set(image_ids[:val_count])

    raw = CACHE / "images"
    raw.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed = 0

    def fetch(image_id: str) -> tuple[str, bool]:
        target = raw / f"{image_id}.jpg"
        if target.is_file() and target.stat().st_size > 2000:
            return image_id, True
        url = urls[image_id]
        try:
            response = requests.get(url, headers=H, timeout=25)
            if response.status_code != 200 or len(response.content) < 2000:
                return image_id, False
            target.write_bytes(response.content)
            return image_id, True
        except Exception:  # noqa: BLE001 - ağ hataları beklenir
            return image_id, False

    print(f"\n{len(image_ids)} görsel indiriliyor ({args.workers} paralel) …")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, image_id) for image_id in image_ids]
        for count, future in enumerate(as_completed(futures), 1):
            image_id, success = future.result()
            if not success:
                failed += 1
                continue
            frame = cv2.imread(str(raw / f"{image_id}.jpg"))
            if frame is None or frame.size == 0:
                failed += 1
                continue

            height, width = frame.shape[:2]
            split = "val" if image_id in val_ids else "train"
            dest_image = out / "images" / split / f"{image_id}.jpg"
            shutil.copyfile(raw / f"{image_id}.jpg", dest_image)

            lines: list[str] = []
            for row in selected[image_id]:
                xmin = float(row["XMin"])  # type: ignore[arg-type]
                xmax = float(row["XMax"])  # type: ignore[arg-type]
                ymin = float(row["YMin"])  # type: ignore[arg-type]
                ymax = float(row["YMax"])  # type: ignore[arg-type]
                w = max(0.0, xmax - xmin)
                h = max(0.0, ymax - ymin)
                if w < 0.005 or h < 0.005:
                    continue
                cx = xmin + w / 2
                cy = ymin + h / 2
                index = mid_to_index[str(row["LabelName"])]
                lines.append(f"{index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not lines:
                dest_image.unlink(missing_ok=True)
                failed += 1
                continue

            (out / "labels" / split / f"{image_id}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            ok += 1
            if count % 50 == 0:
                print(f"  {count}/{len(image_ids)} … ({ok} başarılı)")

    if not args.keep_cache:
        shutil.rmtree(raw, ignore_errors=True)

    # --- dataset.yaml -----------------------------------------------------
    yaml_lines = [
        "# EvrenAid özel veri seti — tools/build_dataset.py tarafından üretildi",
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines += [f"  {index}: {name}" for index, name in enumerate(names)]
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    summary = {
        "classes": names,
        "per_class_target": args.per_class,
        "images_ok": ok,
        "images_failed": failed,
        "train": len(list((out / "images" / "train").glob("*.jpg"))),
        "val": len(list((out / "images" / "val").glob("*.jpg"))),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nBitti: {ok} görsel işlendi, {failed} başarısız")
    print(f"  train={summary['train']}  val={summary['val']}")
    print(f"  yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
