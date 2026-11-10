"""Open Images sınıf kataloğunu indirir ve hedef sınıflarımızı eşler."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import requests

H = {"User-Agent": "Mozilla/5.0 (EvrenAid dataset tool)"}
CLASS_URL = "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv"

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets/oid_classes.csv")

# Türkçe ad -> Open Images İngilizce adında aranacak anahtar(lar)
WANTED = {
    "su ısıtıcısı": ["kettle"],
    "kitap": ["book"],
    "fincan": ["coffee cup", "mug", "cup"],
    "dizüstü bilgisayar": ["laptop"],
    "şarap kadehi": ["wine glass"],
    "şişe": ["bottle"],
    "kalem": ["pen"],
    "telefon": ["mobile phone", "telephone"],
    "bilgisayar faresi": ["computer mouse", "mouse"],
    "klavye": ["keyboard"],
    "monitör": ["monitor", "computer monitor"],
    "kulaklık": ["headphones", "headphone"],
    "tablet": ["tablet computer", "tablet"],
    "makas": ["scissors"],
    "sandalye": ["chair"],
    "masa": ["table", "desk"],
    "televizyon": ["television", "tv"],
    "kitap rafı": ["bookcase"],
    "kase": ["bowl"],
    "tabak": ["plate"],
    "çatal": ["fork"],
    "bıçak": ["knife"],
    "kaşık": ["spoon"],
    "tencere": ["pot", "saucepan"],
    "tava": ["frying pan", "pan"],
    "muz": ["banana"],
    "elma": ["apple"],
    "portakal": ["orange"],
    "sandviç": ["sandwich"],
    "pizza": ["pizza"],
    "kek": ["cake"],
    "domates": ["tomato"],
    "havuç": ["carrot"],
    "yumurta": ["egg"],
    "sırt çantası": ["backpack"],
    "şemsiye": ["umbrella"],
    "gözlük": ["glasses", "sunglasses"],
    "saat": ["watch", "clock"],
    "ayakkabı": ["shoe", "footwear"],
    "kedi": ["cat"],
    "köpek": ["dog"],
    "araba": ["car"],
    "bisiklet": ["bicycle"],
    "otobüs": ["bus"],
    "kamyon": ["truck"],
    "trafik ışığı": ["traffic light"],
    "dur tabelası": ["stop sign"],
    "bank": ["bench"],
    "koltuk": ["couch", "sofa"],
    "yatak": ["bed"],
    "vazo": ["vase"],
    "saksı bitkisi": ["potted plant", "houseplant"],
    "çöp kovası": ["waste container", "trash"],
    "havlu": ["towel"],
    "diş fırçası": ["toothbrush"],
    "tarak": ["comb"],
    "anahtar": ["key"],
    "cüzdan": ["wallet"],
    "bavul": ["suitcase", "luggage"],
}


def main() -> None:
    response = requests.get(CLASS_URL, headers=H, timeout=120)
    response.raise_for_status()
    rows = list(csv.reader(io.StringIO(response.text)))
    catalogue = {row[1].lower(): row[0] for row in rows if len(row) >= 2}
    print(f"katalog: {len(catalogue)} sınıf")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    matched: list[tuple[str, str, str]] = []
    missing: list[str] = []

    for turkish, keys in WANTED.items():
        found = None
        for key in keys:
            for name, mid in catalogue.items():
                if name == key:
                    found = (name, mid)
                    break
            if found:
                break
        if not found:
            for key in keys:
                for name, mid in catalogue.items():
                    if key in name:
                        found = (name, mid)
                        break
                if found:
                    break
        if found:
            matched.append((turkish, found[0], found[1]))
        else:
            missing.append(turkish)

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["turkish", "english", "mid"])
        writer.writerows(matched)

    print(f"\neşleşen: {len(matched)} / {len(WANTED)}  -> {OUT}")
    for turkish, english, mid in matched:
        print(f"  {turkish:22s} = {english:24s} {mid}")
    if missing:
        print(f"\nOpen Images'da YOK ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
