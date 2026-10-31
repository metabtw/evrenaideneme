"""COCO sınıf kimlikleri -> Türkçe etiketler.

YOLOv8 (``yolov8n.pt``) COCO'nun 80 sınıfıyla eğitilmiştir. HUD'da İngilizce
``person`` yerine ``kişi`` göstermek için bu tablo kullanılır. OpenCV yalnızca
ASCII çizebildiği için :func:`ascii_label` ayrıca sunulur.
"""

from __future__ import annotations

from app.utils.text import to_ascii

#: COCO sınıf indeksi -> Türkçe ad (Ultralytics sıralamasıyla birebir).
COCO_TR: dict[int, str] = {
    0: "kişi",
    1: "bisiklet",
    2: "araba",
    3: "motosiklet",
    4: "uçak",
    5: "otobüs",
    6: "tren",
    7: "kamyon",
    8: "tekne",
    9: "trafik lambası",
    10: "yangın musluğu",
    11: "dur tabelası",
    12: "parkmetre",
    13: "bank",
    14: "kuş",
    15: "kedi",
    16: "köpek",
    17: "at",
    18: "koyun",
    19: "inek",
    20: "fil",
    21: "ayı",
    22: "zebra",
    23: "zürafa",
    24: "sırt çantası",
    25: "şemsiye",
    26: "el çantası",
    27: "kravat",
    28: "valiz",
    29: "frizbi",
    30: "kayak",
    31: "snowboard",
    32: "top",
    33: "uçurtma",
    34: "beyzbol sopası",
    35: "beyzbol eldiveni",
    36: "kaykay",
    37: "sörf tahtası",
    38: "tenis raketi",
    39: "şişe",
    40: "şarap kadehi",
    41: "bardak",
    42: "çatal",
    43: "bıçak",
    44: "kaşık",
    45: "kase",
    46: "muz",
    47: "elma",
    48: "sandviç",
    49: "portakal",
    50: "brokoli",
    51: "havuç",
    52: "sosisli sandviç",
    53: "pizza",
    54: "donut",
    55: "pasta",
    56: "sandalye",
    57: "kanepe",
    58: "saksı bitkisi",
    59: "yatak",
    60: "yemek masası",
    61: "tuvalet",
    62: "televizyon",
    63: "dizüstü bilgisayar",
    64: "fare",
    65: "kumanda",
    66: "klavye",
    67: "cep telefonu",
    68: "mikrodalga",
    69: "fırın",
    70: "tost makinesi",
    71: "lavabo",
    72: "buzdolabı",
    73: "kitap",
    74: "saat",
    75: "vazo",
    76: "makas",
    77: "oyuncak ayı",
    78: "saç kurutma makinesi",
    79: "diş fırçası",
}


def turkish_label(class_id: int, fallback: str | None = None) -> str:
    """Sınıf kimliği için Türkçe etiket; bilinmiyorsa modelin adı döner."""
    if class_id in COCO_TR:
        return COCO_TR[class_id]
    return fallback or f"sinif-{class_id}"


def ascii_label(class_id: int, fallback: str | None = None) -> str:
    """OpenCV ile çizilebilir ASCII etiket (``kişi`` -> ``kisi``)."""
    return to_ascii(turkish_label(class_id, fallback))
