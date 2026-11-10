"""COCO sınıf kimlikleri -> Türkçe etiketler.

YOLOv8 (``yolov8n.pt``) COCO'nun 80 sınıfıyla eğitilmiştir; HUD'da İngilizce
``person`` yerine ``kişi`` gösterilir.

**Öncelik kuralı (önemli):** özel eğitilmiş bir model kendi sınıf adlarını
verir (``su ısıtıcısı`` gibi). Bu adlar COCO eşlemesinden **önce** gelir;
aksi halde sınıf kimliği 0 olan "su ısıtıcısı" yanlışlıkla "kişi" görünür.
COCO eşlemesi yalnızca model İngilizce COCO adı döndürdüğünde devreye girer.
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

#: Ultralytics'in COCO modelinin döndürdüğü İngilizce adlar -> sınıf kimliği.
COCO_EN_TO_ID: dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}


def resolve_label(class_id: int, model_name: str | None) -> str:
    """Etiketi çözer: modelin kendi adı > COCO İngilizce -> Türkçe > yedek.

    Args:
        class_id: Modelin sınıf kimliği.
        model_name: Modelin bu kimlik için verdiği ad (``result.names[id]``).

    Returns:
        Kullanıcıya gösterilecek Türkçe etiket.
    """
    if model_name:
        cleaned = model_name.strip()
        # Model İngilizce COCO adı döndürdüyse Türkçeye çevir.
        mapped = COCO_EN_TO_ID.get(cleaned.lower())
        if mapped is not None and mapped in COCO_TR:
            return COCO_TR[mapped]
        # Özel model (Türkçe ya da kendi adı): olduğu gibi kullan.
        return cleaned

    if class_id in COCO_TR:
        return COCO_TR[class_id]
    return f"sinif-{class_id}"


def turkish_label(class_id: int, fallback: str | None = None) -> str:
    """Sınıf kimliği için Türkçe etiket (``resolve_label`` kısayolu)."""
    return resolve_label(class_id, fallback)


def ascii_label(class_id: int, fallback: str | None = None) -> str:
    """OpenCV ile çizilebilir ASCII etiket (``kişi`` -> ``kisi``)."""
    return to_ascii(resolve_label(class_id, fallback))
