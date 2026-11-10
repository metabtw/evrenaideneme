"""Sınıf kataloğu: kullanıcının tanımladığı nesne listesi (metin tabanlı).

Bu, "veri seti tanımlamak" isteğinin **eğitim gerektirmeyen** karşılığıdır:

    * Klasik YOLO eğitimi sınıf başına 100-300 **etiketli görsel** ister
      (Open Images indirmesi bu makinede başarısız oldu: 467 görselden 392'si
      inmedi, Flickr kaynakları ölü).
    * CLIP tabanlı sınıflandırma ise sınıfları **İngilizce metin** olarak alır;
      hiç görsel toplamadan çalışır. Türkçe adları biz İngilizce istemlere
      çeviriyoruz (CLIP İngilizce eğitildi).

Katalog bir YAML/JSON dosyasıdır; kullanıcı kendi listesini yazar:

    names:
      0: {tr: su ısıtıcısı, en: an electric kettle}
      1: {tr: kitap, en: a book}

Ayrıca her sınıfa ``hazard`` (engel uyarısı) ve ``min_confidence`` verilebilir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Türkçe ad -> CLIP için İngilizce istem. Kullanıcı listesinde ``en`` yoksa
#: buradan tamamlanır; burada da yoksa Türkçe ad olduğu gibi kullanılır.
DEFAULT_EN: dict[str, str] = {
    "su ısıtıcısı": "an electric kettle",
    "kitap": "a book",
    "fincan": "a coffee mug",
    "dizüstü bilgisayar": "a laptop computer",
    "şarap kadehi": "a wine glass",
    "şişe": "a bottle",
    "kalem": "a pen",
    "telefon": "a mobile phone",
    "bilgisayar faresi": "a computer mouse",
    "klavye": "a computer keyboard",
    "monitör": "a computer monitor",
    "kulaklık": "a pair of headphones",
    "kulak içi kulaklık": "a pair of earbuds",
    "usb bellek": "a usb flash drive",
    "tablet": "a tablet computer",
    "hesap makinesi": "a calculator",
    "defter": "a notebook",
    "zımba": "a stapler",
    "makas": "a pair of scissors",
    "ataş": "a paper clip",
    "masa lambası": "a desk lamp",
    "yazıcı": "a printer",
    "yönlendirici": "a wifi router",
    "web kamerası": "a webcam",
    "hoparlör": "a speaker",
    "şarj cihazı": "a phone charger",
    "güç bankası": "a power bank",
    "uzaktan kumanda": "a remote control",
    "gözlük": "a pair of glasses",
    "kalemlik": "a pen holder",
    "silgi": "an eraser",
    "cetvel": "a ruler",
    "tabak": "a plate",
    "kase": "a bowl",
    "kaşık": "a spoon",
    "çatal": "a fork",
    "bıçak": "a knife",
    "çaydanlık": "a teapot",
    "tava": "a frying pan",
    "tencere": "a cooking pot",
    "kesme tahtası": "a cutting board",
    "mikrodalga fırın": "a microwave oven",
    "fırın": "an oven",
    "buzdolabı": "a refrigerator",
    "ekmek kızartma makinesi": "a toaster",
    "tepsi": "a tray",
    "kavanoz": "a glass jar",
    "teneke kutu": "an aluminum can",
    "su bardağı": "a drinking glass",
    "termos": "a thermos flask",
    "sürahi": "a water pitcher",
    "elma": "an apple",
    "muz": "a banana",
    "portakal": "an orange",
    "sandviç": "a sandwich",
    "pizza": "a pizza",
    "ekmek": "a loaf of bread",
    "pasta": "a cake",
    "kurabiye": "a cookie",
    "peynir": "a piece of cheese",
    "yumurta": "an egg",
    "domates": "a tomato",
    "patates": "a potato",
    "havuç": "a carrot",
    "cüzdan": "a wallet",
    "el çantası": "a handbag",
    "sırt çantası": "a backpack",
    "bavul": "a suitcase",
    "anahtar": "a key",
    "anahtarlık": "a keychain",
    "kol saati": "a wristwatch",
    "yüzük": "a ring",
    "kolye": "a necklace",
    "şemsiye": "an umbrella",
    "tara": "a comb",
    "diş fırçası": "a toothbrush",
    "diş macunu": "a tube of toothpaste",
    "parfüm şişesi": "a perfume bottle",
    "ayna": "a mirror",
    "maske": "a face mask",
    "güneş gözlüğü": "a pair of sunglasses",
    "saç kurutma makinesi": "a hair dryer",
    "havlu": "a towel",
    "çakmak": "a lighter",
    "sandalye": "a chair",
    "koltuk": "a couch",
    "masa": "a table",
    "yatak": "a bed",
    "yastık": "a pillow",
    "battaniye": "a blanket",
    "lamba": "a lamp",
    "duvar saati": "a wall clock",
    "vazo": "a vase",
    "saksı bitkisi": "a potted plant",
    "televizyon": "a television",
    "çöp kovası": "a trash can",
    "perde": "a curtain",
    "halı": "a rug",
    "raf": "a shelf",
    "kitaplık": "a bookshelf",
    "tabure": "a stool",
    "ayakkabı": "a shoe",
    "otomobil": "a car",
    "bisiklet": "a bicycle",
    "motosiklet": "a motorcycle",
    "otobüs": "a bus",
    "kamyon": "a truck",
    "trafik ışığı": "a traffic light",
    "dur işareti": "a stop sign",
    "kask": "a helmet",
    "köpek": "a dog",
    "kedi": "a cat",
    "kuş": "a bird",
    "tenis raketi": "a tennis racket",
    "basketbol topu": "a basketball",
    "futbol topu": "a soccer ball",
    "gitar": "a guitar",
    "oyuncak ayı": "a teddy bear",
    "fotoğraf makinesi": "a camera",
    "banknot": "a banknote",
    "türk lirası": "a Turkish lira banknote",
}

#: Varsayılan tehlike sınıfları (otopilot sesli uyarı üretir).
DEFAULT_HAZARDS = frozenset(
    {
        "kişi",
        "sandalye",
        "masa",
        "koltuk",
        "yatak",
        "kapı",
        "duvar",
        "raf",
        "kitaplık",
        "çöp kovası",
        "saksı bitkisi",
        "vazo",
        "bisiklet",
        "otomobil",
        "motosiklet",
        "otobüs",
        "kamyon",
        "tabure",
        "ayakkabı",
    }
)


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Katalogdaki tek sınıf."""

    index: int
    tr: str
    en: str
    hazard: bool = False
    min_confidence: float = 0.25

    @property
    def prompt(self) -> str:
        """CLIP'e verilecek istem (İngilizce, doğal cümle)."""
        return self.en


@dataclass(frozen=True, slots=True)
class ClassCatalog:
    """Sınıf kataloğu (sıralı, indeksli)."""

    entries: tuple[CatalogEntry, ...] = field(default_factory=tuple)
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def labels(self) -> list[str]:
        return [entry.tr for entry in self.entries]

    @property
    def prompts(self) -> list[str]:
        return [entry.prompt for entry in self.entries]

    def by_index(self, index: int) -> CatalogEntry | None:
        for entry in self.entries:
            if entry.index == index:
                return entry
        return None

    def hazard_labels(self) -> tuple[str, ...]:
        hazards = [entry.tr for entry in self.entries if entry.hazard]
        return tuple(hazards) if hazards else tuple(DEFAULT_HAZARDS)

    @classmethod
    def from_names(cls, names: list[str]) -> ClassCatalog:
        """Sadece Türkçe ad listesinden katalog kurar (İngilizce tamamlanır)."""
        entries = tuple(
            CatalogEntry(
                index=index,
                tr=name,
                en=DEFAULT_EN.get(name, name),
                hazard=name in DEFAULT_HAZARDS,
            )
            for index, name in enumerate(names)
        )
        return cls(entries=entries)

    @classmethod
    def load(cls, path: str | Path) -> ClassCatalog:
        """JSON veya basit YAML-benzeri katalog dosyasını okur."""
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"Katalog bulunamadı: {file}")

        text = file.read_text(encoding="utf-8")
        if file.suffix.lower() == ".json":
            return cls._from_json(json.loads(text), file)
        return cls._from_simple_yaml(text, file)

    @classmethod
    def _from_json(cls, data: Any, source: Path) -> ClassCatalog:
        raw = data.get("names", data) if isinstance(data, dict) else data
        entries: list[CatalogEntry] = []
        items = raw.items() if isinstance(raw, dict) else enumerate(raw)
        for key, value in items:
            index = int(key)
            if isinstance(value, str):
                entries.append(
                    CatalogEntry(
                        index=index,
                        tr=value,
                        en=DEFAULT_EN.get(value, value),
                        hazard=value in DEFAULT_HAZARDS,
                    )
                )
            elif isinstance(value, dict):
                tr = str(value.get("tr") or value.get("name") or f"sinif-{index}")
                entries.append(
                    CatalogEntry(
                        index=index,
                        tr=tr,
                        en=str(value.get("en") or DEFAULT_EN.get(tr, tr)),
                        hazard=bool(value.get("hazard", tr in DEFAULT_HAZARDS)),
                        min_confidence=float(value.get("min_confidence", 0.25)),
                    )
                )
        return cls(entries=tuple(sorted(entries, key=lambda e: e.index)), source=source)

    @classmethod
    def _from_simple_yaml(cls, text: str, source: Path) -> ClassCatalog:
        """Bağımlılıksız mini YAML okuyucu (yalnızca ``names:`` bloğu).

        Beklenen biçim::

            names:
              0: su ısıtıcısı
              1: {tr: kitap, en: a book, hazard: true}
        """
        entries: list[CatalogEntry] = []
        in_names = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith((" ", "\t")):
                in_names = line.strip().startswith("names:")
                continue
            if not in_names or ":" not in line:
                continue

            key, _, rest = line.strip().partition(":")
            try:
                index = int(key.strip())
            except ValueError:
                continue
            rest = rest.strip()

            if rest.startswith("{"):
                fields = _parse_inline_map(rest)
                tr = fields.get("tr") or fields.get("name") or f"sinif-{index}"
                entries.append(
                    CatalogEntry(
                        index=index,
                        tr=tr,
                        en=fields.get("en") or DEFAULT_EN.get(tr, tr),
                        hazard=_as_bool(fields.get("hazard"), tr in DEFAULT_HAZARDS),
                        min_confidence=float(fields.get("min_confidence", 0.25)),
                    )
                )
            elif rest:
                tr = rest.strip("'\"")
                entries.append(
                    CatalogEntry(
                        index=index,
                        tr=tr,
                        en=DEFAULT_EN.get(tr, tr),
                        hazard=tr in DEFAULT_HAZARDS,
                    )
                )

        if not entries:
            raise ValueError(f"Katalogda 'names:' bloğu okunamadı: {source}")
        return cls(entries=tuple(sorted(entries, key=lambda e: e.index)), source=source)


def _parse_inline_map(text: str) -> dict[str, str]:
    """``{tr: kitap, en: a book}`` -> sözlük."""
    inner = text.strip().lstrip("{").rstrip("}")
    result: dict[str, str] = {}
    for part in inner.split(","):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        result[key.strip()] = value.strip().strip("'\"")
    return result


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "evet", "e")
