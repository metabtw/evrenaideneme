"""Sınıf kataloğu testleri: JSON/YAML okuma, İngilizce tamamlama, tehlike listesi."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.catalog import DEFAULT_EN, DEFAULT_HAZARDS, CatalogEntry, ClassCatalog


def test_from_names_fills_english_prompts() -> None:
    catalog = ClassCatalog.from_names(["su ısıtıcısı", "kitap", "bilinmeyen nesne"])

    assert len(catalog) == 3
    assert catalog.labels == ["su ısıtıcısı", "kitap", "bilinmeyen nesne"]
    assert catalog.prompts[0] == "an electric kettle"
    assert catalog.prompts[1] == "a book"
    # Bilinmeyen ad olduğu gibi kullanılır (en azından çalışır)
    assert catalog.prompts[2] == "bilinmeyen nesne"


def test_hazard_flags_come_from_defaults() -> None:
    catalog = ClassCatalog.from_names(["sandalye", "kitap", "masa"])

    assert catalog.by_index(0).hazard is True
    assert catalog.by_index(1).hazard is False
    assert "sandalye" in catalog.hazard_labels()
    assert "kitap" not in catalog.hazard_labels()


def test_hazard_labels_falls_back_to_defaults() -> None:
    catalog = ClassCatalog.from_names(["kitap", "kalem"])
    hazards = catalog.hazard_labels()

    assert hazards == tuple(DEFAULT_HAZARDS)  # hiç tehlike işaretli değil


def test_load_simple_yaml(tmp_path: Path) -> None:
    file = tmp_path / "classes.yaml"
    file.write_text(
        "# EvrenAid katalog\n"
        "names:\n"
        "  0: su ısıtıcısı\n"
        "  1: kitap\n"
        "  2: {tr: fincan, en: a coffee cup, hazard: true, min_confidence: 0.4}\n",
        encoding="utf-8",
    )

    catalog = ClassCatalog.load(file)

    assert len(catalog) == 3
    assert catalog.labels == ["su ısıtıcısı", "kitap", "fincan"]
    assert catalog.prompts[2] == "a coffee cup"
    assert catalog.by_index(2).hazard is True
    assert catalog.by_index(2).min_confidence == pytest.approx(0.4)


def test_load_json(tmp_path: Path) -> None:
    file = tmp_path / "classes.json"
    file.write_text(
        json.dumps(
            {
                "names": {
                    "0": "kitap",
                    "1": {"tr": "masa", "en": "a desk"},
                }
            }
        ),
        encoding="utf-8",
    )

    catalog = ClassCatalog.load(file)

    assert catalog.labels == ["kitap", "masa"]
    assert catalog.prompts[1] == "a desk"


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Katalog bulunamadı"):
        ClassCatalog.load("yok/boyle/bir/dosya.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    file = tmp_path / "bos.yaml"
    file.write_text("baska: bir sey\n", encoding="utf-8")

    with pytest.raises(ValueError, match="names"):
        ClassCatalog.load(file)


def test_by_index_returns_none_for_unknown() -> None:
    catalog = ClassCatalog.from_names(["kitap"])
    assert catalog.by_index(99) is None


def test_default_english_map_covers_core_classes() -> None:
    for name in ("su ısıtıcısı", "kitap", "fincan", "dizüstü bilgisayar", "şarap kadehi"):
        assert name in DEFAULT_EN, f"{name} için İngilizce istem yok"


def test_entry_prompt_property() -> None:
    entry = CatalogEntry(index=0, tr="kitap", en="a book", hazard=False)
    assert entry.prompt == "a book"
