"""Etiket çözümleme testleri: özel model adları COCO eşlemesini ezmemeli."""

from __future__ import annotations

from app.detection.labels import (
    COCO_EN_TO_ID,
    COCO_TR,
    ascii_label,
    resolve_label,
    turkish_label,
)


def test_custom_model_label_wins_over_coco_mapping() -> None:
    """Sınıf kimliği 0 COCO'da 'person' ama özel modelde 'su ısıtıcısı'."""
    assert resolve_label(0, "su ısıtıcısı") == "su ısıtıcısı"
    assert resolve_label(0, "kitap") == "kitap"
    assert resolve_label(3, "fincan") == "fincan"


def test_english_coco_name_is_translated() -> None:
    assert resolve_label(0, "person") == "kişi"
    assert resolve_label(2, "car") == "araba"
    assert resolve_label(63, "laptop") == "dizüstü bilgisayar"
    assert resolve_label(73, "book") == "kitap"


def test_english_name_translation_uses_name_not_index() -> None:
    """Özel modelde kitap kimliği 0 olsa bile 'book' -> 'kitap' çevrilir."""
    assert resolve_label(0, "book") == "kitap"
    assert resolve_label(7, "book") == "kitap"


def test_falls_back_to_coco_index_when_no_name() -> None:
    assert resolve_label(0, None) == "kişi"
    assert resolve_label(56, None) == "sandalye"
    assert resolve_label(999, None) == "sinif-999"


def test_unknown_custom_label_is_preserved() -> None:
    assert resolve_label(5, "hesap makinesi") == "hesap makinesi"
    assert resolve_label(5, "  kalem  ") == "kalem"


def test_ascii_label_strips_turkish_characters() -> None:
    assert ascii_label(0, "su ısıtıcısı") == "su isiticisi"
    assert ascii_label(0, "person") == "kisi"


def test_turkish_label_shortcut_matches_resolve() -> None:
    assert turkish_label(73, "book") == resolve_label(73, "book")


def test_coco_tables_are_consistent() -> None:
    assert len(COCO_TR) == 80
    assert len(COCO_EN_TO_ID) == 80
    for name, index in COCO_EN_TO_ID.items():
        assert index in COCO_TR, f"{name} -> {index} COCO_TR'de yok"
