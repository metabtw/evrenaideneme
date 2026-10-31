"""Metin yardımcıları.

Neden gerekli? ``cv2.putText`` yalnızca ASCII (Hershey) gliflerini çizer;
Türkçe karakterler ``?`` olarak görünür. Ekrana çizilecek tüm metinler
:func:`to_ascii` ile sadeleştirilir. Terminal/log çıktısı ise Türkçe kalır.
"""

from __future__ import annotations

_TRANSLATION = str.maketrans(
    {
        "ı": "i",
        "İ": "I",
        "ş": "s",
        "Ş": "S",
        "ğ": "g",
        "Ğ": "G",
        "ç": "c",
        "Ç": "C",
        "ö": "o",
        "Ö": "O",
        "ü": "u",
        "Ü": "U",
        "â": "a",
        "Â": "A",
        "î": "i",
        "Î": "I",
        "û": "u",
        "Û": "U",
        "₺": "TL",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
    }
)


def to_ascii(text: str) -> str:
    """Türkçe karakterleri OpenCV'nin çizebileceği ASCII karşılıklarına çevirir."""
    return text.translate(_TRANSLATION)


def wrap_text(text: str, max_chars: int) -> list[str]:
    """Metni kelime sınırlarından bölerek satırlara ayırır."""
    if max_chars <= 0:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]
