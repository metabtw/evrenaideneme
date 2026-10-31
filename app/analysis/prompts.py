"""Türkçe prompt şablonları (M2).

Tüm prompt'lar tek yerde tutulur; model adına göre kısa bir yönlendirme eklenir.
Amaç: kısa (≤ 3 cümle), somut, Türkçe yanıtlar.
"""

from __future__ import annotations

from enum import Enum


class AnalysisMode(str, Enum):
    """Analiz türleri (``s`` = DESCRIBE, ``o`` = OCR)."""

    DESCRIBE = "describe"
    OCR = "ocr"
    COUNT = "count"


DESCRIBE = (
    "Sahneyi en fazla 3 cümleyle Türkçe açıkla. "
    "Görünen nesneleri, kişileri ve konumlarını (sol/sağ/ön/arka) belirt. "
    "Tahminlerini 'görünüyor' diye ifade et; kesin olmayan şeyleri uydurma."
)

OCR = (
    "Görseldeki tüm yazıyı olduğu gibi, satır satır yaz. "
    "Yorum, çeviri veya açıklama ekleme. Okunamayan kısımlar için [okunamadı] yaz."
)

COUNT_TEMPLATE = (
    "Görseldeki '{target}' nesnelerini say. "
    "Yanıtı tek satırda 'Sayı: N - kısa gerekçe' biçiminde Türkçe ver."
)

_MODEL_HINTS: dict[str, str] = {
    "llava": "Yanıtı kısa tut; en fazla 3 cümle.",
    "qwen": "Yanıtı kısa tut; metin okurken Türkçe karakterleri koru.",
    "gemma": "Yanıtı kısa tut; yalnızca görünen bilgiye dayan.",
}


def build_prompt(
    mode: AnalysisMode | str = AnalysisMode.DESCRIBE,
    *,
    target: str | None = None,
    model: str | None = None,
) -> str:
    """Moda (ve varsa modele) uygun Türkçe prompt üretir.

    Raises:
        ValueError: Bilinmeyen mod veya ``COUNT`` için hedef nesne verilmediyse.
    """
    try:
        selected = AnalysisMode(mode)
    except ValueError as exc:
        raise ValueError(f"Bilinmeyen analiz modu: {mode!r}") from exc

    if selected is AnalysisMode.DESCRIBE:
        prompt = DESCRIBE
    elif selected is AnalysisMode.OCR:
        prompt = OCR
    else:
        if not target:
            raise ValueError("COUNT modu için 'target' nesne adı gerekli")
        prompt = COUNT_TEMPLATE.format(target=target)

    hint = _model_hint(model)
    return f"{prompt} {hint}".strip() if hint else prompt


def _model_hint(model: str | None) -> str:
    if not model:
        return ""
    lowered = model.lower()
    for keyword, hint in _MODEL_HINTS.items():
        if keyword in lowered:
            return hint
    return ""
