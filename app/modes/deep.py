"""Mod 2 — Derin analiz: istek üzerine kare dondurma + Vision LLM + seslendirme.

Akış (kullanıcı düğmeye bastığında):

    1. **Dondur (freeze):** o anki kare kopyalanır; otopilot uyarıları susar.
    2. **Sor (ask):** kare JPEG'e çevrilip Ollama'ya gönderilir (worker thread).
    3. **Oku (speak):** yanıt Türkçe seslendirilir; STT modunda tarih ayrıştırılır.
    4. **Dön (resume):** canlı moda dönülür; otopilot sayaçları sıfırlanır.

Neden kare dondurulur?
    Kullanıcı ürünün üzerindeki *belirli bir* STT'yi sorar. Kamera akmaya devam
    ederse model başka bir kareyi görebilir ve yanlış tarih okunur. Dondurma,
    "baktığım şeyi sor" garantisidir.

STT (Son Tüketim Tarihi) ayrıştırma:
    Model serbest metin döner; biz tarih kalıplarını yakalayıp kullanıcıya
    *kısa* ve *net* bir cümle söyleriz ("Son tüketim tarihi 12 Mart 2027").
    Tarih bulunamazsa ham yanıt okunur — asla uydurulmaz.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date

from app.analysis import AnalysisMode, AnalysisResult
from app.config import SpatialSettings
from app.tts import Priority, TextToSpeech, to_speech_text
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Türkçe ay adları (model 12.03.2027 veya "12 Mart 2027" dönebilir).
_MONTHS_TR = {
    1: "Ocak",
    2: "Şubat",
    3: "Mart",
    4: "Nisan",
    5: "Mayıs",
    6: "Haziran",
    7: "Temmuz",
    8: "Ağustos",
    9: "Eylül",
    10: "Ekim",
    11: "Kasım",
    12: "Aralık",
}

_MONTH_NAMES = {name.lower(): number for number, name in _MONTHS_TR.items()}
_MONTH_NAMES.update(
    {"subat": 2, "mayis": 5, "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12}
)

#: 12.03.2027 | 12/03/2027 | 12-03-2027 | 12.03.27
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")
#: 12 Mart 2027 | 12 MART 27
_NAMED_DATE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-zÇĞİÖŞÜçğıöşü]{3,10})\.?\s+(\d{2,4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ExpiryInfo:
    """Bir görselden çıkarılan son tüketim tarihi."""

    raw: str
    parsed: date | None = None
    expired: bool | None = None
    days_left: int | None = None

    @property
    def spoken(self) -> str:
        """Seslendirilecek kısa, net cümle."""
        if self.parsed is None:
            return ""
        base = f"Son tüketim tarihi {self.parsed.day} {_MONTHS_TR[self.parsed.month]} {self.parsed.year}."
        if self.expired is True:
            days = abs(self.days_left or 0)
            return f"{base} Ürünün süresi {days} gün önce dolmuş."
        if self.days_left is not None and self.days_left <= 3:
            return f"{base} Son {self.days_left} gün."
        return base


@dataclass(slots=True)
class DeepAnalysisState:
    """Derin analiz modunun anlık durumu."""

    active: bool = False
    frozen_at: float | None = None
    last_result: AnalysisResult | None = None
    last_expiry: ExpiryInfo | None = None
    requests: int = 0

    @property
    def frozen_seconds(self) -> float:
        if self.frozen_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self.frozen_at)


class DeepAnalysisMode:
    """İstek üzerine derin analiz modu (kare dondurma + VLM + TTS)."""

    def __init__(
        self,
        *,
        tts: TextToSpeech | None = None,
        spatial_settings: SpatialSettings | None = None,
        speak_raw_on_no_date: bool = True,
    ) -> None:
        self._tts = tts
        self._speak_raw_on_no_date = speak_raw_on_no_date
        self._spatial_settings = spatial_settings or SpatialSettings()
        self.state = DeepAnalysisState()

    # -- yaşam döngüsü -----------------------------------------------------

    def begin(self) -> None:
        """Kare dondurulur: otopilot uyarıları bu süre boyunca susar."""
        self.state.active = True
        self.state.frozen_at = time.monotonic()
        self.state.requests += 1
        if self._tts is not None:
            # Kısa onay: kullanıcı düğmenin çalıştığını duyar.
            self._tts.say("Bakıyorum.", priority=Priority.BACKGROUND)

    def complete(
        self, result: AnalysisResult, *, mode: AnalysisMode = AnalysisMode.DESCRIBE
    ) -> str:
        """VLM yanıtını işler, seslendirir ve canlı moda döner.

        Returns:
            Kullanıcıya söylenen (veya söylenecek) metin.
        """
        self.state.last_result = result

        if mode is AnalysisMode.EXPIRY:
            expiry = parse_expiry(result.answer)
            self.state.last_expiry = expiry
            if expiry.parsed is not None:
                message = expiry.spoken
            elif self._speak_raw_on_no_date:
                message = to_speech_text(result.answer)
            else:
                message = "Son tüketim tarihini okuyamadım. Ürünü biraz yaklaştırıp tekrar deneyin."
        else:
            message = to_speech_text(result.answer)

        if self._tts is not None and message:
            self._tts.say(message, priority=Priority.NORMAL)

        self.end()
        return message

    def fail(self, message: str) -> str:
        """Analiz başarısız: kullanıcıya Türkçe, kısa bir açıklama söyler."""
        spoken = f"Analiz yapılamadı. {message}"
        if self._tts is not None:
            self._tts.say(spoken, priority=Priority.NORMAL)
        self.end()
        return spoken

    def end(self) -> None:
        """Canlı moda dön."""
        self.state.active = False
        self.state.frozen_at = None

    @property
    def is_active(self) -> bool:
        return self.state.active


def parse_expiry(text: str, *, today: date | None = None) -> ExpiryInfo:
    """Serbest metinden son tüketim tarihini çıkarır.

    Desteklenen biçimler: ``12.03.2027``, ``12/03/27``, ``12 Mart 2027``.
    Yorum yapmaz; tarih yoksa ``parsed=None`` döner (uydurma yok).
    """
    reference = today or date.today()

    match = _NUMERIC_DATE.search(text)
    if match:
        day, month, year = (int(part) for part in match.groups())
        return _build_expiry(text, day, month, _normalize_year(year), reference)

    match = _NAMED_DATE.search(text)
    if match:
        day = int(match.group(1))
        month_name = match.group(2).lower()
        month = _MONTH_NAMES.get(month_name)
        if month is not None:
            year = _normalize_year(int(match.group(3)))
            return _build_expiry(text, day, month, year, reference)

    return ExpiryInfo(raw=text.strip())


def _normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def _build_expiry(text: str, day: int, month: int, year: int, today: date) -> ExpiryInfo:
    try:
        parsed = date(year, month, day)
    except ValueError:
        return ExpiryInfo(raw=text.strip())

    delta = (parsed - today).days
    return ExpiryInfo(
        raw=text.strip(),
        parsed=parsed,
        expired=delta < 0,
        days_left=delta,
    )
