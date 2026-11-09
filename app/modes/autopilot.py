"""Mod 1 — Otopilot: sürekli, ucuz, kesintisiz tarama.

Sorumluluk:
    * Her karede YOLO tespiti (worker thread'de).
    * Yakınlık analizi (ana thread'de, saf ve hızlı).
    * Kritik engel varsa **sesli** uyarı (barge-in ile).
    * Banknot tespiti (varsa) → tutar bilgisini periyodik seslendir.

Neden bu mod "ultra hafif" olmalı?
    Giyilebilir cihazda pil ve gecikme kritik. Ağır iş (VLM) buraya girmez;
    yalnızca kutu sezgisi ve kısa cümleler kullanılır.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import SpatialSettings
from app.detection import Detection
from app.spatial import ProximityLevel, SpatialAnalyzer, SpatialReport
from app.tts import Priority, TextToSpeech
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Banknot sınıf adları (özel modelle eğitildiğinde devreye girer).
BANKNOTE_LABELS = frozenset(
    {
        "banknot",
        "türk lirası",
        "turk lirasi",
        "turkish lira",
        "para",
        "5 lira",
        "10 lira",
        "20 lira",
        "50 lira",
        "100 lira",
        "200 lira",
    }
)


@dataclass(slots=True)
class AutopilotStats:
    """Otopilot sayaçları (HUD/log için)."""

    frames: int = 0
    critical_alerts: int = 0
    near_alerts: int = 0
    banknote_reports: int = 0
    last_labels: tuple[str, ...] = ()
    last_report: SpatialReport | None = None
    started_at: float = field(default_factory=time.monotonic)

    @property
    def uptime(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)


class AutopilotMode:
    """Sürekli tarama modu.

    Durum tutar (yakınlık sayaçları, banknot cooldown'ı) fakat I/O yapmaz;
    kareyi ve tespitleri dışarıdan alır. Bu sayede test edilebilir kalır.
    """

    def __init__(
        self,
        tts: TextToSpeech | None = None,
        *,
        spatial: SpatialAnalyzer | None = None,
        settings: SpatialSettings | None = None,
        banknote_cooldown: float = 8.0,
        announce_labels: bool = False,
    ) -> None:
        self._tts = tts
        self.analyzer = spatial or SpatialAnalyzer(settings)
        self._banknote_cooldown = banknote_cooldown
        self._announce_labels = announce_labels
        self._last_banknote_at = 0.0
        self._last_banknote_total: int | None = None
        self.stats = AutopilotStats()

    # -- her karede çağrılır ----------------------------------------------

    def process(
        self,
        detections: list[Detection] | tuple[Detection, ...],
        frame_shape: tuple[int, ...],
        *,
        now: float | None = None,
    ) -> SpatialReport:
        """Tespitleri değerlendirir, gerekli sesli uyarıları yayınlar."""
        timestamp = time.monotonic() if now is None else now
        report = self.analyzer.analyze(detections, frame_shape, now=timestamp)

        self.stats.frames += 1
        self.stats.last_labels = tuple(detection.label for detection in detections)
        self.stats.last_report = report

        for alert in report.announcements:
            if alert.level is ProximityLevel.CRITICAL:
                self.stats.critical_alerts += 1
            else:
                self.stats.near_alerts += 1

            if self._tts is not None:
                if alert.level is ProximityLevel.CRITICAL:
                    # Kesme: sürmekte olan açıklamayı durdurur (güvenlik).
                    self._tts.say_critical(alert.message)
                else:
                    self._tts.say(alert.message, priority=Priority.NORMAL)
            logger.info("Yakınlık uyarısı: %s (%.0f%%)", alert.message, alert.vertical_ratio * 100)

        self._report_banknotes(detections, timestamp)

        if self._announce_labels and not report.announcements and detections:
            unique = list(dict.fromkeys(d.label for d in detections))[:3]
            if self._tts is not None:
                self._tts.say(", ".join(unique), priority=Priority.BACKGROUND)

        return report

    def _report_banknotes(
        self,
        detections: list[Detection] | tuple[Detection, ...],
        timestamp: float,
    ) -> None:
        """Banknot gördüyse tutarı periyodik olarak seslendirir."""
        found = [d for d in detections if d.label.lower() in BANKNOTE_LABELS]
        if not found:
            self._last_banknote_total = None
            return

        total = self._sum_banknotes(found)
        if total is None:
            return

        if (
            total == self._last_banknote_total
            and timestamp - self._last_banknote_at < self._banknote_cooldown
        ):
            return

        self._last_banknote_at = timestamp
        self._last_banknote_total = total
        self.stats.banknote_reports += 1
        message = f"Toplam {total} lira görüyorum."
        if self._tts is not None:
            self._tts.say(message, priority=Priority.NORMAL)
        logger.info("Banknot raporu: %s", message)

    @staticmethod
    def _sum_banknotes(found: list[Detection]) -> int | None:
        """Etiketlerden tutar toplar; bilinmeyen etiket varsa ``None``.

        Not: varsayılan COCO modeli banknot tanımaz. Bu yol, kendi banknot
        modeliniz ``labels`` içinde "100 lira" gibi adlar ürettiğinde çalışır.
        """
        total = 0
        for detection in found:
            label = detection.label.lower()
            for value in (200, 100, 50, 20, 10, 5):
                if str(value) in label:
                    total += value
                    break
            else:
                return None
        return total or None

    def reset(self) -> None:
        """Kare dondurma / mod geçişi sonrası sayaçları temizler."""
        self.analyzer.reset()
        self._last_banknote_total = None
