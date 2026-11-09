"""Yakınlık / engel analizi: sınırlayıcı kutu sezgisi (bounding-box heuristic).

Neden kutu sezgisi?
    Giyilebilir gözlükte stereo çift kamera veya LiDAR yoksa tek karede mutlak
    derinlik ölçülemez. Kutu büyüklüğü, nesne kameraya yaklaştıkça artar; bu
    yüzden "karenin dikey alanını ne kadar kaplıyor" sorusu ucuz ve deterministik
    bir **vekil (proxy)** metriktir. Gerçek derinlik değildir.

Dürüst sınırlar:
    * Uzaktaki bir otobüs de kareyi doldurabilir → yanlış "kritik" uyarı.
    * Çok yakındaki küçük bir nesne (anahtar) kareyi doldurmaz → kaçırılır.
    Bu yüzden üç koruma vardır: tehlike sınıfı beyaz listesi, art arda N kare
    onayı (debounce) ve etiket başına cooldown. Böylece "sürekli bağıran"
    bir cihaz yerine ölçülü bir uyarı üretilir.

Sorumluluk sınırı:
    Bu modül **karar verir, seslendirmez**. Seslendirme çağıranın işidir
    (bkz. ``modes.autopilot``); böylece analiz saf ve test edilebilir kalır.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

from app.config import SpatialSettings
from app.detection import Detection


class ProximityLevel(str, Enum):
    """Kutu büyüklüğüne göre yakınlık sınıfı."""

    SAFE = "guvenli"
    NEAR = "yakin"
    CRITICAL = "kritik"

    @property
    def rank(self) -> int:
        return {ProximityLevel.SAFE: 0, ProximityLevel.NEAR: 1, ProximityLevel.CRITICAL: 2}[self]


class Zone(str, Enum):
    """Nesnenin yatay konumu (yönlendirme için)."""

    LEFT = "solunuzda"
    CENTER = "onunuzde"
    RIGHT = "saginizda"


@dataclass(frozen=True, slots=True)
class ProximityAlert:
    """Tek bir nesne için yakınlık değerlendirmesi."""

    level: ProximityLevel
    label: str
    vertical_ratio: float
    area_ratio: float
    zone: Zone
    box: tuple[int, int, int, int]

    @property
    def message(self) -> str:
        """Türkçe, kısa, eyleme dönük uyarı metni."""
        if self.level is ProximityLevel.CRITICAL:
            return f"Dikkat, {self.label} çok yakın, {self.zone.value}."
        return f"{self.label} yaklaşıyor, {self.zone.value}."


@dataclass(frozen=True, slots=True)
class SpatialReport:
    """Bir kareye ait yakınlık raporu."""

    alerts: tuple[ProximityAlert, ...] = ()
    announcements: tuple[ProximityAlert, ...] = ()
    nearest: ProximityAlert | None = None
    occupied_ratio: float = 0.0
    frame_size: tuple[int, int] = (0, 0)

    @property
    def is_critical(self) -> bool:
        return any(alert.level is ProximityLevel.CRITICAL for alert in self.alerts)


class SpatialAnalyzer:
    """Tespitleri yakınlık uyarılarına çeviren durum makinesi.

    Durum tutar (sayaç/cooldown), bu yüzden aynı örnek ana döngüde tek
    thread'den çağrılmalıdır. İç kilit yine de koruma sağlar.
    """

    def __init__(self, settings: SpatialSettings | None = None) -> None:
        self.settings = settings or SpatialSettings()
        self._lock = threading.Lock()
        # etiket -> art arda kaç karedir eşiği aştığı
        self._streak: dict[str, int] = {}
        # etiket -> son seslendirme zamanı (monotonic)
        self._last_announced: dict[str, float] = {}

    def reset(self) -> None:
        """Durumu sıfırlar (mod değişimi / kare dondurma sonrası)."""
        with self._lock:
            self._streak.clear()
            self._last_announced.clear()

    def analyze(
        self,
        detections: list[Detection] | tuple[Detection, ...],
        frame_shape: tuple[int, ...],
        *,
        now: float | None = None,
    ) -> SpatialReport:
        """Tespitleri değerlendirir.

        Args:
            detections: YOLO tespitleri (Türkçe etiketli).
            frame_shape: ``frame.shape`` (H, W, ...) veya ``(H, W)``.
            now: Testler için enjekte edilebilir zaman (``time.monotonic``).

        Returns:
            Anlık uyarılar + bu karede **seslendirilmesi gereken** yeniler.
        """
        if len(frame_shape) < 2:
            raise ValueError(f"Geçersiz kare şekli: {frame_shape!r}")

        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        if frame_height <= 0 or frame_width <= 0:
            return SpatialReport(frame_size=(frame_width, frame_height))

        timestamp = time.monotonic() if now is None else now
        frame_area = float(frame_height * frame_width)

        alerts: list[ProximityAlert] = []
        announcements: list[ProximityAlert] = []
        occupied = 0.0

        with self._lock:
            for detection in detections:
                alert = self._evaluate(detection, frame_height, frame_width, frame_area)
                if alert is None:
                    self._streak.pop(detection.label, None)
                    continue

                occupied = max(occupied, alert.area_ratio)
                alerts.append(alert)

                if not self._allowed(alert.label):
                    continue

                streak = self._streak.get(alert.label, 0) + 1
                self._streak[alert.label] = streak
                if streak < self.settings.confirm_frames:
                    continue

                last = self._last_announced.get(alert.label)
                if last is not None and timestamp - last < self.settings.announce_cooldown:
                    continue

                self._last_announced[alert.label] = timestamp
                announcements.append(alert)

            # Karede görünmeyen etiketlerin sayacı sıfırlanır (birikmesin).
            visible = {alert.label for alert in alerts}
            for label in list(self._streak):
                if label not in visible:
                    self._streak.pop(label, None)

        alerts.sort(key=lambda alert: (alert.level.rank, alert.vertical_ratio), reverse=True)
        announcements.sort(key=lambda alert: alert.level.rank, reverse=True)

        return SpatialReport(
            alerts=tuple(alerts),
            announcements=tuple(announcements),
            nearest=alerts[0] if alerts else None,
            occupied_ratio=occupied,
            frame_size=(frame_width, frame_height),
        )

    # -- iç işleyiş --------------------------------------------------------

    def _evaluate(
        self,
        detection: Detection,
        frame_height: int,
        frame_width: int,
        frame_area: float,
    ) -> ProximityAlert | None:
        """Tek tespiti seviyeye çevirir; ``SAFE`` ise ``None`` döner."""
        x1, y1, x2, y2 = detection.box
        box_height = max(0, min(y2, frame_height) - max(y1, 0))
        box_width = max(0, min(x2, frame_width) - max(x1, 0))
        if box_height <= 0 or box_width <= 0:
            return None

        vertical_ratio = box_height / frame_height
        area_ratio = (box_height * box_width) / frame_area

        if vertical_ratio >= self.settings.critical_ratio:
            level = ProximityLevel.CRITICAL
        elif vertical_ratio >= self.settings.near_ratio:
            level = ProximityLevel.NEAR
        else:
            return None

        center_x = (x1 + x2) / 2.0
        third = frame_width / 3.0
        if center_x < third:
            zone = Zone.LEFT
        elif center_x > 2 * third:
            zone = Zone.RIGHT
        else:
            zone = Zone.CENTER

        return ProximityAlert(
            level=level,
            label=detection.label,
            vertical_ratio=vertical_ratio,
            area_ratio=area_ratio,
            zone=zone,
            box=detection.box,
        )

    def _allowed(self, label: str) -> bool:
        """Tehlike beyaz listesi: sadece bu sınıflar sesli uyarı üretir."""
        if not self.settings.hazard_only:
            return True
        if not self.settings.hazard_classes:
            return True
        return label in self.settings.hazard_classes


def default_hazard_classes() -> tuple[str, ...]:
    """Yürüyüş sırasında çarpma riski taşıyan sınıflar (Türkçe etiketler)."""
    return (
        "kişi",
        "araba",
        "bisiklet",
        "motosiklet",
        "otobüs",
        "kamyon",
        "tren",
        "sandalye",
        "masa",
        "kanepe",
        "yatak",
        "kapı",
        "duvar",
        "dolap",
        "raf",
        "saksı bitkisi",
        "vazo",
        "çöp kovası",
        "merdiven",
        "bank",
    )
