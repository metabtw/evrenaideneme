"""Giyilebilir asistan orkestratörü.

Bu sınıf **yeni bir iş mantığı içermez**; parçaları doğru sırayla bağlar:

    CameraStream ──► DetectionWorker ──► AutopilotMode ──► TextToSpeech
                            │                   (Mod 1: sürekli)
                            │
                            └──► DeepAnalysisMode ──► VisionLLM ──► TextToSpeech
                                       (Mod 2: istek üzerine, kare dondurma)

Tasarım kararları:
    * Ana döngü **hiçbir zaman** ağ beklemez: VLM çağrısı ``AnalysisWorker``
      thread'inde, TTS ``TextToSpeech`` thread'inde çalışır.
    * Mod geçişi tek bir bayrakla yapılır (``_mode``); dondurulmuş kare
      ``_frozen_frame`` içinde tutulur.
    * Her parça enjekte edilebilir (kamera, dedektör, VLM, TTS). Testler
      donanım/ağ olmadan tüm akışı sürebilir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from app.analysis import AnalysisMode, AnalysisResult, AnalysisWorker
from app.camera import CameraStream
from app.config import Settings
from app.detection import DetectionWorker, ObjectDetector
from app.modes import AutopilotMode, DeepAnalysisMode
from app.spatial import SpatialReport
from app.tts import Priority, TextToSpeech
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModeName(str, Enum):
    """Çalışma modu."""

    AUTOPILOT = "autopilot"
    DEEP = "deep"


class Clock(Protocol):
    """Testlerde zamanı kontrol edebilmek için minimal saat arayüzü."""

    def monotonic(self) -> float: ...


class WearableAssistant:
    """Giyilebilir akıllı gözlük asistanının çekirdeği."""

    def __init__(
        self,
        settings: Settings,
        *,
        camera: CameraStream | None = None,
        detector: ObjectDetector | None = None,
        detection_worker: DetectionWorker | None = None,
        analysis_worker: AnalysisWorker | None = None,
        tts: TextToSpeech | None = None,
        autopilot: AutopilotMode | None = None,
        deep: DeepAnalysisMode | None = None,
    ) -> None:
        self.settings = settings
        self.camera = camera
        self.detector = detector
        self.detection_worker = detection_worker
        self.analysis_worker = analysis_worker
        self.tts = tts
        self.autopilot = autopilot or AutopilotMode(tts, settings=settings.spatial)
        self.deep = deep or DeepAnalysisMode(tts=tts, spatial_settings=settings.spatial)

        self._mode = ModeName(settings.wearable.start_mode)
        self._frozen_frame: np.ndarray | None = None
        self._last_report: SpatialReport | None = None
        self._running = False
        self._frames = 0
        self._started_at = 0.0
        self._deep_started_at: float | None = None
        #: Derin analiz bu süreden uzun sürerse kullanıcıya bilgi verilir.
        self.deep_timeout = 25.0

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> WearableAssistant:
        if self.camera is not None:
            self.camera.start()
        if self.detection_worker is not None:
            self.detection_worker.start()
        if self.analysis_worker is not None:
            self.analysis_worker.start()

        self._running = True
        self._started_at = time.monotonic()

        if self.settings.wearable.speak_on_start and self.tts is not None:
            self.tts.say("Asistan hazır. Otopilot modunda.", priority=Priority.BACKGROUND)
        logger.info("Giyilebilir asistan başladı (mod=%s)", self._mode.value)
        return self

    def close(self) -> None:
        """Tüm thread'leri ve donanımı sırayla kapatır (asılı thread bırakmaz)."""
        self._running = False
        for part in (self.analysis_worker, self.detection_worker):
            if part is not None:
                part.stop(timeout=3.0)
        if self.camera is not None:
            self.camera.stop()
        if self.tts is not None:
            self.tts.close()
        logger.info("Giyilebilir asistan kapandı (%s kare)", self._frames)

    def __enter__(self) -> WearableAssistant:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    # -- mod yönetimi ------------------------------------------------------

    @property
    def mode(self) -> ModeName:
        return self._mode

    def request_deep_analysis(
        self,
        frame: np.ndarray | None = None,
        *,
        mode: AnalysisMode = AnalysisMode.DESCRIBE,
        target: str | None = None,
    ) -> bool:
        """Kullanıcı düğmesi / sesli komutu: kareyi dondur, VLM'e gönder.

        Args:
            frame: Dondurulacak kare. ``None`` ise son kare kullanılır.
            mode: Analiz türü (``EXPIRY`` = son tüketim tarihi).
            target: ``COUNT`` modu için nesne adı.

        Returns:
            İstek kabul edildiyse ``True`` (meşgulse ``False``).
        """
        if self.analysis_worker is None:
            logger.warning("Analiz işçisi yok; derin analiz yapılamaz")
            return False
        if self.analysis_worker.is_busy:
            logger.info("Derin analiz isteği yok sayıldı: meşgul")
            return False

        frozen = frame if frame is not None else self._frozen_frame
        if frozen is None:
            logger.warning("Dondurulacak kare yok")
            return False

        self._frozen_frame = frozen.copy()
        self._mode = ModeName.DEEP
        self._deep_started_at = time.monotonic()
        self.deep.begin()

        try:
            self.analysis_worker.submit(self._frozen_frame, mode=mode, target=target)
        except ValueError as exc:
            logger.error("Analiz isteği reddedildi: %s", exc)
            self._exit_deep()
            return False
        return True

    def resume(self) -> None:
        """Canlı (otopilot) moda dön."""
        self._exit_deep()

    def _exit_deep(self) -> None:
        if self._mode is ModeName.AUTOPILOT:
            return
        self._mode = ModeName.AUTOPILOT
        self._deep_started_at = None
        self._frozen_frame = None
        self.deep.end()
        self.autopilot.reset()
        logger.debug("Otopilot moduna dönüldü")

    @property
    def is_frozen(self) -> bool:
        return self._mode is ModeName.DEEP

    @property
    def frozen_frame(self) -> np.ndarray | None:
        return self._frozen_frame

    # -- ana adım ----------------------------------------------------------

    def handle_frame(
        self,
        frame: np.ndarray,
        *,
        process: bool = True,
    ) -> SpatialReport | None:
        """Bir kareyi işler; ana döngüden çağrılır (bloklamaz).

        Args:
            frame: BGR kare.
            process: ``False`` ise yalnızca dondurulmuş kare korunur.

        Returns:
            Otopilot modundaysa yakınlık raporu, derin modda ``None``.
        """
        self._frames += 1

        if self._mode is ModeName.DEEP:
            self._check_deep_timeout()
            return None

        if not process or self.detection_worker is None:
            return None

        self.detection_worker.submit(frame)
        result = self.detection_worker.latest_result()
        detections = result.detections if result else ()
        self._last_report = self.autopilot.process(detections, frame.shape)
        return self._last_report

    def on_analysis_result(
        self, result: AnalysisResult, *, mode: AnalysisMode | None = None
    ) -> str:
        """Analiz işçisinin geri çağrısı: yanıtı seslendirir, moda döner.

        Bu metot ``AnalysisWorker(on_result=...)`` ile bağlanmalıdır.
        """
        selected = mode or result.mode
        message = self.deep.complete(result, mode=selected)
        self._exit_deep()
        return message

    def on_analysis_error(self, message: str) -> None:
        """Analiz hatası: kullanıcıya Türkçe bilgi ver, akışı sürdür."""
        if self.settings.wearable.speak_errors:
            self.deep.fail(message)
        self._exit_deep()

    def _check_deep_timeout(self) -> None:
        if self._deep_started_at is None:
            return
        elapsed = time.monotonic() - self._deep_started_at
        if elapsed > self.deep_timeout:
            logger.warning("Derin analiz %.0f sn sürdü; canlı moda dönülüyor", elapsed)
            if self.tts is not None:
                self.tts.say("Analiz uzun sürüyor, canlı moda dönüyorum.")
            self._exit_deep()

    # -- durum -------------------------------------------------------------

    @property
    def last_report(self) -> SpatialReport | None:
        return self._last_report

    @property
    def stats(self) -> dict[str, object]:
        return {
            "mode": self._mode.value,
            "frames": self._frames,
            "uptime": max(0.0, time.monotonic() - self._started_at),
            "autopilot": self.autopilot.stats,
            "deep_requests": self.deep.state.requests,
            "tts_busy": self.tts.is_busy if self.tts is not None else False,
        }


@dataclass(frozen=True, slots=True)
class WearableStatus:
    """HUD / log için sadeleştirilmiş anlık durum."""

    mode: ModeName
    fps: float
    detection_count: int
    nearest_label: str
    nearest_ratio: float
    is_critical: bool
    tts_busy: bool
