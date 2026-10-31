"""Thread tabanlı kamera yakalama.

Tasarım:
    * Kamera **kendi thread'inde** çalışır; ana döngü yalnızca en son kareyi
      okur. Böylece tespit/analiz gecikmesi akışı kilitlemez.
    * Kare erişimi ``threading.Condition`` ile korunur; okuyucu kopya alır.
    * Okuma hatasında (kablo çıktı, kamera meşgul) otomatik yeniden bağlanma
      denenir. Denemeler tükenirse ``last_error`` dolar ve thread durur.

Örnek::

    with CameraStream(settings.camera) as camera:
        frame = camera.read()          # None ise henüz kare yok
        print(camera.stats.fps)
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import CameraSettings
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: (index, OpenCV backend sabiti veya None) -> VideoCapture benzeri nesne
CaptureFactory = Callable[[int, "int | None"], "CaptureLike"]


class CameraError(RuntimeError):
    """Kamera açılamadı veya okunamadı."""


class CaptureLike(Protocol):
    """Testlerde taklit edilebilmesi için gereken minimum ``VideoCapture`` arayüzü."""

    def isOpened(self) -> bool: ...  # noqa: N802 - OpenCV adlandırması

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CameraStats:
    """Kamera thread'inin anlık sayaçları."""

    frames_read: int = 0
    read_failures: int = 0
    reconnects: int = 0
    fps: float = 0.0
    frame_size: tuple[int, int] = (0, 0)
    running: bool = False
    last_error: str = ""


def _default_capture_factory(index: int, backend: int | None) -> CaptureLike:
    try:
        import cv2  # noqa: PLC0415 - ağır bağımlılık, isteğe bağlı içe aktarım
    except ImportError as exc:  # pragma: no cover - ortama bağlı
        raise CameraError("OpenCV bulunamadı. Kurulum: pip install -r requirements.txt") from exc

    capture = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    return capture


def _backend_flag(name: str) -> int | None:
    """Ayar adını OpenCV backend sabitine çevirir (``auto`` -> platforma göre)."""
    if name == "auto":
        if sys.platform == "win32":
            return _backend_flag("dshow")
        if sys.platform == "darwin":
            return _backend_flag("avfoundation")
        return None

    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return None

    mapping = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", None),
        "v4l2": getattr(cv2, "CAP_V4L2", None),
    }
    return mapping.get(name)


class CameraStream:
    """Webcam akışını arka planda yakalayan, thread-güvenli okuyucu."""

    def __init__(
        self,
        settings: CameraSettings | None = None,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.settings = settings or CameraSettings()
        self._capture_factory = capture_factory or _default_capture_factory

        self._capture: CaptureLike | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()

        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._timestamps: deque[float] = deque(maxlen=30)

        self._frames_read = 0
        self._read_failures = 0
        self._reconnects = 0
        self._last_error = ""

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> CameraStream:
        """Kamerayı açar ve yakalama thread'ini başlatır.

        Raises:
            CameraError: Kamera açılamazsa (indeks yanlış, izin yok, cihaz yok).
        """
        if self.is_running:
            return self

        self._stop_event.clear()
        self._open()

        self._thread = threading.Thread(
            target=self._run,
            name="evrenaidd-camera",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Kamera açıldı (indeks=%s, %sx%s, hedef=%s FPS)",
            self.settings.index,
            self.settings.width,
            self.settings.height,
            self.settings.target_fps,
        )
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Thread'i durdurur ve kamerayı serbest bırakır. Idempotenttir."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Kamera thread'i %.1f sn içinde durmadı", timeout)

        self._thread = None
        self._release()
        logger.info("Kamera kapatıldı")

    def __enter__(self) -> CameraStream:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    # -- okuma -------------------------------------------------------------

    def read(self, *, copy: bool = True) -> np.ndarray | None:
        """En son kareyi döndürür; henüz kare yoksa ``None``.

        Args:
            copy: ``True`` (varsayılan) ise kopya döner. Ana döngü kareyi
                yerinde çizeceği için kopya önerilir; sıcak yolda kopya
                maliyetinden kaçınmak için ``False`` verilebilir.
        """
        with self._condition:
            frame = self._frame
            if frame is None:
                return None
            return frame.copy() if copy else frame

    def read_latest(
        self, previous_sequence: int = -1, *, copy: bool = True
    ) -> tuple[int, np.ndarray | None]:
        """Yeni bir kare varsa ``(sıra_no, kare)`` döndürür.

        Kamera thread'i hedef FPS'te çalışırken ana döngünün aynı kareyi
        tekrar tekrar işlemesini engeller. Kare değişmediyse sıra numarası
        ``previous_sequence`` ile aynı kalır ve kare ``None`` olur.
        """
        with self._condition:
            if self._frame is None or self._sequence == previous_sequence:
                return previous_sequence, None
            frame = self._frame
            return self._sequence, (frame.copy() if copy else frame)

    @property
    def sequence(self) -> int:
        """Son yakalanan karenin artan sıra numarası."""
        with self._condition:
            return self._sequence

    def wait_for_frame(self, timeout: float = 2.0) -> np.ndarray | None:
        """Yeni bir kare gelene kadar (en fazla ``timeout`` sn) bekler."""
        with self._condition:
            if self._frame is None:
                self._condition.wait(timeout=timeout)
            frame = self._frame
            return frame.copy() if frame is not None else None

    def list_cameras(self, max_index: int = 5) -> list[int]:
        """Bu örnekteki fabrikayı kullanarak erişilebilir kamera indekslerini döndürür."""
        return _probe_cameras(self._capture_factory, max_index)

    # -- durum -------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def fps(self) -> float:
        with self._condition:
            return _calculate_fps(self._timestamps)

    @property
    def stats(self) -> CameraStats:
        with self._condition:
            frame = self._frame
            size = (int(frame.shape[1]), int(frame.shape[0])) if frame is not None else (0, 0)
            return CameraStats(
                frames_read=self._frames_read,
                read_failures=self._read_failures,
                reconnects=self._reconnects,
                fps=_calculate_fps(self._timestamps),
                frame_size=size,
                running=self.is_running,
                last_error=self._last_error,
            )

    # -- iç işleyiş --------------------------------------------------------

    def _open(self) -> None:
        backend = _backend_flag(self.settings.backend)
        try:
            capture = self._capture_factory(self.settings.index, backend)
        except CameraError:
            raise
        except Exception as exc:  # noqa: BLE001 - donanım sürücüsü her şeyi fırlatabilir
            raise CameraError(f"Kamera açılamadı: {exc}") from exc

        if not capture.isOpened():
            try:
                capture.release()
            except Exception:  # noqa: BLE001 - kapanış hatası asıl hatayı gizlemesin
                logger.debug("Açılamayan kamera serbest bırakılırken hata", exc_info=True)
            raise CameraError(
                f"Kamera açılamadı (indeks={self.settings.index}). "
                "Cihazın bağlı ve başka uygulama tarafından kullanılmıyor "
                "olduğunu kontrol edin. Mevcut indeksler için: "
                "python -m app.main --list-cameras"
            )

        self._apply_properties(capture)
        self._capture = capture

    def _apply_properties(self, capture: CaptureLike) -> None:
        setter = getattr(capture, "set", None)
        if setter is None:
            return
        try:
            import cv2  # noqa: PLC0415
        except ImportError:
            return

        setter(cv2.CAP_PROP_FRAME_WIDTH, float(self.settings.width))
        setter(cv2.CAP_PROP_FRAME_HEIGHT, float(self.settings.height))
        setter(cv2.CAP_PROP_FPS, float(self.settings.target_fps))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            # Gecikmeyi azaltmak için sürücü tamponunu 1'e indir.
            setter(cv2.CAP_PROP_BUFFERSIZE, 1.0)

    def _release(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001 - kapanışta hata yutulur
                logger.debug("Kamera serbest bırakılırken hata", exc_info=True)

    def _run(self) -> None:
        interval = 1.0 / self.settings.target_fps if self.settings.target_fps else 0.0

        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                with self._condition:
                    self._read_failures += 1
                if not self._reconnect():
                    break
                continue

            now = time.monotonic()
            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._frames_read += 1
                self._timestamps.append(now)
                self._condition.notify_all()

            # Kamera hedef FPS'ten hızlıysa (ör. 60 FPS cihaz) nefes al.
            elapsed = time.monotonic() - now
            if interval > elapsed:
                self._stop_event.wait(interval - elapsed)

    def _reconnect(self) -> bool:
        """Ardışık okuma hatalarında kamerayı yeniden açar.

        Returns:
            ``True`` -> yeniden bağlandı, döngü devam edebilir.
            ``False`` -> denemeler tükendi veya durdurma istendi.
        """
        self._release()
        attempts = self.settings.reconnect_attempts
        for attempt in range(1, attempts + 1):
            if self._stop_event.wait(self.settings.reconnect_delay):
                return False
            try:
                self._open()
            except CameraError as exc:
                logger.warning("Kamera yeniden bağlanamadı (%s/%s): %s", attempt, attempts, exc)
                continue

            with self._condition:
                self._reconnects += 1
                self._last_error = ""
            logger.info("Kamera yeniden bağlandı (%s/%s)", attempt, attempts)
            return True

        self._last_error = (
            "Kamera bağlantısı kaybedildi ve yeniden kurulamadı. "
            "Cihazı kontrol edip uygulamayı yeniden başlatın."
        )
        logger.error(self._last_error)
        return False


def _calculate_fps(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    span = timestamps[-1] - timestamps[0]
    if span <= 0:
        return 0.0
    return (len(timestamps) - 1) / span


def _probe_cameras(factory: CaptureFactory, max_index: int = 5) -> list[int]:
    """0..max_index aralığındaki erişilebilir kamera indekslerini dener."""
    available: list[int] = []
    for index in range(max_index + 1):
        capture: CaptureLike | None = None
        try:
            capture = factory(index, _backend_flag("auto"))
            if capture.isOpened():
                available.append(index)
        except Exception:  # noqa: BLE001 - olmayan indeks normaldir
            continue
        finally:
            if capture is not None:
                # Olmayan indekslerde sürücü hata fırlatabilir; yok sayılır.
                with contextlib.suppress(Exception):
                    capture.release()
    return available


def list_cameras(max_index: int = 5) -> list[int]:
    """Sistemde erişilebilen kamera indekslerini listeler."""
    return _probe_cameras(_default_capture_factory, max_index)
