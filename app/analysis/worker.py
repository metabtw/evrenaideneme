"""Analiz işçisi: Vision LLM çağrısını ana döngüden ayırır.

Kurallar (ROADMAP M4 ile uyumlu):
    * **Tek uçuş (single-flight):** aynı anda yalnızca bir VLM isteği açık.
    * **Latest-frame-wins:** istek sürerken gelen yeni tetikleme kuyruğa
      yığılmaz; bekleyen kare son kareyle değiştirilir.
    * **Kopya:** kare ana thread'de kopyalanır, JPEG kodlaması işçi thread'inde
      yapılır (ana döngüde CPU harcanmaz).
    * **Dayanıklılık:** istemci hatası işçiyi öldürmez; çağırana Türkçe mesaj
      iletilir, akış devam eder.

    worker = AnalysisWorker(client, model=settings.vlm.model, on_result=...)
    worker.start()
    worker.submit(frame, mode=AnalysisMode.DESCRIBE)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.analysis.prompts import AnalysisMode, build_prompt
from app.analysis.vlm_client import VLMClient, VLMError
from app.utils.frames import FrameEncodingError, encode_jpeg
from app.utils.logging import get_logger

logger = get_logger(__name__)

ResultCallback = Callable[["AnalysisResult"], None]
ErrorCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Başarılı bir analizin sonucu."""

    answer: str
    mode: AnalysisMode
    elapsed_ms: float
    frame_id: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """İşçiye bırakılan tek istek."""

    frame_id: int
    frame: np.ndarray
    mode: AnalysisMode
    target: str | None


class AnalysisWorker:
    """Tek worker thread'i ile VLM isteklerini sıraya koyar."""

    def __init__(
        self,
        client: VLMClient,
        *,
        model: str | None = None,
        on_result: ResultCallback | None = None,
        on_error: ErrorCallback | None = None,
        jpeg_quality: int = 85,
        max_side: int | None = 1024,
        name: str = "evrenaidd-analysis",
    ) -> None:
        self._client = client
        self._model = model
        self._on_result = on_result
        self._on_error = on_error
        self._jpeg_quality = jpeg_quality
        self._max_side = max_side
        self._name = name

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._pending: AnalysisRequest | None = None
        self._in_flight = False
        self._latest: AnalysisResult | None = None
        self._next_frame_id = 0
        self._replaced = 0
        self._completed = 0
        self._last_error = ""

    # -- yaşam döngüsü -----------------------------------------------------

    def start(self) -> AnalysisWorker:
        if self.is_running:
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        logger.debug("Analiz işçisi başladı (%s)", self._name)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """İşçiyi durdurur; süren istek varsa bitmesini (en fazla ``timeout``) bekler."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Analiz işçisi %.1f sn içinde durmadı", timeout)
        self._thread = None
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - kapanış hatası akışı etkilemesin
            logger.debug("VLM istemcisi kapatılırken hata", exc_info=True)

    def __enter__(self) -> AnalysisWorker:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_callbacks(
        self,
        *,
        on_result: ResultCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        """Geri çağrıları (yeniden) bağlar.

        İşçi kurulduktan sonra orkestratörün kancalarını bağlamak için gerekir;
        özel alanlara dokunmadan aynı işlevi sunar.
        """
        self._on_result = on_result
        self._on_error = on_error

    # -- üretici tarafı ----------------------------------------------------

    def submit(
        self,
        frame: np.ndarray,
        *,
        mode: AnalysisMode | str = AnalysisMode.DESCRIBE,
        target: str | None = None,
    ) -> int:
        """Kareyi analiz kuyruğuna bırakır ve hemen döner.

        Bekleyen bir istek varsa (henüz başlamamış) yenisiyle değiştirilir.

        Returns:
            İsteğe atanan artan kimlik.

        Raises:
            ValueError: Kare boşsa veya mod geçersizse.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Boş kare analiz edilemez")
        selected = AnalysisMode(mode)
        if selected is AnalysisMode.COUNT and not target:
            raise ValueError("COUNT modu için hedef nesne adı gerekli")

        payload = frame.copy()
        with self._condition:
            if self._pending is not None:
                self._replaced += 1
            self._next_frame_id += 1
            frame_id = self._next_frame_id
            self._pending = AnalysisRequest(frame_id, payload, selected, target)
            self._condition.notify()
        return frame_id

    def cancel(self) -> None:
        """Henüz başlamamış isteği iptal eder (süren istek durdurulamaz)."""
        with self._condition:
            if self._pending is not None:
                self._pending = None
                logger.debug("Bekleyen analiz isteği iptal edildi")

    # -- tüketici tarafı ---------------------------------------------------

    def latest_result(self) -> AnalysisResult | None:
        with self._condition:
            return self._latest

    @property
    def is_busy(self) -> bool:
        """İstek sürüyor veya kuyrukta bekliyor mu?"""
        with self._condition:
            return self._in_flight or self._pending is not None

    @property
    def last_error(self) -> str:
        with self._condition:
            return self._last_error

    @property
    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "completed": self._completed,
                "replaced": self._replaced,
                "pending": int(self._pending is not None),
                "in_flight": int(self._in_flight),
            }

    def wait_for_result(self, timeout: float = 30.0) -> AnalysisResult | None:
        """Bir sonraki sonucu (en fazla ``timeout`` sn) bekler."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._latest

    # -- thread gövdesi ----------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            request = self._take_request()
            if request is None:
                return

            started = time.perf_counter()
            try:
                answer = self._analyze(request)
            except (VLMError, FrameEncodingError) as exc:
                self._fail(request, str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - işçi asla ölmemeli
                logger.exception("Analizde beklenmeyen hata")
                self._fail(request, f"Analiz sırasında beklenmeyen hata: {exc}")
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result = AnalysisResult(
                answer=answer,
                mode=request.mode,
                elapsed_ms=elapsed_ms,
                frame_id=request.frame_id,
                timestamp=time.time(),
            )
            with self._condition:
                self._latest = result
                self._completed += 1
                self._last_error = ""
                self._in_flight = False
                self._condition.notify_all()

            logger.info(
                "Analiz tamamlandı (%.0f ms, mod=%s): %s",
                elapsed_ms,
                request.mode.value,
                answer[:120],
            )
            if self._on_result is not None:
                self._safe_call(self._on_result, result)

    def _take_request(self) -> AnalysisRequest | None:
        with self._condition:
            while self._pending is None and not self._stop_event.is_set():
                self._condition.wait(timeout=0.2)
            if self._stop_event.is_set() and self._pending is None:
                return None
            request, self._pending = self._pending, None
            self._in_flight = request is not None
            return request

    def _analyze(self, request: AnalysisRequest) -> str:
        prompt = build_prompt(request.mode, target=request.target, model=self._model)
        frame_jpeg = encode_jpeg(request.frame, quality=self._jpeg_quality, max_side=self._max_side)
        logger.debug(
            "VLM isteği: %s, mod=%s, jpeg=%s KB",
            self._client.name,
            request.mode.value,
            len(frame_jpeg) // 1024,
        )
        return self._client.describe(frame_jpeg, prompt)

    def _fail(self, request: AnalysisRequest, message: str) -> None:
        with self._condition:
            self._last_error = message
            self._in_flight = False
            self._condition.notify_all()
        logger.error("Analiz başarısız (kare #%s): %s", request.frame_id, message)
        if self._on_error is not None:
            self._safe_call(self._on_error, message)

    def _safe_call(self, callback: Callable[[object], None], value: object) -> None:
        try:
            callback(value)
        except Exception:  # noqa: BLE001 - callback hatası işçiyi öldürmemeli
            logger.exception("Analiz callback'i hata verdi")
