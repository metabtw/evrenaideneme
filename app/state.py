"""Thread'ler arasında paylaşılan, kilitli uygulama durumu.

Ana döngü her karede :meth:`AppState.snapshot` ile tutarlı bir kopya alır;
arka plan işçileri (tespit, VLM, TTS) durumu bu sınıf üzerinden günceller.
Böylece çizim kodu hiçbir zaman yarım kalmış bir durum görmez.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from enum import Enum


class AppStatus(str, Enum):
    """Durum makinesi (M4'te genişletilecek)."""

    IDLE = "HAZIR"
    CAPTURING = "İZLENİYOR"
    ANALYZING = "ANALİZ EDİLİYOR"
    SPEAKING = "KONUŞUYOR"
    ERROR = "HATA"

    @property
    def ascii(self) -> str:
        """OpenCV HUD'ında çizilebilir ASCII karşılığı."""
        return {
            AppStatus.IDLE: "HAZIR",
            AppStatus.CAPTURING: "IZLENIYOR",
            AppStatus.ANALYZING: "ANALIZ EDILIYOR",
            AppStatus.SPEAKING: "KONUSUYOR",
            AppStatus.ERROR: "HATA",
        }[self]


@dataclass(frozen=True, slots=True)
class AppSnapshot:
    """Kilit alınmadan okunabilen tutarlı durum kopyası."""

    status: AppStatus = AppStatus.IDLE
    fps: float = 0.0
    detection_count: int = 0
    inference_ms: float = 0.0
    last_answer: str = ""
    last_error: str = ""
    tts_enabled: bool = True
    updated_at: float = 0.0


class AppState:
    """Paylaşılan durumu kilit altında tutan küçük yardımcı."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = AppSnapshot(updated_at=time.time())

    def snapshot(self) -> AppSnapshot:
        with self._lock:
            return self._snapshot

    def set_status(self, status: AppStatus) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, status=status, updated_at=time.time())

    def set_detections(self, count: int, inference_ms: float) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                detection_count=count,
                inference_ms=inference_ms,
                updated_at=time.time(),
            )

    def set_fps(self, fps: float) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, fps=fps, updated_at=time.time())

    def set_answer(self, answer: str) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                last_answer=answer,
                last_error="",
                updated_at=time.time(),
            )

    def set_error(self, message: str) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                last_error=message,
                status=AppStatus.ERROR,
                updated_at=time.time(),
            )

    def clear_error(self) -> None:
        """Hata mesajını siler ve durumu akışa döndürür (rozet sıfırlama)."""
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                last_error="",
                status=AppStatus.CAPTURING,
                updated_at=time.time(),
            )

    def clear_answer(self) -> None:
        """Son yanıtı temizler (``c`` tuşu)."""
        with self._lock:
            self._snapshot = replace(
                self._snapshot, last_answer="", last_error="", updated_at=time.time()
            )

    def set_tts_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, tts_enabled=enabled)

    def toggle_tts(self) -> bool:
        with self._lock:
            self._snapshot = replace(self._snapshot, tts_enabled=not self._snapshot.tts_enabled)
            return self._snapshot.tts_enabled
