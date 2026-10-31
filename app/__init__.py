"""EvrenAid — hibrit YOLOv8 + Vision LLM masaüstü kamera asistanı.

Paket giriş noktası: ``python -m app.main``.

Modüler yapı:
    config/     .env -> tip güvenli ayarlar
    camera/     webcam yakalama (thread)
    detection/  YOLOv8 tespiti, çizim, tespit işçisi
    analysis/   Vision LLM istemcisi ve Türkçe prompt'lar (M2)
    tts/        sesli geri bildirim motorları (M3)
    ui/         HUD / overlay
    utils/      loglama, kare kodlama, metin yardımcıları
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
