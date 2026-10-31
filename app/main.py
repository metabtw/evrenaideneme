"""EvrenAid giriş noktası.

Çalıştırma::

    python -m app.main                 # canlı kamera + YOLO + HUD
    python -m app.main --list-cameras  # erişilebilir kamera indeksleri
    python -m app.main --no-gui        # başsız (CI / SSH) mod

Mimari not: Kamera ve tespit ayrı thread'lerde çalışır. Ana thread yalnızca
en son kareyi okur, son tespit sonucunu çizer ve tuşları işler; hiçbir ağır iş
ana döngüyü bloklamaz.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

from app import __version__
from app.camera import CameraError, CameraStream, list_cameras
from app.config import Settings, get_settings
from app.detection import DetectionWorker, ObjectDetector
from app.detection.draw import draw_detections
from app.state import AppState, AppStatus
from app.ui import draw_hud, latest_labels
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.main",
        description="EvrenAid — hibrit YOLOv8 + Vision LLM kamera asistanı",
    )
    parser.add_argument("--version", action="version", version=f"EvrenAid {__version__}")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Pencere açmadan çalıştır (başsız test/CI için)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Erişilebilir kamera indekslerini listele ve çık",
    )
    parser.add_argument("--camera-index", type=int, help="Kamera indeksini geçersiz kıl")
    parser.add_argument("--model", help="YOLO ağırlık dosyası (örn. yolov8n.pt)")
    parser.add_argument("--confidence", type=float, help="Güven eşiği (0.0 - 1.0)")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Belirtilen saniye sonra otomatik çık (0 = sonsuz)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Log seviyesini geçersiz kıl",
    )
    return parser


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    camera = settings.camera
    detection = settings.detection
    log = settings.log

    if args.camera_index is not None:
        camera = replace(camera, index=args.camera_index)
    if args.model:
        detection = replace(detection, model_path=args.model)
    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            raise SystemExit("--confidence 0.0 ile 1.0 arasında olmalı")
        detection = replace(detection, confidence=args.confidence)
    if args.log_level:
        log = replace(log, level=args.log_level)

    return replace(settings, camera=camera, detection=detection, log=log)


def _import_cv2():
    try:
        import cv2  # noqa: PLC0415 - ağır bağımlılık
    except ImportError:
        print(
            "OpenCV bulunamadı. Kurulum: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    return cv2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_cameras:
        available = list_cameras()
        if available:
            print("Erişilebilir kameralar: " + ", ".join(str(index) for index in available))
        else:
            print("Kamera bulunamadı. Cihaz bağlı ve izinler açık mı?")
        return 0

    settings = _apply_overrides(get_settings(), args)
    setup_logging(settings.log, force=True)
    logger.info("EvrenAid %s başlıyor", __version__)
    if settings.loaded_env_file:
        logger.info("Yapılandırma dosyası: %s", settings.loaded_env_file)
    else:
        logger.info(".env bulunamadı; varsayılan ayarlar kullanılıyor")

    detector = ObjectDetector(settings.detection)
    try:
        detector.load()
    except Exception as exc:  # noqa: BLE001 - kullanıcıya net mesaj
        logger.error("%s", exc)
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    state = AppState()
    state.set_status(AppStatus.CAPTURING)

    worker = DetectionWorker(detector)
    camera = CameraStream(settings.camera)
    cv2 = None if args.no_gui else _import_cv2()

    exit_code = 0
    started = time.monotonic()
    frames = 0
    last_sequence = -1
    try:
        camera.start()
        worker.start()

        while True:
            sequence, frame = camera.read_latest(last_sequence)
            if frame is None:
                if not camera.is_running:
                    raise CameraError(camera.last_error or "Kamera akışı durdu")
                time.sleep(0.002)
                continue

            last_sequence = sequence
            frames += 1
            worker.submit(frame)

            result = worker.latest_result()
            detections = result.detections if result else ()
            state.set_detections(len(detections), result.inference_ms if result else 0.0)
            state.set_fps(camera.fps)

            if cv2 is not None:
                draw_detections(frame, detections)
                snapshot = state.snapshot()
                if settings.ui.hud_enabled:
                    message = snapshot.last_answer or snapshot.last_error
                    draw_hud(
                        frame,
                        fps=snapshot.fps,
                        status=snapshot.status,
                        detection_count=snapshot.detection_count,
                        inference_ms=snapshot.inference_ms,
                        message=message,
                        show_fps=settings.ui.show_fps,
                    )
                cv2.imshow(settings.ui.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Çıkış tuşuna basıldı")
                    break
                if key == ord("t"):
                    enabled = state.toggle_tts()
                    logger.info("Sesli yanıt: %s", "açık" if enabled else "kapalı")
                if key == ord(settings.ui.trigger_key):
                    note = "Analiz M2 fazında eklenecek."
                    state.set_answer(note)
                    labels = result.labels() if result else []
                    logger.info("Analiz tetiklendi: %s", latest_labels(labels) or "nesne yok")
            else:
                labels = result.labels() if result else []
                if frames % 30 == 0:
                    logger.info(
                        "kare=%s fps=%.1f tespit=%s %s",
                        frames,
                        camera.fps,
                        len(labels),
                        latest_labels(labels),
                    )

            if args.duration and time.monotonic() - started >= args.duration:
                logger.info("--duration %.1f sn doldu", args.duration)
                break
    except CameraError as exc:
        logger.error("Kamera hatası: %s", exc)
        print(f"HATA: {exc}", file=sys.stderr)
        exit_code = 1
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu")
    finally:
        worker.stop(timeout=3.0)
        camera.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started, 1e-6)
    stats = worker.stats
    logger.info(
        "Özet: %.1f sn, %s kare, ortalama %.1f FPS, %s tespit işlendi, %s kare düşürüldü",
        elapsed,
        frames,
        frames / elapsed,
        stats.processed,
        stats.dropped,
    )
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
