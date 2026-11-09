"""Giyilebilir asistan giriş noktası.

Çalıştırma::

    python -m app.main_wearable                  # kamera + otopilot + ses
    python -m app.main_wearable --no-gui        # başsız (gözlük donanımı / CI)
    python -m app.main_wearable --bench 15      # 15 sn ölçüm, FPS raporu
    python -m app.main_wearable --no-tts        # sessiz (ses aygıtı yoksa)

Tuşlar (pencereli mod):

    boşluk / s   Derin analiz (sahne açıklaması)
    e            Son tüketim tarihi (STT) oku
    o            Yalnızca metin oku (OCR)
    m            Mod bilgisini seslendir
    t            Sesi aç/kapat
    q / ESC      Çıkış

Mimari: Kamera, YOLO ve Vision LLM ayrı thread'lerde çalışır; bu döngü
yalnızca kare okur, HUD çizer ve tuş işler — hiçbir ağ işlemi beklemez.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

from app import __version__
from app.analysis import AnalysisMode, AnalysisWorker, create_vlm_client
from app.camera import CameraError, CameraStream, list_cameras
from app.config import Settings, get_settings
from app.detection import DetectionWorker, ObjectDetector
from app.modes import AutopilotMode
from app.tts import Priority, create_tts
from app.ui.wearable_hud import draw_wearable_hud
from app.utils.logging import get_logger, setup_logging
from app.wearable import ModeName, WearableAssistant

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.main_wearable",
        description="EvrenAid Giyilebilir — gözlük asistanı (otopilot + derin analiz)",
    )
    parser.add_argument("--version", action="version", version=f"EvrenAid {__version__}")
    parser.add_argument("--no-gui", action="store_true", help="Pencere açma (başsız)")
    parser.add_argument("--no-tts", action="store_true", help="Sesi kapat")
    parser.add_argument("--no-vlm", action="store_true", help="Vision LLM'i kapat")
    parser.add_argument("--list-cameras", action="store_true", help="Kameraları listele")
    parser.add_argument("--camera-index", type=int, help="Kamera indeksi")
    parser.add_argument("--model", help="YOLO ağırlığı (örn. models/evrenaidd-v1.pt)")
    parser.add_argument("--vlm-model", help="Vision model (örn. qwen2.5vl:7b)")
    parser.add_argument("--confidence", type=float, help="Güven eşiği (0-1)")
    parser.add_argument(
        "--bench",
        type=float,
        default=0.0,
        metavar="SN",
        help="Belirtilen süre çalış, FPS/gecikme raporu yazdır ve çık",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Süre sonunda çık (0 = sonsuz)")
    parser.add_argument(
        "--deep-every",
        type=float,
        default=0.0,
        help="Her N saniyede otomatik derin analiz (0 = kapalı)",
    )
    parser.add_argument(
        "--announce-labels",
        action="store_true",
        help="Her karede algılanan nesneleri sesle bildir (stres testi / sürekli anlatım)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Log seviyesi",
    )
    return parser


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    camera, detection, vlm, log = (
        settings.camera,
        settings.detection,
        settings.vlm,
        settings.log,
    )
    if args.camera_index is not None:
        camera = replace(camera, index=args.camera_index)
    if args.model:
        detection = replace(detection, model_path=args.model)
    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            raise SystemExit("--confidence 0.0 ile 1.0 arasında olmalı")
        detection = replace(detection, confidence=args.confidence)
    if args.vlm_model:
        vlm = replace(vlm, model=args.vlm_model, api_model="")
    if args.log_level:
        log = replace(log, level=args.log_level)
    return replace(settings, camera=camera, detection=detection, vlm=vlm, log=log)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_cameras:
        available = list_cameras()
        print(
            "Erişilebilir kameralar: " + ", ".join(map(str, available))
            if available
            else "Kamera bulunamadı."
        )
        return 0

    settings = _apply_overrides(get_settings(), args)
    setup_logging(settings.log, force=True)
    logger.info("EvrenAid Giyilebilir %s başlıyor", __version__)

    # --- parçaları kur (donanım hatası kullanıcıya Türkçe anlatılır) ------
    detector = ObjectDetector(settings.detection)
    try:
        detector.load()
    except Exception as exc:  # noqa: BLE001
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    tts = None if args.no_tts else create_tts(settings.tts)
    if tts is not None and not tts.enabled:
        logger.warning("Sesli geri bildirim kapalı (motor yok)")

    analysis_worker: AnalysisWorker | None = None
    if not args.no_vlm:
        client = create_vlm_client(settings.vlm)
        analysis_worker = AnalysisWorker(
            client,
            model=settings.vlm.model,
            jpeg_quality=85,
            max_side=1024,
        )

    camera = CameraStream(settings.camera)
    detection_worker = DetectionWorker(detector)
    autopilot = AutopilotMode(
        tts,
        settings=settings.spatial,
        announce_labels=args.announce_labels,
    )
    assistant = WearableAssistant(
        settings,
        camera=camera,
        detector=detector,
        detection_worker=detection_worker,
        analysis_worker=analysis_worker,
        tts=tts,
        autopilot=autopilot,
    )

    # Analiz sonucu → seslendirme → canlı moda dönüş
    if analysis_worker is not None:
        analysis_worker.set_callbacks(
            on_result=assistant.on_analysis_result,
            on_error=assistant.on_analysis_error,
        )

    cv2 = None
    if not args.no_gui:
        try:
            import cv2 as _cv2  # noqa: PLC0415
        except ImportError:
            print("OpenCV bulunamadı. Kurulum: pip install -r requirements.txt", file=sys.stderr)
            return 1
        cv2 = _cv2

    exit_code = 0
    started = time.monotonic()
    frames = 0
    last_sequence = -1
    fps_samples: list[float] = []
    loop_gaps: list[float] = []
    prev_frame_started = time.monotonic()
    next_deep = time.monotonic() + args.deep_every if args.deep_every > 0 else float("inf")
    bench_until = started + args.bench if args.bench > 0 else None
    duration_until = started + args.duration if args.duration > 0 else None

    try:
        assistant.start()
        if tts is not None and tts.enabled:
            tts.say(
                "Gözlük asistanı hazır. Otopilot açık.",
                priority=Priority.BACKGROUND,
            )

        while True:
            sequence, frame = camera.read_latest(last_sequence)
            if frame is None:
                if not camera.is_running:
                    raise CameraError(camera.last_error or "Kamera akışı durdu")
                time.sleep(0.002)
                continue

            last_sequence = sequence
            frames += 1

            report = assistant.handle_frame(frame)
            fps_samples.append(camera.fps)

            frame_started = time.monotonic()
            now = frame_started
            if now >= next_deep and not assistant.is_frozen:
                next_deep = now + args.deep_every
                assistant.request_deep_analysis(frame, mode=AnalysisMode.DESCRIBE)

            if cv2 is not None:
                draw_wearable_hud(
                    frame,
                    mode=assistant.mode,
                    report=report if report is not None else assistant.last_report,
                    fps=camera.fps,
                    tts_busy=tts.is_busy if tts is not None else False,
                    tts_text=tts.current_text if tts is not None else "",
                    deep_state=assistant.deep.state,
                    frozen=assistant.is_frozen,
                )
                cv2.imshow(settings.ui.window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("s"), 13):
                    assistant.request_deep_analysis(frame, mode=AnalysisMode.DESCRIBE)
                if key == ord("e"):
                    assistant.request_deep_analysis(frame, mode=AnalysisMode.EXPIRY)
                if key == ord("o"):
                    assistant.request_deep_analysis(frame, mode=AnalysisMode.OCR)
                if key == ord("m") and tts is not None:
                    mode_name = (
                        "Otopilot" if assistant.mode is ModeName.AUTOPILOT else "Derin analiz"
                    )
                    tts.say(f"{mode_name} modundasınız.")
                if key == ord("t") and tts is not None:
                    tts.toggle()

            # Döngü tıkanmasını ölç: ardışık kare başlangıçları arası süre.
            # (Kamera FPS'i donanımla sınırlı; asıl risk uzun tıkanmalar.)
            loop_gaps.append(frame_started - prev_frame_started)
            prev_frame_started = frame_started

            if bench_until is not None and now >= bench_until:
                break
            if duration_until is not None and now >= duration_until:
                break
    except CameraError as exc:
        logger.error("Kamera hatası: %s", exc)
        print(f"HATA: {exc}", file=sys.stderr)
        exit_code = 1
    except KeyboardInterrupt:
        logger.info("Kullanıcı durdurdu")
    finally:
        assistant.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started, 1e-6)
    average_fps = frames / elapsed
    stats = assistant.stats
    autopilot_stats = stats["autopilot"]

    logger.info(
        "Özet: %.1f sn | %s kare | %.1f FPS | tespit=%s | kritik=%s | yakın=%s | derin=%s",
        elapsed,
        frames,
        average_fps,
        getattr(autopilot_stats, "frames", 0),
        getattr(autopilot_stats, "critical_alerts", 0),
        getattr(autopilot_stats, "near_alerts", 0),
        stats["deep_requests"],
    )

    if bench_until is not None:
        # Isınma payı: model/kamera/ses ilk saniyelerde yavaştır.
        warmup = 3.0
        gaps = sorted(loop_gaps[int(len(loop_gaps) * 0.2) :]) or [0.0]
        p50 = gaps[len(gaps) // 2]
        p95 = gaps[int(len(gaps) * 0.95)] if len(gaps) > 1 else gaps[0]
        print(
            f"BENCH  süre={elapsed:.1f}s  kare={frames}  ort={average_fps:.1f} FPS  "
            f"medyan={p50 * 1000:.1f} ms  p95={p95 * 1000:.1f} ms  "
            f"max={gaps[-1] * 1000:.1f} ms  kritik={getattr(autopilot_stats, 'critical_alerts', 0)}"
        )
        _ = warmup  # belgelendirme amaçlı
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
