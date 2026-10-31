"""Tek görsel üzerinde Vision LLM analizi (M2 komut satırı aracı).

Kullanım::

    python -m app.analyze --image ornek.jpg
    python -m app.analyze --image ekran.png --mode ocr
    python -m app.analyze --image mutfak.jpg --mode count --target bardak
    python -m app.analyze --check          # Ollama + model hazır mı?

Kamera açılmaz; yalnızca verilen dosya analiz edilir. Uygulamanın ana döngüsü
ile aynı istemci ve prompt kodunu kullanır.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.analysis.prompts import AnalysisMode, build_prompt
from app.analysis.vlm_client import (
    OllamaClient,
    VLMError,
    create_vlm_client,
)
from app.config import get_settings
from app.utils.frames import FrameEncodingError, encode_jpeg
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.analyze",
        description="EvrenAid — tek görsel üzerinde Türkçe Vision LLM analizi",
    )
    parser.add_argument("--image", type=Path, help="Analiz edilecek görsel dosyası")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in AnalysisMode],
        default=AnalysisMode.DESCRIBE.value,
        help="Analiz modu (varsayılan: describe)",
    )
    parser.add_argument("--target", help="COUNT modu için nesne adı (örn. bardak)")
    parser.add_argument("--model", help="Model adını geçersiz kıl (örn. qwen2.5vl:7b)")
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai-compatible"),
        help="Sağlayıcıyı geçersiz kıl",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Yalnızca sağlayıcı/model hazır mı kontrol et",
    )
    parser.add_argument("--json", action="store_true", help="Sonucu JSON olarak yazdır")
    parser.add_argument("--quality", type=int, default=85, help="JPEG kalitesi (1-100)")
    parser.add_argument(
        "--max-side", type=int, default=1024, help="Görselin uzun kenarı en fazla (px)"
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="WARNING",
        help="Log seviyesi (varsayılan: WARNING)",
    )
    return parser


def _load_frame(path: Path) -> np.ndarray:
    try:
        import cv2  # noqa: PLC0415 - ağır bağımlılık
    except ImportError:
        raise SystemExit("OpenCV bulunamadı. Kurulum: pip install -r requirements.txt") from None

    if not path.is_file():
        raise SystemExit(f"Görsel bulunamadı: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Görsel okunamadı: {path} (desteklenen biçimler: jpg, png, bmp, webp)")
    return frame


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = get_settings()
    vlm = settings.vlm
    if args.model:
        vlm = replace(vlm, model=args.model, api_model="")
    if args.provider:
        vlm = replace(vlm, provider=args.provider)

    setup_logging(replace(settings.log, level=args.log_level), force=True)

    if args.check:
        return _check(vlm)

    if args.image is None:
        raise SystemExit("--image zorunlu (veya --check kullanın)")

    mode = AnalysisMode(args.mode)
    if mode is AnalysisMode.COUNT and not args.target:
        raise SystemExit("--mode count için --target gerekli (örn. --target bardak)")

    client = create_vlm_client(vlm)
    try:
        frame = _load_frame(args.image)
        jpeg = encode_jpeg(frame, quality=args.quality, max_side=args.max_side)
        prompt = build_prompt(mode, target=args.target, model=vlm.model)

        started = time.perf_counter()
        answer = client.describe(jpeg, prompt)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if args.json:
            print(
                json.dumps(
                    {
                        "provider": client.name,
                        "mode": mode.value,
                        "image": str(args.image),
                        "elapsed_ms": round(elapsed_ms, 1),
                        "jpeg_kb": round(len(jpeg) / 1024, 1),
                        "answer": answer,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(answer)
            print(f"\n[{client.name} | {mode.value} | {elapsed_ms / 1000:.2f} sn]")
        return 0
    except (VLMError, FrameEncodingError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


def _check(vlm) -> int:
    """Sağlayıcının ve modelin hazır olup olmadığını bildirir."""
    if vlm.provider != "ollama":
        try:
            client = create_vlm_client(vlm)
        except VLMError as exc:
            print(f"HATA: {exc}", file=sys.stderr)
            return 1
        print(f"{client.name}: yapılandırma tamam (istek atılmadı).")
        client.close()
        return 0

    client = OllamaClient(vlm)
    ready, message = client.health_check()
    print(("HAZIR: " if ready else "EKSİK: ") + message)
    client.close()
    return 0 if ready else 1


__all__ = ["build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
