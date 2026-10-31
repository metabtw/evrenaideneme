"""Konsol + döner dosya loglaması.

Kullanım::

    from app.config import get_settings
    from app.utils.logging import setup_logging

    setup_logging(get_settings().log)

Kurulum idempotenttir; ikinci çağrı yeni handler eklemez.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import LogSettings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(settings: LogSettings | None = None, *, force: bool = False) -> None:
    """Kök logger'ı yapılandırır.

    Args:
        settings: Log ayarları. ``None`` ise varsayılanlar kullanılır.
        force: Daha önce yapılandırılmış olsa bile yeniden kur.
    """
    global _configured
    if _configured and not force:
        return

    settings = settings or LogSettings()
    level = getattr(logging, settings.level.upper(), logging.INFO)

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    if settings.console:
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)
        console.setLevel(level)
        root.addHandler(console)

    try:
        log_path = settings.path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)
    except OSError as exc:  # salt-okunur dizin, izin hatası vb.
        root.warning("Log dosyası açılamadı (%s); yalnızca konsola yazılacak.", exc)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Modül logger'ı döndürür."""
    return logging.getLogger(name)
