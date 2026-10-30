"""
utils/logger.py
================
Centralized logging configuration.

Every module calls `get_logger(__name__)` to obtain a logger that writes to
both the console and a rotating file (logs/app.log), so errors anywhere in
the system are captured for later debugging -- as required by the project spec.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_FILE

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger("attendance_system")
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (e.g. 'attendance_system.recognition.face_detector')."""
    _configure_root()
    return logging.getLogger(f"attendance_system.{name}")
