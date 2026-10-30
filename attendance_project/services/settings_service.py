"""
services/settings_service.py
==============================
Persists user-configurable settings (recognition threshold, camera
resolution, export folder, etc.) to database/settings.json, and applies
them onto the in-memory `config.settings` singleton at app startup.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from config import REPORTS_DIR, SETTINGS_FILE, settings
from utils.logger import get_logger

logger = get_logger("services.settings_service")


def load_settings() -> None:
    """Load settings.json (if present) and apply values onto config.settings."""
    if not SETTINGS_FILE.exists():
        return

    try:
        data: Dict[str, Any] = json.loads(SETTINGS_FILE.read_text())
    except Exception:
        logger.exception("Failed to parse settings.json, using defaults.")
        return

    recognition_data = data.get("recognition", {})
    for key, value in recognition_data.items():
        if hasattr(settings.recognition, key):
            setattr(settings.recognition, key, value)

    camera_data = data.get("camera", {})
    for key, value in camera_data.items():
        if hasattr(settings.camera, key):
            setattr(settings.camera, key, value)

    logger.info("Loaded settings from %s", SETTINGS_FILE)


def save_settings(
    similarity_threshold: float | None = None,
    camera_resolution: str | None = None,
    detector_confidence: float | None = None,
    duplicate_face_threshold: float | None = None,
) -> None:
    """Update config.settings in-memory and persist the full settings snapshot to disk."""
    if similarity_threshold is not None:
        settings.recognition.similarity_threshold = similarity_threshold
    if detector_confidence is not None:
        settings.recognition.detector_confidence = detector_confidence
    if camera_resolution is not None:
        settings.camera.resolution = camera_resolution
    if duplicate_face_threshold is not None:
        settings.recognition.duplicate_face_threshold = duplicate_face_threshold

    data = {
        "recognition": {
            "similarity_threshold": settings.recognition.similarity_threshold,
            "detector_confidence": settings.recognition.detector_confidence,
            "min_face_size_px": settings.recognition.min_face_size_px,
            "blur_threshold": settings.recognition.blur_threshold,
            "min_brightness": settings.recognition.min_brightness,
            "duplicate_face_threshold": settings.recognition.duplicate_face_threshold,
        },
        "camera": {
            "resolution": settings.camera.resolution,
            "jpeg_quality": settings.camera.jpeg_quality,
        },
        "export_folder": str(REPORTS_DIR),
    }
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))
    logger.info("Settings saved to %s", SETTINGS_FILE)