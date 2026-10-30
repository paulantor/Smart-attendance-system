"""
config.py
=========
Central configuration for the AI Smart Classroom Attendance System.

Every path, threshold, and tunable parameter used across the project is
defined here so that the rest of the codebase never hardcodes a value.
Settings can be overridden at runtime (see services/settings_service.py),
which persists changes to database/settings.json.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"          # raw + cropped face images per student
EMBEDDINGS_DIR = BASE_DIR / "embeddings"    # cached .npy embeddings (backup of DB)
REPORTS_DIR = BASE_DIR / "reports"          # exported CSV / Excel / PDF
DATABASE_DIR = BASE_DIR / "database"        # sqlite file + backups + settings.json
LOGS_DIR = BASE_DIR / "logs"                # rotating log files
MODELS_DIR = BASE_DIR / "models_weights"    # downloaded InsightFace/RetinaFace weights

for _dir in (DATASET_DIR, EMBEDDINGS_DIR, REPORTS_DIR, DATABASE_DIR, LOGS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'attendance.db'}"
SETTINGS_FILE = DATABASE_DIR / "settings.json"
LOG_FILE = LOGS_DIR / "app.log"


@dataclass
class RecognitionConfig:
    """Parameters controlling face detection & recognition behaviour."""

    # Cosine similarity above which two embeddings are considered a match.
    # Higher = stricter (fewer false positives, more false negatives).
    similarity_threshold: float = 0.45

    # Threshold used ONLY when checking if a newly captured registration face
    # already belongs to a different student (duplicate-registration guard).
    # Deliberately LOWER (more lenient) than similarity_threshold, because a
    # live-camera capture has more lighting/angle variation than attendance
    # photos, and it's safer to over-flag a possible duplicate than miss one.
    duplicate_face_threshold: float = 0.35

    # Minimum RetinaFace detector confidence to accept a detected face box.
    detector_confidence: float = 0.90

    # Faces smaller than this (in pixels, on the shorter side) are rejected
    # during registration as "too small / too far from camera".
    min_face_size_px: int = 80

    # Laplacian variance below this is considered "blurred".
    blur_threshold: float = 60.0

    # Mean pixel brightness (0-255) below this is considered "too dark".
    min_brightness: float = 40.0

    # Target number of images to capture per student during registration.
    # A small guided-pose sequence: one frame per pose, each pose checked via
    # utils/pose_estimation.py before being accepted.
    min_registration_images: int = 5
    max_registration_images: int = 10

    # The guided pose sequence shown one-at-a-time during registration.
    # Each entry is (pose_key, instruction_text). pose_key is passed to
    # utils.pose_estimation.matches_expected_pose().
    guided_pose_sequence: tuple = (
        ("center", "Look straight at the camera"),
        ("left", "Slowly turn your face to the LEFT"),
        ("right", "Slowly turn your face to the RIGHT"),
        ("up", "Tilt your head slightly UP"),
        ("down", "Tilt your head slightly DOWN"),
    )

    # Embedding vector size produced by the recognition model (ArcFace / InsightFace buffalo_l = 512-d).
    embedding_dim: int = 512

    # If a student is detected in two overlapping boxes (duplicate), boxes whose
    # IoU exceeds this are merged into a single detection.
    duplicate_iou_threshold: float = 0.5

    # A face recognized with similarity between threshold and late_grace_margin
    # is still marked Present; this margin exists only for future "Late" logic.
    late_grace_minutes: int = 10


@dataclass
class CameraConfig:
    """Defaults for the iPhone / webcam capture widgets in Streamlit."""

    resolution: str = "1280x960"      # shown in Settings page, informational
    jpeg_quality: int = 92


@dataclass
class AppConfig:
    app_name: str = "AI Smart Classroom Attendance System"
    version: str = "1.0.0"
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    export_formats: tuple = ("CSV", "Excel", "PDF")
    timezone: str = "Asia/Dhaka"


# Singleton config instance used throughout the app.
settings = AppConfig()