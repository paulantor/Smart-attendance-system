"""
utils/image_quality.py
=======================
Image-quality gatekeeping used during student registration.

A captured frame is only accepted into dataset/<student_id>/ if it passes
all checks below. This keeps the embedding database free of noisy,
low-quality samples that would otherwise hurt recognition accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from config import settings


@dataclass
class QualityResult:
    accepted: bool
    reason: Optional[str] = None          # human-readable rejection reason (None if accepted)
    brightness: Optional[float] = None
    blur_score: Optional[float] = None
    face_count: int = 0


def compute_brightness(image_bgr: np.ndarray) -> float:
    """Mean pixel intensity on the grayscale image (0-255)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian -- low values indicate a blurred image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate_frame(
    image_bgr: np.ndarray,
    face_boxes: List[tuple],
) -> QualityResult:
    """
    Validate a captured frame against all registration quality rules.

    Parameters
    ----------
    image_bgr : the raw captured frame (BGR, as returned by OpenCV)
    face_boxes : list of (x1, y1, x2, y2) boxes already found by the detector
                 for this frame (pass the detector's output in -- this
                 function does not perform detection itself).

    Returns
    -------
    QualityResult with `accepted=True` only if every check passes.
    """
    cfg = settings.recognition

    brightness = compute_brightness(image_bgr)
    blur_score = compute_blur_score(image_bgr)
    face_count = len(face_boxes)

    if face_count == 0:
        return QualityResult(False, "No face detected", brightness, blur_score, face_count)

    if face_count > 1:
        return QualityResult(False, "More than one face detected", brightness, blur_score, face_count)

    x1, y1, x2, y2 = face_boxes[0]
    face_w, face_h = x2 - x1, y2 - y1
    if min(face_w, face_h) < cfg.min_face_size_px:
        return QualityResult(
            False, "Face too small / too far from camera", brightness, blur_score, face_count
        )

    if brightness < cfg.min_brightness:
        return QualityResult(False, "Image too dark / low brightness", brightness, blur_score, face_count)

    if blur_score < cfg.blur_threshold:
        return QualityResult(False, "Image is blurred", brightness, blur_score, face_count)

    return QualityResult(True, None, brightness, blur_score, face_count)


def crop_face(image_bgr: np.ndarray, box: tuple, margin: float = 0.2) -> np.ndarray:
    """Crop a face region with a proportional margin around the detected box."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    mx, my = int(bw * margin), int(bh * margin)
    x1c = max(0, x1 - mx)
    y1c = max(0, y1 - my)
    x2c = min(w, x2 + mx)
    y2c = min(h, y2 + my)

    return image_bgr[y1c:y2c, x1c:x2c].copy()
