"""
recognition/face_detector.py
=============================
Thin, clean wrapper around InsightFace's RetinaFace detector.

InsightFace ships RetinaFace as the detector inside its `FaceAnalysis`
pipeline. We use that pipeline for detection only here (recognition is
handled separately in face_recognizer.py via the same underlying model
pack, but exposed through its own class for clarity / single-responsibility).

Detected faces are returned as plain `DetectedFace` dataclasses so callers
never depend on InsightFace's internal types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import MODELS_DIR, settings
from utils.logger import get_logger

logger = get_logger("recognition.face_detector")


@dataclass
class DetectedFace:
    box: Tuple[int, int, int, int]        # x1, y1, x2, y2 in pixel coordinates
    confidence: float
    landmarks: np.ndarray                  # 5x2 array: eyes, nose, mouth corners
    embedding: np.ndarray | None = None    # populated later by FaceRecognizer if needed


class FaceDetector:
    """
    Detects all faces in an image using RetinaFace (via insightface.app.FaceAnalysis).

    The heavy model is loaded once (lazily, on first use) and reused for the
    lifetime of the process -- loading it per-request would be far too slow
    for an interactive Streamlit app.
    """

    _shared_app = None  # class-level cache: the FaceAnalysis model is expensive to load

    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        self.det_size = det_size
        self._app = self._get_shared_app()

    @classmethod
    def _get_shared_app(cls):
        if cls._shared_app is None:
            from insightface.app import FaceAnalysis  # imported lazily: heavy dependency

            logger.info("Loading InsightFace model pack (buffalo_l) for detection...")
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(MODELS_DIR),
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            cls._shared_app = app
            logger.info("InsightFace model pack loaded.")
        return cls._shared_app

    def detect(self, image_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR image (as produced by cv2.imread / camera capture).

        Returns a list of DetectedFace, already filtered by the configured
        detector confidence threshold, sorted left-to-right (stable ordering
        makes duplicate-suppression and UI display deterministic).
        """
        min_conf = settings.recognition.detector_confidence

        try:
            raw_faces = self._app.get(image_bgr)
        except Exception:
            logger.exception("Face detection failed")
            return []

        results: List[DetectedFace] = []
        for face in raw_faces:
            confidence = float(getattr(face, "det_score", 0.0))
            if confidence < min_conf:
                continue

            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            landmarks = np.array(face.kps) if getattr(face, "kps", None) is not None else np.zeros((5, 2))

            results.append(
                DetectedFace(
                    box=(x1, y1, x2, y2),
                    confidence=confidence,
                    landmarks=landmarks,
                    embedding=getattr(face, "normed_embedding", None),
                )
            )

        results.sort(key=lambda f: f.box[0])
        return self._suppress_duplicates(results)

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _suppress_duplicates(self, faces: List[DetectedFace]) -> List[DetectedFace]:
        """Remove overlapping duplicate detections of the same physical face."""
        threshold = settings.recognition.duplicate_iou_threshold
        kept: List[DetectedFace] = []
        for face in faces:
            if not any(self._iou(face.box, k.box) > threshold for k in kept):
                kept.append(face)
        return kept
