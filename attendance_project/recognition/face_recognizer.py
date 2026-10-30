"""
recognition/face_recognizer.py
================================
Face recognition using InsightFace's pretrained ArcFace embedding model
(no CNN training happens anywhere in this project -- we only use pretrained
embeddings, as required).

Responsibilities:
  1. Generate a 512-d L2-normalized embedding for a cropped face image.
  2. Compare a query embedding against a gallery of known embeddings using
     cosine similarity, returning the best match if it clears the
     configurable threshold.

Embeddings are cached in the `embeddings` table (see models/models.py) so
they are computed once at registration time and never recomputed during
attendance -- satisfying the "cache embeddings, don't recompute" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from recognition.face_detector import FaceDetector
from utils.logger import get_logger

logger = get_logger("recognition.face_recognizer")


@dataclass
class MatchResult:
    student_db_id: Optional[int]   # primary key in `students` table, None if unknown
    similarity: float
    is_match: bool


class FaceRecognizer:
    """
    Wraps InsightFace's recognition head. Reuses the same shared FaceAnalysis
    model instance as FaceDetector (InsightFace bundles detection +
    recognition in a single pipeline), so no extra model is loaded.
    """

    def __init__(self):
        # Ensures the shared model is loaded (idempotent / cached at class level).
        self._detector = FaceDetector()

    def embed_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run detection + recognition on an image expected to contain exactly
        one face (e.g. a registration photo) and return its embedding.

        Returns None if no face (or more than one) was found.
        """
        faces = self._detector.detect(image_bgr)
        if len(faces) != 1:
            logger.warning("embed_face expected exactly 1 face, found %d", len(faces))
            return None

        embedding = faces[0].embedding
        if embedding is None:
            return None

        return self._normalize(np.asarray(embedding, dtype=np.float32))

    def embed_all_faces(self, image_bgr: np.ndarray):
        """
        Detect + embed every face in a classroom photo. Used during attendance.

        Returns a list of (DetectedFace, embedding) tuples.
        """
        faces = self._detector.detect(image_bgr)
        output = []
        for face in faces:
            if face.embedding is not None:
                output.append((face, self._normalize(np.asarray(face.embedding, dtype=np.float32))))
        return output

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    @staticmethod
    def serialize(vector: np.ndarray) -> bytes:
        """Convert an embedding to bytes for storage in the `embeddings.vector` column."""
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def deserialize(blob: bytes) -> np.ndarray:
        """Convert stored bytes back into a float32 embedding array."""
        return np.frombuffer(blob, dtype=np.float32).copy()

    def match(
        self,
        query_embedding: np.ndarray,
        gallery: List[Tuple[int, np.ndarray]],
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """
        Compare a query embedding against a gallery of (student_db_id, embedding)
        pairs using cosine similarity, returning the best match.

        Parameters
        ----------
        gallery : list of (student_db_id, embedding_vector). Typically this is
            one row per stored embedding (a student may have several); the
            best-scoring row across all of a student's embeddings wins.
        threshold : override config.recognition.similarity_threshold if given.
        """
        if not gallery:
            return MatchResult(None, 0.0, False)

        threshold = threshold if threshold is not None else settings.recognition.similarity_threshold

        gallery_ids = [g[0] for g in gallery]
        gallery_matrix = np.vstack([g[1] for g in gallery])

        sims = cosine_similarity(query_embedding.reshape(1, -1), gallery_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_student_id = gallery_ids[best_idx]

        is_match = best_score >= threshold
        return MatchResult(
            student_db_id=best_student_id if is_match else None,
            similarity=best_score,
            is_match=is_match,
        )
