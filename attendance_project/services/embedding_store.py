"""
services/embedding_store.py
=============================
In-memory cache of the embedding gallery, backed by the `embeddings` table.

Attendance-taking needs to compare against *all* known embeddings on every
run. Re-querying + re-deserializing the DB on every single face in every
classroom photo would be wasteful, so this module loads the gallery once
and exposes an explicit `invalidate()` to call after registering a new
student or deleting one.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

import numpy as np

from database import get_session
from models.models import Embedding, Student
from recognition.face_recognizer import FaceRecognizer
from utils.logger import get_logger

logger = get_logger("services.embedding_store")


class EmbeddingGallery:
    """Thread-safe, lazily-populated cache of (student_db_id, embedding) pairs."""

    _lock = threading.Lock()
    _gallery: List[Tuple[int, np.ndarray]] | None = None
    _student_lookup: Dict[int, Student] | None = None

    @classmethod
    def get(cls) -> List[Tuple[int, np.ndarray]]:
        with cls._lock:
            if cls._gallery is None:
                cls._load()
            return cls._gallery

    @classmethod
    def get_student_lookup(cls) -> Dict[int, "StudentSnapshot"]:
        with cls._lock:
            if cls._student_lookup is None:
                cls._load()
            return cls._student_lookup

    @classmethod
    def invalidate(cls) -> None:
        """Force a reload on next access. Call after any student/embedding change."""
        with cls._lock:
            cls._gallery = None
            cls._student_lookup = None
        logger.info("Embedding gallery cache invalidated.")

    @classmethod
    def _load(cls) -> None:
        logger.info("Loading embedding gallery from database...")
        gallery: List[Tuple[int, np.ndarray]] = []
        lookup: Dict[int, "StudentSnapshot"] = {}

        with get_session() as session:
            rows = session.query(Embedding).all()
            for row in rows:
                vector = FaceRecognizer.deserialize(row.vector)
                gallery.append((row.student_id, vector))

            students = session.query(Student).filter(Student.is_active.is_(True)).all()
            for s in students:
                lookup[s.id] = StudentSnapshot(
                    db_id=s.id,
                    student_id=s.student_id,
                    name=s.name,
                    department=s.department,
                    section=s.section,
                )

        cls._gallery = gallery
        cls._student_lookup = lookup
        logger.info("Gallery loaded: %d embeddings, %d active students.", len(gallery), len(lookup))


class StudentSnapshot:
    """Lightweight, detached copy of student fields safe to use after the DB session closes."""

    def __init__(self, db_id: int, student_id: str, name: str, department: str | None, section: str | None):
        self.db_id = db_id
        self.student_id = student_id
        self.name = name
        self.department = department
        self.section = section
