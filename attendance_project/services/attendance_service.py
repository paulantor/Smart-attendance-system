"""
services/attendance_service.py
================================
Implements Steps 3-6 of the spec: detect every face in a classroom photo,
recognize each one against the cached embedding gallery, mark attendance
(Present / Absent / Unknown), and persist everything to the database.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import DATASET_DIR, settings
from database import get_session
from models.models import Attendance, AttendanceDetail, Student
from recognition.face_detector import DetectedFace
from recognition.face_recognizer import FaceRecognizer
from services.embedding_store import EmbeddingGallery
from utils.logger import get_logger

logger = get_logger("services.attendance_service")

CLASSROOM_PHOTOS_DIR = DATASET_DIR.parent / "reports" / "classroom_photos"
CLASSROOM_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RecognizedFace:
    box: tuple
    confidence: float
    similarity: float
    student_db_id: Optional[int]
    student_id: Optional[str]
    student_name: Optional[str]
    status: str  # "Present" or "Unknown"


@dataclass
class AttendanceResult:
    session_id: int
    recognized_faces: List[RecognizedFace] = field(default_factory=list)
    total_detected: int = 0
    total_recognized: int = 0
    total_unknown: int = 0
    absent_students: List[Student] = field(default_factory=list)


class AttendanceService:
    def __init__(self):
        self._recognizer = FaceRecognizer()

    def take_attendance(
        self,
        classroom_image_bgr: np.ndarray,
        class_id: Optional[int] = None,
        session_date: Optional[dt.date] = None,
    ) -> AttendanceResult:
        """
        Full pipeline: detect -> recognize -> mark -> persist.

        Every *active* student not recognized in the photo is automatically
        marked Absent for this session, matching "Present = Recognized,
        Absent = Not Recognized" from the spec.
        """
        session_date = session_date or dt.date.today()

        # Save the raw classroom photo for audit purposes.
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = CLASSROOM_PHOTOS_DIR / f"classroom_{timestamp}.jpg"
        cv2.imwrite(str(photo_path), classroom_image_bgr)

        gallery = EmbeddingGallery.get()
        student_lookup = EmbeddingGallery.get_student_lookup()

        faces_with_embeddings = self._recognizer.embed_all_faces(classroom_image_bgr)

        recognized: List[RecognizedFace] = []
        matched_student_ids: set[int] = set()

        for face, embedding in faces_with_embeddings:
            match = self._recognizer.match(embedding, gallery)

            if match.is_match and match.student_db_id in student_lookup:
                snapshot = student_lookup[match.student_db_id]
                recognized.append(
                    RecognizedFace(
                        box=face.box,
                        confidence=face.confidence,
                        similarity=match.similarity,
                        student_db_id=snapshot.db_id,
                        student_id=snapshot.student_id,
                        student_name=snapshot.name,
                        status="Present",
                    )
                )
                matched_student_ids.add(snapshot.db_id)
            else:
                recognized.append(
                    RecognizedFace(
                        box=face.box,
                        confidence=face.confidence,
                        similarity=match.similarity,
                        student_db_id=None,
                        student_id=None,
                        student_name="Unknown",
                        status="Unknown",
                    )
                )

        # Persist the session + per-student details.
        with get_session() as session:
            attendance_session = Attendance(
                class_id=class_id,
                date=session_date,
                time=dt.datetime.now(),
                classroom_image_path=str(photo_path),
                total_detected_faces=len(faces_with_embeddings),
                total_recognized=len(matched_student_ids),
                total_unknown=len(recognized) - len(matched_student_ids),
            )
            session.add(attendance_session)
            session.flush()
            session_id = attendance_session.id

            all_active_students = session.query(Student).filter(Student.is_active.is_(True)).all()
            absent_students: List[Student] = []

            for student in all_active_students:
                if student.id in matched_student_ids:
                    # Find the best similarity score recorded for this student in this photo.
                    best_sim = max(
                        (r.similarity for r in recognized if r.student_db_id == student.id),
                        default=0.0,
                    )
                    detail = AttendanceDetail(
                        session_id=session_id,
                        student_id=student.id,
                        status="Present",
                        similarity_score=best_sim,
                    )
                else:
                    detail = AttendanceDetail(
                        session_id=session_id,
                        student_id=student.id,
                        status="Absent",
                        similarity_score=None,
                    )
                    absent_students.append(student)
                session.add(detail)

        logger.info(
            "Attendance session %d: %d detected, %d recognized, %d unknown, %d absent",
            session_id, len(faces_with_embeddings), len(matched_student_ids),
            len(recognized) - len(matched_student_ids), len(absent_students),
        )

        return AttendanceResult(
            session_id=session_id,
            recognized_faces=recognized,
            total_detected=len(faces_with_embeddings),
            total_recognized=len(matched_student_ids),
            total_unknown=len(recognized) - len(matched_student_ids),
            absent_students=absent_students,
        )
