"""
services/registration_service.py
==================================
Orchestrates the student registration workflow described in the spec:

  1. Validate the student's basic info (unique ID, required fields).
  2. For every captured image: detect the face, run quality checks,
     reject bad frames with a reason, crop + save good frames to
     dataset/<student_id>/, and generate + store its embedding.
  3. Prevent duplicate registration (same student_id twice).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import DATASET_DIR, settings
from database import get_session
from models.models import Embedding, Student
from recognition.face_detector import FaceDetector
from recognition.face_recognizer import FaceRecognizer
from services.embedding_store import EmbeddingGallery
from utils.image_quality import compute_blur_score, compute_brightness, crop_face, validate_frame
from utils.pose_estimation import matches_expected_pose
from utils.auth import hash_password
from utils.logger import get_logger

logger = get_logger("services.registration_service")


class DuplicateStudentError(Exception):
    """Raised when attempting to register a student_id that already exists."""


class DuplicateFaceError(Exception):
    """Raised when a captured face already matches a DIFFERENT, already-registered student."""

    def __init__(self, existing_student_id: str, similarity: float):
        self.existing_student_id = existing_student_id
        self.similarity = similarity
        super().__init__(
            f"This face already appears to be registered as student '{existing_student_id}' "
            f"(similarity {similarity:.2f}). Duplicate face registration under a new ID is not allowed."
        )


class IncompleteRegistrationError(Exception):
    """Raised when finalize_registration() is called without enough accepted face images."""

    def __init__(self, accepted: int, required: int):
        self.accepted = accepted
        self.required = required
        super().__init__(
            f"Cannot complete registration: only {accepted} valid face image(s) captured, "
            f"minimum {required} required."
        )


@dataclass
class FrameResult:
    accepted: bool
    reason: Optional[str]
    saved_path: Optional[str] = None


@dataclass
class RegistrationSummary:
    student_db_id: int
    accepted_count: int
    rejected_count: int
    frame_results: List[FrameResult] = field(default_factory=list)


class RegistrationService:
    def __init__(self):
        self._detector = FaceDetector()
        self._recognizer = FaceRecognizer()

    # ------------------------------------------------------------------ #
    # Student record management
    # ------------------------------------------------------------------ #
    def validate_unique_student_id(self, student_id: str) -> None:
        with get_session() as session:
            existing = session.query(Student).filter(Student.student_id == student_id).first()
        if existing:
            raise DuplicateStudentError(f"Student ID '{student_id}' is already registered.")

    def create_student_record(
        self,
        student_id: str,
        name: str,
        department: str,
        semester: str,
        section: str,
        phone: str = "",
        email: str = "",
    ) -> int:
        """Create the Student row (0 images so far). Raises DuplicateStudentError if taken."""
        student_id = student_id.strip()
        if not student_id or not name.strip():
            raise ValueError("Student ID and Name are required.")

        self.validate_unique_student_id(student_id)

        with get_session() as session:
            student = Student(
                student_id=student_id,
                name=name.strip(),
                department=department,
                semester=semester,
                batch=None,
                section=section,
                phone=phone,
                email=email,
                registered_images_count=0,
                password_hash=hash_password(student_id),  # default login password = their Student ID
            )
            session.add(student)
            session.flush()  # populate student.id before session closes
            db_id = student.id

        (DATASET_DIR / student_id).mkdir(parents=True, exist_ok=True)
        logger.info("Created student record for %s (db_id=%d)", student_id, db_id)
        return db_id

    def check_duplicate_face(
        self, embedding: np.ndarray, current_student_db_id: int
    ) -> Optional[tuple[str, float]]:
        """
        Compare a newly captured embedding against every OTHER student's cached
        embeddings (the current student's own images, e.g. during recapture,
        are excluded so they don't falsely "duplicate" against themselves).

        Returns (existing_student_id, similarity) if this face already belongs
        to a different registered student, else None.
        """
        gallery = EmbeddingGallery.get()
        other_students_gallery = [
            (sid, vec) for sid, vec in gallery if sid != current_student_db_id
        ]
        if not other_students_gallery:
            return None

        match = self._recognizer.match(
            embedding,
            other_students_gallery,
            threshold=settings.recognition.duplicate_face_threshold,
        )
        logger.debug(
            "Duplicate-face check for student_db_id=%d: best similarity=%.3f against student_db_id=%s (threshold=%.2f)",
            current_student_db_id, match.similarity, match.student_db_id, settings.recognition.duplicate_face_threshold,
        )
        if not match.is_match or match.student_db_id is None:
            return None

        with get_session() as session:
            existing_student = session.get(Student, match.student_db_id)
            existing_id = existing_student.student_id if existing_student else "unknown"

        return existing_id, match.similarity

    # ------------------------------------------------------------------ #
    # Frame-by-frame capture processing
    # ------------------------------------------------------------------ #
    def process_capture_frame(
        self,
        student_db_id: int,
        student_id: str,
        frame_bgr: np.ndarray,
        frame_index: int,
        expected_pose: Optional[str] = None,
    ) -> FrameResult:
        """
        Validate one captured frame and, if it passes, crop/save it and store
        its embedding immediately (so registration progress is never lost
        even if the session is interrupted).

        If `expected_pose` is given ("center"/"left"/"right"/"up"/"down"),
        the frame is also checked against the guided-pose sequence using
        utils.pose_estimation, so the person must actually turn their head
        the requested direction before the frame is accepted.
        """
        faces = self._detector.detect(frame_bgr)
        boxes = [f.box for f in faces]

        quality = validate_frame(frame_bgr, boxes)
        if not quality.accepted:
            return FrameResult(accepted=False, reason=quality.reason)

        face = faces[0]

        if expected_pose is not None and not matches_expected_pose(face.landmarks, expected_pose):
            return FrameResult(
                accepted=False,
                reason=f"Pose not detected -- please follow the instruction: turn/tilt as guided.",
            )

        cropped = crop_face(frame_bgr, face.box)

        embedding = face.embedding
        if embedding is None:
            embedding = self._recognizer.embed_face(frame_bgr)
        if embedding is None:
            return FrameResult(accepted=False, reason="Could not generate embedding for this frame")
        embedding = self._recognizer._normalize(np.asarray(embedding, dtype=np.float32))

        # --- Face-level duplicate check: is this actually a DIFFERENT student's face? ---
        duplicate = self.check_duplicate_face(embedding, student_db_id)
        if duplicate is not None:
            existing_id, similarity = duplicate
            logger.warning(
                "Duplicate face detected for student_db_id=%d: matches existing student '%s' (sim=%.3f)",
                student_db_id, existing_id, similarity,
            )
            return FrameResult(
                accepted=False,
                reason=(
                    f"This face is already registered as student '{existing_id}' "
                    f"(match confidence {similarity:.2f}). Cannot register the same face under a new ID."
                ),
            )

        student_dir = DATASET_DIR / student_id
        student_dir.mkdir(parents=True, exist_ok=True)
        save_path = student_dir / f"{student_id}_{frame_index:03d}.jpg"
        cv2.imwrite(str(save_path), cropped)

        quality_score = min(1.0, compute_blur_score(frame_bgr) / 200.0)  # rough 0-1 normalization

        with get_session() as session:
            session.add(
                Embedding(
                    student_id=student_db_id,
                    vector=FaceRecognizer.serialize(embedding),
                    source_image_path=str(save_path),
                    quality_score=quality_score,
                )
            )
            student = session.get(Student, student_db_id)
            if student:
                student.registered_images_count += 1

        return FrameResult(accepted=True, reason=None, saved_path=str(save_path))

    def finalize_registration(self, student_db_id: int) -> RegistrationSummary:
        """
        Call after all capture frames are processed to refresh the recognition cache.

        HARD RULE: registration cannot be finalized with fewer than
        settings.recognition.min_registration_images accepted face images.
        This is enforced here at the service layer (not just hidden in the UI),
        so it can never be bypassed regardless of which page/flow calls it.
        """
        required = settings.recognition.min_registration_images

        with get_session() as session:
            student = session.get(Student, student_db_id)
            accepted = student.registered_images_count if student else 0

        if accepted < required:
            logger.warning(
                "Blocked finalize for student_db_id=%d: only %d/%d images captured",
                student_db_id, accepted, required,
            )
            raise IncompleteRegistrationError(accepted=accepted, required=required)

        EmbeddingGallery.invalidate()
        logger.info("Finalized registration for student_db_id=%d (%d images)", student_db_id, accepted)
        return RegistrationSummary(student_db_id=student_db_id, accepted_count=accepted, rejected_count=0)

    def delete_incomplete_registrations(self) -> int:
        """
        Admin cleanup: remove any Student rows left behind with ZERO captured
        images -- e.g. from someone closing the browser mid-registration
        instead of clicking Cancel. Returns how many were deleted.
        """
        with get_session() as session:
            incomplete = session.query(Student).filter(Student.registered_images_count == 0).all()
            count = len(incomplete)
            for student in incomplete:
                session.delete(student)

        if count:
            EmbeddingGallery.invalidate()
        logger.info("Deleted %d incomplete (zero-image) student registrations.", count)
        return count

    # ------------------------------------------------------------------ #
    # Admin operations
    # ------------------------------------------------------------------ #
    def delete_student(self, student_db_id: int) -> None:
        with get_session() as session:
            student = session.get(Student, student_db_id)
            if student:
                session.delete(student)  # cascades to embeddings & attendance_details
        EmbeddingGallery.invalidate()
        logger.info("Deleted student_db_id=%d", student_db_id)

    def update_student(self, student_db_id: int, **fields) -> None:
        with get_session() as session:
            student = session.get(Student, student_db_id)
            if not student:
                raise ValueError("Student not found.")
            for key, value in fields.items():
                if hasattr(student, key):
                    setattr(student, key, value)
        EmbeddingGallery.invalidate()
        logger.info("Updated student_db_id=%d fields=%s", student_db_id, list(fields.keys()))

    def recapture_images(self, student_db_id: int) -> None:
        """Delete all existing embeddings/images for a student so they can re-register images."""
        with get_session() as session:
            session.query(Embedding).filter(Embedding.student_id == student_db_id).delete()
            student = session.get(Student, student_db_id)
            if student:
                student.registered_images_count = 0
        EmbeddingGallery.invalidate()
        logger.info("Cleared images for recapture, student_db_id=%d", student_db_id)