"""
services/auth_service.py
==========================
Two separate login portals:
  - Teacher / Admin  -- username + password, full app access
  - Student           -- Student ID + password, access only to their own
                         attendance history (see app_pages/student_portal.py)

A default admin account is auto-created on first run (see ensure_default_admin,
called once from database.init_db()) so there's always a way to log in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from database import get_session
from models.models import Student, Teacher
from utils.auth import hash_password, verify_password
from utils.logger import get_logger

logger = get_logger("services.auth_service")

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"  # must be changed after first login


@dataclass
class AuthenticatedTeacher:
    db_id: int
    name: str
    username: str
    is_admin: bool


@dataclass
class AuthenticatedStudent:
    db_id: int
    student_id: str
    name: str


def ensure_default_admin() -> None:
    """Create a default admin Teacher account if no teacher accounts exist yet."""
    with get_session() as session:
        any_teacher = session.query(Teacher).filter(Teacher.username.isnot(None)).first()
        if any_teacher:
            return

        admin = Teacher(
            name="Administrator",
            username=DEFAULT_ADMIN_USERNAME,
            password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
            is_admin=True,
        )
        session.add(admin)
    logger.warning(
        "No teacher accounts found -- created default admin account "
        "(username='%s', password='%s'). CHANGE THIS PASSWORD IMMEDIATELY after first login.",
        DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD,
    )


def authenticate_teacher(username: str, password: str) -> Optional[AuthenticatedTeacher]:
    with get_session() as session:
        teacher = session.query(Teacher).filter(Teacher.username == username.strip()).first()
        if not teacher or not teacher.password_hash:
            return None
        if not verify_password(password, teacher.password_hash):
            return None
        return AuthenticatedTeacher(
            db_id=teacher.id, name=teacher.name, username=teacher.username, is_admin=teacher.is_admin
        )


def authenticate_student(student_id: str, password: str) -> Optional[AuthenticatedStudent]:
    with get_session() as session:
        student = session.query(Student).filter(Student.student_id == student_id.strip()).first()
        if not student or not student.password_hash:
            return None
        if not verify_password(password, student.password_hash):
            return None
        return AuthenticatedStudent(db_id=student.id, student_id=student.student_id, name=student.name)


def create_teacher_account(name: str, username: str, password: str, email: str = "", is_admin: bool = True) -> int:
    """Admin-only: create an additional teacher/admin login."""
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")

    with get_session() as session:
        existing = session.query(Teacher).filter(Teacher.username == username).first()
        if existing:
            raise ValueError(f"Username '{username}' is already taken.")

        teacher = Teacher(
            name=name.strip() or username,
            username=username,
            password_hash=hash_password(password),
            email=email or None,
            is_admin=is_admin,
        )
        session.add(teacher)
        session.flush()
        db_id = teacher.id

    logger.info("Created teacher account '%s' (db_id=%d)", username, db_id)
    return db_id


def set_student_password(student_db_id: int, new_password: str) -> None:
    """Admin action: set/reset a student's login password (e.g. after registration)."""
    with get_session() as session:
        student = session.get(Student, student_db_id)
        if not student:
            raise ValueError("Student not found.")
        student.password_hash = hash_password(new_password)
    logger.info("Password set for student_db_id=%d", student_db_id)


def change_teacher_password(teacher_db_id: int, new_password: str) -> None:
    with get_session() as session:
        teacher = session.get(Teacher, teacher_db_id)
        if not teacher:
            raise ValueError("Teacher not found.")
        teacher.password_hash = hash_password(new_password)
    logger.info("Password changed for teacher_db_id=%d", teacher_db_id)