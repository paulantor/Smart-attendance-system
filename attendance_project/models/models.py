"""
models/models.py
=================
SQLAlchemy ORM models for the AI Smart Classroom Attendance System.

Schema
------
Teacher            -- teaching staff who take attendance
Course              -- a subject/course
Class               -- a scheduled class session template (course + teacher + section)
Student              -- registered student, unique student_id
Embedding            -- one or more 512-d face embeddings per student (cached)
Attendance           -- one row per "attendance session" (a specific class taken on a date)
AttendanceDetail     -- one row per student within an Attendance session (Present/Absent/Late)
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Teacher(Base):
    __tablename__ = "teachers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(120), unique=True, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    username: Mapped[Optional[str]] = mapped_column(String(60), unique=True, nullable=True, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    classes: Mapped[List["Class"]] = relationship(back_populates="teacher")

    def __repr__(self) -> str:
        return f"<Teacher id={self.id} name={self.name!r}>"


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(150), nullable=False)
    department: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    classes: Mapped[List["Class"]] = relationship(back_populates="course")

    def __repr__(self) -> str:
        return f"<Course code={self.code!r}>"


class Class(Base):
    """A scheduled class 'template' -- e.g. CSE101 / Section A / Teacher X."""

    __tablename__ = "classes"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), nullable=False)
    teacher_id: Mapped[int] = mapped_column(ForeignKey("teachers.id"), nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    batch: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    course: Mapped["Course"] = relationship(back_populates="classes")
    teacher: Mapped["Teacher"] = relationship(back_populates="classes")
    attendance_sessions: Mapped[List["Attendance"]] = relationship(back_populates="class_")

    def __repr__(self) -> str:
        return f"<Class id={self.id} course_id={self.course_id} section={self.section!r}>"


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    department: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    semester: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    batch: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    registered_images_count: Mapped[int] = mapped_column(default=0)
    password_hash: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    embeddings: Mapped[List["Embedding"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )
    attendance_details: Mapped[List["AttendanceDetail"]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Student student_id={self.student_id!r} name={self.name!r}>"


class Embedding(Base):
    """A single cached 512-d ArcFace embedding vector for a student.

    Multiple rows per student are allowed (e.g. one per good registration
    image) so that recognition can match against the centroid or the best
    of several embeddings for higher accuracy.
    """

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False, index=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)  # np.float32 array, serialized
    source_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    student: Mapped["Student"] = relationship(back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<Embedding id={self.id} student_id={self.student_id}>"


class Attendance(Base):
    """One row per attendance-taking session (a specific class, on a specific date/time)."""

    __tablename__ = "attendance_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    class_id: Mapped[Optional[int]] = mapped_column(ForeignKey("classes.id"), nullable=True)
    date: Mapped[dt.date] = mapped_column(default=dt.date.today, index=True)
    time: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    classroom_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    total_detected_faces: Mapped[int] = mapped_column(default=0)
    total_recognized: Mapped[int] = mapped_column(default=0)
    total_unknown: Mapped[int] = mapped_column(default=0)

    class_: Mapped[Optional["Class"]] = relationship(back_populates="attendance_sessions")
    details: Mapped[List["AttendanceDetail"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Attendance id={self.id} date={self.date}>"


class AttendanceDetail(Base):
    """One row per student within a single attendance session."""

    __tablename__ = "attendance_details"
    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_session_student"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("attendance_sessions.id"), nullable=False)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="Absent")  # Present / Absent / Late
    similarity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    marked_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    session: Mapped["Attendance"] = relationship(back_populates="details")
    student: Mapped["Student"] = relationship(back_populates="attendance_details")

    def __repr__(self) -> str:
        return f"<AttendanceDetail session={self.session_id} student={self.student_id} status={self.status!r}>"
    