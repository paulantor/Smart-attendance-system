"""
database.py
============
SQLAlchemy engine, session factory, and declarative base.

This module owns the *connection* to SQLite. The actual table definitions
live in models/models.py. Any module needing DB access should do:

    from database import get_session
    with get_session() as session:
        session.query(...)
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_DIR, DATABASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # required for Streamlit's threading model
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they do not already exist. Call once at app startup."""
    # Import models here (not at module top) to avoid circular imports, while
    # still ensuring their metadata is registered on Base before create_all().
    from models import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", DATABASE_URL)

    from services.auth_service import ensure_default_admin
    ensure_default_admin()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context-managed session: commits on success, rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database session rolled back due to an error")
        raise
    finally:
        session.close()


def backup_database() -> Path:
    """Copy the current SQLite file into database/backups/ with a timestamp."""
    backup_dir = DATABASE_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    src = DATABASE_DIR / "attendance.db"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"attendance_backup_{timestamp}.db"

    if src.exists():
        shutil.copy2(src, dst)
        logger.info("Database backed up to %s", dst)
    else:
        logger.warning("No database file found to back up at %s", src)

    return dst


def restore_database(backup_path: Path) -> None:
    """Restore the SQLite file from a given backup path, overwriting the current DB."""
    dst = DATABASE_DIR / "attendance.db"
    shutil.copy2(backup_path, dst)
    logger.info("Database restored from %s", backup_path)