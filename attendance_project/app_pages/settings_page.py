"""pages/settings_page.py -- Recognition threshold, camera, export folder, and DB backup/restore."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import DATABASE_DIR, REPORTS_DIR, settings
from database import backup_database, get_session, restore_database
from models.models import Student
from services.auth_service import change_teacher_password, create_teacher_account, set_student_password
from services.embedding_store import EmbeddingGallery
from services.settings_service import save_settings


def render() -> None:
    st.title("⚙️ Settings")

    st.subheader("Recognition")
    threshold = st.slider(
        "Recognition similarity threshold (cosine)",
        min_value=0.20, max_value=0.80,
        value=settings.recognition.similarity_threshold, step=0.01,
        help="Higher = stricter matching (fewer false positives, more false negatives).",
    )
    detector_conf = st.slider(
        "Detector confidence threshold",
        min_value=0.5, max_value=0.99,
        value=settings.recognition.detector_confidence, step=0.01,
    )

    duplicate_threshold = st.slider(
        "Duplicate-face detection threshold (registration)",
        min_value=0.15, max_value=0.60,
        value=settings.recognition.duplicate_face_threshold, step=0.01,
        help="LOWER = catches more duplicate faces during registration (stricter duplicate blocking), "
             "but may occasionally flag two very similar-looking different people. If real duplicate "
             "faces are still slipping through during registration, lower this value.",
    )

    st.subheader("Camera")
    resolution = st.selectbox(
        "Camera resolution", ["640x480", "1280x960", "1920x1080"],
        index=["640x480", "1280x960", "1920x1080"].index(settings.camera.resolution)
        if settings.camera.resolution in ["640x480", "1280x960", "1920x1080"] else 1,
    )

    st.subheader("Export Folder")
    st.text_input("Reports export folder", value=str(REPORTS_DIR), disabled=True)

    if st.button("💾 Save Settings", type="primary"):
        save_settings(
            similarity_threshold=threshold,
            detector_confidence=detector_conf,
            camera_resolution=resolution,
            duplicate_face_threshold=duplicate_threshold,
        )
        EmbeddingGallery.invalidate()
        st.success("Settings saved.")

    st.divider()
    st.subheader("Database Backup & Restore")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📦 Backup Database Now", use_container_width=True):
            path = backup_database()
            st.success(f"Backup created: {path.name}")

    with c2:
        backups_dir = DATABASE_DIR / "backups"
        backups = sorted(backups_dir.glob("*.db"), reverse=True) if backups_dir.exists() else []
        if backups:
            chosen = st.selectbox("Restore from backup", [b.name for b in backups])
            if st.button("♻️ Restore Selected Backup", use_container_width=True):
                restore_database(backups_dir / chosen)
                EmbeddingGallery.invalidate()
                st.warning("Database restored. Please refresh the app.")
        else:
            st.caption("No backups available yet.")

    st.divider()
    st.subheader("👥 Manage Accounts (Admin)")

    tab1, tab2, tab3 = st.tabs(["➕ Add Teacher/Admin", "🔑 Reset Student Password", "🔑 Change My Password"])

    with tab1:
        with st.form("add_teacher_form"):
            t_name = st.text_input("Full Name")
            t_username = st.text_input("Username")
            t_password = st.text_input("Password", type="password")
            t_email = st.text_input("Email (optional)")
            t_is_admin = st.checkbox("Grant full admin rights", value=True)
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            try:
                create_teacher_account(
                    name=t_name, username=t_username, password=t_password,
                    email=t_email, is_admin=t_is_admin,
                )
                st.success(f"Teacher account '{t_username}' created.")
            except ValueError as e:
                st.error(str(e))

    with tab2:
        with get_session() as session:
            students = session.query(Student).all()
            options = {f"{s.student_id} - {s.name}": s.id for s in students}

        if options:
            chosen = st.selectbox("Select Student", list(options.keys()))
            new_password = st.text_input("New Password", type="password", key="student_new_pw")
            if st.button("Reset Student Password", use_container_width=True):
                if not new_password:
                    st.error("Please enter a new password.")
                else:
                    set_student_password(options[chosen], new_password)
                    st.success("Student password updated.")
        else:
            st.caption("No students registered yet.")

    with tab3:
        current_teacher = st.session_state.get("auth_teacher")
        if current_teacher:
            new_own_password = st.text_input("New Password", type="password", key="own_new_pw")
            if st.button("Change My Password", use_container_width=True):
                if not new_own_password:
                    st.error("Please enter a new password.")
                else:
                    change_teacher_password(current_teacher.db_id, new_own_password)
                    st.success("Your password has been changed. Use it next time you log in.")
        else:
            st.caption("Log in as a teacher/admin to change your own password.")