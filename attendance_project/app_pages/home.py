"""pages/home.py -- Landing page with a quick overview and shortcuts."""

from __future__ import annotations

import streamlit as st

from config import settings
from database import get_session
from models.models import Attendance, Student


def render() -> None:
    st.title(f"👋 Welcome to {settings.app_name}")
    st.caption(f"Version {settings.version} · Facial-recognition based classroom attendance")

    with get_session() as session:
        total_students = session.query(Student).filter(Student.is_active.is_(True)).count()
        total_sessions = session.query(Attendance).count()

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Students", total_students)
    col2.metric("Attendance Sessions Taken", total_sessions)
    col3.metric("Recognition Threshold", f"{settings.recognition.similarity_threshold:.2f}")

    st.divider()
    st.subheader("Quick Actions")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧑‍🎓 Register a Student")
        st.write("Capture 20-50 face images and enroll a new student.")
        if st.button("Go to Registration", use_container_width=True):
            st.session_state["nav_page"] = "Student Registration"
            st.rerun()
    with c2:
        st.markdown("### 📸 Take Attendance")
        st.write("Capture a classroom photo and auto-mark attendance.")
        if st.button("Go to Take Attendance", use_container_width=True):
            st.session_state["nav_page"] = "Take Attendance"
            st.rerun()
    with c3:
        st.markdown("### 📊 View Dashboard")
        st.write("See attendance trends, charts and statistics.")
        if st.button("Go to Dashboard", use_container_width=True):
            st.session_state["nav_page"] = "Dashboard"
            st.rerun()
