"""
app_pages/student_portal.py
=============================
Read-only self-service view for a logged-in STUDENT: their profile,
their own attendance history, and their attendance percentage.
Students cannot see any other student's data, cannot register faces,
and cannot take attendance -- this page is intentionally limited.
"""

from __future__ import annotations

import streamlit as st

from database import get_session
from models.models import Student
from services.report_service import ReportService

report_service = ReportService()


def render() -> None:
    auth_student = st.session_state.get("auth_student")
    if not auth_student:
        st.error("Not logged in.")
        return

    with get_session() as session:
        student = session.get(Student, auth_student.db_id)
        if not student:
            st.error("Student record not found.")
            return
        name, student_id, department, section, semester = (
            student.name, student.student_id, student.department, student.section, student.semester,
        )

    st.title(f"🧑‍🎓 Welcome, {name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student ID", student_id)
    c2.metric("Department", department or "-")
    c3.metric("Section", section or "-")
    c4.metric("Semester", semester or "-")

    st.divider()
    st.subheader("📊 My Attendance")

    df = report_service.query_attendance(student_search=student_id)
    df = df[df["Student ID"] == student_id]  # exact match safety (search is substring-based)

    if df.empty:
        st.info("No attendance records yet.")
        return

    total = len(df)
    present = int((df["Status"] == "Present").sum())
    absent = total - present
    rate = (present / total * 100) if total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sessions", total)
    c2.metric("Present", present)
    c3.metric("Absent", absent)
    st.progress(min(1.0, rate / 100), text=f"Attendance rate: {rate:.1f}%")

    st.divider()
    st.subheader("History")
    st.dataframe(
        df[["Date", "Time", "Course", "Teacher", "Status"]].sort_values("Date", ascending=False),
        use_container_width=True, hide_index=True,
    )