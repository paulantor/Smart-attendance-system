"""pages/attendance_history.py -- Browse & search past attendance sessions with filters."""

from __future__ import annotations

import datetime as dt

import streamlit as st

from database import get_session
from models.models import Attendance, AttendanceDetail
from services.report_service import ReportService

report_service = ReportService()


def render() -> None:
    st.title("🗓️ Attendance History")

    c1, c2, c3, c4 = st.columns(4)
    start_date = c1.date_input("Start Date", value=dt.date.today() - dt.timedelta(days=30))
    end_date = c2.date_input("End Date", value=dt.date.today())
    department = c3.text_input("Department filter")
    student_search = c4.text_input("Search Student ID / Name")

    df = report_service.query_attendance(
        start_date=start_date,
        end_date=end_date,
        department=department or None,
        student_search=student_search or None,
    )

    if df.empty:
        st.info("No attendance records found for the selected filters.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"{len(df)} attendance records found.")

    st.divider()
    st.subheader("🗑️ Admin: Delete a Session")
    with get_session() as session:
        sessions = (
            session.query(Attendance)
            .filter(Attendance.date >= start_date, Attendance.date <= end_date)
            .order_by(Attendance.date.desc())
            .all()
        )
        session_options = {f"#{s.id} · {s.date} · {s.total_detected_faces} faces": s.id for s in sessions}

    if session_options:
        chosen = st.selectbox("Select a session to delete", list(session_options.keys()))
        if st.button("Delete Selected Session", type="secondary"):
            session_id = session_options[chosen]
            with get_session() as session:
                session.query(AttendanceDetail).filter(AttendanceDetail.session_id == session_id).delete()
                obj = session.get(Attendance, session_id)
                if obj:
                    session.delete(obj)
            st.success(f"Session #{session_id} deleted.")
            st.rerun()
