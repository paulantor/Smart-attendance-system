"""pages/dashboard.py -- Real-time attendance statistics and charts."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from database import get_session
from models.models import Attendance, AttendanceDetail, Student
from services.report_service import ReportService

report_service = ReportService()


def render() -> None:
    st.title("📊 Dashboard")

    with get_session() as session:
        total_students = session.query(Student).filter(Student.is_active.is_(True)).count()
        total_sessions = session.query(Attendance).count()
        total_present = session.query(AttendanceDetail).filter(AttendanceDetail.status == "Present").count()
        total_absent = session.query(AttendanceDetail).filter(AttendanceDetail.status == "Absent").count()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Students", total_students)
    col2.metric("Sessions Taken", total_sessions)
    col3.metric("Total Present Marks", total_present)
    col4.metric("Total Absent Marks", total_absent)

    if total_present + total_absent > 0:
        attendance_rate = total_present / (total_present + total_absent) * 100
        st.progress(min(1.0, attendance_rate / 100), text=f"Overall attendance rate: {attendance_rate:.1f}%")

    st.divider()

    start, end = report_service.period_range("Monthly")
    df = report_service.query_attendance(start_date=start, end_date=end)

    if df.empty:
        st.info("No attendance data yet. Take attendance to see charts here.")
        return

    tab1, tab2, tab3 = st.tabs(["Daily Trend", "By Department", "By Student"])

    with tab1:
        daily = df.groupby(["Date", "Status"]).size().reset_index(name="Count")
        fig = px.bar(daily, x="Date", y="Count", color="Status", barmode="group",
                     title="Attendance by Day (last 30 days)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dept = df[df["Status"] == "Present"].groupby("Department").size().reset_index(name="Present Count")
        if not dept.empty:
            fig2 = px.pie(dept, names="Department", values="Present Count", title="Present Marks by Department")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No department data available yet.")

    with tab3:
        per_student = (
            df.groupby(["Student ID", "Student Name", "Status"]).size().reset_index(name="Count")
        )
        pivot = per_student.pivot_table(
            index=["Student ID", "Student Name"], columns="Status", values="Count", fill_value=0
        ).reset_index()
        st.dataframe(pivot, use_container_width=True)
