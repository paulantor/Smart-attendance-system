"""pages/student_list.py -- Browse, search, filter, and manage registered students."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import get_session
from models.models import Student
from services.registration_service import RegistrationService

registration_service = RegistrationService()


def _load_students_df() -> pd.DataFrame:
    with get_session() as session:
        students = session.query(Student).all()
        data = [
            {
                "DB ID": s.id,
                "Student ID": s.student_id,
                "Name": s.name,
                "Department": s.department,
                "Semester": s.semester,
                "Section": s.section,
                "Phone": s.phone,
                "Email": s.email,
                "Images": s.registered_images_count,
                "Active": s.is_active,
            }
            for s in students
        ]
    return pd.DataFrame(data)


def render() -> None:
    st.title("📋 Student List")

    df = _load_students_df()
    if df.empty:
        st.info("No students registered yet. Go to Student Registration to add one.")
        return

    c1, c2, c3 = st.columns(3)
    search = c1.text_input("🔍 Search by ID or Name")
    dept_filter = c2.selectbox("Department", ["All"] + sorted(df["Department"].dropna().unique().tolist()))
    section_filter = c3.selectbox("Section", ["All"] + sorted(df["Section"].dropna().unique().tolist()))

    filtered = df.copy()
    if search:
        mask = filtered["Student ID"].str.contains(search, case=False, na=False) | \
               filtered["Name"].str.contains(search, case=False, na=False)
        filtered = filtered[mask]
    if dept_filter != "All":
        filtered = filtered[filtered["Department"] == dept_filter]
    if section_filter != "All":
        filtered = filtered[filtered["Section"] == section_filter]

    st.dataframe(filtered.drop(columns=["DB ID"]), use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} students.")

    incomplete_count = int((df["Images"] == 0).sum())
    if incomplete_count > 0:
        st.warning(
            f"⚠️ {incomplete_count} student(s) have **0 registered images** -- likely an abandoned "
            f"registration (e.g. the browser was closed before finishing). These students can never "
            f"be recognized during attendance. Use the cleanup button below to remove them."
        )
        if st.button(f"🧹 Delete all {incomplete_count} incomplete (0-image) registrations"):
            deleted = registration_service.delete_incomplete_registrations()
            st.success(f"Deleted {deleted} incomplete registration(s).")
            st.rerun()

    st.divider()
    st.subheader("⚙️ Admin Actions")
    if filtered.empty:
        return

    selected_label = st.selectbox(
        "Select a student",
        options=filtered["DB ID"].tolist(),
        format_func=lambda db_id: f"{filtered.loc[filtered['DB ID'] == db_id, 'Student ID'].values[0]} - "
                                   f"{filtered.loc[filtered['DB ID'] == db_id, 'Name'].values[0]}",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Delete Student", use_container_width=True):
            registration_service.delete_student(selected_label)
            st.success("Student deleted.")
            st.rerun()
    with c2:
        if st.button("🔁 Recapture Images", use_container_width=True):
            registration_service.recapture_images(selected_label)
            st.success("Images cleared. Go to Student Registration flow to recapture (existing record reused).")
    with c3:
        toggle_label = "🚫 Deactivate" if bool(
            filtered.loc[filtered["DB ID"] == selected_label, "Active"].values[0]
        ) else "✅ Reactivate"
        if st.button(toggle_label, use_container_width=True):
            current_active = bool(filtered.loc[filtered["DB ID"] == selected_label, "Active"].values[0])
            registration_service.update_student(selected_label, is_active=not current_active)
            st.rerun()