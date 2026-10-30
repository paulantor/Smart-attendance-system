"""
app.py
======
Main entry point for the AI Smart Classroom Attendance System.

Responsibilities:
  - Initialize the database and load persisted settings on first run.
  - Show a login gate with TWO portals: Teacher/Admin and Student.
  - Render role-appropriate sidebar navigation.
  - Dispatch to the selected page's `render()` function.

Run with:
    streamlit run app.py

Default admin login (auto-created on first run, CHANGE IMMEDIATELY):
    username: admin
    password: admin123

Students log in with their Student ID as both username and initial
password (set automatically at registration time); they can be given a
new password via Settings > Manage Accounts (admin-only).
"""

from __future__ import annotations

import importlib

import streamlit as st

from config import settings
from database import init_db
from services.auth_service import authenticate_student, authenticate_teacher
from services.settings_service import load_settings
from utils.logger import get_logger

logger = get_logger("app")

st.set_page_config(
    page_title=settings.app_name,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _bootstrap() -> None:
    """Runs exactly once per server process: create tables, default admin, load settings."""
    init_db()
    load_settings()
    logger.info("Application bootstrap complete.")


_bootstrap()

# ---------------------------------------------------------------------- #
# Page registries per role
# ---------------------------------------------------------------------- #
TEACHER_PAGES = {
    "Home": "app_pages.home",
    "Dashboard": "app_pages.dashboard",
    "Student Registration": "app_pages.student_registration",
    "Student List": "app_pages.student_list",
    "Take Attendance": "app_pages.take_attendance",
    "Attendance History": "app_pages.attendance_history",
    "Reports": "app_pages.reports",
    "Settings": "app_pages.settings_page",
}

STUDENT_PAGES = {
    "My Attendance": "app_pages.student_portal",
}

ICONS = {
    "Home": "🏠", "Dashboard": "📊", "Student Registration": "🧑‍🎓",
    "Student List": "📋", "Take Attendance": "📸", "Attendance History": "🗓️",
    "Reports": "📑", "Settings": "⚙️", "My Attendance": "📊",
}


def _logout() -> None:
    for key in ("auth_role", "auth_teacher", "auth_student", "nav_page"):
        st.session_state.pop(key, None)
    st.rerun()


def _render_login_screen() -> None:
    st.title(f"🎓 {settings.app_name}")
    st.caption("Please log in to continue.")

    tab_teacher, tab_student = st.tabs(["🧑‍🏫 Teacher / Admin Login", "🧑‍🎓 Student Login"])

    with tab_teacher:
        with st.form("teacher_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login as Teacher/Admin", use_container_width=True)
        if submitted:
            auth = authenticate_teacher(username, password)
            if auth:
                st.session_state["auth_role"] = "teacher"
                st.session_state["auth_teacher"] = auth
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_student:
        with st.form("student_login_form"):
            student_id = st.text_input("Student ID")
            password = st.text_input("Password", type="password", help="Default password is your Student ID.")
            submitted = st.form_submit_button("Login as Student", use_container_width=True)
        if submitted:
            auth = authenticate_student(student_id, password)
            if auth:
                st.session_state["auth_role"] = "student"
                st.session_state["auth_student"] = auth
                st.rerun()
            else:
                st.error("Invalid Student ID or password.")


# ---------------------------------------------------------------------- #
# Main gate
# ---------------------------------------------------------------------- #
if "auth_role" not in st.session_state:
    _render_login_screen()
    st.stop()

role = st.session_state["auth_role"]
pages = TEACHER_PAGES if role == "teacher" else STUDENT_PAGES

if "nav_page" not in st.session_state or st.session_state["nav_page"] not in pages:
    st.session_state["nav_page"] = list(pages.keys())[0]

with st.sidebar:
    st.markdown(f"## 🎓 {settings.app_name}")
    st.caption(f"v{settings.version}")

    if role == "teacher":
        who = st.session_state["auth_teacher"]
        st.success(f"👤 {who.name} ({'Admin' if who.is_admin else 'Teacher'})")
    else:
        who = st.session_state["auth_student"]
        st.success(f"👤 {who.name} ({who.student_id})")

    st.divider()

    selected = st.radio(
        "Navigation",
        list(pages.keys()),
        format_func=lambda p: f"{ICONS.get(p, '📄')}  {p}",
        index=list(pages.keys()).index(st.session_state["nav_page"]),
        label_visibility="collapsed",
    )
    st.session_state["nav_page"] = selected

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        _logout()

module = importlib.import_module(pages[st.session_state["nav_page"]])

try:
    module.render()
except Exception:
    logger.exception("Unhandled error while rendering page '%s'", st.session_state["nav_page"])
    st.error(
        "Something went wrong while loading this page. The error has been logged. "
        "Please check logs/app.log for details."
    )