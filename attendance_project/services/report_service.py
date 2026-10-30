"""
services/report_service.py
============================
Builds attendance reports (with search/filter support) and exports them
to CSV, Excel, or PDF -- Phase 7 of the spec.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from config import REPORTS_DIR
from database import get_session
from models.models import Attendance, AttendanceDetail, Class, Course, Student, Teacher
from utils.logger import get_logger

logger = get_logger("services.report_service")


class ReportService:
    # ------------------------------------------------------------------ #
    # Query building
    # ------------------------------------------------------------------ #
    def query_attendance(
        self,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
        department: Optional[str] = None,
        section: Optional[str] = None,
        course_id: Optional[int] = None,
        teacher_id: Optional[int] = None,
        student_search: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a flat DataFrame of attendance rows matching the given filters:
        Date, Time, Course, Teacher, Student ID, Student Name, Status.
        """
        with get_session() as session:
            q = (
                session.query(
                    Attendance.date,
                    Attendance.time,
                    Course.title.label("course"),
                    Teacher.name.label("teacher"),
                    Student.student_id,
                    Student.name.label("student_name"),
                    Student.department,
                    Student.section,
                    AttendanceDetail.status,
                    AttendanceDetail.similarity_score,
                )
                .join(AttendanceDetail, AttendanceDetail.session_id == Attendance.id)
                .join(Student, Student.id == AttendanceDetail.student_id)
                .outerjoin(Class, Class.id == Attendance.class_id)
                .outerjoin(Course, Course.id == Class.course_id)
                .outerjoin(Teacher, Teacher.id == Class.teacher_id)
            )

            if start_date:
                q = q.filter(Attendance.date >= start_date)
            if end_date:
                q = q.filter(Attendance.date <= end_date)
            if department:
                q = q.filter(Student.department == department)
            if section:
                q = q.filter(Student.section == section)
            if course_id:
                q = q.filter(Class.course_id == course_id)
            if teacher_id:
                q = q.filter(Class.teacher_id == teacher_id)
            if student_search:
                like = f"%{student_search}%"
                q = q.filter(
                    (Student.student_id.ilike(like)) | (Student.name.ilike(like))
                )

            rows = q.all()

        df = pd.DataFrame(
            rows,
            columns=[
                "Date", "Time", "Course", "Teacher", "Student ID",
                "Student Name", "Department", "Section", "Status", "Similarity",
            ],
        )
        return df

    def period_range(self, period: str) -> tuple[dt.date, dt.date]:
        """Return (start, end) dates for the named period: Daily/Weekly/Monthly/Semester."""
        today = dt.date.today()
        if period == "Daily":
            return today, today
        if period == "Weekly":
            return today - dt.timedelta(days=7), today
        if period == "Monthly":
            return today - dt.timedelta(days=30), today
        if period == "Semester":
            return today - dt.timedelta(days=120), today
        return today - dt.timedelta(days=365), today  # fallback: last year

    # ------------------------------------------------------------------ #
    # Exports
    # ------------------------------------------------------------------ #
    def export_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> Path:
        filename = filename or f"attendance_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
        path = REPORTS_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Exported CSV report to %s", path)
        return path

    def export_excel(self, df: pd.DataFrame, filename: Optional[str] = None) -> Path:
        filename = filename or f"attendance_{dt.datetime.now():%Y%m%d_%H%M%S}.xlsx"
        path = REPORTS_DIR / filename

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Attendance")
            workbook = writer.book
            worksheet = writer.sheets["Attendance"]

            header_format = workbook.add_format(
                {"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1}
            )
            for col_num, col_name in enumerate(df.columns):
                worksheet.write(0, col_num, col_name, header_format)
                worksheet.set_column(col_num, col_num, max(12, len(col_name) + 2))

        logger.info("Exported Excel report to %s", path)
        return path

    def export_pdf(self, df: pd.DataFrame, title: str = "Attendance Report", filename: Optional[str] = None) -> Path:
        filename = filename or f"attendance_{dt.datetime.now():%Y%m%d_%H%M%S}.pdf"
        path = REPORTS_DIR / filename

        doc = SimpleDocTemplate(str(path), pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        elements = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

        table_data = [list(df.columns)] + df.astype(str).values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ]
            )
        )
        elements.append(table)
        doc.build(elements)

        logger.info("Exported PDF report to %s", path)
        return path
