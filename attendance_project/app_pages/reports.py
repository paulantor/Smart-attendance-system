"""pages/reports.py -- Generate & export Daily/Weekly/Monthly/Semester reports as CSV/Excel/PDF."""

from __future__ import annotations

import streamlit as st

from services.report_service import ReportService

report_service = ReportService()


def render() -> None:
    st.title("📑 Reports")

    period = st.selectbox("Report Period", ["Daily", "Weekly", "Monthly", "Semester"])
    department = st.text_input("Filter by Department (optional)")

    start, end = report_service.period_range(period)
    st.caption(f"Date range: {start} to {end}")

    df = report_service.query_attendance(start_date=start, end_date=end, department=department or None)

    if df.empty:
        st.info("No attendance data available for this period.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("⬇️ Export as CSV", use_container_width=True):
            path = report_service.export_csv(df, filename=f"{period.lower()}_report.csv")
            with open(path, "rb") as f:
                st.download_button("Download CSV", f, file_name=path.name, mime="text/csv")

    with c2:
        if st.button("⬇️ Export as Excel", use_container_width=True):
            path = report_service.export_excel(df, filename=f"{period.lower()}_report.xlsx")
            with open(path, "rb") as f:
                st.download_button(
                    "Download Excel", f, file_name=path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with c3:
        if st.button("⬇️ Export as PDF", use_container_width=True):
            path = report_service.export_pdf(df, title=f"{period} Attendance Report", filename=f"{period.lower()}_report.pdf")
            with open(path, "rb") as f:
                st.download_button("Download PDF", f, file_name=path.name, mime="application/pdf")
