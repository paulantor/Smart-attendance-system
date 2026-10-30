"""pages/take_attendance.py -- Teacher captures a classroom photo (live camera or upload); the system auto-marks attendance."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from database import get_session
from models.models import Class, Course, Teacher
from services.attendance_service import AttendanceService

attendance_service = AttendanceService()


def _decode_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def _draw_annotations(image_bgr: np.ndarray, faces) -> np.ndarray:
    annotated = image_bgr.copy()
    for face in faces:
        x1, y1, x2, y2 = face.box
        color = (0, 200, 0) if face.status == "Present" else (0, 0, 220)
        label = face.student_name or "Unknown"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def _process_and_show_result(frame: np.ndarray, class_id) -> None:
    with st.spinner("Detecting and recognizing faces..."):
        result = attendance_service.take_attendance(frame, class_id=class_id)

    st.success(
        f"Attendance session #{result.session_id} recorded: "
        f"{result.total_detected} faces detected, {result.total_recognized} recognized, "
        f"{result.total_unknown} unknown."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Faces", result.total_detected)
    col2.metric("Recognized (Present)", result.total_recognized)
    col3.metric("Unknown", result.total_unknown)

    st.image(_draw_annotations(frame, result.recognized_faces), caption="Recognized faces", use_container_width=True)

    st.subheader("Recognized Students")
    if result.recognized_faces:
        rows = [
            {
                "Student ID": f.student_id or "-",
                "Name": f.student_name or "Unknown",
                "Status": f.status,
                "Similarity": f"{f.similarity:.3f}",
                "Detector Confidence": f"{f.confidence:.3f}",
            }
            for f in result.recognized_faces
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No faces were detected in this photo.")

    st.subheader("Absent Students")
    if result.absent_students:
        st.dataframe(
            [{"Student ID": s.student_id, "Name": s.name} for s in result.absent_students],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No absent students -- everyone active was recognized.")


def render() -> None:
    st.title("📸 Take Attendance")

    with get_session() as session:
        classes = session.query(Class).join(Course).join(Teacher).all()
        class_options = {
            f"{c.course.title} · {c.teacher.name} · Section {c.section or '-'}": c.id for c in classes
        }

    class_label = st.selectbox("Select Class (optional)", ["-- No specific class --"] + list(class_options.keys()))
    class_id = class_options.get(class_label)

    st.divider()

    # --- Choose input method: Live Camera or Upload Photo ---
    input_method = st.radio(
        "Choose how to provide the classroom photo",
        ["📸 Live Camera", "📁 Upload Photo"],
        horizontal=True,
    )

    if input_method == "📸 Live Camera":
        camera_image = st.camera_input("Capture the classroom photo (use your iPhone/webcam camera)")
        if camera_image is None:
            st.info("Waiting for a classroom photo to be captured...")
            return
        frame = _decode_uploaded_image(camera_image)
        _process_and_show_result(frame, class_id)

    else:  # Upload Photo
        uploaded_file = st.file_uploader(
            "Upload a classroom photo",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is None:
            st.info("Waiting for a classroom photo to be uploaded...")
            return

        frame = _decode_uploaded_image(uploaded_file)
        if frame is None:
            st.error("Could not read the uploaded image. Please try a different file.")
            return

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded photo preview", use_container_width=True)

        if st.button("✅ Process Attendance from this Photo", type="primary"):
            _process_and_show_result(frame, class_id)