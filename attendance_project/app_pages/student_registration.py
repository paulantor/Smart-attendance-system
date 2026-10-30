"""
pages/student_registration.py
===============================
Registration form + guided-pose capture flow using Streamlit's reliable
st.camera_input widget (click-to-capture).

User-friendly UI updates applied to clearly instruct the user.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from config import settings
from services.registration_service import DuplicateStudentError, IncompleteRegistrationError, RegistrationService

registration_service = RegistrationService()


def _decode_image_bytes(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def render() -> None:
    st.title("🧑‍🎓 Student Registration")

    if "reg_student_db_id" not in st.session_state:
        st.session_state.reg_student_db_id = None
        st.session_state.reg_student_id = None
        st.session_state.reg_accepted = 0
        st.session_state.reg_rejected = 0
        st.session_state.reg_frame_index = 0

    # ------------------------------------------------------------------ #
    # Step 1: student details form (only shown before a record is created)
    # ------------------------------------------------------------------ #
    if st.session_state.reg_student_db_id is None:
        st.subheader("Step 1 · Student Details")
        with st.form("registration_form"):
            c1, c2 = st.columns(2)
            student_id = c1.text_input("Student ID *")
            name = c2.text_input("Full Name *")
            department = c1.text_input("Department")
            semester = c2.text_input("Semester")
            section = c1.text_input("Section")
            phone = c2.text_input("Phone")
            email = st.text_input("Email")

            submitted = st.form_submit_button("Register & Start Capturing Images", use_container_width=True)

        if submitted:
            try:
                db_id = registration_service.create_student_record(
                    student_id=student_id, name=name, department=department,
                    semester=semester, section=section, phone=phone, email=email,
                )
                st.session_state.reg_student_db_id = db_id
                st.session_state.reg_student_id = student_id.strip()
                st.rerun()
            except DuplicateStudentError as e:
                st.error(str(e))
            except ValueError as e:
                st.error(str(e))
        return

    # ------------------------------------------------------------------ #
    # Step 2: capture images -- Guided Camera OR Upload Photos
    # ------------------------------------------------------------------ #
    pose_sequence = settings.recognition.guided_pose_sequence
    target = settings.recognition.min_registration_images
    max_images = settings.recognition.max_registration_images

    st.subheader(f"Step 2 · Capture Face Images for {st.session_state.reg_student_id}")

    accepted = st.session_state.reg_accepted
    st.progress(min(1.0, accepted / target), text=f"{accepted} / {target} guided poses captured")

    input_method = st.radio(
        "Choose how to provide face images",
        ["📸 Guided Camera Capture", "📁 Upload Photos"],
        horizontal=True,
    )

    col_input, col_status = st.columns([2, 1])

    # -------------------- Guided Camera Capture path -------------------- #
    if input_method == "📸 Guided Camera Capture":
        pose_index = min(st.session_state.reg_accepted, len(pose_sequence) - 1)
        pose_key, instruction = pose_sequence[pose_index]

        if st.session_state.reg_accepted < len(pose_sequence):
            # User-Friendly Instruction Box
            st.info(f"### 🎯 Step {st.session_state.reg_accepted + 1}/{len(pose_sequence)}: \n## **{instruction}**")
            st.caption("💡 **Tip:** Keep your face exactly in the center of the camera view and click 'Take Photo'.")
        else:
            st.success("🎉 All guided poses captured! You can finish registration below.")

        with col_input:
            camera_image = st.camera_input(
                "Face Scanner Camera",
                key=f"cam_{st.session_state.reg_frame_index}",
            )

        if camera_image is not None and st.session_state.reg_accepted < len(pose_sequence):
            frame = _decode_image_bytes(camera_image)
            with st.spinner("Validating your face pose..."):
                result = registration_service.process_capture_frame(
                    student_db_id=st.session_state.reg_student_db_id,
                    student_id=st.session_state.reg_student_id,
                    frame_bgr=frame,
                    frame_index=st.session_state.reg_frame_index,
                    expected_pose=pose_key,
                )

            st.session_state.reg_frame_index += 1
            if result.accepted:
                st.session_state.reg_accepted += 1
                st.success(f"✅ Awesome! Pose captured successfully ({st.session_state.reg_accepted}/{len(pose_sequence)}).")
                st.rerun()
            else:
                st.session_state.reg_rejected += 1
                st.error(f"❌ Could not verify pose: {result.reason}. Please try again!")

    # -------------------- Upload Photos path -------------------- #
    else:
        with col_input:
            uploaded_files = st.file_uploader(
                "Select multiple existing face photos to upload at once",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) selected.")
                if st.button("⬆️ Process All Uploaded Photos", type="primary", use_container_width=True):
                    progress_bar = st.progress(0.0)
                    status_area = st.container()

                    for i, uploaded_file in enumerate(uploaded_files):
                        frame = _decode_image_bytes(uploaded_file)
                        if frame is None:
                            status_area.error(f"❌ {uploaded_file.name}: could not read this file, skipped.")
                            st.session_state.reg_rejected += 1
                        else:
                            result = registration_service.process_capture_frame(
                                student_db_id=st.session_state.reg_student_db_id,
                                student_id=st.session_state.reg_student_id,
                                frame_bgr=frame,
                                frame_index=st.session_state.reg_frame_index,
                                expected_pose=None,  # uploaded photos: skip pose matching
                            )
                            st.session_state.reg_frame_index += 1

                            if result.accepted:
                                st.session_state.reg_accepted += 1
                                status_area.success(f"✅ {uploaded_file.name}: accepted ({st.session_state.reg_accepted}/{target}).")
                            else:
                                st.session_state.reg_rejected += 1
                                status_area.error(f"❌ {uploaded_file.name}: rejected -- {result.reason}")

                        progress_bar.progress((i + 1) / len(uploaded_files))

                    st.rerun()

    with col_status:
        st.metric("Accepted", st.session_state.reg_accepted)
        st.metric("Rejected", st.session_state.reg_rejected)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.reg_accepted >= target:
            if st.button("✅ Finish Registration", type="primary", use_container_width=True):
                try:
                    registration_service.finalize_registration(st.session_state.reg_student_db_id)
                    st.success("Registration complete! Embeddings cached and ready for attendance.")
                    for key in ("reg_student_db_id", "reg_student_id", "reg_accepted", "reg_rejected", "reg_frame_index"):
                        st.session_state.pop(key, None)
                    st.rerun()
                except IncompleteRegistrationError as e:
                    st.error(f"❌ {e}")
        else:
            st.info(f"Capture at least {target - st.session_state.reg_accepted} more good images to finish.")
    with c2:
        if st.button("Cancel Registration", use_container_width=True):
            registration_service.delete_student(st.session_state.reg_student_db_id)
            for key in ("reg_student_db_id", "reg_student_id", "reg_accepted", "reg_rejected", "reg_frame_index"):
                st.session_state.pop(key, None)
            st.rerun()

    if st.session_state.reg_accepted >= max_images:
        st.warning(f"Maximum of {max_images} images reached. Please finish registration.")