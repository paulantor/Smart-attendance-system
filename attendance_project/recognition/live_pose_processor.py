"""
recognition/live_pose_processor.py
====================================
Continuous "face-lock scanner" video processor for guided registration
capture, used via streamlit-webrtc.

Instead of clicking "take photo" repeatedly, the person clicks Start ONCE.
Every incoming video frame is analyzed:
  - RED box drawn  -> no face / wrong pose / too blurry / too small / multiple faces
  - GREEN box drawn -> face detected, pose correct, quality good
  - Holding a GREEN box steady for `stable_frames_required` frames triggers
    an automatic capture (no click needed) and advances to the next pose
    in the guided sequence (center -> left -> right -> up -> down -> center).

An oval NID-card-style guide frame is drawn in the center so the person
knows where to position their face, matching the familiar national-ID
scan UX.

If the detector itself throws an error, it is now shown directly on the
video feed (and logged to logs/app.log) instead of silently being treated
as "no face detected" -- this was the bug causing the false "no face
detected" message on your previous test.
"""

from __future__ import annotations

import queue
import time

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from config import settings
from recognition.face_detector import FaceDetector
from utils.image_quality import compute_blur_score, compute_brightness
from utils.pose_estimation import matches_expected_pose
from utils.logger import get_logger

logger = get_logger("recognition.live_pose_processor")

RED = (0, 0, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)


class PoseGuidedVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = FaceDetector()
        self.pose_sequence = settings.recognition.guided_pose_sequence
        self.pose_index = 0
        self.stable_count = 0
        self.stable_frames_required = 8   # ~consecutive good frames before auto-capture
        self.cooldown_until = 0.0
        self.capture_queue: "queue.Queue" = queue.Queue()
        self.completed = False

        self._frame_counter = 0
        self._detect_every_n = 2          # throttle detection for performance
        self._last_faces = []
        self._last_error = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        self._frame_counter += 1
        if self._frame_counter % self._detect_every_n == 0:
            try:
                self._last_faces = self.detector.detect(img)
                self._last_error = None
            except Exception as exc:  # noqa: BLE001 -- intentionally broad: we show it on-screen + log it
                self._last_faces = []
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("Live scanner detection failed")
        faces = self._last_faces

        if self._last_error:
            cv2.putText(img, "DETECTOR ERROR (see below):", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            cv2.putText(img, self._last_error[:70], (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.completed:
            cv2.putText(img, "All poses captured! Click Finish below.", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        pose_key, instruction = self.pose_sequence[self.pose_index]
        box_color = RED
        status_text = "No face detected"

        # --- NID-card-style oval guide frame (visual alignment target, like national-ID scan apps) --- #
        h, w = img.shape[:2]
        guide_center = (w // 2, h // 2)
        guide_axes = (int(w * 0.22), int(h * 0.32))
        guide_color = GREEN if (faces and len(faces) == 1) else WHITE
        cv2.ellipse(img, guide_center, guide_axes, 0, 0, 360, guide_color, 2)

        if faces and now >= self.cooldown_until:
            face = faces[0]
            x1, y1, x2, y2 = face.box
            brightness = compute_brightness(img)
            blur = compute_blur_score(img)

            pose_ok = matches_expected_pose(face.landmarks, pose_key)
            quality_ok = (
                brightness > settings.recognition.min_brightness
                and blur > settings.recognition.blur_threshold
            )
            size_ok = min(x2 - x1, y2 - y1) > settings.recognition.min_face_size_px
            single_face = len(faces) == 1

            if pose_ok and quality_ok and size_ok and single_face:
                self.stable_count += 1
                box_color = GREEN
                status_text = f"Good! Hold still... {self.stable_count}/{self.stable_frames_required}"
            else:
                self.stable_count = 0
                box_color = RED
                if not single_face:
                    status_text = "Multiple faces detected"
                elif not size_ok:
                    status_text = "Move closer to the camera"
                elif not quality_ok:
                    status_text = "Improve lighting / hold the camera steady"
                else:
                    status_text = "Adjust your pose to match the instruction"

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)

            if self.stable_count >= self.stable_frames_required:
                self.capture_queue.put((pose_key, img.copy()))
                self.stable_count = 0
                self.cooldown_until = now + 1.2
                self.pose_index += 1
                if self.pose_index >= len(self.pose_sequence):
                    self.completed = True
        elif faces and now < self.cooldown_until:
            status_text = "Captured! Get ready for the next pose..."
            box_color = GREEN

        cv2.putText(
            img, f"Step {min(self.pose_index + 1, len(self.pose_sequence))}/{len(self.pose_sequence)}: {instruction}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2,
        )
        cv2.putText(img, status_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def retry_current_pose(self) -> None:
        """Called from the main thread if RegistrationService rejects an auto-captured
        frame (e.g. duplicate face) -- rolls the pose index back so the person retries."""
        self.pose_index = max(0, self.pose_index - 1)
        self.completed = False
        self.cooldown_until = time.time() + 1.0