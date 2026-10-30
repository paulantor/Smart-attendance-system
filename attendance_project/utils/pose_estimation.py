"""
utils/pose_estimation.py
==========================
Lightweight head-pose estimation from the 5-point facial landmarks that
RetinaFace/InsightFace already return with every detection (no extra model
needed). Landmark order (InsightFace convention): 
    [0] left_eye, [1] right_eye, [2] nose, [3] left_mouth, [4] right_mouth
("left"/"right" as seen in the image, not anatomical).

This is a simple geometric heuristic (nose-offset for yaw, nose-to-eye/mouth
ratio for pitch) -- good enough to confirm someone actually turned their head
roughly the requested direction during guided registration capture. It is
NOT a precise 3D pose estimator and is not used for liveness/anti-spoofing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseEstimate:
    yaw: str    # "left", "right", or "center"
    pitch: str  # "up", "down", or "center"
    yaw_ratio: float
    pitch_ratio: float


def estimate_pose(landmarks: np.ndarray, yaw_threshold: float = 0.15, pitch_threshold: float = 0.15) -> PoseEstimate:
    """
    Estimate a coarse yaw/pitch direction from 5-point landmarks.

    yaw_ratio  > 0  -> nose shifted toward the image-right eye  -> person turned LEFT (their left)
    yaw_ratio  < 0  -> nose shifted toward the image-left eye   -> person turned RIGHT (their right)
    pitch_ratio > threshold -> looking DOWN, < -threshold -> looking UP
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks[:5]

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    inter_eye_dist = np.linalg.norm(right_eye - left_eye)
    face_height = np.linalg.norm(mouth_center - eye_center)

    if inter_eye_dist < 1e-6 or face_height < 1e-6:
        return PoseEstimate("center", "center", 0.0, 0.0)

    # --- Yaw: horizontal nose deviation from the eye midpoint, normalized by inter-eye distance ---
    yaw_ratio = float((nose[0] - eye_center[0]) / inter_eye_dist)
    if yaw_ratio > yaw_threshold:
        yaw = "right"   # subject turned their head so their RIGHT cheek shows more (our "turn right" prompt)
    elif yaw_ratio < -yaw_threshold:
        yaw = "left"
    else:
        yaw = "center"

    # --- Pitch: where the nose sits between the eye-line and mouth-line ---
    pitch_ratio = float((nose[1] - eye_center[1]) / face_height - 0.5)
    if pitch_ratio > pitch_threshold:
        pitch = "down"
    elif pitch_ratio < -pitch_threshold:
        pitch = "up"
    else:
        pitch = "center"

    return PoseEstimate(yaw=yaw, pitch=pitch, yaw_ratio=yaw_ratio, pitch_ratio=pitch_ratio)


def matches_expected_pose(landmarks: np.ndarray, expected: str) -> bool:
    """
    expected is one of: "center", "left", "right", "up", "down".
    Returns True if the estimated pose satisfies the expected direction.
    """
    pose = estimate_pose(landmarks)
    if expected == "center":
        return pose.yaw == "center" and pose.pitch == "center"
    if expected in ("left", "right"):
        return pose.yaw == expected
    if expected in ("up", "down"):
        return pose.pitch == expected
    return True