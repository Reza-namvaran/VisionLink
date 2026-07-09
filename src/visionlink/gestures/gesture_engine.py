"""Gesture recognition — FR-4."""

import math
from dataclasses import dataclass
from typing import Literal

from visionlink.gestures.landmark_indices import (
    CHIN,
    FOREHEAD,
    LEFT_CHEEK,
    LEFT_EYE,
    LOWER_LIP,
    MIN_LANDMARKS,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    NOSE_TIP,
    RIGHT_CHEEK,
    RIGHT_EYE,
    UPPER_LIP,
)
from visionlink.models import Face

EyeState = Literal["open", "closed"]
HeadPose = Literal["center", "left", "right", "up", "down"]


@dataclass
class Gestures:
    """Recognized facial gestures for a single face."""

    smile: bool
    mouth_open: bool
    left_eye: EyeState
    right_eye: EyeState
    head_pose: HeadPose

    def as_dict(self) -> dict[str, object]:
        return {
            "smile": self.smile,
            "mouth_open": self.mouth_open,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "both_eyes_closed": self.left_eye == "closed" and self.right_eye == "closed",
            "head_pose": self.head_pose,
        }


@dataclass
class GestureThresholds:
    """Tunable thresholds for geometric gesture detection."""

    eye_aspect_ratio: float = 0.22
    mouth_aspect_ratio: float = 0.35
    smile_width_ratio: float = 0.42
    smile_corner_lift: float = 3.0
    head_yaw_ratio: float = 0.04
    head_pitch_up: float = 0.46
    head_pitch_down: float = 0.54


class GestureEngine:
    """Detects facial gestures from landmark coordinates."""

    def __init__(self, thresholds: GestureThresholds | None = None) -> None:
        self._thresholds = thresholds or GestureThresholds()

    def analyze(self, face: Face) -> Face:
        """Fill ``face.gestures`` from landmarks. Returns the same face object."""
        if len(face.landmarks) < MIN_LANDMARKS:
            face.gestures = Gestures(
                smile=False,
                mouth_open=False,
                left_eye="open",
                right_eye="open",
                head_pose="center",
            ).as_dict()
            return face

        lm = face.landmarks
        thresholds = self._thresholds

        left_ear = _eye_aspect_ratio(lm, LEFT_EYE)
        right_ear = _eye_aspect_ratio(lm, RIGHT_EYE)
        left_eye: EyeState = "closed" if left_ear < thresholds.eye_aspect_ratio else "open"
        right_eye: EyeState = (
            "closed" if right_ear < thresholds.eye_aspect_ratio else "open"
        )

        mouth_open = _mouth_aspect_ratio(lm) > thresholds.mouth_aspect_ratio
        smile = _is_smiling(lm, face.bbox.width, thresholds)

        face.gestures = Gestures(
            smile=smile,
            mouth_open=mouth_open,
            left_eye=left_eye,
            right_eye=right_eye,
            head_pose=_head_pose(lm, thresholds),
        ).as_dict()
        return face

    def analyze_all(self, faces: list[Face]) -> list[Face]:
        for face in faces:
            self.analyze(face)
        return faces


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _eye_aspect_ratio(
    landmarks: list[tuple[float, float, float]], indices: tuple[int, ...]
) -> float:
    points = [landmarks[i] for i in indices]
    vertical = _dist(points[1], points[5]) + _dist(points[2], points[4])
    horizontal = _dist(points[0], points[3])
    if horizontal == 0:
        return 1.0
    return vertical / (2.0 * horizontal)


def _mouth_aspect_ratio(landmarks: list[tuple[float, float, float]]) -> float:
    vertical = _dist(landmarks[UPPER_LIP], landmarks[LOWER_LIP])
    horizontal = _dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def _is_smiling(
    landmarks: list[tuple[float, float, float]],
    face_width: int,
    thresholds: GestureThresholds,
) -> bool:
    mouth_width = _dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    width_ratio = mouth_width / face_width if face_width > 0 else 0.0

    lip_center_y = (landmarks[UPPER_LIP][1] + landmarks[LOWER_LIP][1]) / 2
    corner_avg_y = (landmarks[MOUTH_LEFT][1] + landmarks[MOUTH_RIGHT][1]) / 2
    corner_lift = corner_avg_y - lip_center_y  # positive when corners are higher on face

    return (
        width_ratio > thresholds.smile_width_ratio
        and corner_lift > thresholds.smile_corner_lift
    )


def _head_pose(
    landmarks: list[tuple[float, float, float]], thresholds: GestureThresholds
) -> HeadPose:
    nose_x = landmarks[NOSE_TIP][0]
    face_center_x = (landmarks[LEFT_CHEEK][0] + landmarks[RIGHT_CHEEK][0]) / 2
    face_width = abs(landmarks[RIGHT_CHEEK][0] - landmarks[LEFT_CHEEK][0])
    yaw_offset = (nose_x - face_center_x) / face_width if face_width > 0 else 0.0

    forehead_y = landmarks[FOREHEAD][1]
    chin_y = landmarks[CHIN][1]
    nose_y = landmarks[NOSE_TIP][1]
    face_height = chin_y - forehead_y
    pitch_ratio = (nose_y - forehead_y) / face_height if face_height > 0 else 0.5

    if pitch_ratio < thresholds.head_pitch_up:
        return "up"
    if pitch_ratio > thresholds.head_pitch_down:
        return "down"
    if yaw_offset > thresholds.head_yaw_ratio:
        return "left"
    if yaw_offset < -thresholds.head_yaw_ratio:
        return "right"
    return "center"
