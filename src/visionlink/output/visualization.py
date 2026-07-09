"""Draw face analysis results on images — FR-5."""

from pathlib import Path

import cv2

from visionlink.models import Face, ImageArray


class Visualizer:
    """Renders bounding boxes, landmarks, and gesture labels on an image."""

    def __init__(
        self,
        draw_boxes: bool = True,
        draw_landmarks: bool = True,
        draw_labels: bool = True,
    ) -> None:
        self.draw_boxes = draw_boxes
        self.draw_landmarks = draw_landmarks
        self.draw_labels = draw_labels

    def annotate(self, image: ImageArray, faces: list[Face]) -> ImageArray:
        """Return a copy of the image with annotations drawn."""
        canvas = image.copy()

        for face in faces:
            if self.draw_boxes:
                _draw_bbox(canvas, face)
            if self.draw_landmarks and face.landmarks:
                _draw_landmarks(canvas, face.landmarks)
            if self.draw_labels and face.gestures:
                _draw_labels(canvas, face)

        return canvas

    def save(self, image: ImageArray, path: Path) -> Path:
        """Save an annotated image as PNG."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to write image: {path}")
        return path


def _draw_bbox(canvas: ImageArray, face: Face) -> None:
    bbox = face.bbox
    cv2.rectangle(
        canvas,
        (bbox.x, bbox.y),
        (bbox.x + bbox.width, bbox.y + bbox.height),
        (0, 255, 0),
        2,
    )


def _draw_landmarks(
    canvas: ImageArray, landmarks: list[tuple[float, float, float]]
) -> None:
    for x, y, _z in landmarks:
        cv2.circle(canvas, (int(x), int(y)), 1, (255, 200, 0), -1)


def _draw_labels(canvas: ImageArray, face: Face) -> None:
    gestures = face.gestures
    lines = _gesture_label_lines(gestures)
    if not lines:
        return

    x = face.bbox.x
    y = max(face.bbox.y - 10, 20)
    for line in lines:
        (text_w, text_h), _baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            canvas,
            (x, y - text_h - 4),
            (x + text_w + 4, y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            canvas,
            line,
            (x + 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y -= text_h + 8


def _gesture_label_lines(gestures: dict[str, object]) -> list[str]:
    lines: list[str] = []

    active: list[str] = []
    if gestures.get("smile"):
        active.append("smile")
    if gestures.get("mouth_open"):
        active.append("mouth open")
    if gestures.get("both_eyes_closed"):
        active.append("eyes closed")
    elif gestures.get("left_eye") == "closed":
        active.append("left eye closed")
    elif gestures.get("right_eye") == "closed":
        active.append("right eye closed")
    if active:
        lines.append(", ".join(active))

    head_pose = gestures.get("head_pose", "center")
    if head_pose != "center":
        lines.append(f"head: {head_pose}")

    return lines


def draw_hud(
    canvas: ImageArray,
    *,
    fps: float,
    face_count: int,
) -> None:
    """Draw FPS and keyboard hints on a live frame."""
    cv2.putText(
        canvas,
        f"{fps:.0f} FPS  |  {face_count} face(s)",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "q quit   s snapshot",
        (10, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
