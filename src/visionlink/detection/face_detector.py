from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from visionlink.models import BoundingBox, Face, ImageArray

MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "blaze_face_short_range.tflite"
)


class FaceDetector:
    """Detects faces in images using MediaPipe BlazeFace."""

    def __init__(
        self,
        model_path: Path | None = None,
        min_detection_confidence: float = 0.5,
    ) -> None:
        path = model_path or MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Face detection model not found: {path}")

        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image: ImageArray) -> list[Face]:
        """Detect faces and return a list of Face objects with bounding boxes.

        Returns an empty list when no faces are found — never raises for that case.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        faces: list[Face] = []
        for detection in result.detections:
            bbox = detection.bounding_box
            faces.append(
                Face(
                    bbox=BoundingBox(
                        x=bbox.origin_x,
                        y=bbox.origin_y,
                        width=bbox.width,
                        height=bbox.height,
                    )
                )
            )
        return faces

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
