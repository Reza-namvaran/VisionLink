"""Landmark extraction — FR-3."""

from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from visionlink.mediapipe_utils import to_mediapipe_image
from visionlink.models import Face, ImageArray
from visionlink.paths import model_path

MODEL_PATH = model_path("face_landmarker.task")


class LandmarkDetector:
    """Extracts facial landmarks using MediaPipe Face Landmarker."""

    def __init__(
        self,
        model_path: Path | None = None,
        num_faces: int = 5,
        min_detection_confidence: float = 0.5,
    ) -> None:
        path = model_path or MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Face landmarker model not found: {path}")

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def add_landmarks(self, image: ImageArray, faces: list[Face]) -> list[Face]:
        """Attach landmark coordinates to each detected face.

        Landmarks are stored as (x, y, z) pixel coordinates. MediaPipe returns
        up to 478 points per face (face mesh + iris landmarks).

        Faces without a matching landmark set are returned unchanged.
        """
        if not faces:
            return faces

        landmark_sets = self._detect_landmarks(image)
        for i, face in enumerate(faces):
            if i < len(landmark_sets):
                face.landmarks = landmark_sets[i]
        return faces

    def _detect_landmarks(
        self, image: ImageArray
    ) -> list[list[tuple[float, float, float]]]:
        height, width = image.shape[:2]
        mp_image = to_mediapipe_image(image)
        result = self._landmarker.detect(mp_image)

        all_landmarks: list[list[tuple[float, float, float]]] = []
        for face_landmarks in result.face_landmarks:
            points = [
                (lm.x * width, lm.y * height, lm.z) for lm in face_landmarks
            ]
            all_landmarks.append(points)
        return all_landmarks

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "LandmarkDetector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
