"""Webcam capture — M8."""

import cv2

from visionlink.exceptions import WebcamError
from visionlink.models import ImageArray


class WebcamCapture:
    """Reads frames from a USB camera via OpenCV."""

    def __init__(self, camera_id: int = 0) -> None:
        self._camera_id = camera_id
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._camera_id)
        if not capture.isOpened():
            raise WebcamError(
                f"Could not open camera {self._camera_id}. "
                "Check that a webcam is connected and not in use."
            )
        self._capture = capture

    def read(self) -> ImageArray | None:
        if self._capture is None:
            raise WebcamError("Webcam is not open. Call open() first.")
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
