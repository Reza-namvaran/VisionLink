"""Shared MediaPipe image conversion."""

import cv2
import mediapipe as mp

from visionlink.models import ImageArray


def to_mediapipe_image(image: ImageArray) -> mp.Image:
    """Convert an OpenCV BGR image to a MediaPipe SRGB image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
