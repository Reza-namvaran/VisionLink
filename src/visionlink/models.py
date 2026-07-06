"""Shared data models."""
"""TODO: Add this to a folder with better structure in future"""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# OpenCV images: H×W×C uint8 arrays.
ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned rectangle around a detected face."""

    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Face:
    """A detected face with optional analysis results."""

    bbox: BoundingBox
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)
    gestures: dict[str, object] = field(default_factory=dict)
