"""Shared data models."""

from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class AnalysisResult:
    """Complete analysis output for a single image."""

    source: Path
    faces: list[Face] = field(default_factory=list)
    error: str | None = None

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": str(self.source),
            "face_count": self.face_count,
            "faces": [
                {"bbox": face.bbox.as_dict(), **face.gestures} for face in self.faces
            ],
        }
        if self.error:
            payload["error"] = self.error
        return payload

