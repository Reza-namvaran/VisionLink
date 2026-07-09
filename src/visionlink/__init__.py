"""VisionLink — modular facial gesture recognition framework."""

from visionlink.config import Config
from visionlink.controller import Pipeline
from visionlink.models import AnalysisResult, BoundingBox, Face

__version__ = "0.2.0"
__all__ = [
    "AnalysisResult",
    "BoundingBox",
    "Config",
    "Face",
    "Pipeline",
    "__version__",
]
