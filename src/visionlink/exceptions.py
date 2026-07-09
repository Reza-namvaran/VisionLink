"""Custom exceptions for VisionLink."""

class VisionLinkError(Exception):
    """Base exception for all VisionLink errors."""


class ImageLoadError(VisionLinkError):
    """Raised when an image cannot be loaded or is invalid."""


class WebcamError(VisionLinkError):
    """Raised when the webcam cannot be opened or read."""
