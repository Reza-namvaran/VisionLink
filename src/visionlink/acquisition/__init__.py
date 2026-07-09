"""Image acquisition module."""

from visionlink.acquisition.batch import iter_images
from visionlink.acquisition.image_loader import SUPPORTED_EXTENSIONS, load_image
from visionlink.acquisition.webcam import WebcamCapture

__all__ = ["SUPPORTED_EXTENSIONS", "WebcamCapture", "iter_images", "load_image"]
