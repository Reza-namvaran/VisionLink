from pathlib import Path

import cv2

from visionlink.exceptions import ImageLoadError
from visionlink.models import ImageArray

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(path: str | Path) -> ImageArray:
    """Load a JPEG or PNG image from disk.

    Args:
        path: Filesystem path to the image.

    Returns:
        The image as a BGR numpy array (OpenCV convention btw).

    Raises:
        ImageLoadError: If the file is missing, unsupported, or unreadable.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise ImageLoadError(f"Image not found: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image format '{file_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    image: ImageArray | None = cv2.imread(str(file_path))
    if image is None:
        raise ImageLoadError(f"Could not read image (corrupt or invalid): {file_path}")

    return image
