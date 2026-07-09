"""Batch image discovery — FR-7."""

from collections.abc import Iterator
from pathlib import Path

from visionlink.acquisition.image_loader import SUPPORTED_EXTENSIONS

_IMAGE_GLOB = {f"*{ext}" for ext in SUPPORTED_EXTENSIONS}


def iter_images(directory: Path) -> Iterator[Path]:
    """Yield supported image files in a directory (non-recursive)."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    paths: list[Path] = []
    for pattern in _IMAGE_GLOB:
        paths.extend(directory.glob(pattern))
        paths.extend(directory.glob(pattern.upper()))

    yield from sorted({path.resolve() for path in paths})
