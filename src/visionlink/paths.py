"""Project path helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the VisionLink repository root directory."""
    return Path(__file__).resolve().parents[2]


def model_path(filename: str) -> Path:
    """Resolve a path to a file inside the ``models/`` directory."""
    return project_root() / "models" / filename
