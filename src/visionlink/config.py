"""Application configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Runtime configuration for a VisionLink session."""

    input_path: Path
    output_dir: Path = Path("output")
    draw_landmarks: bool = True
    draw_boxes: bool = True
    save_json: bool = True
    batch: bool = False

    @property
    def is_directory(self) -> bool:
        return self.input_path.is_dir()
