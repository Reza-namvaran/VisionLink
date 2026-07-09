"""Application configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Runtime configuration for a VisionLink session."""

    input_path: Path | None = None
    output_dir: Path = Path("output")
    camera_id: int | None = None
    draw_landmarks: bool = True
    draw_boxes: bool = True
    save_json: bool = True
    save_report: bool = True
    save_annotated: bool = True
    batch: bool = False
    recursive: bool = False
    quiet: bool = False
    serial_port: str | None = None
    serial_baud: int = 9600

    @property
    def is_webcam(self) -> bool:
        return self.camera_id is not None

    @property
    def is_directory(self) -> bool:
        return self.input_path is not None and self.input_path.is_dir()

    @property
    def serial_enabled(self) -> bool:
        return self.serial_port is not None
