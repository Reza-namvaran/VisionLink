"""Format and export analysis results — FR-6."""

import json
from pathlib import Path

from visionlink.models import AnalysisResult, Face


def format_results(
    faces: list[Face], source: Path | None = None
) -> dict[str, object]:
    """Build a JSON-serializable result dict matching the SRS schema."""
    payload: dict[str, object] = {
        "faces": [
            {
                "bbox": face.bbox.as_dict(),
                **face.gestures,
            }
            for face in faces
        ]
    }
    if source is not None:
        payload["source"] = str(source)
        payload["face_count"] = len(faces)
    return payload


def save_json(results: dict[str, object], path: Path) -> Path:
    """Write results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return path
