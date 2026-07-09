"""Command-line interface for VisionLink."""

import argparse
import json
import sys
from pathlib import Path

from visionlink import __version__
from visionlink.acquisition import load_image
from visionlink.detection import FaceDetector
from visionlink.exceptions import ImageLoadError
from visionlink.gestures import GestureEngine
from visionlink.landmarks import LandmarkDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionlink",
        description="Detect facial gestures from images.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a JPEG or PNG image",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        image = load_image(args.image)
    except ImageLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    height, width = image.shape[:2]
    print(f"Loaded {args.image} ({width}×{height}, {image.shape[2]} channels)")

    with FaceDetector() as detector, LandmarkDetector() as landmarker:
        faces = detector.detect(image)
        faces = landmarker.add_landmarks(image, faces)

    faces = GestureEngine().analyze_all(faces)

    print(f"Detected {len(faces)} face(s)")
    if faces:
        for i, face in enumerate(faces, start=1):
            print(f"\nFace {i}:")
            print(json.dumps({"bbox": face.bbox.as_dict(), **face.gestures}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
