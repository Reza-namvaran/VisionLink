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
from visionlink.output import Visualizer, format_results, save_json


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
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for annotated image and JSON (default: output/)",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Skip drawing landmark dots",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing JSON results file",
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
    results = format_results(faces)
    if faces:
        print(json.dumps(results, indent=2))

    visualizer = Visualizer(draw_landmarks=not args.no_landmarks)
    annotated = visualizer.annotate(image, faces)
    image_out = args.output / f"{args.image.stem}_annotated.png"
    visualizer.save(annotated, image_out)
    print(f"Saved annotated image: {image_out}")

    if not args.no_json:
        json_out = args.output / f"{args.image.stem}.json"
        save_json(results, json_out)
        print(f"Saved JSON: {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
