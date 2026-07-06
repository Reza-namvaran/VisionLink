"""Command-line interface for VisionLink."""

import argparse
import sys
from pathlib import Path

from visionlink import __version__
from visionlink.acquisition import load_image
from visionlink.exceptions import ImageLoadError


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
    print(f"Loaded {args.image} ({width}x{height}, {image.shape[2]} channels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
