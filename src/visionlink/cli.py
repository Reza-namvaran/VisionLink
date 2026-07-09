"""Command-line interface for VisionLink."""

import argparse
import json
import sys
from pathlib import Path

from visionlink import __version__
from visionlink.config import Config
from visionlink.controller import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionlink",
        description="Detect facial gestures from images.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an image or a directory of images (batch mode)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory for annotated images and JSON (default: output/)",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Skip drawing landmark dots",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing JSON results files",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"Error: Path not found: {args.input}", file=sys.stderr)
        return 1

    config = Config(
        input_path=args.input,
        output_dir=args.output,
        draw_landmarks=not args.no_landmarks,
        save_json=not args.no_json,
        batch=args.input.is_dir(),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with Pipeline(config) as pipeline:
        results = pipeline.run()

    if config.batch:
        print(f"Processed {len(results)} image(s) from {args.input}")
    else:
        result = results[0]
        if result.error:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1
        print(
            f"Loaded {result.source} — detected {result.face_count} face(s)"
        )

    for result in results:
        if result.error:
            print(f"  ✗ {result.source.name}: {result.error}", file=sys.stderr)
            continue
        print(f"  ✓ {result.source.name}: {result.face_count} face(s)")
        if result.faces and not config.batch:
            print(json.dumps(result.as_dict(), indent=2))

        stem = result.source.stem
        print(f"    → {config.output_dir / f'{stem}_annotated.png'}")
        if config.save_json:
            print(f"    → {config.output_dir / f'{stem}.json'}")

    if config.batch and config.save_json:
        print(f"Summary: {config.output_dir / 'batch_summary.json'}")

    failed = sum(1 for result in results if result.error)
    return 1 if failed and not config.batch else (1 if failed == len(results) else 0)


if __name__ == "__main__":
    raise SystemExit(main())
