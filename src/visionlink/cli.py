"""Command-line interface for VisionLink."""

import argparse
import json
import logging
import sys
from pathlib import Path

from visionlink.logging_config import quiet_third_party_logs

quiet_third_party_logs()

from visionlink import __version__
from visionlink.config import Config
from visionlink.controller import Pipeline
from visionlink.models import AnalysisResult


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
        "--no-image",
        action="store_true",
        help="Skip saving annotated PNG files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when input is a folder",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress normal output (errors only)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _print_results(results: list[AnalysisResult], config: Config) -> None:
    if config.quiet:
        return

    if config.batch:
        print(f"Processed {len(results)} image(s) from {config.input_path}")
    elif results:
        result = results[0]
        if result.ok:
            size = ""
            if result.image_size:
                w, h = result.image_size
                size = f" ({w}×{h})"
            timing = ""
            if result.elapsed_ms is not None:
                timing = f" in {result.elapsed_ms:.0f} ms"
            print(
                f"Loaded {result.source.name}{size} — "
                f"{result.face_count} face(s){timing}"
            )

    for result in results:
        if result.error:
            print(f"  ✗ {result.source.name}: {result.error}", file=sys.stderr)
            continue
        if config.batch or config.quiet:
            timing = f" ({result.elapsed_ms:.0f} ms)" if result.elapsed_ms else ""
            print(f"  ✓ {result.source.name}: {result.face_count} face(s){timing}")
        elif result.faces:
            print(json.dumps(result.as_dict(), indent=2))

        if not config.quiet:
            if config.save_annotated:
                stem = result.source.stem
                print(f"    → {config.output_dir / f'{stem}_annotated.png'}")
            if config.save_json:
                print(f"    → {config.output_dir / f'{stem}.json'}")

    if config.batch and config.save_json and not config.quiet:
        print(f"Summary: {config.output_dir / 'batch_summary.json'}")
    if config.batch and config.save_report and len(results) > 1 and not config.quiet:
        print(f"Report:  {config.output_dir / 'batch_report.txt'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose, args.quiet)

    if not args.input.exists():
        print(f"Error: Path not found: {args.input}", file=sys.stderr)
        return 1

    config = Config(
        input_path=args.input,
        output_dir=args.output,
        draw_landmarks=not args.no_landmarks,
        save_json=not args.no_json,
        save_annotated=not args.no_image,
        batch=args.input.is_dir(),
        recursive=args.recursive,
        quiet=args.quiet,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with Pipeline(config) as pipeline:
        results = pipeline.run()

    if not config.batch and results and results[0].error:
        print(f"Error: {results[0].error}", file=sys.stderr)
        return 1

    _print_results(results, config)

    failed = sum(1 for result in results if result.error)
    if not results:
        return 1
    return 1 if failed == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
