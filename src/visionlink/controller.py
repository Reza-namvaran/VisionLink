"""Vision pipeline controller."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from visionlink.acquisition import load_image
from visionlink.acquisition.batch import iter_images
from visionlink.acquisition.webcam import WebcamCapture
from visionlink.config import Config
from visionlink.detection import FaceDetector
from visionlink.exceptions import ImageLoadError
from visionlink.gestures import GestureEngine
from visionlink.landmarks import LandmarkDetector
from visionlink.models import AnalysisResult, Face, ImageArray
from visionlink.output import Visualizer, draw_hud, format_results, save_json
from visionlink.output.report import build_batch_stats, save_batch_report

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full image analysis workflow."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._face_detector = FaceDetector()
        self._landmark_detector = LandmarkDetector()
        self._gesture_engine = GestureEngine()
        self._visualizer = Visualizer(
            draw_boxes=config.draw_boxes,
            draw_landmarks=config.draw_landmarks,
        )

    def close(self) -> None:
        self._face_detector.close()
        self._landmark_detector.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def run(self) -> list[AnalysisResult]:
        """Process a single image, batch directory, or webcam stream."""
        if self._config.is_webcam:
            self.run_webcam()
            return []
        if self._config.input_path is None:
            raise ValueError("input_path is required for image/batch mode")
        if self._config.is_directory:
            return self.process_batch(self._config.input_path, self._config.output_dir)
        return [self.process_image(self._config.input_path, self._config.output_dir)]

    def run_webcam(self) -> None:
        """Run real-time analysis on a USB camera. Press q to quit, s to snapshot."""
        camera_id = self._config.camera_id if self._config.camera_id is not None else 0
        window = "VisionLink"
        logger.info("Starting webcam %d — press q to quit, s to snapshot", camera_id)

        self._config.output_dir.mkdir(parents=True, exist_ok=True)

        with WebcamCapture(camera_id) as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    logger.error("Failed to read frame from camera %d", camera_id)
                    break

                started = time.perf_counter()
                faces = self.analyze_frame(frame)
                elapsed_ms = (time.perf_counter() - started) * 1000

                annotated = self._visualizer.annotate(frame, faces)
                fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
                draw_hud(annotated, fps=fps, face_count=len(faces))

                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    self._save_webcam_snapshot(annotated, faces)

        cv2.destroyAllWindows()
        logger.info("Webcam session ended")

    def analyze_frame(self, image: ImageArray) -> list[Face]:
        """Run detection, landmarks, and gestures on a single frame."""
        faces = self._face_detector.detect(image)
        faces = self._landmark_detector.add_landmarks(image, faces)
        return self._gesture_engine.analyze_all(faces)

    def process_batch(
        self, input_dir: Path, output_dir: Path
    ) -> list[AnalysisResult]:
        results: list[AnalysisResult] = []
        images = list(
            iter_images(input_dir, recursive=self._config.recursive)
        )
        if not images:
            logger.warning("No images found in %s", input_dir)
            return results

        for image_path in images:
            results.append(self.process_image(image_path, output_dir))

        if self._config.save_json:
            summary_path = output_dir / "batch_summary.json"
            save_json(
                {
                    **build_batch_stats(results),
                    "results": [result.as_dict() for result in results],
                },
                summary_path,
            )

        if self._config.save_report and len(results) > 1:
            report_path = output_dir / "batch_report.txt"
            save_batch_report(results, report_path)
            logger.info("Saved batch report: %s", report_path)

        return results

    def process_image(self, image_path: Path, output_dir: Path) -> AnalysisResult:
        started = time.perf_counter()
        try:
            image = load_image(image_path)
        except ImageLoadError as exc:
            logger.error("Failed to load %s: %s", image_path.name, exc)
            return AnalysisResult(
                source=image_path,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        height, width = image.shape[:2]
        faces = self.analyze_frame(image)
        elapsed_ms = (time.perf_counter() - started) * 1000
        result = AnalysisResult(
            source=image_path,
            faces=faces,
            elapsed_ms=elapsed_ms,
            image_size=(width, height),
        )

        if self._config.save_annotated:
            annotated = self._visualizer.annotate(image, faces)
            image_out = output_dir / f"{image_path.stem}_annotated.png"
            self._visualizer.save(annotated, image_out)

        if self._config.save_json:
            json_out = output_dir / f"{image_path.stem}.json"
            save_json(format_results(faces, source=image_path), json_out)

        logger.debug(
            "Processed %s in %.0f ms (%d face(s))",
            image_path.name,
            elapsed_ms,
            result.face_count,
        )
        return result

    def _save_webcam_snapshot(self, frame: ImageArray, faces: list[Face]) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"webcam_{stamp}"
        image_path = self._config.output_dir / f"{stem}.png"
        json_path = self._config.output_dir / f"{stem}.json"

        self._visualizer.save(frame, image_path)
        if self._config.save_json:
            save_json(format_results(faces, source=image_path), json_path)

        logger.info("Saved snapshot: %s", image_path)
        return image_path

