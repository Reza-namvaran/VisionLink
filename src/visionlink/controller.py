"""Vision pipeline controller."""

from pathlib import Path

from visionlink.acquisition import load_image
from visionlink.acquisition.batch import iter_images
from visionlink.config import Config
from visionlink.detection import FaceDetector
from visionlink.exceptions import ImageLoadError
from visionlink.gestures import GestureEngine
from visionlink.landmarks import LandmarkDetector
from visionlink.models import AnalysisResult, ImageArray
from visionlink.output import Visualizer, format_results, save_json


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
        """Process a single image or every image in a directory."""
        if self._config.is_directory:
            return self.process_batch(self._config.input_path, self._config.output_dir)
        return [self.process_image(self._config.input_path, self._config.output_dir)]

    def process_batch(
        self, input_dir: Path, output_dir: Path
    ) -> list[AnalysisResult]:
        results: list[AnalysisResult] = []
        for image_path in iter_images(input_dir):
            results.append(self.process_image(image_path, output_dir))

        if self._config.save_json:
            summary_path = output_dir / "batch_summary.json"
            save_json(
                {
                    "images_processed": len(results),
                    "results": [result.as_dict() for result in results],
                },
                summary_path,
            )
        return results

    def process_image(self, image_path: Path, output_dir: Path) -> AnalysisResult:
        try:
            image = load_image(image_path)
        except ImageLoadError as exc:
            return AnalysisResult(source=image_path, error=str(exc))

        faces = self._analyze(image)
        result = AnalysisResult(source=image_path, faces=faces)

        annotated = self._visualizer.annotate(image, faces)
        image_out = output_dir / f"{image_path.stem}_annotated.png"
        self._visualizer.save(annotated, image_out)

        if self._config.save_json:
            json_out = output_dir / f"{image_path.stem}.json"
            save_json(format_results(faces, source=image_path), json_out)

        return result

    def _analyze(self, image: ImageArray):
        faces = self._face_detector.detect(image)
        faces = self._landmark_detector.add_landmarks(image, faces)
        return self._gesture_engine.analyze_all(faces)
