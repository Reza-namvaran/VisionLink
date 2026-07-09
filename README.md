# VisionLink

Modular computer vision framework for **facial gesture recognition** from images and live webcam feeds.  
VisionLink runs fully offline on your machine — no cloud required.

Process a single photo, an entire folder, or a live camera feed.

---

## What it does

| Step | Capability |
|------|------------|
| **Load** | Read JPEG / PNG images from disk |
| **Detect** | Find faces and return bounding boxes (MediaPipe BlazeFace) |
| **Landmark** | Extract 478 facial landmark points per face |
| **Recognize** | Classify gestures from landmark geometry |
| **Visualize** | Draw boxes, landmarks, and gesture labels on the image |
| **Export** | Save annotated PNG + JSON results |
| **Batch** | Process every image in a folder independently |
| **Webcam** | Real-time gesture detection from a USB camera |

### Gestures detected

- Smile
- Mouth open
- Left / right eye open or closed
- Both eyes closed
- Head pose — left, right, up, down, or center

---

## Example

**Input** → **Output**

| Original | Annotated result |
|----------|------------------|
| ![Input image](images/face.jpg) | ![Annotated result](images/results/face_annotated.png) |

```bash
visionlink images/face.jpg -o images/results
```

The annotated image shows:
- **Green box** — detected face
- **Cyan dots** — 478 facial landmarks
- **Label** — recognized head pose (`head: down` in this example)

### Sample JSON output

```json
{
  "source": "images/face.jpg",
  "face_count": 1,
  "faces": [
    {
      "bbox": { "x": 197, "y": 221, "width": 160, "height": 160 },
      "smile": false,
      "mouth_open": false,
      "left_eye": "open",
      "right_eye": "open",
      "both_eyes_closed": false,
      "head_pose": "down"
    }
  ]
}
```

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Download models (first-time setup)
bash scripts/download_models.sh

# Single image
visionlink images/face.jpg

# Batch — process every image in a folder
visionlink images/ -o results/

# Live webcam (q to quit, s to snapshot)
visionlink --webcam
```

---

## CLI usage

```text
visionlink [input] [--webcam [ID]] [-o OUTPUT] [options]
```

| Argument / Flag | Description |
|-----------------|-------------|
| `input` | Image file or directory (batch). Omit when using `--webcam` |
| `--webcam [ID]` | Live USB camera (default ID `0`) |
| `-o`, `--output` | Output directory (default: `output/`) |
| `--no-landmarks` | Skip drawing landmark dots |
| `--no-json` | Skip writing JSON result files |
| `--no-image` | Skip saving annotated PNGs (image/batch mode) |
| `--recursive` | Include subfolders in batch mode |
| `-q` / `--quiet` | Suppress normal output |
| `-v` / `--verbose` | Debug logging |
| `--version` | Print version and exit |

### Webcam mode

```bash
visionlink --webcam          # camera 0
visionlink --webcam 1        # second camera
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save snapshot to `output/webcam_YYYYMMDD_HHMMSS.png` |

Live overlay shows FPS, face count, bounding boxes, landmarks, and gesture labels.

### Batch mode

Pass a directory instead of a file:

```bash
visionlink images/ -o results/
```

Each image is processed independently. Outputs per image:

```text
results/
├── face_annotated.png
├── face.json
├── test_annotated.png
├── test.json
└── batch_summary.json    # combined results for all images
```

---

## Architecture

Each step is a separate, replaceable module:

```text
CLI  →  Controller  →  Image Loader
                    →  Face Detector
                    →  Landmark Detector
                    →  Gesture Engine
                    →  Visualizer / JSON
```

```text
src/visionlink/
├── acquisition/    # Image loading, batch, webcam
├── controller.py   # Pipeline orchestrator
├── detection/      # Face detection (MediaPipe BlazeFace)
├── landmarks/      # Landmark extraction (478 points)
├── gestures/       # Gesture recognition (EAR, MAR, head pose)
├── output/         # Visualization + JSON export
├── cli.py          # Command-line interface
└── models.py       # Shared data types (Face, BoundingBox, …)
```

---

## Requirements

- Python 3.11+
- Linux, macOS, or Windows
- OpenCV + MediaPipe (see `requirements.txt`)

---

## Roadmap

| Phase | Status |
|-------|--------|
| Phase 1 — Single image analysis (M1–M6) | ✓ |
| Phase 2 — Batch + evaluation reports (M7) | ✓ |
| Phase 4 — Webcam / real-time (M8) | ✓ |
| Phase 3 — Video file processing | Planned |
| Phase 5 — Arduino serial communication (M9) | Planned |

---

## License

MIT — see [LICENSE](LICENSE).
