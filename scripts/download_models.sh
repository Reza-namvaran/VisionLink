#!/usr/bin/env bash
# Download MediaPipe model files required by VisionLink.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT/models"
mkdir -p "$MODEL_DIR"

FACE_DETECTOR_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_LANDMARKER_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "$(basename "$dest") already exists"
        return
    fi
    echo "[Downloading] $(basename "$dest")..."
    wget -q -O "$dest" "$url"
    echo "Saved to $dest"
}

download "$FACE_DETECTOR_URL" "$MODEL_DIR/blaze_face_short_range.tflite"
download "$FACE_LANDMARKER_URL" "$MODEL_DIR/face_landmarker.task"

echo "All models ready in $MODEL_DIR"
