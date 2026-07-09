# VisionLink

Modular computer vision framework for facial gesture recognition from images.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Single image
visionlink images/your_photo.jpg

# Batch directory
visionlink images/ -o results/
```

## CLI options

| Flag | Description |
|------|-------------|
| `-o output/` | Output directory for PNG + JSON |
| `--no-landmarks` | Skip drawing landmark dots |
| `--no-json` | Skip JSON export |

