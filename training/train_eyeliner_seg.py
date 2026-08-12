#!/usr/bin/env python3
"""Train the first eyeliner segmentation model.

This script expects a YOLO segmentation dataset at datasets/eyeliner_seg.
It uses Ultralytics YOLO, so install training/requirements.txt first.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "eyeliner_seg" / "data.yaml"
DEFAULT_PROJECT = ROOT / "models" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train swatch'd eyeliner segmentation.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base YOLO segmentation model")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output directory")
    parser.add_argument("--name", default="eyeliner_seg_v0", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()

    if not data_path.exists():
      raise FileNotFoundError(f"Missing dataset config: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install with:\n"
            "  python3 -m pip install -r training/requirements.txt"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project.resolve()),
        name=args.name,
        task="segment",
        patience=20,
        plots=True,
    )


if __name__ == "__main__":
    main()
