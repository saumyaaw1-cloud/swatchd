#!/usr/bin/env python3
"""Export a trained eyeliner segmentation model for browser inference."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "training_runs" / "eyeliner_seg_v1" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "models" / "eyeliner-seg.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export swatch'd eyeliner segmentation model.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Final ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    output = args.output.resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Missing trained weights: {weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing ultralytics. Install with:\n"
            "  python3 -m pip install -r training/requirements.txt"
        ) from exc

    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output)
    print(f"Exported browser model: {output}")


if __name__ == "__main__":
    main()
