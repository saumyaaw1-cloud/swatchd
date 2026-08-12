#!/usr/bin/env python3
"""Normalize intake images and create LabelMe-ready draft files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "intake" / "raw"
DEFAULT_OUTPUT = ROOT / "datasets" / "intake" / "labelme"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert intake images to normalized JPEGs with blank LabelMe JSON files."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_stem(source_root: Path, image_path: Path) -> str:
    if source_root.is_file():
        return image_path.stem
    relative = image_path.relative_to(source_root)
    parts = [*relative.parts[:-1], image_path.stem]
    return "__".join(part.replace(" ", "_") for part in parts)


def prepare_image(image_path: Path, output_path: Path, max_size: int) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, "JPEG", quality=92, optimize=True)
        return image.size


def write_labelme_json(json_path: Path, image_name: str, width: int, height: int) -> None:
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Missing source: {source}")
    if args.max_size < 256:
        raise ValueError("--max-size must be at least 256")

    images = iter_images(source)
    converted = 0
    skipped = 0

    for image_path in images:
        stem = output_stem(source, image_path)
        output_image = output / f"{stem}.jpg"
        output_json = output / f"{stem}.json"

        if not args.overwrite and output_image.exists() and output_json.exists():
            skipped += 1
            continue

        width, height = prepare_image(image_path, output_image, args.max_size)
        write_labelme_json(output_json, output_image.name, width, height)
        converted += 1

    print(json.dumps({
        "source": str(source),
        "output": str(output),
        "imagesFound": len(images),
        "converted": converted,
        "skipped": skipped,
    }, indent=2))


if __name__ == "__main__":
    main()
