# Eyeliner Segmentation Training

This folder is the training pipeline for the first `swatch'd` eyeliner vision model.

## Current Dataset

The converted Labelme dataset lives at:

```text
datasets/eyeliner_seg/data.yaml
```

It currently has one class:

```text
eyeliner
```

This is enough to prove the pipeline, but not enough for product-quality detection yet.

## Install

```bash
python3 -m pip install -r training/requirements.txt
```

## Train

```bash
python3 training/train_eyeliner_seg.py
```

Default output:

```text
models/training_runs/eyeliner_seg_v0/weights/best.pt
```

## Export To Browser Model

```bash
python3 training/export_eyeliner_seg.py
```

Default output:

```text
models/eyeliner-seg.onnx
```

The app can serve that file from:

```text
http://127.0.0.1:5174/models/eyeliner-seg.onnx
```

## Next Data Labels

The next dataset should add:

```text
pen_or_hand
eye_area
```

The app needs `pen_or_hand` to understand occlusion while the user is actively drawing.
