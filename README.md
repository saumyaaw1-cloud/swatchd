# Swatch'd

Swatch'd is a browser-based makeup coach that helps beginners practice winged eyeliner. It combines live face tracking, eye-aware AR guides, eyeliner segmentation, and step-by-step coaching that responds to the marks appearing on the user's face.

## Experience

- Detects and tracks facial landmarks in the browser
- Fits eyeliner guides to the selected eye
- Segments visible eyeliner with a custom ONNX model
- Breaks application into small, non-reversing coaching milestones
- Supports left-eye, right-eye, and automatic focus modes
- Opens with an interactive Three.js compact mirror

## Stack

- Vanilla JavaScript, HTML, and CSS
- MediaPipe Tasks Vision for facial landmarks
- ONNX Runtime Web for eyeliner segmentation
- Three.js for the interactive compact
- YOLO segmentation for model training and export

## Run Locally

Install the browser dependency and start the local server:

```bash
npm install
npm start
```

Open [http://127.0.0.1:5174](http://127.0.0.1:5174). Camera access requires localhost or HTTPS.

## Model Workflow

The browser-ready model is stored at `models/eyeliner-seg.onnx`. Training and export instructions are documented in [`training/README.md`](training/README.md).

The included model is an early prototype trained on a limited dataset. Product-quality performance will require broader coverage across skin tones, eye shapes, lighting conditions, eyeliner styles, hand occlusion, and camera quality.
