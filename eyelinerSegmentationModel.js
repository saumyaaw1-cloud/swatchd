const MODEL_URL = "/models/eyeliner-seg.onnx?v=20260806-v1";
const MODEL_SIZE = 640;
const MASK_SIZE = 160;
const MASK_CHANNELS = 32;
const CONFIDENCE_THRESHOLD = 0.28;
const MASK_THRESHOLD = 0.42;
const MAX_DETECTIONS = 4;

let eyelinerSessionPromise = null;
let ortModulePromise = null;
const inputCanvas = document.createElement("canvas");
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });

export async function loadEyelinerSegmentationModel() {
  if (eyelinerSessionPromise) return eyelinerSessionPromise;

  eyelinerSessionPromise = loadOnnxSession();
  return eyelinerSessionPromise;
}

export async function hasEyelinerSegmentationModel() {
  const response = await fetch(MODEL_URL, { method: "HEAD" });
  return response.ok;
}

export async function runEyelinerSegmentation(sourceCanvas, geometry, helpers = {}) {
  const session = await loadEyelinerSegmentationModel();

  if (!session) {
    return {
      available: false,
      eyelinerMask: null,
      penOrHandMask: null
    };
  }

  const frame = prepareFrame(sourceCanvas);
  const ort = await getOrt();
  const inputName = session.inputNames[0];
  const feeds = {
    [inputName]: new ort.Tensor("float32", frame.tensor, [1, 3, MODEL_SIZE, MODEL_SIZE])
  };
  const outputs = await session.run(feeds);
  const decoded = decodeSegmentation(outputs, frame, geometry, helpers);

  return {
    available: true,
    eyelinerMask: decoded,
    penOrHandMask: null
  };
}

async function loadOnnxSession() {
  const modelExists = await hasEyelinerSegmentationModel();
  if (!modelExists) return null;

  const ort = await getOrt();
  return ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"]
  });
}

async function getOrt() {
  if (!ortModulePromise) {
    ortModulePromise = import("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.mjs")
      .then((ort) => {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        return ort;
      });
  }

  return ortModulePromise;
}

function prepareFrame(sourceCanvas) {
  const sourceWidth = sourceCanvas.width;
  const sourceHeight = sourceCanvas.height;
  const scale = Math.min(MODEL_SIZE / sourceWidth, MODEL_SIZE / sourceHeight);
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);
  const padX = Math.floor((MODEL_SIZE - drawWidth) / 2);
  const padY = Math.floor((MODEL_SIZE - drawHeight) / 2);

  inputCanvas.width = MODEL_SIZE;
  inputCanvas.height = MODEL_SIZE;
  inputCtx.fillStyle = "rgb(114, 114, 114)";
  inputCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  inputCtx.drawImage(sourceCanvas, 0, 0, sourceWidth, sourceHeight, padX, padY, drawWidth, drawHeight);

  const image = inputCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const tensor = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  const planeSize = MODEL_SIZE * MODEL_SIZE;

  for (let i = 0; i < planeSize; i += 1) {
    const pixelIndex = i * 4;
    tensor[i] = image.data[pixelIndex] / 255;
    tensor[planeSize + i] = image.data[pixelIndex + 1] / 255;
    tensor[planeSize * 2 + i] = image.data[pixelIndex + 2] / 255;
  }

  return {
    tensor,
    sourceWidth,
    sourceHeight,
    scale,
    padX,
    padY
  };
}

function decodeSegmentation(outputs, frame, geometry, helpers) {
  const tensors = Object.values(outputs);
  const detectionTensor = tensors.find((tensor) => tensor.dims.includes(37));
  const maskTensor = tensors.find((tensor) => tensor.dims.includes(MASK_CHANNELS) && tensor.dims.includes(MASK_SIZE));
  if (!detectionTensor || !maskTensor) return null;

  const detections = extractDetections(detectionTensor)
    .filter((detection) => detection.score >= CONFIDENCE_THRESHOLD)
    .filter((detection) => isBoxNearEye(detection, frame, geometry, helpers))
    .sort((a, b) => b.score - a.score);

  const kept = nonMaxSuppress(detections).slice(0, MAX_DETECTIONS);
  if (!kept.length) return null;

  const pixels = [];
  kept.forEach((detection) => {
    pixels.push(...decodeMaskPixels(detection, maskTensor.data, frame, geometry, helpers));
  });

  const uniquePixels = dedupePixels(pixels);
  if (uniquePixels.length < 12) return null;

  const line = helpers.fitLineToPixels ? helpers.fitLineToPixels(uniquePixels) : null;
  if (!line || line.length < geometry.eyeWidth * 0.1) return null;

  const zones = helpers.analyzeLinerZones ? helpers.analyzeLinerZones({ pixels: uniquePixels }, geometry) : null;
  const tip = zones?.tip || pickTip(line, geometry, helpers);

  return {
    pixels: uniquePixels,
    line,
    tip,
    zones,
    confidence: clamp(kept[0].score * 0.72 + Math.min(uniquePixels.length / 120, 1) * 0.28, 0, 1),
    source: "model"
  };
}

function extractDetections(tensor) {
  const dims = tensor.dims;
  const data = tensor.data;
  const detections = [];
  const channelsFirst = dims[1] === 37;
  const count = channelsFirst ? dims[2] : dims[1];
  const channels = channelsFirst ? dims[1] : dims[2];

  for (let index = 0; index < count; index += 1) {
    const value = (channel) => channelsFirst
      ? data[channel * count + index]
      : data[index * channels + channel];
    const score = value(4);
    if (score < CONFIDENCE_THRESHOLD) continue;

    const cx = value(0);
    const cy = value(1);
    const width = value(2);
    const height = value(3);
    const coeffs = new Float32Array(MASK_CHANNELS);
    for (let c = 0; c < MASK_CHANNELS; c += 1) {
      coeffs[c] = value(5 + c);
    }

    detections.push({
      score,
      box: {
        x1: cx - width / 2,
        y1: cy - height / 2,
        x2: cx + width / 2,
        y2: cy + height / 2
      },
      coeffs
    });
  }

  return detections;
}

function decodeMaskPixels(detection, maskData, frame, geometry, helpers) {
  const pixels = [];
  const maskX1 = clamp(Math.floor(detection.box.x1 / 4), 0, MASK_SIZE - 1);
  const maskY1 = clamp(Math.floor(detection.box.y1 / 4), 0, MASK_SIZE - 1);
  const maskX2 = clamp(Math.ceil(detection.box.x2 / 4), maskX1 + 1, MASK_SIZE);
  const maskY2 = clamp(Math.ceil(detection.box.y2 / 4), maskY1 + 1, MASK_SIZE);

  for (let y = maskY1; y < maskY2; y += 1) {
    for (let x = maskX1; x < maskX2; x += 1) {
      let logit = 0;
      const maskIndex = y * MASK_SIZE + x;
      for (let c = 0; c < MASK_CHANNELS; c += 1) {
        logit += detection.coeffs[c] * maskData[c * MASK_SIZE * MASK_SIZE + maskIndex];
      }

      if (sigmoid(logit) < MASK_THRESHOLD) continue;

      const point = modelToCanvasPoint({ x: x * 4 + 2, y: y * 4 + 2 }, frame);
      if (!point) continue;
      if (helpers.isLinerSearchPixel && !helpers.isLinerSearchPixel(point, geometry)) continue;
      pixels.push(point);
    }
  }

  return pixels;
}

function isBoxNearEye(detection, frame, geometry, helpers) {
  const center = modelToCanvasPoint({
    x: (detection.box.x1 + detection.box.x2) / 2,
    y: (detection.box.y1 + detection.box.y2) / 2
  }, frame);

  if (!center) return false;
  if (helpers.isLinerSearchPixel && helpers.isLinerSearchPixel(center, geometry)) return true;

  return helpers.distanceToSegment
    ? helpers.distanceToSegment(center, geometry.outer, geometry.wingEnd) < geometry.eyeWidth * 0.55
    : true;
}

function modelToCanvasPoint(point, frame) {
  const x = (point.x - frame.padX) / frame.scale;
  const y = (point.y - frame.padY) / frame.scale;
  if (x < 0 || x > frame.sourceWidth || y < 0 || y > frame.sourceHeight) return null;
  return { x, y };
}

function nonMaxSuppress(detections) {
  const kept = [];

  detections.forEach((detection) => {
    if (kept.every((other) => boxIou(detection.box, other.box) < 0.45)) {
      kept.push(detection);
    }
  });

  return kept;
}

function boxIou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);

  return intersection / Math.max(areaA + areaB - intersection, 1);
}

function dedupePixels(pixels) {
  const seen = new Set();
  const unique = [];

  pixels.forEach((point) => {
    const key = `${Math.round(point.x / 2)}:${Math.round(point.y / 2)}`;
    if (seen.has(key)) return;
    seen.add(key);
    unique.push(point);
  });

  return unique;
}

function pickTip(line, geometry, helpers) {
  if (!helpers.projectProgress) return line.end;
  const startProgress = helpers.projectProgress(line.start, geometry.outer, geometry.wingEnd);
  const endProgress = helpers.projectProgress(line.end, geometry.outer, geometry.wingEnd);

  return endProgress >= startProgress ? line.end : line.start;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
