import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { runEyelinerSegmentation } from "./eyelinerSegmentationModel.js?v=20260806-v1";
import { initCompactScene } from "./compactScene.js?v=20260810-compact-15";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const analysisCanvas = document.createElement("canvas");
const analysisCtx = analysisCanvas.getContext("2d", { willReadFrequently: true });

const getStatusTarget = (id) => document.getElementById(id) || { textContent: "" };
const scanStatus = getStatusTarget("scanStatus");
const eyeStatus = getStatusTarget("eyeStatus");
const accuracyStatus = getStatusTarget("accuracyStatus");
const handStatus = getStatusTarget("handStatus");
const guideStatus = getStatusTarget("guideStatus");
const coachMessage = document.getElementById("coachMessage");
const stepLabel = document.getElementById("stepLabel");
const stepTitle = document.getElementById("stepTitle");
const coachTip = document.getElementById("coachTip");
const nextAction = document.getElementById("nextAction");
const startCameraButton = document.getElementById("startCameraButton");
const calibrateButton = document.getElementById("calibrateButton");
const resetButton = document.getElementById("resetButton");
const eyeFocusButtons = [...document.querySelectorAll(".eye-focus-button")];
const homeView = document.getElementById("homeView");
const coachView = document.getElementById("coachView");
const experienceIntro = document.getElementById("experienceIntro");
const enterExperienceButton = document.getElementById("enterExperienceButton");
const enterExperienceControls = [...document.querySelectorAll("[data-enter-experience]")];
const compactSceneTarget = document.getElementById("compactScene");
const compactHoverTarget = document.querySelector("[data-compact-hover]");
const customCursor = document.getElementById("customCursor");
const USE_TRAINED_LINER_MODEL = new URLSearchParams(window.location.search).get("vision") === "model"
  || window.localStorage.getItem("swatchUseVisionModel") === "true";
const MODEL_LINER_INTERVAL_MS = 1800;
const COACH_STEP_DURATION_MS = 10000;

const TIMED_LESSON_SEQUENCE = {
  placeTip: "outerCornerAnchor",
  limitedLidMap: "outerCornerAnchor",
  downturnedMap: "outerCornerAnchor",
  roundMap: "outerCornerAnchor",
  upturnedMap: "outerCornerAnchor",
  findOuterCorner: "outerCornerAnchor",
  outerCornerAnchor: "firstDot",
  firstDot: "tailDirection",
  firstMark: "tailDirection",
  stampCorner: "tailDirection",
  anchorHand: "tailDirection",
  dontStretch: "tailDirection",
  openEyeStamps: "tailLength",
  downturnedLift: "tailLength",
  roundElongate: "tailLength",
  upturnedBalance: "tailLength",
  tailDirection: "tailLength",
  softPressure: "tailLength",
  featherStrokes: "tailLength",
  buildSlowly: "steadyPull",
  tailLength: "steadyPull",
  rotateHand: "steadyPull",
  tooHigh: "steadyPull",
  tooLow: "steadyPull",
  steadyPull: "returnStart",
  extendTip: "returnStart",
  tooLong: "returnStart",
  keepLashThin: "returnStart",
  returnStart: "lashConnection",
  lashConnection: "connectTriangle",
  connectTriangle: "halfCloseEye",
  halfCloseEye: "fillTriangle",
  fillTriangle: "cleanWithAngle",
  cleanWithAngle: "finished",
  cleanEdge: "finished"
};

const EYES = {
  left: {
    label: "Left eye",
    inner: 133,
    outer: 33,
    upper: 159,
    lower: 145,
    browOuter: 70
  },
  right: {
    label: "Right eye",
    inner: 362,
    outer: 263,
    upper: 386,
    lower: 374,
    browOuter: 300
  }
};

const state = {
  faceLandmarker: null,
  handLandmarker: null,
  cameraStarted: false,
  activeEye: "left",
  focusEyeKey: "auto",
  lockedEyeKey: null,
  currentGeometry: null,
  pointer: null,
  inputMode: "pointer",
  handTip: null,
  penTip: null,
  penLine: null,
  penGripOffset: null,
  penLockedUntil: 0,
  detectedLiner: null,
  lastDetectedLiner: null,
  lastLinerSeenAt: 0,
  pendingLiner: null,
  pendingLinerFrames: 0,
  modelDetectedLiner: null,
  modelLinerSeenAt: 0,
  modelLinerPending: false,
  modelLinerLastRunAt: 0,
  modelLinerAvailable: false,
  modelLinerError: null,
  ignoreLinerUntil: 0,
  needsResetBaseline: false,
  baselineModel: null,
  linerTrackingArmed: false,
  candidateLessonKey: "placeTip",
  candidateLessonFrames: 0,
  handPinching: false,
  lastHandSeen: 0,
  cameraListenersAttached: false,
  isDrawing: false,
  stroke: [],
  strokes: [],
  currentLessonKey: "prep",
  pendingLessonKey: "prep",
  pendingLessonSince: 0,
  lessonChangedAt: 0,
  coachMetrics: null,
  coachMetricsFrames: 0,
  coachPhase: "start",
  missedFaceFrames: 0,
  lastFrameTime: -1
};

let appInitialized = false;
let compactScene = null;

function initHome() {
  compactScene = initCompactScene(compactSceneTarget);

  homeView?.addEventListener("pointermove", (event) => {
    const x = event.clientX / window.innerWidth - 0.5;
    const y = event.clientY / window.innerHeight - 0.5;
    homeView.style.setProperty("--pointer-x", x.toFixed(3));
    homeView.style.setProperty("--pointer-y", y.toFixed(3));
    compactScene?.setPointer(x, y);
  });

  homeView?.addEventListener("pointerleave", () => {
    homeView.style.setProperty("--pointer-x", "0");
    homeView.style.setProperty("--pointer-y", "0");
    compactScene?.setPointer(0, 0);
  });

  const previewDoor = (isOpening) => {
    if (homeView.classList.contains("is-opening")) return;
    compactScene?.setOpen(isOpening);
  };

  compactHoverTarget?.addEventListener("mouseenter", () => previewDoor(true));
  compactHoverTarget?.addEventListener("mouseleave", () => previewDoor(false));
  enterExperienceButton?.addEventListener("focus", () => previewDoor(true));
  enterExperienceButton?.addEventListener("blur", () => previewDoor(false));

  enterExperienceControls.forEach((control) => {
    control.addEventListener("click", enterExperience);
  });
}

function initCustomCursor() {
  const canUseCustomCursor = window.matchMedia("(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)");
  if (!customCursor || !canUseCustomCursor.matches) return;

  const spark = customCursor.querySelector(".cursor-spark");
  let targetX = -40;
  let targetY = -40;
  let sparkX = -40;
  let sparkY = -40;

  document.body.classList.add("has-custom-cursor");

  const renderCursor = () => {
    sparkX += (targetX - sparkX) * 0.5;
    sparkY += (targetY - sparkY) * 0.5;
    spark.style.transform = `translate3d(${sparkX}px, ${sparkY}px, 0) translate(-50%, -50%)`;
    window.requestAnimationFrame(renderCursor);
  };

  window.addEventListener("pointermove", (event) => {
    targetX = event.clientX;
    targetY = event.clientY;
    customCursor.classList.add("is-visible");
  });

  document.addEventListener("pointerover", (event) => {
    const isInteractive = Boolean(event.target.closest("a, button, input, select, textarea, [role='button']"));
    customCursor.classList.toggle("is-interactive", isInteractive);
  });

  document.addEventListener("pointerdown", () => customCursor.classList.add("is-pressed"));
  document.addEventListener("pointerup", () => customCursor.classList.remove("is-pressed"));
  window.addEventListener("blur", () => customCursor.classList.remove("is-visible"));
  document.documentElement.addEventListener("mouseleave", () => customCursor.classList.remove("is-visible"));

  renderCursor();
}

async function enterExperience() {
  if (homeView.classList.contains("is-opening")) return;

  homeView.classList.add("is-opening");
  enterExperienceControls.forEach((control) => {
    control.disabled = true;
  });
  compactScene?.setOpen(true);

  await new Promise((resolve) => window.setTimeout(resolve, 700));
  experienceIntro.hidden = false;
  window.requestAnimationFrame(() => experienceIntro.classList.add("is-active"));

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  await new Promise((resolve) => window.setTimeout(resolve, prefersReducedMotion ? 180 : 11800));

  coachView.hidden = false;
  homeView.hidden = true;
  document.body.classList.add("coach-active");
  experienceIntro.classList.add("is-leaving");

  await new Promise((resolve) => window.setTimeout(resolve, prefersReducedMotion ? 20 : 460));
  experienceIntro.hidden = true;
  experienceIntro.classList.remove("is-active", "is-leaving");

  if (!appInitialized) {
    appInitialized = true;
    await initCoach();
  }
}

async function initCoach() {
  startCameraButton.addEventListener("click", startCamera);
  calibrateButton.addEventListener("click", captureBaselineFrame);
  resetButton.addEventListener("click", resetStroke);
  eyeFocusButtons.forEach((button) => {
    button.addEventListener("click", () => setEyeFocus(button.dataset.eyeFocus));
  });
  updateEyeFocusButtons();

  try {
    updateHud("Starting camera", "Find face", "--", getLesson("prep"));
    startCameraButton.disabled = true;

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
      },
      runningMode: "VIDEO",
      numFaces: 1
    });

    try {
      state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        },
        runningMode: "VIDEO",
        numHands: 1
      });
      handStatus.textContent = "No liner detected";
    } catch (handError) {
      console.warn("Hand tracking unavailable", handError);
      handStatus.textContent = "Unavailable";
    }

    startCameraButton.disabled = false;
    await startCamera();
  } catch (error) {
    console.error(error);
    startCameraButton.disabled = false;
    updateHud("Model error", "Unavailable", "--", getLesson("cameraBlocked"));
  }
}

async function startCamera() {
  try {
    startCameraButton.disabled = true;
    updateHud("Starting camera", "Find face", "--", getLesson("prep"));
    guideStatus.textContent = "Waiting";

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });

    if (video.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop());
    }

    video.srcObject = stream;
    state.cameraStarted = true;
    startCameraButton.textContent = "Camera active";
    guideStatus.textContent = "Scanning";

    video.addEventListener("loadedmetadata", () => {
      resizeCanvasToVideo();
      requestAnimationFrame(loop);
    }, { once: true });

    if (!state.cameraListenersAttached) {
      window.addEventListener("resize", resizeCanvasToVideo);
      state.cameraListenersAttached = true;
    }
  } catch (error) {
    console.error(error);
    startCameraButton.disabled = false;
    startCameraButton.textContent = "Start camera";
    state.cameraStarted = false;
    guideStatus.textContent = "Blocked";
    updateHud("Camera blocked", "Unavailable", "--", getCameraErrorLesson(error));
  }
}

function loop() {
  try {
    if (!state.faceLandmarker || video.readyState < HTMLMediaElement.HAVE_METADATA) {
      requestAnimationFrame(loop);
      return;
    }

    const now = performance.now();
    const faceResults = state.faceLandmarker.detectForVideo(video, now);
    const handResults = state.handLandmarker
      ? state.handLandmarker.detectForVideo(video, now)
      : { landmarks: [] };

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!faceResults.faceLandmarks.length) {
      processHandTracking(handResults, now, null);
      state.currentGeometry = null;
      state.missedFaceFrames += 1;
      guideStatus.textContent = "Finding face";

      if (state.missedFaceFrames < 12) {
        requestAnimationFrame(loop);
        return;
      }

      state.currentLessonKey = "findFace";
      state.currentGeometry = null;
      state.pendingLessonKey = "findFace";
      state.pendingLessonSince = performance.now();
      updateHud("Searching", "Find face", "--", getLesson("findFace"));
      requestAnimationFrame(loop);
      return;
    }

    state.missedFaceFrames = 0;

    const landmarks = faceResults.faceLandmarks[0];
    const eye = chooseActiveEye(landmarks);
    const geometry = getEyeGeometry(landmarks, eye);
    state.currentGeometry = geometry;
    processHandTracking(handResults, now, geometry);

    if (state.needsResetBaseline && captureBaselineIfReady()) {
      state.needsResetBaseline = false;
      armLinerTracking(1100);
      handStatus.textContent = "Reset from current eye";
    }

    const toolOccludingEye = isToolOccludingEye(geometry, now);
    prepareAnalysisFrame();
    requestModelLinerDetection(geometry, now);
    const freshLiner = now >= state.ignoreLinerUntil ? getConfirmedLiner(detectActualLiner(geometry), geometry) : null;
    if (freshLiner) {
      state.detectedLiner = freshLiner;
      state.lastDetectedLiner = freshLiner;
      state.lastLinerSeenAt = now;
    } else if (now >= state.ignoreLinerUntil && shouldHoldLastLiner(now, toolOccludingEye)) {
      state.detectedLiner = state.lastDetectedLiner;
    } else {
      state.detectedLiner = null;
    }

    const guidance = evaluateStroke(geometry);
    handStatus.textContent = freshLiner
      ? `Line observed (${freshLiner.pixels.length}, ${Math.round(guidance.progress * 100)}%, r${Math.round((guidance.returnProgress || 0) * 100)}, f${Math.round((guidance.fillProgress || 0) * 100)}, ${state.coachPhase}, ${guidance.lessonKey})`
      : state.detectedLiner ? `Line observed (${Math.round(guidance.progress * 100)}%, ${state.coachPhase}, ${guidance.lessonKey})` : getNoLinerStatus(toolOccludingEye);

    updateHud("Locked", eye.label, guidance.accuracy, guidance.lesson);
    guideStatus.textContent = "Visible";

    drawFaceRulers(landmarks, geometry);
    drawTargetWing(geometry, guidance);
  } catch (error) {
    console.error(error);
    guideStatus.textContent = "Overlay error";
    updateHud("Scan paused", state.activeEye ? EYES[state.activeEye].label : "Find face", "--", getLesson("findFace"));
  }

  requestAnimationFrame(loop);
}

function chooseActiveEye(landmarks) {
  if (state.focusEyeKey !== "auto") {
    state.activeEye = state.focusEyeKey;
    return EYES[state.focusEyeKey];
  }

  if (state.lockedEyeKey) {
    state.activeEye = state.lockedEyeKey;
    return EYES[state.lockedEyeKey];
  }

  const candidates = Object.entries(EYES).map(([key, eye]) => {
    const geometry = getEyeGeometry(landmarks, eye);
    const strokeTip = getStrokeTip();
    const target = strokeTip || state.pointer || centerPoint(geometry.inner, geometry.outer);
    return {
      key,
      eye,
      distance: distance(target, geometry.outer)
    };
  });

  candidates.sort((a, b) => a.distance - b.distance);
  state.activeEye = candidates[0].key;
  return candidates[0].eye;
}

function isToolOccludingEye(geometry, now = performance.now()) {
  if (!geometry || now - state.lastHandSeen > 850) return false;

  const activeTip = state.penTip || state.handTip;
  if (!activeTip) return false;

  const eyeCenter = centerPoint(geometry.inner, geometry.outer);
  const nearOuterCorner = distance(activeTip, geometry.outer) < geometry.eyeWidth * 1.05;
  const nearEye = distance(activeTip, eyeCenter) < geometry.eyeWidth * 1.25;
  const onWingPath = distanceToSegment(activeTip, geometry.outer, geometry.wingEnd) < geometry.eyeWidth * 0.7;

  return nearOuterCorner || nearEye || onWingPath || isTipInEyelinerZone(activeTip, geometry);
}

function shouldHoldLastLiner(now, toolOccludingEye) {
  if (!state.lastDetectedLiner) return false;

  const normalHold = now - state.lastLinerSeenAt < 1000;
  const occludedHold = toolOccludingEye && now - state.lastLinerSeenAt < 3000;

  return normalHold || occludedHold;
}

function getNoLinerStatus(toolOccludingEye) {
  if (!state.baselineModel) return "Calibrate first";
  if (toolOccludingEye) return "Pen blocking view";
  return "No liner detected";
}

function setEyeFocus(key) {
  if (!["auto", "left", "right"].includes(key)) return;

  state.focusEyeKey = key;
  state.lockedEyeKey = null;
  state.activeEye = key === "auto" ? state.activeEye : key;
  resetFocusSensitiveTracking();
  updateEyeFocusButtons();
}

function resetFocusSensitiveTracking() {
  state.pointer = null;
  state.stroke = [];
  state.strokes = [];
  state.handTip = null;
  state.penTip = null;
  state.penLine = null;
  state.detectedLiner = null;
  state.lastDetectedLiner = null;
  state.pendingLiner = null;
  state.pendingLinerFrames = 0;
  state.lastLinerSeenAt = 0;
  state.coachMetrics = null;
  state.coachMetricsFrames = 0;
  state.candidateLessonKey = "placeTip";
  state.candidateLessonFrames = 0;

  if (state.cameraStarted) {
    state.needsResetBaseline = true;
    clearLinerTracking(900);
    handStatus.textContent = "Focus switched";
  } else {
    clearLinerTracking(700);
  }

  setImmediateLesson("placeTip");
}

function updateEyeFocusButtons() {
  eyeFocusButtons.forEach((button) => {
    const isActive = button.dataset.eyeFocus === state.focusEyeKey;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}

function getEyeGeometry(landmarks, eye) {
  const inner = toXY(landmarks[eye.inner]);
  const outer = toXY(landmarks[eye.outer]);
  const upper = toXY(landmarks[eye.upper]);
  const lower = toXY(landmarks[eye.lower]);
  const browOuter = toXY(landmarks[eye.browOuter]);
  const eyeWidth = distance(inner, outer);
  const eyeHeight = distance(upper, lower);
  const lashAngle = Math.atan2(outer.y - inner.y, outer.x - inner.x);
  const side = outer.x < inner.x ? -1 : 1;
  const eyeProfile = getEyeProfile(inner, outer, upper, lower, eyeWidth, eyeHeight);
  const overlayProfile = getOverlayProfile(eyeProfile);
  const outwardAngle = lashAngle;
  const classicLiftAngle = outwardAngle - side * overlayProfile.lift;
  const browLiftAngle = Math.atan2(browOuter.y - outer.y, browOuter.x - outer.x);
  const lowerLiftAngle = Math.atan2(outer.y - lower.y, outer.x - lower.x) - side * 0.04;
  const wingAngle = blendAngles(blendAngles(classicLiftAngle, lowerLiftAngle, 0.14), browLiftAngle, 0.03);
  const wingLength = clamp(eyeWidth * overlayProfile.length, 22, 62);
  const wingEnd = {
    x: outer.x + Math.cos(wingAngle) * wingLength,
    y: outer.y + Math.sin(wingAngle) * wingLength
  };
  const outerWingBase = {
    x: outer.x + Math.cos(lashAngle) * eyeWidth * overlayProfile.baseOut,
    y: outer.y + Math.sin(lashAngle) * eyeWidth * overlayProfile.baseOut
  };
  const returnJoin = {
    x: outer.x + Math.cos(lashAngle + Math.PI) * eyeWidth * overlayProfile.returnJoin,
    y: outer.y + Math.sin(lashAngle + Math.PI) * eyeWidth * overlayProfile.returnJoin
  };
  const returnBase = {
    x: outer.x + Math.cos(lashAngle + Math.PI) * eyeWidth * overlayProfile.returnIn,
    y: outer.y + Math.sin(lashAngle + Math.PI) * eyeWidth * overlayProfile.returnIn
  };
  const rulerEnd = {
    x: outer.x + Math.cos(lashAngle) * side * eyeWidth * 1.05,
    y: outer.y + Math.sin(lashAngle) * side * eyeWidth * 1.05
  };

  return {
    inner,
    outer,
    upper,
    lower,
    browOuter,
    eyeWidth,
    eyeHeight,
    eyeProfile,
    overlayProfile,
    lashAngle,
    side,
    wingAngle,
    wingLength,
    wingEnd,
    outerWingBase,
    returnJoin,
    returnBase,
    rulerEnd
  };
}

function evaluateStroke(geometry) {
  const liner = state.detectedLiner;
  const input = liner ? liner.tip : null;

  if (!input) {
    const lessonKey = getStartLessonKey(geometry);
    state.coachMetrics = null;
    state.coachMetricsFrames = 0;
    return {
      accuracy: "--",
      lessonKey,
      lesson: getStableLesson(lessonKey),
      lineDistance: 0,
      tipDistance: 0,
      progress: 0
    };
  }

  const zones = liner.zones || analyzeLinerZones(liner, geometry);
  const zoneTip = zones.tip || input;
  const lineDistance = signedDistanceToLine(zoneTip, geometry.outer, geometry.wingEnd);
  const tipDistance = distance(zoneTip, geometry.wingEnd);
  const progress = zones.tailProgress;
  const returnProgress = zones.returnProgress;
  const fillProgress = zones.fillProgress;
  const fillPixelRatio = zones.fillPixelRatio;
  const angleError = liner ? getLineAngleError(liner.line, geometry) : 0;
  const metrics = getSmoothedCoachMetrics({
    lineDistance,
    tipDistance,
    progress,
    returnProgress,
    fillProgress,
    fillPixelRatio,
    angleError,
    markLength: liner.line.length
  });
  const score = getAccuracyScore(metrics.lineDistance, metrics.tipDistance, metrics.progress, metrics.angleError, geometry.eyeWidth);
  const startDistance = distance(zoneTip, geometry.outer);
  const lessonKey = getMilestoneLesson(metrics, geometry, score, startDistance);

  return {
    accuracy: `${score}%`,
    lessonKey,
    lesson: getStableLesson(lessonKey),
    lineDistance: metrics.lineDistance,
    tipDistance: metrics.tipDistance,
    progress: metrics.progress,
    returnProgress: metrics.returnProgress,
    fillProgress: Math.max(metrics.fillProgress, metrics.fillPixelRatio)
  };
}

function getMilestoneLesson(metrics, geometry, score, startDistance) {
  const markRatio = metrics.markLength / Math.max(geometry.eyeWidth, 1);
  const offGuide = Math.abs(metrics.lineDistance) > geometry.eyeWidth * 0.2;

  if (startDistance > geometry.eyeWidth * 0.68 && metrics.progress < 0.18) {
    return "findOuterCorner";
  }

  if (metrics.progress > 1.15) return "tooLong";

  if (markRatio < 0.14 || metrics.progress < 0.12) {
    if (Math.abs(metrics.angleError) > 30) return "anchorHand";
    return "outerCornerAnchor";
  }

  if (metrics.progress < 0.28 || markRatio < 0.24) {
    if (Math.abs(metrics.angleError) > 28) return "anchorHand";
    return "firstDot";
  }

  if (metrics.progress < 0.46) {
    if (offGuide) return "softPressure";
    return getTailDirectionLesson(geometry);
  }

  if (metrics.progress < 0.68) {
    if (offGuide) return "softPressure";
    if (Math.abs(metrics.angleError) > 24) return "rotateHand";
    return "tailLength";
  }

  if (metrics.progress < 0.86) {
    if (metrics.lineDistance * geometry.side > 12) return "tooHigh";
    if (metrics.lineDistance * geometry.side < -12) return "tooLow";
    return score > 66 ? "steadyPull" : "rotateHand";
  }

  if (metrics.returnProgress < 0.12) {
    if (metrics.tipDistance > geometry.eyeWidth * 0.28) return "extendTip";
    return "returnStart";
  }

  if (metrics.returnProgress < 0.36) {
    return "lashConnection";
  }

  if (metrics.returnProgress < 0.65) {
    return "connectTriangle";
  }

  const fillEvidence = Math.max(metrics.fillProgress, metrics.fillPixelRatio || 0);
  if (fillEvidence < 0.14) return "halfCloseEye";
  if (fillEvidence < 0.78) return "fillTriangle";
  return "cleanWithAngle";
}

function getTailDirectionLesson(geometry) {
  if (geometry.eyeProfile.limitedLid) return "openEyeStamps";
  if (geometry.eyeProfile.downturned) return "downturnedLift";
  if (geometry.eyeProfile.round) return "roundElongate";
  if (geometry.eyeProfile.upturned) return "upturnedBalance";
  return "tailDirection";
}

function getSmoothedCoachMetrics(nextMetrics) {
  const previous = state.coachMetrics;
  const frameCount = state.coachMetricsFrames + 1;
  state.coachMetricsFrames = frameCount;
  const progressKeys = ["progress", "returnProgress", "fillProgress", "fillPixelRatio"];

  if (!previous) {
    const conservative = {};

    Object.entries(nextMetrics).forEach(([key, value]) => {
      conservative[key] = progressKeys.includes(key) ? value * 0.2 : value;
    });

    state.coachMetrics = conservative;
    return conservative;
  }

  if (frameCount < 6) {
    const warmup = {};

    Object.entries(nextMetrics).forEach(([key, value]) => {
      const previousValue = Number.isFinite(previous[key]) ? previous[key] : value;
      warmup[key] = progressKeys.includes(key)
        ? Math.min(previousValue + 0.04, value)
        : previousValue + (value - previousValue) * 0.28;
    });

    state.coachMetrics = warmup;
    return warmup;
  }

  const smoothed = {};

  Object.entries(nextMetrics).forEach(([key, value]) => {
    const previousValue = Number.isFinite(previous[key]) ? previous[key] : value;
    smoothed[key] = previousValue + (value - previousValue) * 0.28;
  });

  state.coachMetrics = smoothed;
  return smoothed;
}

function getEyeProfile(inner, outer, upper, lower, eyeWidth, eyeHeight) {
  const openness = eyeHeight / Math.max(eyeWidth, 1);
  const outerDrop = (outer.y - inner.y) / Math.max(eyeWidth, 1);

  return {
    openness,
    outerDrop,
    limitedLid: openness < 0.22,
    round: openness > 0.34,
    downturned: outerDrop > 0.06,
    upturned: outerDrop < -0.06
  };
}

function getOverlayProfile(profile) {
  if (profile.limitedLid) {
    return {
      baseOut: 0.1,
      returnIn: 0.02,
      returnJoin: 0.2,
      lift: 0.18,
      length: 0.4,
      topWidth: 0.07,
      lowerWidth: 0.12,
      curve: 0.045
    };
  }

  if (profile.downturned) {
    return {
      baseOut: 0.1,
      returnIn: 0.03,
      returnJoin: 0.24,
      lift: 0.28,
      length: 0.46,
      topWidth: 0.078,
      lowerWidth: 0.14,
      curve: 0.06
    };
  }

  if (profile.round) {
    return {
      baseOut: 0.12,
      returnIn: 0.02,
      returnJoin: 0.26,
      lift: 0.2,
      length: 0.54,
      topWidth: 0.076,
      lowerWidth: 0.145,
      curve: 0.045
    };
  }

  if (profile.upturned) {
    return {
      baseOut: 0.08,
      returnIn: 0.025,
      returnJoin: 0.22,
      lift: 0.16,
      length: 0.42,
      topWidth: 0.065,
      lowerWidth: 0.115,
      curve: 0.035
    };
  }

  return {
    baseOut: 0.1,
    returnIn: 0.025,
    returnJoin: 0.24,
    lift: 0.22,
    length: 0.46,
    topWidth: 0.075,
    lowerWidth: 0.135,
    curve: 0.045
  };
}

function getStartLessonKey(geometry) {
  if (!state.linerTrackingArmed) return "placeTip";
  if (geometry.eyeProfile.limitedLid) return "limitedLidMap";
  if (geometry.eyeProfile.downturned) return "downturnedMap";
  if (geometry.eyeProfile.round) return "roundMap";
  if (geometry.eyeProfile.upturned) return "upturnedMap";
  return "placeTip";
}

function drawFaceRulers(landmarks, geometry) {
  ctx.save();
  drawEyeOutline(landmarks, state.activeEye);
  drawLashRootGuide(geometry);
  ctx.restore();
}

function drawEyeOutline(landmarks, activeKey) {
  const eyeIndexes = activeKey === "left"
    ? [33, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
    : [263, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263];
  const points = eyeIndexes.map((index) => toXY(landmarks[index]));

  ctx.beginPath();
  drawSmoothClosedCurve(points, 0.42);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(18, 14, 20, 0.42)";
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.98)";
  ctx.lineWidth = 1.15;
  ctx.stroke();
}

function drawLashRootGuide(geometry) {
  ctx.save();
  ctx.lineCap = "round";
  drawDashedMapLine(geometry.outer, geometry.outerWingBase, "rgba(255, 255, 255, 0.82)", 1.35, [2, 5]);
  drawMapPoint(geometry.outer, 2.5, false);
  ctx.restore();
}

function drawTargetWing(geometry, guidance) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const guideColor = getGuideColor(guidance);
  const activeReturn = guidance.lessonKey === "connectTriangle" || guidance.lessonKey === "fillTriangle";
  const activeTail = !activeReturn;

  const returnGuideStart = getWingEdgePoint(geometry, geometry.returnBase, -geometry.overlayProfile.lowerWidth);
  const topGuideStart = getWingEdgePoint(geometry, geometry.outerWingBase, geometry.overlayProfile.topWidth);
  const topControl = getWingCurveControl(geometry, topGuideStart, geometry.wingEnd, geometry.overlayProfile.curve * 0.65);
  const lowerControl = getWingCurveControl(geometry, returnGuideStart, geometry.wingEnd, -geometry.overlayProfile.curve * 0.35);

  drawDashedMapCurve(
    topGuideStart,
    topControl,
    geometry.wingEnd,
    activeTail ? guideColor : "rgba(255, 255, 255, 0.66)",
    activeTail ? 1.55 : 1.15,
    [2.2, 5.2]
  );

  drawDashedMapCurve(
    geometry.wingEnd,
    lowerControl,
    returnGuideStart,
    activeReturn ? guideColor : "rgba(255, 255, 255, 0.66)",
    activeReturn ? 1.55 : 1.15,
    [2.2, 5.2]
  );

  drawMapPoint(geometry.wingEnd, 2.6, true);
  if (activeReturn) drawMapPoint(returnGuideStart, 2.3, true);
  ctx.restore();
}

function getWingEdgePoint(geometry, point, offset) {
  const normal = {
    x: -Math.sin(geometry.wingAngle),
    y: Math.cos(geometry.wingAngle)
  };

  return {
    x: point.x + normal.x * geometry.eyeWidth * offset,
    y: point.y + normal.y * geometry.eyeWidth * offset
  };
}

function getWingCurveControl(geometry, start, end, curve) {
  const mid = lerpPoint(start, end, 0.52);
  const normal = {
    x: -Math.sin(geometry.wingAngle),
    y: Math.cos(geometry.wingAngle)
  };

  return {
    x: mid.x + normal.x * geometry.eyeWidth * curve,
    y: mid.y + normal.y * geometry.eyeWidth * curve
  };
}

function drawDashedMapLine(start, end, color, width, dash) {
  ctx.save();
  ctx.setLineDash(dash);
  ctx.lineDashOffset = 0;
  ctx.lineCap = "round";
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  drawLine(start, end);
  ctx.setLineDash([]);
  ctx.restore();
}

function drawDashedMapCurve(start, control, end, color, width, dash) {
  ctx.save();
  ctx.setLineDash(dash);
  ctx.lineDashOffset = 0;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.quadraticCurveTo(control.x, control.y, end.x, end.y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function drawMapPoint(point, radius, active) {
  ctx.save();
  ctx.fillStyle = active ? "#ff4f9a" : "rgba(255, 255, 255, 0.96)";
  ctx.strokeStyle = active ? "rgba(255, 255, 255, 0.98)" : "rgba(18, 14, 20, 0.46)";
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawStroke() {
  const strokes = [...state.strokes];
  if (state.stroke.length) strokes.push(state.stroke);
  if (!strokes.length) return;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(12, 12, 14, 0.92)";
  ctx.lineWidth = 7;

  strokes.forEach((stroke) => {
    if (!stroke.length) return;

    ctx.beginPath();
    stroke.forEach((point, index) => {
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.stroke();
  });

  const tip = getStrokeTip();
  if (tip) {
    ctx.fillStyle = "#f04468";
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawPointer(guidance) {
  if (!state.pointer) return;

  ctx.save();
  ctx.fillStyle = getGuideColor(guidance);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.5)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(state.pointer.x, state.pointer.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawDetectedLiner() {
  if (!state.detectedLiner) return;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineWidth = 4;
  ctx.strokeStyle = "rgba(240, 90, 118, 0.92)";
  drawLine(state.detectedLiner.line.start, state.detectedLiner.line.end);

  ctx.fillStyle = "rgba(240, 90, 118, 0.95)";
  ctx.beginPath();
  ctx.arc(state.detectedLiner.tip.x, state.detectedLiner.tip.y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function captureBaselineFrame() {
  const geometry = state.currentGeometry;

  if (!video.videoWidth || !video.videoHeight || !canvas.width || !canvas.height || !geometry) {
    handStatus.textContent = "Calibrate after face locks";
    return;
  }

  if (analysisCanvas.width !== canvas.width || analysisCanvas.height !== canvas.height) {
    analysisCanvas.width = canvas.width;
    analysisCanvas.height = canvas.height;
  }

  analysisCtx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);
  state.baselineModel = buildFaceAlignedBaseline(geometry);
  armLinerTracking(900);
  handStatus.textContent = "Calibrated to face";
  setImmediateLesson("placeTip");
}

function captureBaselineIfReady() {
  const geometry = state.currentGeometry;

  if (!video.videoWidth || !video.videoHeight || !canvas.width || !canvas.height || !geometry) {
    return false;
  }

  if (analysisCanvas.width !== canvas.width || analysisCanvas.height !== canvas.height) {
    analysisCanvas.width = canvas.width;
    analysisCanvas.height = canvas.height;
  }

  analysisCtx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);
  state.baselineModel = buildFaceAlignedBaseline(geometry);
  return true;
}

function clearLinerTracking(cooldown = 700) {
  state.detectedLiner = null;
  state.lastDetectedLiner = null;
  state.pendingLiner = null;
  state.pendingLinerFrames = 0;
  state.lastLinerSeenAt = 0;
  state.ignoreLinerUntil = performance.now() + cooldown;
  state.candidateLessonKey = "placeTip";
  state.candidateLessonFrames = 0;
  state.coachMetrics = null;
  state.coachMetricsFrames = 0;
  state.coachPhase = "start";
}

function armLinerTracking(cooldown = 700) {
  clearLinerTracking(cooldown);
  state.linerTrackingArmed = true;
}

function buildFaceAlignedBaseline(geometry) {
  const columns = 72;
  const rows = 25;
  const uMin = -0.22;
  const uMax = 1.26;
  const vMin = -0.46;
  const vMax = 0.46;
  const samples = new Float32Array(columns * rows);

  for (let row = 0; row < rows; row += 1) {
    const v = vMin + (vMax - vMin) * (row / (rows - 1));

    for (let column = 0; column < columns; column += 1) {
      const u = uMin + (uMax - uMin) * (column / (columns - 1));
      const point = pointFromWingSpace(geometry, u, v);
      samples[row * columns + column] = sampleVideoLuminance(point);
    }
  }

  return {
    columns,
    rows,
    uMin,
    uMax,
    vMin,
    vMax,
    samples
  };
}

function pointFromWingSpace(geometry, u, v) {
  const axis = {
    x: Math.cos(geometry.wingAngle),
    y: Math.sin(geometry.wingAngle)
  };
  const normal = {
    x: -axis.y,
    y: axis.x
  };

  return {
    x: geometry.outer.x + axis.x * geometry.wingLength * u + normal.x * geometry.eyeWidth * v,
    y: geometry.outer.y + axis.y * geometry.wingLength * u + normal.y * geometry.eyeWidth * v
  };
}

function getWingSpace(point, geometry) {
  const axis = {
    x: Math.cos(geometry.wingAngle),
    y: Math.sin(geometry.wingAngle)
  };
  const normal = {
    x: -axis.y,
    y: axis.x
  };
  const dx = point.x - geometry.outer.x;
  const dy = point.y - geometry.outer.y;

  return {
    u: (dx * axis.x + dy * axis.y) / Math.max(geometry.wingLength, 1),
    v: (dx * normal.x + dy * normal.y) / Math.max(geometry.eyeWidth, 1)
  };
}

function sampleVideoLuminance(point) {
  const x = clamp(Math.round(point.x), 0, analysisCanvas.width - 1);
  const y = clamp(Math.round(point.y), 0, analysisCanvas.height - 1);
  const pixel = analysisCtx.getImageData(x, y, 1, 1).data;

  return pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114;
}

function detectActualLiner(geometry) {
  if (!video.videoWidth || !video.videoHeight) return null;
  if (!state.linerTrackingArmed) return null;

  const modelLiner = getRecentModelLiner();
  if (modelLiner) return modelLiner;

  if (!state.baselineModel) return null;

  prepareAnalysisFrame();

  const bounds = getExpandedBounds([
    geometry.inner,
    geometry.outer,
    geometry.upper,
    geometry.lower,
    geometry.wingEnd
  ], geometry.eyeWidth * 0.82);
  const image = analysisCtx.getImageData(bounds.x, bounds.y, bounds.width, bounds.height);
  const pixels = collectLinerPixels(image, bounds, geometry);

  if (pixels.length < 18) return null;

  const line = fitLineToPixels(pixels);
  if (!line || line.length < geometry.eyeWidth * 0.16) return null;

  const zones = analyzeLinerZones({ pixels }, geometry);
  const strongLine = line.ratio >= 2.15 && isPlausibleLinerLine(line, geometry);
  const strongShape = zones.tailPixelCount >= 10 || zones.returnPixelCount >= 16 || zones.fillPixelCount >= 8;
  if (!strongLine && !strongShape) return null;

  const startProgress = projectProgress(line.start, geometry.outer, geometry.wingEnd);
  const endProgress = projectProgress(line.end, geometry.outer, geometry.wingEnd);
  const tip = zones.tip || (endProgress >= startProgress ? line.end : line.start);

  return {
    tip,
    line,
    pixels,
    zones,
    confidence: getLinerConfidence(pixels, line, geometry, zones)
  };
}

function prepareAnalysisFrame() {
  if (analysisCanvas.width !== canvas.width || analysisCanvas.height !== canvas.height) {
    analysisCanvas.width = canvas.width;
    analysisCanvas.height = canvas.height;
  }

  analysisCtx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);
}

function requestModelLinerDetection(geometry, now) {
  if (!USE_TRAINED_LINER_MODEL) return;
  if (!geometry || !state.linerTrackingArmed || state.modelLinerPending) return;
  if (now - state.modelLinerLastRunAt < MODEL_LINER_INTERVAL_MS) return;
  if (!analysisCanvas.width || !analysisCanvas.height) return;

  state.modelLinerPending = true;
  state.modelLinerLastRunAt = now;

  runEyelinerSegmentation(analysisCanvas, geometry, {
    fitLineToPixels,
    analyzeLinerZones,
    isLinerSearchPixel,
    distanceToSegment,
    projectProgress
  }).then((result) => {
    state.modelLinerPending = false;
    state.modelLinerAvailable = Boolean(result?.available);
    state.modelLinerError = null;

    if (result?.eyelinerMask?.confidence >= 0.3) {
      state.modelDetectedLiner = result.eyelinerMask;
      state.modelLinerSeenAt = performance.now();
    }
  }).catch((error) => {
    state.modelLinerPending = false;
    state.modelLinerAvailable = false;
    state.modelLinerError = error;
    console.warn("Eyeliner segmentation model unavailable", error);
  });
}

function getRecentModelLiner() {
  if (!state.modelDetectedLiner) return null;
  if (performance.now() - state.modelLinerSeenAt > 900) return null;

  return state.modelDetectedLiner;
}

function getConfirmedLiner(candidate, geometry) {
  if (!candidate || candidate.confidence < 0.62) {
    state.pendingLiner = null;
    state.pendingLinerFrames = 0;
    return null;
  }

  if (
    state.pendingLiner
    && distance(candidate.tip, state.pendingLiner.tip) < geometry.eyeWidth * 0.14
  ) {
    state.pendingLinerFrames += 1;
  } else {
    state.pendingLinerFrames = 1;
  }

  state.pendingLiner = candidate;
  return state.pendingLinerFrames >= 6 ? candidate : null;
}

function getLinerConfidence(pixels, line, geometry, zones = null) {
  const lengthScore = clamp(line.length / Math.max(geometry.eyeWidth * 0.55, 1), 0, 1);
  const ratioScore = clamp((line.ratio - 2.8) / 4, 0, 1);
  const countScore = clamp(pixels.length / 90, 0, 1);
  const tipProgress = clamp(projectProgress(line.end, geometry.outer, geometry.wingEnd), 0, 1.2);
  const startProgress = clamp(projectProgress(line.start, geometry.outer, geometry.wingEnd), 0, 1.2);
  const progressScore = clamp(Math.max(tipProgress, startProgress) / 0.6, 0, 1);
  const zoneScore = zones
    ? clamp((zones.tailPixelCount + zones.returnPixelCount + zones.fillPixelCount) / 46, 0, 1)
    : 0;

  return lengthScore * 0.28 + ratioScore * 0.18 + countScore * 0.18 + progressScore * 0.12 + zoneScore * 0.24;
}

function isPlausibleLinerLine(line, geometry) {
  const lineAngle = Math.atan2(line.end.y - line.start.y, line.end.x - line.start.x);
  const wingError = Math.abs(normalizeDegrees((lineAngle - geometry.wingAngle) * 180 / Math.PI));
  const returnAngle = Math.atan2(geometry.returnJoin.y - geometry.wingEnd.y, geometry.returnJoin.x - geometry.wingEnd.x);
  const returnError = Math.abs(normalizeDegrees((lineAngle - returnAngle) * 180 / Math.PI));
  const reverseReturnError = Math.abs(normalizeDegrees((lineAngle - returnAngle + Math.PI) * 180 / Math.PI));

  return wingError < 36 || returnError < 36 || reverseReturnError < 36;
}

function collectLinerPixels(image, bounds, geometry) {
  const pixels = [];
  const step = 2;

  for (let y = 0; y < image.height; y += step) {
    for (let x = 0; x < image.width; x += step) {
      const index = (y * image.width + x) * 4;
      const red = image.data[index];
      const green = image.data[index + 1];
      const blue = image.data[index + 2];
      const luminance = red * 0.299 + green * 0.587 + blue * 0.114;
      const colorSpread = Math.max(red, green, blue) - Math.min(red, green, blue);
      const point = {
        x: bounds.x + x,
        y: bounds.y + y
      };
      if (!isLinerSearchPixel(point, geometry)) continue;

      const baselineDrop = getBaselineDrop(point, luminance, geometry);
      const newlyDark = baselineDrop > 34 && luminance < 155 && colorSpread < 72;

      if (!newlyDark) continue;

      pixels.push(point);
    }
  }

  return pixels;
}

function getBaselineDrop(point, currentLuminance, geometry) {
  const baseline = state.baselineModel;
  if (!baseline || !geometry) return 0;

  const wingSpace = getWingSpace(point, geometry);
  const uRatio = (wingSpace.u - baseline.uMin) / (baseline.uMax - baseline.uMin);
  const vRatio = (wingSpace.v - baseline.vMin) / (baseline.vMax - baseline.vMin);

  if (uRatio < 0 || uRatio > 1 || vRatio < 0 || vRatio > 1) return 0;

  const column = uRatio * (baseline.columns - 1);
  const row = vRatio * (baseline.rows - 1);
  const c0 = Math.floor(column);
  const r0 = Math.floor(row);
  const c1 = Math.min(c0 + 1, baseline.columns - 1);
  const r1 = Math.min(r0 + 1, baseline.rows - 1);
  const cx = column - c0;
  const ry = row - r0;
  const top = lerpNumber(
    baseline.samples[r0 * baseline.columns + c0],
    baseline.samples[r0 * baseline.columns + c1],
    cx
  );
  const bottom = lerpNumber(
    baseline.samples[r1 * baseline.columns + c0],
    baseline.samples[r1 * baseline.columns + c1],
    cx
  );
  const expectedLuminance = lerpNumber(top, bottom, ry);

  return expectedLuminance - currentLuminance;
}

function isLinerSearchPixel(point, geometry) {
  const wingSpace = getWingSpace(point, geometry);
  const closeToWing = wingSpace.u > 0.1
    && wingSpace.u < 1.08
    && Math.abs(wingSpace.v) < 0.14;
  const returnProgress = projectProgress(point, geometry.wingEnd, geometry.returnJoin);
  const closeToReturn = returnProgress > 0.05
    && returnProgress < 0.95
    && distanceToSegment(point, geometry.wingEnd, geometry.returnJoin) < geometry.eyeWidth * 0.13;

  return closeToWing || closeToReturn;
}

function processHandTracking(results, now, geometry) {
  const hand = results.landmarks && results.landmarks[0];

  if (!hand) {
    if (state.inputMode === "hand" && now - state.lastHandSeen > 220) {
      finishActiveStroke();
      state.handPinching = false;
      state.penTip = null;
      state.penLine = null;
      state.penGripOffset = null;
      state.penLockedUntil = 0;
      handStatus.textContent = "Searching";
    }
    return;
  }

  const indexTip = toXY(hand[8]);
  const thumbTip = toXY(hand[4]);
  const gripCenter = centerPoint(indexTip, thumbTip);
  const wrist = toXY(hand[0]);
  const middleBase = toXY(hand[9]);
  const handScale = distance(wrist, middleBase);
  const pinchDistance = distance(indexTip, thumbTip);
  const pinchThreshold = clamp(handScale * 0.48, 22, 54);
  const releasedThreshold = pinchThreshold * 1.35;
  const wasPinching = state.handPinching;
  const rawPinching = wasPinching
    ? pinchDistance < releasedThreshold
    : pinchDistance < pinchThreshold;

  state.inputMode = "hand";
  state.lastHandSeen = now;
  state.handPinching = rawPinching;
  state.handTip = state.handTip ? lerpPoint(state.handTip, indexTip, 0.42) : indexTip;
  const penCandidate = !rawPinching ? detectPenTip(hand, geometry, gripCenter) : null;
  const lockedPenTip = getLockedPenTip(gripCenter, now);

  if (lockedPenTip) {
    state.penTip = state.penTip ? lerpPoint(state.penTip, lockedPenTip, 0.58) : lockedPenTip;
    state.penLine = null;
  } else if (penCandidate) {
    state.penTip = state.penTip ? lerpPoint(state.penTip, penCandidate.tip, 0.5) : penCandidate.tip;
    state.penLine = penCandidate.line;
    state.penGripOffset = {
      x: state.penTip.x - gripCenter.x,
      y: state.penTip.y - gripCenter.y
    };
    state.penLockedUntil = now + 1400;
  } else {
    if (!rawPinching && now > state.penLockedUntil) {
      state.penTip = null;
      state.penGripOffset = null;
    }
    state.penLine = null;
  }

  const activeTip = state.penTip || state.handTip;
  const tipOnFace = isTipInEyelinerZone(activeTip, geometry);
  state.pointer = activeTip;
  if (!state.detectedLiner) {
    handStatus.textContent = getHandStatusText(tipOnFace);
  }

  if (state.handPinching && state.penTip) {
    state.penLockedUntil = now + 1200;
  }
}

function getHandStatusText(tipOnFace) {
  if (state.detectedLiner) return "Line observed";
  if (state.penTip && tipOnFace) return "Pen near face";
  if (state.penTip) return "Pen found";
  return "No line yet";
}

function isTipInEyelinerZone(point, geometry) {
  if (!point || !geometry) return false;

  const eyePad = geometry.eyeWidth * 0.72;
  const minX = Math.min(geometry.inner.x, geometry.outer.x, geometry.wingEnd.x) - eyePad;
  const maxX = Math.max(geometry.inner.x, geometry.outer.x, geometry.wingEnd.x) + eyePad;
  const minY = Math.min(geometry.upper.y, geometry.lower.y, geometry.outer.y, geometry.wingEnd.y) - eyePad;
  const maxY = Math.max(geometry.upper.y, geometry.lower.y, geometry.outer.y, geometry.wingEnd.y) + eyePad;

  if (point.x < minX || point.x > maxX || point.y < minY || point.y > maxY) return false;

  const closeToWing = distanceToSegment(point, geometry.outer, geometry.wingEnd) < geometry.eyeWidth * 0.95;
  const closeToLash = distanceToSegment(point, geometry.inner, geometry.outer) < geometry.eyeWidth * 0.82;
  const closeToOuter = distance(point, geometry.outer) < geometry.eyeWidth * 1.25;

  return closeToWing || closeToLash || closeToOuter;
}

function getLockedPenTip(gripCenter, now) {
  if (!state.penGripOffset || now > state.penLockedUntil) return null;

  return {
    x: gripCenter.x + state.penGripOffset.x,
    y: gripCenter.y + state.penGripOffset.y
  };
}

function detectPenTip(hand, geometry, gripCenter) {
  if (!video.videoWidth || !video.videoHeight) return null;

  if (analysisCanvas.width !== canvas.width || analysisCanvas.height !== canvas.height) {
    analysisCanvas.width = canvas.width;
    analysisCanvas.height = canvas.height;
  }

  analysisCtx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);

  const handPoints = [0, 1, 2, 3, 4, 5, 8, 9, 12].map((index) => toXY(hand[index]));
  const indexTip = toXY(hand[8]);
  const thumbTip = toXY(hand[4]);
  const targetPoint = geometry ? centerPoint(geometry.outer, geometry.wingEnd) : indexTip;
  const bounds = getExpandedBounds(handPoints, 92);
  const image = analysisCtx.getImageData(bounds.x, bounds.y, bounds.width, bounds.height);
  const pixels = collectDarkPixels(image, bounds, gripCenter, geometry);

  if (pixels.length < 24) return null;

  const line = fitLineToPixels(pixels);
  if (!line || line.length < 34 || line.ratio < 3.2) return null;

  const tip = distance(line.start, targetPoint) < distance(line.end, targetPoint)
    ? line.start
    : line.end;

  const otherEnd = tip === line.start ? line.end : line.start;
  const nearGrip = distanceToSegment(gripCenter, line.start, line.end);
  const gripProjection = projectProgress(gripCenter, line.start, line.end);
  const lineRunsThroughGrip = nearGrip < 34 && gripProjection > -0.18 && gripProjection < 1.18;
  const endpointNearHand = Math.min(distance(gripCenter, line.start), distance(gripCenter, line.end)) < 92;
  const tipAwayFromGrip = distance(tip, gripCenter) > 24;
  const mostlyOnEye = geometry
    && distanceToSegment(line.start, geometry.outer, geometry.wingEnd) < 26
    && distanceToSegment(line.end, geometry.outer, geometry.wingEnd) < 26;

  if (mostlyOnEye || !lineRunsThroughGrip || !endpointNearHand || !tipAwayFromGrip || distance(tip, otherEnd) < 38) return null;

  return {
    tip,
    line: {
      start: line.start,
      end: line.end
    }
  };
}

function getExpandedBounds(points, padding) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const left = clamp(Math.floor(Math.min(...xs) - padding), 0, canvas.width - 1);
  const top = clamp(Math.floor(Math.min(...ys) - padding), 0, canvas.height - 1);
  const right = clamp(Math.ceil(Math.max(...xs) + padding), left + 1, canvas.width);
  const bottom = clamp(Math.ceil(Math.max(...ys) + padding), top + 1, canvas.height);

  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top
  };
}

function collectDarkPixels(image, bounds, gripCenter, geometry) {
  const pixels = [];
  const step = 3;

  for (let y = 0; y < image.height; y += step) {
    for (let x = 0; x < image.width; x += step) {
      const index = (y * image.width + x) * 4;
      const red = image.data[index];
      const green = image.data[index + 1];
      const blue = image.data[index + 2];
      const alpha = image.data[index + 3];
      if (alpha < 180) continue;

      const maxChannel = Math.max(red, green, blue);
      const minChannel = Math.min(red, green, blue);
      const luminance = red * 0.299 + green * 0.587 + blue * 0.114;
      const point = {
        x: bounds.x + x,
        y: bounds.y + y
      };

      if (isEyeLinePixel(point, gripCenter, geometry)) continue;

      if (luminance < 86 && maxChannel - minChannel < 58 && distance(point, gripCenter) < 118) {
        pixels.push(point);
      }
    }
  }

  return pixels;
}

function isEyeLinePixel(point, gripCenter, geometry) {
  if (!geometry || distance(point, gripCenter) < 42) return false;

  const onWingGuide = distanceToSegment(point, geometry.outer, geometry.wingEnd) < 20;
  const onLashLine = distanceToSegment(point, geometry.inner, geometry.outer) < 18;
  const closeToOuterCorner = distance(point, geometry.outer) < geometry.eyeWidth * 0.7;

  return onWingGuide || onLashLine || closeToOuterCorner;
}

function fitLineToPixels(points) {
  const mean = points.reduce((sum, point) => ({
    x: sum.x + point.x,
    y: sum.y + point.y
  }), { x: 0, y: 0 });
  mean.x /= points.length;
  mean.y /= points.length;

  let xx = 0;
  let xy = 0;
  let yy = 0;

  points.forEach((point) => {
    const dx = point.x - mean.x;
    const dy = point.y - mean.y;
    xx += dx * dx;
    xy += dx * dy;
    yy += dy * dy;
  });

  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const axis = {
    x: Math.cos(angle),
    y: Math.sin(angle)
  };
  const cross = {
    x: -axis.y,
    y: axis.x
  };
  let minProjection = Infinity;
  let maxProjection = -Infinity;
  let axisVariance = 0;
  let crossVariance = 0;

  points.forEach((point) => {
    const dx = point.x - mean.x;
    const dy = point.y - mean.y;
    const projection = dx * axis.x + dy * axis.y;
    const side = dx * cross.x + dy * cross.y;
    minProjection = Math.min(minProjection, projection);
    maxProjection = Math.max(maxProjection, projection);
    axisVariance += projection * projection;
    crossVariance += side * side;
  });

  const length = maxProjection - minProjection;
  const ratio = axisVariance / Math.max(crossVariance, 1);

  return {
    start: {
      x: mean.x + axis.x * minProjection,
      y: mean.y + axis.y * minProjection
    },
    end: {
      x: mean.x + axis.x * maxProjection,
      y: mean.y + axis.y * maxProjection
    },
    length,
    ratio
  };
}

function startStroke(event) {
  if (state.inputMode === "hand" && performance.now() - state.lastHandSeen < 800) return;

  canvas.setPointerCapture(event.pointerId);
  state.inputMode = "pointer";
  beginStrokeAt(eventToCanvasPoint(event));
}

function movePointer(event) {
  if (state.inputMode === "hand" && performance.now() - state.lastHandSeen < 800) return;

  addPointerPoint(event);
}

function endStroke(event) {
  if (state.inputMode === "hand" && performance.now() - state.lastHandSeen < 800) return;

  addPointerPoint(event);
  finishActiveStroke();
}

function addPointerPoint(event) {
  const point = eventToCanvasPoint(event);
  state.pointer = point;

  if (state.isDrawing && isTipInEyelinerZone(point, state.currentGeometry)) {
    addStrokePoint(point);
  } else if (state.isDrawing) {
    finishActiveStroke();
  }
}

function beginStrokeAt(point) {
  if (!isTipInEyelinerZone(point, state.currentGeometry)) {
    state.pointer = point;
    return;
  }

  state.lockedEyeKey = state.activeEye;
  state.isDrawing = true;
  state.stroke = [];
  state.pointer = point;
  addStrokePoint(point);
}

function addStrokePoint(point) {
  const last = getCurrentStrokeTip();
  if (!last || distance(last, point) > 2) {
    state.stroke.push(point);
  }
}

function finishActiveStroke() {
  state.isDrawing = false;

  if (state.stroke.length) {
    state.strokes.push([...state.stroke]);
    state.stroke = [];
  }
}

function resetStroke() {
  state.stroke = [];
  state.strokes = [];
  const recalibrated = captureBaselineIfReady();
  state.needsResetBaseline = !recalibrated;
  if (recalibrated) {
    armLinerTracking(1100);
  } else {
    clearLinerTracking(1100);
    state.linerTrackingArmed = false;
  }
  state.lockedEyeKey = null;
  state.handTip = null;
  state.penTip = null;
  state.penLine = null;
  state.penGripOffset = null;
  state.penLockedUntil = 0;
  state.handPinching = false;
  state.inputMode = "pointer";
  state.pointer = null;
  state.pendingLessonKey = "placeTip";
  state.pendingLessonSince = performance.now();
  state.modelDetectedLiner = null;
  state.modelLinerSeenAt = 0;
  handStatus.textContent = recalibrated ? "Reset from current eye" : "No liner detected";
  accuracyStatus.textContent = "--";
  setImmediateLesson("placeTip");
}

function eventToCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const displayX = (event.clientX - rect.left) * (canvas.width / rect.width);

  return {
    x: canvas.width - displayX,
    y: (event.clientY - rect.top) * (canvas.height / rect.height)
  };
}

function resizeCanvasToVideo() {
  if (!video.videoWidth || !video.videoHeight) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function updateHud(scan, eye, accuracy, lesson) {
  scanStatus.textContent = scan;
  eyeStatus.textContent = eye;
  accuracyStatus.textContent = accuracy;
  setLesson(lesson);
}

function setLesson(lesson) {
  stepLabel.textContent = lesson.label;
  stepTitle.textContent = lesson.title;
  coachMessage.textContent = lesson.message;
  coachTip.textContent = lesson.tip;
  nextAction.textContent = lesson.next;
}

function setImmediateLesson(key) {
  state.currentLessonKey = key;
  state.pendingLessonKey = key;
  state.pendingLessonSince = performance.now();
  state.lessonChangedAt = performance.now();
  updateCoachPhaseForLesson(key);
  setLesson(getLesson(key));
}

function getLiveLinerLesson(key) {
  state.currentLessonKey = key;
  state.pendingLessonKey = key;
  state.pendingLessonSince = performance.now();
  state.lessonChangedAt = performance.now();
  state.candidateLessonKey = key;
  state.candidateLessonFrames = 0;
  return getLesson(key);
}

function getStableLesson(nextKey) {
  const now = performance.now();
  const currentPriority = getLessonPriority(state.currentLessonKey);
  const nextPriority = getLessonPriority(nextKey);
  const currentOrder = getLessonOrder(state.currentLessonKey);
  const nextOrder = getLessonOrder(nextKey);
  const hasDrawing = Boolean(state.detectedLiner);
  const noRecentDrawing = now - state.lastLinerSeenAt > 1600;
  const isRegressing = nextOrder < currentOrder;
  const dwellTime = now - state.lessonChangedAt;
  const timedNextKey = TIMED_LESSON_SEQUENCE[state.currentLessonKey];

  if (timedNextKey && dwellTime >= COACH_STEP_DURATION_MS) {
    state.currentLessonKey = timedNextKey;
    state.pendingLessonKey = timedNextKey;
    state.pendingLessonSince = now;
    state.lessonChangedAt = now;
    state.candidateLessonKey = timedNextKey;
    state.candidateLessonFrames = 0;
    updateCoachPhaseForLesson(timedNextKey);
    return getLesson(timedNextKey);
  }

  if (isRegressing) {
    state.candidateLessonKey = state.currentLessonKey;
    state.candidateLessonFrames = 0;
    return getLesson(state.currentLessonKey);
  }

  if (nextKey === "placeTip") {
    if (hasDrawing || (!noRecentDrawing && currentPriority >= 2)) {
      return getLesson(state.currentLessonKey);
    }
    state.currentLessonKey = nextKey;
    state.pendingLessonKey = nextKey;
    state.pendingLessonSince = now;
    state.lessonChangedAt = now;
    state.candidateLessonKey = nextKey;
    state.candidateLessonFrames = 0;
    updateCoachPhaseForLesson(nextKey);
    return getLesson(state.currentLessonKey);
  }

  if (hasDrawing && currentPriority >= 5 && nextPriority < 5) {
    return getLesson(state.currentLessonKey);
  }

  if (dwellTime < getMinimumLessonDwell(state.currentLessonKey, nextKey)) {
    return getLesson(state.currentLessonKey);
  }

  if (nextKey !== state.candidateLessonKey) {
    state.candidateLessonKey = nextKey;
    state.candidateLessonFrames = 1;
    return getLesson(state.currentLessonKey);
  }

  state.candidateLessonFrames += 1;
  const requiredFrames = getRequiredLessonFrames(nextKey);
  if (state.candidateLessonFrames < requiredFrames) {
    return getLesson(state.currentLessonKey);
  }

  if (hasDrawing && isRegressing && currentPriority >= 3 && currentPriority - nextPriority <= 1) {
    return getLesson(state.currentLessonKey);
  }

  if (hasDrawing && !isRegressing) {
    state.currentLessonKey = nextKey;
    state.pendingLessonKey = nextKey;
    state.pendingLessonSince = now;
    state.lessonChangedAt = now;
    updateCoachPhaseForLesson(nextKey);
    return getLesson(state.currentLessonKey);
  }

  const requiredDelay = isRegressing ? 2800 : 1200;

  if (nextKey === state.currentLessonKey) {
    state.pendingLessonKey = nextKey;
    state.pendingLessonSince = now;
    return getLesson(state.currentLessonKey);
  }

  if (nextKey !== state.pendingLessonKey) {
    state.pendingLessonKey = nextKey;
    state.pendingLessonSince = now;
    return getLesson(state.currentLessonKey);
  }

  if (now - state.pendingLessonSince >= requiredDelay) {
    state.currentLessonKey = nextKey;
    state.lessonChangedAt = now;
    updateCoachPhaseForLesson(nextKey);
  }

  return getLesson(state.currentLessonKey);
}

function updateCoachPhaseForLesson(key) {
  const priority = getLessonPriority(key);
  if (priority >= 6) {
    state.coachPhase = "finish";
  } else if (priority >= 5) {
    state.coachPhase = "shape";
  } else if (priority >= 3) {
    state.coachPhase = "tail";
  } else if (key === "placeTip" || key.endsWith("Map")) {
    state.coachPhase = "start";
  }
}

function getRequiredLessonFrames(key) {
  const priority = getLessonPriority(key);
  if (key === "finished") return 1;
  if (state.currentLessonKey === "halfCloseEye" && key === "fillTriangle") return 8;
  if (priority >= 6) return 24;
  if (priority >= 5) return 18;
  if (priority >= 3) return 12;
  return 8;
}

function getMinimumLessonDwell(currentKey, nextKey) {
  if (currentKey === nextKey) return 0;
  if (TIMED_LESSON_SEQUENCE[currentKey]) return COACH_STEP_DURATION_MS;
  if (getLessonPriority(nextKey) >= 5) return 1800;
  if (getLessonPriority(nextKey) >= 3) return 1400;
  return 1100;
}

function getLessonPriority(key) {
  const priorities = {
    prep: 0,
    cameraBlocked: 0,
    findFace: 0,
    placeTip: 1,
    limitedLidMap: 1,
    downturnedMap: 1,
    roundMap: 1,
    upturnedMap: 1,
    findOuterCorner: 1,
    outerCornerAnchor: 2,
    firstDot: 2,
    firstMark: 2,
    stampCorner: 2,
    anchorHand: 2,
    softPressure: 2,
    dontStretch: 2,
    openEyeStamps: 2,
    downturnedLift: 2,
    roundElongate: 2,
    upturnedBalance: 2,
    steadyPull: 3,
    tailDirection: 3,
    tailLength: 3,
    buildSlowly: 3,
    featherStrokes: 3,
    keepLashThin: 3,
    halfCloseEye: 3,
    tooHigh: 3,
    tooLow: 3,
    rotateHand: 3,
    extendTip: 4,
    tooLong: 4,
    returnStart: 4,
    lashConnection: 5,
    connectTriangle: 5,
    fillTriangle: 6,
    cleanWithAngle: 6,
    cleanEdge: 7,
    finished: 15
  };

  return priorities[key] || 1;
}

function getLessonOrder(key) {
  const order = {
    prep: 0,
    cameraBlocked: 0,
    findFace: 0,
    placeTip: 1,
    limitedLidMap: 1,
    downturnedMap: 1,
    roundMap: 1,
    upturnedMap: 1,
    findOuterCorner: 2,
    outerCornerAnchor: 3,
    firstDot: 4,
    firstMark: 4,
    stampCorner: 4,
    anchorHand: 4,
    dontStretch: 4,
    tailDirection: 5,
    openEyeStamps: 5,
    downturnedLift: 5,
    roundElongate: 5,
    upturnedBalance: 5,
    softPressure: 5,
    featherStrokes: 5,
    buildSlowly: 6,
    tailLength: 6,
    rotateHand: 6,
    tooHigh: 7,
    tooLow: 7,
    steadyPull: 7,
    extendTip: 8,
    tooLong: 8,
    keepLashThin: 8,
    returnStart: 9,
    lashConnection: 10,
    connectTriangle: 11,
    halfCloseEye: 12,
    fillTriangle: 13,
    cleanWithAngle: 14,
    cleanEdge: 14,
    finished: 15
  };

  return order[key] ?? getLessonPriority(key);
}

function getCameraErrorLesson(error) {
  if (error && error.name === "NotAllowedError") {
    return {
      label: "Camera",
      title: "Camera permission is off",
      message: "The browser denied camera access, so I cannot place the AR guide on your face yet.",
      tip: "Open the camera/permissions control in the address bar for this page and allow camera access.",
      next: "After allowing camera access, press Start camera here."
    };
  }

  if (error && error.name === "NotFoundError") {
    return {
      label: "Camera",
      title: "No camera found",
      message: "I could not find a usable camera on this device.",
      tip: "If you use an external webcam, connect it first, then try again.",
      next: "Press Start camera after the camera is available."
    };
  }

  return getLesson("cameraBlocked");
}

function getLesson(key) {
  const lessons = {
    prep: {
      label: "Prep",
      title: "Let me see your eye",
      message: "Face the camera softly, with your gaze straight ahead. Keep your brow relaxed and your chin level so the wing is mapped on the eye you actually wear open.",
      tip: "MUA habit: map first, draw second. Good liner starts with posture and light, not product.",
      next: "Press Calibrate eye, rest your drawing hand lightly on your cheek, then begin."
    },
    cameraBlocked: {
      label: "Camera",
      title: "Camera access needed",
      message: "I need the camera so the wing can be placed on your own eye, not on a generic template.",
      tip: "Artist note: even light is your friend. It reveals the true outer corner and keeps the guide from guessing.",
      next: "Allow camera access, then reload the page."
    },
    findFace: {
      label: "Scan",
      title: "Find the shape",
      message: "Center your face and hold still for a moment. I am looking for the outer corner, the lash line, and the natural lift of your eye.",
      tip: "Artist note: look forward rather than downward. A wing drawn on a folded lid can change once your eyes relax.",
      next: "Hold this position until the scan locks."
    },
    placeTip: {
      label: "Begin",
      title: "Start at the root",
      message: "Place the pen where the upper and lower lashes meet. Rest your pinky or the side of your palm on your cheek so the movement comes from tiny finger pivots, not your whole arm.",
      tip: "Do not pull the skin tight. A hard stretch can make a straight line turn wavy when the lid relaxes.",
      next: "Look forward and make one tiny mark at the outer-corner dot."
    },
    limitedLidMap: {
      label: "Map",
      title: "Keep the eye open",
      message: "Keep your eye relaxed and open while you map the tail. If you close the eye now, the wing can hide in the fold or change angle when you open.",
      tip: "Use tiny stamps with the very tip of the pen. For limited lid space, open-eye placement matters more than one perfect swipe.",
      next: "Look straight ahead, stamp one dot at the outer corner, then pause."
    },
    downturnedMap: {
      label: "Map",
      title: "Lift before the corner falls",
      message: "Your outer corner sits a little lower, so the wing needs to leave the eye slightly before the downward turn takes over.",
      tip: "Think of the wing as a small lift from the upper lash line, not a trace of the lower edge.",
      next: "Start at the outer-corner dot and aim into the dotted lift."
    },
    roundMap: {
      label: "Map",
      title: "Stretch, do not stack",
      message: "Your eye reads round, so keep the wing more horizontal and elegant. Length will flatter more than extra thickness.",
      tip: "Round eyes can get heavy fast if the liner grows upward too soon.",
      next: "Make a narrow outward dash before you add any thickness."
    },
    upturnedMap: {
      label: "Map",
      title: "Balance the natural lift",
      message: "Your eye already lifts at the outer corner. Keep the wing fine and controlled so it enhances the shape without over-tilting it.",
      tip: "A naturally lifted eye usually needs polish, not a dramatic extra angle.",
      next: "Place a small mark along the dotted path and keep the tail short."
    },
    findOuterCorner: {
      label: "Begin",
      title: "Come back to the corner",
      message: "You are a little outside the root. Slide your hand back toward the outer corner and plant your pinky on your cheek before you make the next mark.",
      tip: "The pen should touch down first, then move. If your hand is floating, the wing will float too.",
      next: "Rest the tip at the outer-corner dot and pause for one beat."
    },
    outerCornerAnchor: {
      label: "Anchor",
      title: "Plant the first touch",
      message: "Good, you are at the outer corner. Keep the eye open and relaxed. Let your pinky rest on your cheek, then touch the pen down without dragging yet.",
      tip: "Beginner move: separate touching from drawing. Touch, pause, then decide where the line goes.",
      next: "Make one tiny dot right at the root, then lift the pen."
    },
    firstDot: {
      label: "Dot",
      title: "Make the guide dot",
      message: "You have the first mark. Now place one more tiny dot along the dotted wing path, only a few millimeters away from the corner.",
      tip: "Think connect-the-dots, not one dramatic flick. This keeps the wing calm and symmetrical.",
      next: "Add a second dot in the direction of the wing, then lift."
    },
    firstMark: {
      label: "Mark",
      title: "Whisper the first line",
      message: "Make a tiny, light dash from the corner. This is only the direction mark, not the whole wing.",
      tip: "Artist note: pressure is everything. A soft touch gives you elegance; heavy pressure gives you cleanup.",
      next: "Let that real dash point toward the dotted endpoint."
    },
    stampCorner: {
      label: "Mark",
      title: "Stamp, then pause",
      message: "Use the very tip to stamp a tiny mark. Keep your wrist quiet, press lightly, then lift the pen completely off the skin.",
      tip: "One controlled stamp is easier than one brave swipe. You are placing breadcrumbs for the wing.",
      next: "Lift, breathe, then add the next tiny stamp along the guide."
    },
    anchorHand: {
      label: "Hand",
      title: "Anchor the hand",
      message: "Your mark is changing angle early. Put your elbow on the table if you can, then rest your pinky or palm on your cheek.",
      tip: "Move from the fingers, not the shoulder. The smaller the joint doing the work, the cleaner the liner.",
      next: "Reset the pen at the outer corner and make one short anchored stroke."
    },
    softPressure: {
      label: "Hand",
      title: "Lighten the pressure",
      message: "Your stroke is drifting away from the guide. Hold the pen farther back and let only the tip touch the skin.",
      tip: "Heavy pressure flattens the brush and makes the line blunt. Light pressure keeps the tail tapered.",
      next: "Use a softer touch and add a short feather stroke."
    },
    dontStretch: {
      label: "Lid",
      title: "Do not tug the lid",
      message: "If you need tension, lift the brow slightly instead of pulling the outer corner sideways. Keep the skin close to its natural position.",
      tip: "Many artists avoid hard stretching because the line can shrink, wrinkle, or change direction when the lid releases.",
      next: "Relax the eye, look forward, and place the next small mark."
    },
    openEyeStamps: {
      label: "Mark",
      title: "Use open-eye stamps",
      message: "Keep looking forward and build the tail with tiny stamps. Do not close the eye until the outer shape is mapped.",
      tip: "For hooded or fold-prone lids, open-eye stamps show you exactly where the wing will live.",
      next: "Add two short dots along the guide, then connect them lightly."
    },
    downturnedLift: {
      label: "Lift",
      title: "Leave the downturn",
      message: "Aim the next mark slightly above the outer corner. Avoid following the lash line downward at the very end.",
      tip: "The common mistake on downturned eyes is tracing the drop, which can make the eye look tired.",
      next: "Place the next dot just above the guide tail."
    },
    roundElongate: {
      label: "Shape",
      title: "Pull outward first",
      message: "Keep the line thin and travel outward before you add height. This gives a round eye a softer almond effect.",
      tip: "Length first, thickness last. That order keeps the wing delicate.",
      next: "Add a short outward mark, then pause before thickening."
    },
    upturnedBalance: {
      label: "Shape",
      title: "Keep the tail restrained",
      message: "The eye already has lift, so keep the tail fine and close to the dotted path. Too much angle can look severe.",
      tip: "Use the smallest flick that still reads as intentional.",
      next: "Continue with one small connected dot, not a high flick."
    },
    tailDirection: {
      label: "Angle",
      title: "Set the wing direction",
      message: "Your first dots are there. Connect them with short outward strokes, keeping the pen almost parallel to the skin and the pressure very light.",
      tip: "Move from the fingers. If the wrist or shoulder takes over, the angle usually jumps.",
      next: "Draw one short stroke outward along the dotted tail, then lift."
    },
    tailLength: {
      label: "Length",
      title: "Grow the tail slowly",
      message: "The angle is readable. Add length in tiny sections and stop before the line passes the dotted endpoint.",
      tip: "Length first, thickness later. A beginner wing gets messy when the tail and fill happen at the same time.",
      next: "Add one small section toward the tip, then check the length."
    },
    steadyPull: {
      label: "Lift",
      title: "Follow the eye's lift",
      message: "You have the direction. Pull outward with a soft wrist and barely-there pressure, then lift the pen as you reach the tip.",
      tip: "The flick is a release, not a push. Press at the base, lighten toward the end.",
      next: "Stop adding length when the line reaches the dotted endpoint."
    },
    buildSlowly: {
      label: "Shape",
      title: "Build in small strokes",
      message: "You are close. Add little connected strokes instead of trying to perfect the whole wing at once.",
      tip: "Artist note: beautiful liner is often assembled quietly. Short strokes give you control.",
      next: "Fill the small gap between your line and the guide."
    },
    featherStrokes: {
      label: "Build",
      title: "Feather the line",
      message: "Use short feather strokes that overlap slightly. Touch down, move a few millimeters, lift. Repeat.",
      tip: "Do not saw back and forth. Multiple one-way strokes keep the edge cleaner.",
      next: "Add one short stroke along the guide, then lift the pen."
    },
    keepLashThin: {
      label: "Refine",
      title: "Keep the lash side thin",
      message: "The tail has enough presence. Keep the lash-line side narrow by turning the pen so only the tip, not the side, touches your lashes.",
      tip: "If the lash side gets thick too early, beginners usually keep thickening both eyes to match.",
      next: "Connect back with the lightest possible pressure."
    },
    halfCloseEye: {
      label: "Lash",
      title: "Half-close for the lash line",
      message: "Now you can half-close the eye slightly to fill near the lashes. Do not squeeze shut; keep the lid relaxed so the shape stays visible.",
      tip: "Open eye to map the wing, half-close to fill the lash line. Switching eye position is normal.",
      next: "Use tiny side-to-side touches only at the lash roots."
    },
    tooHigh: {
      label: "Refine",
      title: "Soften the lift",
      message: "The line is climbing above the guide. Ease the next stroke slightly lower so the wing stays connected to the lash line.",
      tip: "Artist note: lift should look intentional, not surprised. Let the eye shape lead.",
      next: "Place the next short stroke just under the guide."
    },
    tooLow: {
      label: "Refine",
      title: "Lift the tail",
      message: "The tail is dipping. Raise your hand slightly and let the next mark travel upward toward the guide.",
      tip: "Artist note: if the tail falls, the eye can look tired. A small lift keeps the shape awake.",
      next: "Aim the next touch a little higher."
    },
    extendTip: {
      label: "Tail",
      title: "Finish the point",
      message: "The direction is lovely; the tail just needs a touch more length. Use the very tip and flick lightly toward the green point.",
      tip: "Artist note: the end should taper. Pressing the brush flat makes the wing look blunt.",
      next: "Add one small flick, then stop."
    },
    tooLong: {
      label: "Refine",
      title: "Pause the length",
      message: "Stop extending. The wing has enough length now; the next move is shaping, not adding more tail.",
      tip: "Artist note: when a wing gets long, refine the point instead of thickening the whole line.",
      next: "Connect the tail back toward the upper lash line."
    },
    rotateHand: {
      label: "Refine",
      title: "Reset the hand",
      message: "Your stroke changed direction. Pause, turn the pencil back to the guide angle, and continue with a small mark.",
      tip: "Artist note: the hand can wander when the face moves. Keep the face quiet and let the wrist do less.",
      next: "Reset your hand, then add one short stroke."
    },
    returnStart: {
      label: "Return",
      title: "Start the lower edge",
      message: "The tail is long enough. Stop extending outward. Put the pen at the tip and angle it back toward your upper lash line.",
      tip: "Keep the eye open for this part. The return line decides the real wing shape.",
      next: "From the tip, draw a tiny inward stroke toward the lashes."
    },
    lashConnection: {
      label: "Connect",
      title: "Meet the lash line",
      message: "You have started the return. Keep the pen light and guide it back until it meets the upper lash line without cutting into the eye.",
      tip: "If it feels scary, use taps instead of a line: tap from the tip inward, then tap from the lashes outward.",
      next: "Close the small gap between the return edge and the lash roots."
    },
    connectTriangle: {
      label: "Shape",
      title: "Make the little triangle",
      message: "The tail is placed. Starting at the tip, angle your pen back toward the lash line and draw the lower return edge with feather-light pressure.",
      tip: "Keep your eye open for the return line. You are checking that the triangle looks right when your face is relaxed.",
      next: "Draw one short return stroke from the tip back toward the lashes."
    },
    fillTriangle: {
      label: "Finish",
      title: "Fill and refine",
      message: "Good. Fill only the little triangle with tiny taps. Keep the pen nearly parallel to the lash line so the base fills without making the whole wing bulky.",
      tip: "Stop before it looks perfect. A clean wing is often edited with cleanup, not more liner.",
      next: "Tap inside the triangle, then check the shape with your eye open."
    },
    cleanWithAngle: {
      label: "Polish",
      title: "Clean with an angle",
      message: "For a sharper edge, use an angled brush or pointed cotton tip with a tiny bit of remover or concealer.",
      tip: "Clean under the wing in one outward motion. Do not scrub back into the liner.",
      next: "Sweep from under the wing outward, then stop."
    },
    cleanEdge: {
      label: "Polish",
      title: "Edit the edge",
      message: "The wing has its shape. Stop adding liner. If you want sharpness, clean the lower edge instead of thickening the top.",
      tip: "Makeup artists often perfect a wing by removing a little, not by adding more.",
      next: "Use an angled brush or cotton tip and sweep outward away from the eye."
    },
    finished: {
      label: "Finished",
      title: "Your wing is complete",
      message: "You have built the tail, connected it to the lash line, and filled the shape. Put the pen down and look straight ahead to see the wing as it will actually be worn.",
      tip: "A finished wing does not need to be perfectly identical to the other eye. It should feel balanced when your face is relaxed.",
      next: "Choose the other eye to continue, or press Reset coaching to practice this eye again."
    }
  };

  return lessons[key] || lessons.placeTip;
}

function getStrokeTip() {
  if (state.stroke.length) {
    return state.stroke[state.stroke.length - 1];
  }

  const lastStroke = state.strokes[state.strokes.length - 1];
  return lastStroke ? lastStroke[lastStroke.length - 1] : null;
}

function getCurrentStrokeTip() {
  return state.stroke[state.stroke.length - 1] || null;
}

function getAccuracyScore(lineDistance, tipDistance, progress, angleError, eyeWidth) {
  const linePenalty = Math.min(42, Math.abs(lineDistance) * 2.2);
  const tipPenalty = Math.min(26, tipDistance / Math.max(eyeWidth, 1) * 20);
  const progressPenalty = Math.min(24, Math.abs(0.9 - progress) * 34);
  const anglePenalty = Math.min(20, Math.abs(angleError) * 0.7);
  return clamp(Math.round(100 - linePenalty - tipPenalty - progressPenalty - anglePenalty), 0, 100);
}

function getStrokeAngleError(geometry) {
  const stroke = state.stroke.length ? state.stroke : state.strokes[state.strokes.length - 1];
  if (!stroke || stroke.length < 4) return 0;

  const start = stroke[Math.max(0, stroke.length - 5)];
  const end = stroke[stroke.length - 1];
  const strokeAngle = Math.atan2(end.y - start.y, end.x - start.x);
  return normalizeDegrees((strokeAngle - geometry.wingAngle) * 180 / Math.PI);
}

function getLineAngleError(line, geometry) {
  if (!line) return 0;

  const lineAngle = Math.atan2(line.end.y - line.start.y, line.end.x - line.start.x);
  return normalizeDegrees((lineAngle - geometry.wingAngle) * 180 / Math.PI);
}

function getReturnLineProgress(liner, geometry) {
  if (!liner || !liner.pixels || !liner.pixels.length) return 0;

  const returnZonePixels = liner.pixels.filter((point) => (
    distanceToSegment(point, geometry.wingEnd, geometry.returnJoin) < geometry.eyeWidth * 0.13
  ));

  if (returnZonePixels.length < 18) return 0;

  let best = 0;

  returnZonePixels.forEach((point) => {
    const progress = projectProgress(point, geometry.wingEnd, geometry.returnJoin);
    if (progress > best && progress <= 1.25) {
      best = progress;
    }
  });

  return clamp(best, 0, 1);
}

function analyzeLinerZones(liner, geometry) {
  const pixels = liner?.pixels || [];
  const tailPixels = [];
  const returnPixels = [];
  const fillPixels = [];
  let tip = null;
  let tailProgress = 0;
  let returnProgress = 0;

  pixels.forEach((point) => {
    const wingSpace = getWingSpace(point, geometry);
    const onTail = wingSpace.u >= -0.03
      && wingSpace.u <= 1.18
      && Math.abs(wingSpace.v) < 0.18;

    if (onTail) {
      tailPixels.push(point);
      const progress = clamp(wingSpace.u, 0, 1.25);
      if (progress > tailProgress) {
        tailProgress = progress;
        tip = point;
      }
    }

    const returnDistance = distanceToSegment(point, geometry.wingEnd, geometry.returnJoin);
    const returnU = projectProgress(point, geometry.wingEnd, geometry.returnJoin);
    const onReturn = returnU >= -0.04
      && returnU <= 1.08
      && returnDistance < geometry.eyeWidth * 0.14;

    if (onReturn) {
      returnPixels.push(point);
      returnProgress = Math.max(returnProgress, clamp(returnU, 0, 1));
    }

    if (isPointInTriangle(point, geometry.outer, geometry.wingEnd, geometry.returnJoin)) {
      const edgeDistance = Math.min(
        distanceToSegment(point, geometry.outer, geometry.wingEnd),
        distanceToSegment(point, geometry.wingEnd, geometry.returnJoin),
        distanceToSegment(point, geometry.returnJoin, geometry.outer)
      );

      if (edgeDistance > geometry.eyeWidth * 0.018) {
        fillPixels.push(point);
      }
    }
  });

  const tailCoverage = getSegmentCoverage(tailPixels, geometry.outer, geometry.wingEnd, 5);
  const returnCoverage = getSegmentCoverage(returnPixels, geometry.wingEnd, geometry.returnJoin, 4);
  const fillPixelRatio = clamp(fillPixels.length / 16, 0, 1);
  const fillProgress = Math.max(getTriangleFillProgress(liner, geometry), fillPixelRatio);

  return {
    tip,
    tailProgress: tailPixels.length >= 6 ? clamp(Math.max(tailProgress, tailCoverage * 0.9), 0, 1.25) : 0,
    returnProgress: returnPixels.length >= 16 ? clamp(Math.max(returnProgress, returnCoverage), 0, 1) : 0,
    fillProgress,
    fillPixelRatio,
    tailCoverage,
    returnCoverage,
    tailPixelCount: tailPixels.length,
    returnPixelCount: returnPixels.length,
    fillPixelCount: fillPixels.length
  };
}

function getSegmentCoverage(points, start, end, bins) {
  if (!points.length) return 0;

  const covered = new Set();

  points.forEach((point) => {
    const progress = projectProgress(point, start, end);
    if (progress < 0 || progress > 1) return;
    covered.add(clamp(Math.floor(progress * bins), 0, bins - 1));
  });

  return covered.size / bins;
}

function getTriangleFillProgress(liner, geometry) {
  if (!liner || !liner.pixels || !liner.pixels.length) return 0;

  const area = Math.max(Math.abs(triangleArea(geometry.outer, geometry.wingEnd, geometry.returnJoin)), 1);
  const targetSamples = clamp(Math.round(area / 34), 24, 90);
  let covered = 0;
  let considered = 0;

  for (let index = 0; index < targetSamples; index += 1) {
    const t = (index + 0.5) / targetSamples;
    if (t < 0.18 || t > 0.88) continue;

    const sample = lerpPoint(
      lerpPoint(geometry.outer, geometry.wingEnd, t),
      lerpPoint(geometry.outer, geometry.returnJoin, t),
      0.5
    );
    const edgeDistance = Math.min(
      distanceToSegment(sample, geometry.outer, geometry.wingEnd),
      distanceToSegment(sample, geometry.wingEnd, geometry.returnJoin),
      distanceToSegment(sample, geometry.returnJoin, geometry.outer)
    );
    if (edgeDistance < geometry.eyeWidth * 0.035) continue;

    considered += 1;
    const hasNearbyLiner = liner.pixels.some((point) => distance(point, sample) < geometry.eyeWidth * 0.055);
    if (hasNearbyLiner) covered += 1;
  }

  return considered ? covered / considered : 0;
}

function getTriangleInteriorInkRatio(liner, geometry) {
  if (!liner || !liner.pixels || !liner.pixels.length) return 0;

  const interiorPixels = liner.pixels.filter((point) => {
    if (!isPointInTriangle(point, geometry.outer, geometry.wingEnd, geometry.returnJoin)) return false;

    const edgeDistance = Math.min(
      distanceToSegment(point, geometry.outer, geometry.wingEnd),
      distanceToSegment(point, geometry.wingEnd, geometry.returnJoin),
      distanceToSegment(point, geometry.returnJoin, geometry.outer)
    );

    return edgeDistance > geometry.eyeWidth * 0.025;
  });

  return clamp(interiorPixels.length / 18, 0, 1);
}

function signedDistanceToLine(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy) || 1;
  return ((point.x - start.x) * dy - (point.y - start.y) * dx) / length;
}

function projectProgress(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSq = dx * dx + dy * dy || 1;
  return ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSq;
}

function normalizeDegrees(value) {
  let degrees = value;
  while (degrees > 180) degrees -= 360;
  while (degrees < -180) degrees += 360;
  return degrees;
}

function getGuideColor(guidance) {
  return "rgba(255, 79, 154, 0.98)";
}

function drawLine(start, end) {
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
}

function drawSmoothClosedCurve(points, tension = 0.5) {
  if (points.length < 3) return;

  const closedPoints = points[0].x === points[points.length - 1].x && points[0].y === points[points.length - 1].y
    ? points.slice(0, -1)
    : points;

  ctx.moveTo(closedPoints[0].x, closedPoints[0].y);

  for (let index = 0; index < closedPoints.length; index += 1) {
    const previous = closedPoints[(index - 1 + closedPoints.length) % closedPoints.length];
    const current = closedPoints[index];
    const next = closedPoints[(index + 1) % closedPoints.length];
    const afterNext = closedPoints[(index + 2) % closedPoints.length];
    const cp1 = {
      x: current.x + (next.x - previous.x) * tension / 6,
      y: current.y + (next.y - previous.y) * tension / 6
    };
    const cp2 = {
      x: next.x - (afterNext.x - current.x) * tension / 6,
      y: next.y - (afterNext.y - current.y) * tension / 6
    };

    ctx.bezierCurveTo(cp1.x, cp1.y, cp2.x, cp2.y, next.x, next.y);
  }

  ctx.closePath();
}

function drawAnchor(point, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
  ctx.fill();
}

function centerPoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2
  };
}

function lerpPoint(a, b, amount) {
  return {
    x: a.x + (b.x - a.x) * amount,
    y: a.y + (b.y - a.y) * amount
  };
}

function lerpNumber(a, b, amount) {
  return a + (b - a) * amount;
}

function blendAngles(a, b, amount) {
  const delta = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + delta * amount;
}

function triangleArea(a, b, c) {
  return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2;
}

function isPointInTriangle(point, a, b, c) {
  const area = Math.abs(triangleArea(a, b, c));
  const areaA = Math.abs(triangleArea(point, b, c));
  const areaB = Math.abs(triangleArea(a, point, c));
  const areaC = Math.abs(triangleArea(a, b, point));

  return Math.abs(area - (areaA + areaB + areaC)) < 0.5;
}

function toXY(point) {
  return {
    x: point.x * canvas.width,
    y: point.y * canvas.height
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function distanceToSegment(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSq = dx * dx + dy * dy;

  if (!lengthSq) return distance(point, start);

  const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSq, 0, 1);
  return distance(point, {
    x: start.x + dx * t,
    y: start.y + dy * t
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

initHome();
initCustomCursor();
