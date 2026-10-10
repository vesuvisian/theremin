import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const MEDIAPIPE_VERSION = "0.10.18";
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const MAX_POSES = 4;
const MAX_HANDS = 8;
const PERFORMER_MATCH_MAX_DIST = 0.55;
const PERFORMER_SILENCE_MS = 350;
const PERFORMER_TIMEOUT_MS = 1200;
const FIST_MUTE_FRAMES = 5;
const WRIST_HOLD_MS = 90;
const GAIN_TIME_CONSTANT = 0.045;
const GAIN_RELEASE_TIME_CONSTANT = 0.06;
const FREQ_TIME_CONSTANT = 0.03;
const FILTER_TIME_CONSTANT = 0.05;

const MIN_FREQ = 110;
const MAX_FREQ = 1400;
const VIBRATO_MIN = 5;
const VIBRATO_MAX = 15;
const FILTER_MIN = 600;
const FILTER_MAX = 9000;
const MAX_GAIN = 0.65;

// Instrument layout in units of shoulder width so it scales with performer size.
const PITCH_ROD_SW = 1.15;
const PITCH_NEAR_SW = 0.08;
const PITCH_FAR_SW = 1.05;
const VOLUME_LOOP_Y_SW = 1.05;
const VOLUME_NEAR_SW = 0.06;
const VOLUME_FAR_SW = 2.55;
// >1 keeps the near-loop region quieter longer, closer to a real volume field.
const VOLUME_RESPONSE_CURVE = 1.65;
const BODY_TOP_SW = 0.72;
const BODY_HEIGHT_SW = 0.55;
const VOLUME_END_SW = 0.75;
const ROD_HEIGHT_SW = 2.0;
const MIN_SHOULDER_WIDTH = 0.07;
const MAX_SHOULDER_WIDTH = 0.42;

// MediaPipe pose landmark indices
const LEFT_SHOULDER = 11;
const RIGHT_SHOULDER = 12;
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;

// Hand landmark indices
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_TIP = 12;
const PINKY_MCP = 17;

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

const POSE_CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

const canvas = document.getElementById("canvas");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const hud = document.getElementById("hud");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const poseOverlayToggle = document.getElementById("poseOverlayToggle");
const thereminOverlayToggle = document.getElementById("thereminOverlayToggle");
const reverseToggle = document.getElementById("reverseToggle");
const statusEl = document.getElementById("status");
const hudTip = document.querySelector(".hud-tip");
const ctx2d = canvas.getContext("2d");

/** @type {AudioContext | null} */
let audioCtx = null;
/** @type {PoseLandmarker | null} */
let poseLandmarker = null;
/** @type {HandLandmarker | null} */
let handLandmarker = null;
/** @type {Map<number, Performer>} */
const performers = new Map();
let nextPerformerId = 1;
let lastVideoTime = -1;
let running = false;
let showPoseOverlay = false;
let showThereminOverlay = false;
let reverseInstrument = false;
/** @type {MediaStream | null} */
let mediaStream = null;
/** @type {Array<{ left: any[] | null, right: any[] | null, torso: {x:number,y:number}, poseLm: any[] }> } */
let lastBundled = [];
/** @type {Map<number, number>} */
let lastAssignment = new Map();

const HUD_TIP_DEFAULT =
  "Right hand → pitch (closer to your right = higher). Left hand → volume (higher = louder). Hands below the instrument mute you only.";
const HUD_TIP_REVERSED =
  "Left hand → pitch (closer to your left = higher). Right hand → volume (higher = louder). Hands below the instrument mute you only.";

/**
 * @typedef {{
 *   id: number,
 *   torso: { x: number, y: number },
 *   shoulderWidth: number,
 *   lastSeen: number,
 *   osc: OscillatorNode,
 *   filter: BiquadFilterNode,
 *   gain: GainNode,
 *   lastFineSpan: number,
 *   lastVolume: number,
 *   lastFrequency: number,
 *   lastCutoff: number,
 *   lastPitchWrist: { x: number, y: number, z?: number } | null,
 *   lastVolumeWrist: { x: number, y: number, z?: number } | null,
 *   pitchWristSeenAt: number,
 *   volumeWristSeenAt: number,
 *   fistFrames: number,
 * }} Performer
 */

/**
 * @param {number} shoulderWidth
 */
function instrumentMetrics(shoulderWidth) {
  const s = clamp(shoulderWidth, MIN_SHOULDER_WIDTH, MAX_SHOULDER_WIDTH);
  return {
    s,
    pitchOffsetX: s * PITCH_ROD_SW,
    pitchNear: s * PITCH_NEAR_SW,
    pitchFar: s * PITCH_FAR_SW,
    volumeOffsetY: s * VOLUME_LOOP_Y_SW,
    volumeNear: s * VOLUME_NEAR_SW,
    volumeFar: s * VOLUME_FAR_SW,
    bodyTop: s * BODY_TOP_SW,
    bodyHeight: s * BODY_HEIGHT_SW,
    volumeEndX: s * VOLUME_END_SW,
    rodHeight: s * ROD_HEIGHT_SW,
  };
}

function landmarkVisible(lm, minVisibility = 0.45) {
  if (!lm) return false;
  if (typeof lm.visibility === "number" && lm.visibility < minVisibility) return false;
  return Number.isFinite(lm.x) && Number.isFinite(lm.y);
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function dist2d(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function dist3d(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.hypot(dx, dy, dz);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function triangleArea3d(a, b, c) {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const abz = (b.z ?? 0) - (a.z ?? 0);
  const acx = c.x - a.x;
  const acy = c.y - a.y;
  const acz = (c.z ?? 0) - (a.z ?? 0);
  const cx = aby * acz - abz * acy;
  const cy = abz * acx - abx * acz;
  const cz = abx * acy - aby * acx;
  return 0.5 * Math.hypot(cx, cy, cz);
}

function mirrorLandmarkX(landmark) {
  return { x: 1 - landmark.x, y: landmark.y, z: landmark.z };
}

function mirrorLandmarks(landmarks) {
  return landmarks.map(mirrorLandmarkX);
}

/** MediaPipe handedness is from the person's view; mirrored selfie view flips Left/Right. */
/** Prefer screen-side vs torso in mirrored space; MediaPipe labels are a fallback. */
function handSideLabel(mirroredWrist, torso, categoryName) {
  const dx = mirroredWrist.x - torso.x;
  if (Math.abs(dx) > 0.04) {
    return dx >= 0 ? "right" : "left";
  }
  // Near midline: trust MediaPipe. Unmirrored webcam input needs a label swap.
  if (categoryName === "Left") return "right";
  if (categoryName === "Right") return "left";
  return dx >= 0 ? "right" : "left";
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = window.innerWidth;
  const h = window.innerHeight;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawMirroredVideo() {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { drawW: 0, drawH: 0, offsetX: 0, offsetY: 0 };

  const canvasW = window.innerWidth;
  const canvasH = window.innerHeight;
  const scale = Math.max(canvasW / vw, canvasH / vh);
  const drawW = vw * scale;
  const drawH = vh * scale;
  const offsetX = (canvasW - drawW) / 2;
  const offsetY = (canvasH - drawH) / 2;

  ctx2d.save();
  ctx2d.translate(canvasW, 0);
  ctx2d.scale(-1, 1);
  ctx2d.drawImage(video, canvasW - offsetX - drawW, offsetY, drawW, drawH);
  ctx2d.restore();

  return { drawW, drawH, offsetX, offsetY };
}

function toCanvasPoint(landmark, layout) {
  return {
    x: layout.offsetX + landmark.x * layout.drawW,
    y: layout.offsetY + landmark.y * layout.drawH,
  };
}

function drawLandmarks(landmarks, layout, connections, color) {
  if (!landmarks?.length) return;

  ctx2d.strokeStyle = color;
  ctx2d.fillStyle = color;
  ctx2d.lineWidth = 1.5;
  ctx2d.globalAlpha = 0.55;

  for (const [a, b] of connections) {
    if (!landmarks[a] || !landmarks[b]) continue;
    const p1 = toCanvasPoint(landmarks[a], layout);
    const p2 = toCanvasPoint(landmarks[b], layout);
    ctx2d.beginPath();
    ctx2d.moveTo(p1.x, p1.y);
    ctx2d.lineTo(p2.x, p2.y);
    ctx2d.stroke();
  }

  for (const lm of landmarks) {
    const p = toCanvasPoint(lm, layout);
    ctx2d.beginPath();
    ctx2d.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
    ctx2d.fill();
  }

  ctx2d.globalAlpha = 1;
}

async function createLandmarkers(delegate) {
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  const pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_MODEL,
      delegate,
    },
    runningMode: "VIDEO",
    numPoses: MAX_POSES,
  });

  const hands = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL,
      delegate,
    },
    runningMode: "VIDEO",
    numHands: MAX_HANDS,
  });

  return { pose, hands };
}

async function initMediaPipe() {
  try {
    return await createLandmarkers("GPU");
  } catch (gpuError) {
    console.warn("GPU delegate failed, retrying with CPU", gpuError);
    setStatus("GPU unavailable — falling back to CPU…");
    return await createLandmarkers("CPU");
  }
}

async function initWebcam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access is not supported in this browser.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });

  mediaStream = stream;
  video.srcObject = stream;
  await video.play();
}

function stopMediaTracks() {
  if (mediaStream) {
    for (const track of mediaStream.getTracks()) track.stop();
    mediaStream = null;
  }
  video.srcObject = null;
}

function createPerformer(torso) {
  if (!audioCtx) throw new Error("AudioContext not ready");

  const osc = audioCtx.createOscillator();
  const filter = audioCtx.createBiquadFilter();
  const gain = audioCtx.createGain();

  osc.type = "sine";
  osc.frequency.value = MIN_FREQ;
  filter.type = "lowpass";
  filter.frequency.value = 3200;
  filter.Q.value = 0.5;
  gain.gain.value = 0;

  osc.connect(filter);
  filter.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();

  const performer = {
    id: nextPerformerId++,
    torso: { x: torso.x, y: torso.y },
    shoulderWidth: torso.shoulderWidth ?? 0.18,
    lastSeen: performance.now(),
    osc,
    filter,
    gain,
    lastFineSpan: 0,
    lastVolume: 0,
    lastFrequency: MIN_FREQ,
    lastCutoff: 3200,
    lastPitchWrist: null,
    lastVolumeWrist: null,
    pitchWristSeenAt: 0,
    volumeWristSeenAt: 0,
    fistFrames: 0,
  };

  performers.set(performer.id, performer);
  return performer;
}

function teardownPerformerNodes(performer) {
  try {
    performer.osc.stop();
  } catch {
    /* already stopped */
  }
  for (const node of [performer.osc, performer.filter, performer.gain]) {
    try {
      node.disconnect();
    } catch {
      /* ignore */
    }
  }
}

function destroyPerformer(performer, { immediate = false } = {}) {
  const now = audioCtx?.currentTime ?? 0;
  try {
    performer.gain.gain.cancelScheduledValues(now);
    performer.gain.gain.setTargetAtTime(0, now, immediate ? 0.001 : 0.015);
  } catch {
    /* ignore */
  }

  if (immediate) {
    teardownPerformerNodes(performer);
  } else {
    window.setTimeout(() => teardownPerformerNodes(performer), 80);
  }

  performers.delete(performer.id);
}

function matchPerformers(poseAnchors, now) {
  const unmatched = new Set(performers.keys());
  /** @type {Map<number, number>} poseIndex -> performerId */
  const assignment = new Map();

  const pairs = [];
  for (let i = 0; i < poseAnchors.length; i++) {
    for (const [id, performer] of performers) {
      pairs.push({
        poseIndex: i,
        id,
        d: dist2d(poseAnchors[i], performer.torso),
      });
    }
  }
  pairs.sort((a, b) => a.d - b.d);

  const usedPoses = new Set();
  for (const pair of pairs) {
    if (usedPoses.has(pair.poseIndex) || !unmatched.has(pair.id)) continue;
    if (pair.d > PERFORMER_MATCH_MAX_DIST) continue;
    usedPoses.add(pair.poseIndex);
    unmatched.delete(pair.id);
    assignment.set(pair.poseIndex, pair.id);
    const performer = performers.get(pair.id);
    const anchor = poseAnchors[pair.poseIndex];
    performer.torso = { x: anchor.x, y: anchor.y };
    if (anchor.shoulderWidth) {
      performer.shoulderWidth = lerp(performer.shoulderWidth, anchor.shoulderWidth, 0.35);
    }
    performer.lastSeen = now;
  }

  for (let i = 0; i < poseAnchors.length; i++) {
    if (assignment.has(i)) continue;
    const performer = createPerformer(poseAnchors[i]);
    assignment.set(i, performer.id);
  }

  for (const id of unmatched) {
    const performer = performers.get(id);
    const missingFor = now - performer.lastSeen;
    // Hold the last note through brief pose flicker; only fade after a grace period.
    if (audioCtx && missingFor > PERFORMER_SILENCE_MS) {
      performer.gain.gain.setTargetAtTime(
        0,
        audioCtx.currentTime,
        GAIN_RELEASE_TIME_CONSTANT
      );
      performer.lastVolume = 0;
    }
    if (missingFor > PERFORMER_TIMEOUT_MS) {
      destroyPerformer(performer);
    }
  }

  return assignment;
}

function associateHandsToPoses(poseResults, handResults) {
  const poses = poseResults.landmarks ?? [];
  /** @type {Array<{ left: any[] | null, right: any[] | null, leftWrist: any, rightWrist: any, torso: {x:number,y:number}, poseLm: any[] }>} */
  const bundled = poses.map((poseLm) => {
    const ls = poseLm[LEFT_SHOULDER];
    const rs = poseLm[RIGHT_SHOULDER];
    if (!landmarkVisible(ls, 0.3) || !landmarkVisible(rs, 0.3)) {
      return null;
    }
    const shoulderWidth = clamp(dist2d(ls, rs), MIN_SHOULDER_WIDTH, MAX_SHOULDER_WIDTH);
    const torso = {
      x: (ls.x + rs.x) / 2,
      y: (ls.y + rs.y) / 2,
      shoulderWidth,
    };
    return {
      left: null,
      right: null,
      // Pose wrists are reliable coarse controls when HandLandmarker misses a frame
      leftWrist: landmarkVisible(poseLm[LEFT_WRIST]) ? poseLm[LEFT_WRIST] : null,
      rightWrist: landmarkVisible(poseLm[RIGHT_WRIST]) ? poseLm[RIGHT_WRIST] : null,
      torso,
      shoulderWidth,
      poseLm,
    };
  }).filter(Boolean);

  const hands = handResults.landmarks ?? [];
  const handedness = handResults.handedness ?? [];

  for (let h = 0; h < hands.length; h++) {
    const mirrored = mirrorLandmarks(hands[h]);
    const wrist = mirrored[WRIST];
    let bestIdx = -1;
    let bestDist = Infinity;

    for (let p = 0; p < bundled.length; p++) {
      const d = dist2d(wrist, bundled[p].torso);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = p;
      }
    }

    if (bestIdx < 0 || bestDist > 0.55) continue;

    const rawLabel = handedness[h]?.[0]?.categoryName ?? "";
    const side = handSideLabel(wrist, bundled[bestIdx].torso, rawLabel);
    if (side === "left") {
      bundled[bestIdx].left = mirrored;
      bundled[bestIdx].leftWrist = wrist;
    } else {
      bundled[bestIdx].right = mirrored;
      bundled[bestIdx].rightWrist = wrist;
    }
  }

  return bundled;
}

function isFist(hand) {
  if (!hand?.[INDEX_TIP] || !hand?.[MIDDLE_TIP] || !hand?.[PINKY_MCP] || !hand?.[THUMB_TIP]) {
    return false;
  }
  const scale = dist3d(hand[WRIST], hand[MIDDLE_MCP]) + 1e-6;
  const tipReach =
    (dist3d(hand[WRIST], hand[INDEX_TIP]) +
      dist3d(hand[WRIST], hand[MIDDLE_TIP]) +
      dist3d(hand[WRIST], hand[THUMB_TIP])) /
    3 /
    scale;
  const palmArea = triangleArea3d(hand[WRIST], hand[INDEX_MCP], hand[PINKY_MCP]);
  const palmNorm = palmArea / (scale * scale);
  // Stricter than before so noisy hand tracking does not false-mute.
  return tipReach < 1.0 && palmNorm < 0.16;
}

function mapPitch(pitchWrist, pitchHand, torso, performer) {
  if (!pitchWrist) {
    return { frequency: MIN_FREQ, cutoff: 3200 };
  }

  const m = instrumentMetrics(performer.shoulderWidth);
  // Approach the vertical rod from the body side: closer => higher pitch.
  const side = reverseInstrument ? -1 : 1;
  const antennaX = torso.x + side * m.pitchOffsetX;
  // Signed distance from hand to rod along the approach axis (0 at the rod).
  const approach = side * (antennaX - pitchWrist.x);
  const distance = Math.max(0, approach);
  const nearness = 1 - clamp((distance - m.pitchNear) / (m.pitchFar - m.pitchNear), 0, 1);

  const octaves = Math.log2(MAX_FREQ / MIN_FREQ);
  let baseFreq = MIN_FREQ * Math.pow(2, nearness * octaves);

  let cutoff = 3200;

  if (pitchHand?.[THUMB_TIP] && pitchHand?.[INDEX_TIP] && pitchHand?.[MIDDLE_TIP]) {
    const thumbIndex = dist3d(pitchHand[THUMB_TIP], pitchHand[INDEX_TIP]);
    const handExt = dist3d(pitchHand[WRIST], pitchHand[MIDDLE_TIP]);
    const fineSpan = thumbIndex * 0.65 + handExt * 0.35;

    const spanNorm = clamp((fineSpan - 0.03) / 0.16, 0, 1);
    const intervalCents = (spanNorm - 0.5) * 250;
    baseFreq *= Math.pow(2, intervalCents / 1200);

    const spanDelta = fineSpan - performer.lastFineSpan;
    performer.lastFineSpan = fineSpan;
    const vibratoDepth = lerp(VIBRATO_MIN, VIBRATO_MAX, clamp(Math.abs(spanDelta) / 0.02, 0, 1));
    const vibrato =
      Math.sin(performance.now() * 0.035) *
      vibratoDepth *
      clamp(Math.abs(spanDelta) / 0.008, 0, 1);
    baseFreq += vibrato;

    const mcp = pitchHand[MIDDLE_MCP];
    const wrist = pitchHand[WRIST];
    const vx = mcp.x - wrist.x;
    const vy = mcp.y - wrist.y;
    const angleFromVertical = Math.atan2(vx, -vy);
    const tiltNorm = clamp(Math.abs(angleFromVertical) / (Math.PI / 2), 0, 1);
    // Neutral upright hand stays fairly bright; extreme tilt opens the filter.
    cutoff = lerp(FILTER_MIN, FILTER_MAX, 0.35 + tiltNorm * 0.65);
  }

  const frequency = clamp(baseFreq, MIN_FREQ * 0.9, MAX_FREQ * 1.1);
  return { frequency, cutoff };
}

function handAboveDevice(wrist, torso, shoulderWidth) {
  if (!wrist) return false;
  const m = instrumentMetrics(shoulderWidth ?? 0.18);
  // Image Y grows downward; above the cabinet means a smaller Y than the body top.
  return wrist.y < torso.y + m.bodyTop;
}

function mapVolume(volumeWrist, volumeHand, torso, fistMuted, shoulderWidth) {
  if (!volumeWrist) return 0;
  if (!handAboveDevice(volumeWrist, torso, shoulderWidth)) return 0;

  const m = instrumentMetrics(shoulderWidth);
  // Higher above the horizontal loop => louder (wide vertical span like a real theremin).
  const antennaY = torso.y + m.volumeOffsetY;
  const heightAbove = antennaY - volumeWrist.y;
  if (heightAbove <= m.volumeNear) return 0;
  const t = clamp((heightAbove - m.volumeNear) / (m.volumeFar - m.volumeNear), 0, 1);
  const heightGain = Math.pow(t, VOLUME_RESPONSE_CURVE);

  if (fistMuted) return 0;

  let openness = 1;
  if (volumeHand?.[INDEX_MCP] && volumeHand?.[PINKY_MCP]) {
    const scale = dist3d(volumeHand[WRIST], volumeHand[MIDDLE_MCP]) + 1e-6;
    const palmArea = triangleArea3d(
      volumeHand[WRIST],
      volumeHand[INDEX_MCP],
      volumeHand[PINKY_MCP]
    );
    openness = clamp(palmArea / (scale * scale * 0.35), 0.55, 1);
  }

  return clamp(heightGain * openness * MAX_GAIN, 0, MAX_GAIN);
}

function resolveWrist(current, performer, which) {
  const seenKey = which === "pitch" ? "pitchWristSeenAt" : "volumeWristSeenAt";
  const lastKey = which === "pitch" ? "lastPitchWrist" : "lastVolumeWrist";
  const wallNow = performance.now();

  if (landmarkVisible(current)) {
    performer[lastKey] = { x: current.x, y: current.y, z: current.z };
    performer[seenKey] = wallNow;
    return current;
  }

  // Only bridge tiny flicker gaps — out-of-frame hands must go silent.
  if (performer[lastKey] && wallNow - performer[seenKey] < WRIST_HOLD_MS) {
    return performer[lastKey];
  }

  performer[lastKey] = null;
  return null;
}

function updatePerformerAudio(performer, person) {
  if (!audioCtx) return;
  const now = audioCtx.currentTime;
  const pitchHand = reverseInstrument ? person.left : person.right;
  const volumeHand = reverseInstrument ? person.right : person.left;
  const pitchWrist = resolveWrist(
    reverseInstrument ? person.leftWrist : person.rightWrist,
    performer,
    "pitch"
  );
  const volumeWrist = resolveWrist(
    reverseInstrument ? person.rightWrist : person.leftWrist,
    performer,
    "volume"
  );

  if (volumeHand && isFist(volumeHand)) {
    performer.fistFrames += 1;
  } else {
    performer.fistFrames = Math.max(0, performer.fistFrames - 2);
  }
  const fistMuted = performer.fistFrames >= FIST_MUTE_FRAMES;

  const { frequency, cutoff } = mapPitch(pitchWrist, pitchHand, performer.torso, performer);

  // No tone unless both hands are tracked and above the theremin body.
  let volume = 0;
  const handsReady =
    pitchWrist &&
    volumeWrist &&
    handAboveDevice(pitchWrist, performer.torso, performer.shoulderWidth) &&
    handAboveDevice(volumeWrist, performer.torso, performer.shoulderWidth);

  if (handsReady) {
    volume = mapVolume(
      volumeWrist,
      volumeHand,
      performer.torso,
      fistMuted,
      performer.shoulderWidth
    );
  }

  performer.lastFrequency = frequency;
  performer.lastCutoff = cutoff;
  performer.lastVolume = volume;

  const gainTau =
    volume < performer.gain.gain.value ? GAIN_RELEASE_TIME_CONSTANT : GAIN_TIME_CONSTANT;
  performer.osc.frequency.setTargetAtTime(frequency, now, FREQ_TIME_CONSTANT);
  performer.filter.frequency.setTargetAtTime(cutoff, now, FILTER_TIME_CONSTANT);
  performer.gain.gain.setTargetAtTime(volume, now, gainTau);
}

function fillRoundRect(x, y, w, h, r) {
  ctx2d.beginPath();
  if (typeof ctx2d.roundRect === "function") {
    ctx2d.roundRect(x, y, w, h, r);
  } else {
    ctx2d.rect(x, y, w, h);
  }
  ctx2d.fill();
}

function strokeRoundRect(x, y, w, h, r) {
  ctx2d.beginPath();
  if (typeof ctx2d.roundRect === "function") {
    ctx2d.roundRect(x, y, w, h, r);
  } else {
    ctx2d.rect(x, y, w, h);
  }
  ctx2d.stroke();
}

function drawChromeKnob(cx, cy, r) {
  const g = ctx2d.createRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 0.1, cx, cy, r);
  g.addColorStop(0, "rgba(250, 250, 255, 0.95)");
  g.addColorStop(0.45, "rgba(190, 195, 205, 0.95)");
  g.addColorStop(1, "rgba(90, 95, 105, 0.95)");
  ctx2d.fillStyle = g;
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, r, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.strokeStyle = "rgba(40, 40, 45, 0.7)";
  ctx2d.lineWidth = 1;
  ctx2d.stroke();
  ctx2d.fillStyle = "rgba(20, 20, 24, 0.85)";
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, r * 0.28, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.strokeStyle = "rgba(30, 30, 35, 0.9)";
  ctx2d.beginPath();
  ctx2d.moveTo(cx, cy);
  ctx2d.lineTo(cx + r * 0.15, cy - r * 0.75);
  ctx2d.stroke();
}

function drawThereminAntennas(layout, torso, shoulderWidth) {
  // Moog Etherwave proportions scaled to this performer's shoulder width.
  const pitchSide = reverseInstrument ? -1 : 1;
  const volumeSide = -pitchSide;
  const m = instrumentMetrics(shoulderWidth);

  const antennaX = torso.x + pitchSide * m.pitchOffsetX;
  const volumeEndX = torso.x + volumeSide * m.volumeEndX;
  const pitchEndX = antennaX + pitchSide * m.s * 0.12;
  const bodyLeftX = Math.min(volumeEndX, pitchEndX);
  const bodyRightX = Math.max(volumeEndX, pitchEndX);
  const bodyTopY = torso.y + m.bodyTop;
  const bodyBotY = bodyTopY + m.bodyHeight;

  const tl = toCanvasPoint({ x: bodyLeftX, y: bodyTopY }, layout);
  const br = toCanvasPoint({ x: bodyRightX, y: bodyBotY }, layout);
  const cabX = Math.min(tl.x, br.x);
  const cabY = Math.min(tl.y, br.y);
  const cabW = Math.abs(br.x - tl.x);
  const cabH = Math.abs(br.y - tl.y);
  const depth = Math.max(6, cabH * 0.28);
  const strokeScale = clamp(cabW / 220, 0.55, 1.6);

  const rodBase = toCanvasPoint({ x: antennaX, y: bodyTopY }, layout);
  const rodTop = toCanvasPoint({ x: antennaX, y: torso.y - m.rodHeight }, layout);

  // Volume loop: two horizontal rails + asymmetric rounded end (Theremini / Etherwave).
  const loopAttachX = volumeSide < 0 ? cabX : cabX + cabW;
  const loopCy = cabY + cabH * 0.55;
  const railHalf = Math.max(10, cabH * 0.36);
  const railLen = Math.max(22, cabH * 1.05);
  const endRx = Math.max(14, cabH * 0.72);
  const endRy = railHalf * 1.08;
  const topY = loopCy - railHalf;
  const botY = loopCy + railHalf;
  const railEndX = loopAttachX + volumeSide * railLen;
  // Tip sits a touch low so the end reads as an asymmetric semicircle, not a perfect U.
  const tipX = railEndX + volumeSide * endRx;
  const tipY = loopCy + railHalf * 0.18;

  ctx2d.save();
  ctx2d.globalAlpha = 0.92;
  ctx2d.lineCap = "round";
  ctx2d.lineJoin = "round";

  // --- Chrome volume loop (draw behind body so the mounts tuck in) ---
  const loopMetal = ctx2d.createLinearGradient(
    tipX,
    topY,
    loopAttachX,
    botY
  );
  loopMetal.addColorStop(0, "rgba(245, 245, 250, 0.95)");
  loopMetal.addColorStop(0.35, "rgba(170, 175, 185, 0.95)");
  loopMetal.addColorStop(0.7, "rgba(220, 225, 235, 0.95)");
  loopMetal.addColorStop(1, "rgba(120, 125, 135, 0.9)");

  /** Two rails + asymmetric end cap as one continuous stroke. */
  const strokeVolumeLoop = (style, width) => {
    ctx2d.strokeStyle = style;
    ctx2d.lineWidth = width;
    ctx2d.beginPath();
    ctx2d.moveTo(loopAttachX, topY);
    ctx2d.lineTo(railEndX, topY);
    // Asymmetric semicircle: wider outward bulge, slightly dropped tip.
    ctx2d.bezierCurveTo(
      railEndX + volumeSide * endRx * 0.55,
      topY,
      tipX,
      tipY - endRy * 0.85,
      tipX,
      tipY
    );
    ctx2d.bezierCurveTo(
      tipX,
      tipY + endRy * 0.95,
      railEndX + volumeSide * endRx * 0.5,
      botY,
      railEndX,
      botY
    );
    ctx2d.lineTo(loopAttachX, botY);
    ctx2d.stroke();
  };

  strokeVolumeLoop("rgba(60, 65, 75, 0.35)", 9 * strokeScale);
  strokeVolumeLoop(loopMetal, 5.5 * strokeScale);

  // Specular highlight on the outer tip
  ctx2d.strokeStyle = "rgba(255, 255, 255, 0.55)";
  ctx2d.lineWidth = 1.5 * strokeScale;
  ctx2d.beginPath();
  ctx2d.moveTo(railEndX + volumeSide * endRx * 0.15, topY + 1);
  ctx2d.bezierCurveTo(
    railEndX + volumeSide * endRx * 0.55,
    topY + 1,
    tipX - volumeSide * 1,
    tipY - endRy * 0.7,
    tipX - volumeSide * 1,
    tipY
  );
  ctx2d.stroke();

  // --- Wooden cabinet (top face + front face for a slight 3D box) ---
  const woodTop = ctx2d.createLinearGradient(cabX, cabY - depth, cabX, cabY);
  woodTop.addColorStop(0, "rgba(196, 120, 58, 0.88)");
  woodTop.addColorStop(1, "rgba(160, 92, 42, 0.9)");
  ctx2d.fillStyle = woodTop;
  ctx2d.beginPath();
  ctx2d.moveTo(cabX + 4, cabY - depth);
  ctx2d.lineTo(cabX + cabW - 4, cabY - depth);
  ctx2d.lineTo(cabX + cabW, cabY);
  ctx2d.lineTo(cabX, cabY);
  ctx2d.closePath();
  ctx2d.fill();

  const woodFront = ctx2d.createLinearGradient(cabX, cabY, cabX, cabY + cabH);
  woodFront.addColorStop(0, "rgba(186, 108, 48, 0.92)");
  woodFront.addColorStop(0.45, "rgba(158, 88, 38, 0.94)");
  woodFront.addColorStop(1, "rgba(118, 62, 28, 0.95)");
  ctx2d.fillStyle = woodFront;
  fillRoundRect(cabX, cabY, cabW, cabH, 5);
  ctx2d.strokeStyle = "rgba(70, 36, 16, 0.75)";
  ctx2d.lineWidth = 1.25;
  strokeRoundRect(cabX, cabY, cabW, cabH, 5);

  // Subtle wood grain lines
  ctx2d.strokeStyle = "rgba(90, 48, 20, 0.12)";
  ctx2d.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const gy = cabY + (cabH * i) / 5;
    ctx2d.beginPath();
    ctx2d.moveTo(cabX + 8, gy);
    ctx2d.lineTo(cabX + cabW - 8, gy + (i % 2 === 0 ? 1.5 : -1));
    ctx2d.stroke();
  }

  // --- Black control panel with knobs (Etherwave-style) ---
  const panelX = cabX + cabW * 0.08;
  const panelY = cabY + cabH * 0.18;
  const panelW = cabW * 0.84;
  const panelH = cabH * 0.58;
  ctx2d.fillStyle = "rgba(18, 18, 20, 0.92)";
  fillRoundRect(panelX, panelY, panelW, panelH, 3);
  ctx2d.strokeStyle = "rgba(0, 0, 0, 0.5)";
  ctx2d.lineWidth = 1;
  strokeRoundRect(panelX, panelY, panelW, panelH, 3);

  // moog wordmark
  ctx2d.fillStyle = "rgba(235, 235, 240, 0.9)";
  ctx2d.font = `600 ${Math.max(9, panelH * 0.28)}px "Avenir Next", "Segoe UI", sans-serif`;
  ctx2d.textBaseline = "middle";
  ctx2d.fillText("moog", panelX + panelW * 0.04, panelY + panelH * 0.45);

  // Power rocker + jack suggestion
  const switchX = panelX + panelW * 0.22;
  const switchY = panelY + panelH * 0.35;
  ctx2d.fillStyle = "rgba(40, 40, 48, 0.95)";
  fillRoundRect(switchX, switchY, panelH * 0.35, panelH * 0.38, 2);
  ctx2d.fillStyle = "rgba(200, 200, 210, 0.85)";
  fillRoundRect(switchX + 2, switchY + 2, panelH * 0.35 - 4, panelH * 0.16, 1);

  ctx2d.fillStyle = "rgba(30, 30, 35, 0.95)";
  ctx2d.beginPath();
  ctx2d.arc(switchX + panelW * 0.09, panelY + panelH * 0.52, panelH * 0.12, 0, Math.PI * 2);
  ctx2d.fill();
  ctx2d.strokeStyle = "rgba(160, 160, 170, 0.5)";
  ctx2d.lineWidth = 1;
  ctx2d.stroke();

  // Four silver knobs
  const knobR = Math.max(4.5, panelH * 0.22);
  const knobYs = panelY + panelH * 0.55;
  const knobStart = panelX + panelW * 0.42;
  const knobStep = panelW * 0.13;
  for (let i = 0; i < 4; i++) {
    drawChromeKnob(knobStart + i * knobStep, knobYs, knobR);
  }

  // Tiny labels under branding area
  ctx2d.fillStyle = "rgba(200, 200, 210, 0.55)";
  ctx2d.font = `${Math.max(6, panelH * 0.14)}px "Avenir Next", "Segoe UI", sans-serif`;
  ctx2d.fillText("ETHERWAVE", panelX + panelW * 0.72, panelY + panelH * 0.22);

  // --- Tall chrome pitch rod ---
  const rodX = rodBase.x;
  const rodGrad = ctx2d.createLinearGradient(rodX - 5, 0, rodX + 5, 0);
  rodGrad.addColorStop(0, "rgba(110, 115, 125, 0.85)");
  rodGrad.addColorStop(0.35, "rgba(245, 245, 250, 0.98)");
  rodGrad.addColorStop(0.55, "rgba(200, 205, 215, 0.95)");
  rodGrad.addColorStop(1, "rgba(100, 105, 115, 0.9)");

  // Base collar on the cabinet top
  const collarRx = Math.max(3.5, 6 * strokeScale);
  const collarRy = Math.max(2, 3.5 * strokeScale);
  ctx2d.fillStyle = "rgba(180, 185, 195, 0.95)";
  ctx2d.beginPath();
  ctx2d.ellipse(rodX, cabY - depth * 0.15, collarRx, collarRy, 0, 0, Math.PI * 2);
  ctx2d.fill();

  ctx2d.strokeStyle = "rgba(40, 40, 50, 0.25)";
  ctx2d.lineWidth = 7 * strokeScale;
  ctx2d.beginPath();
  ctx2d.moveTo(rodX, cabY - depth * 0.1);
  ctx2d.lineTo(rodX, rodTop.y);
  ctx2d.stroke();

  ctx2d.strokeStyle = rodGrad;
  ctx2d.lineWidth = 3.75 * strokeScale;
  ctx2d.beginPath();
  ctx2d.moveTo(rodX, cabY - depth * 0.1);
  ctx2d.lineTo(rodX, rodTop.y);
  ctx2d.stroke();

  // Tip
  const tipR = Math.max(2.2, 3.2 * strokeScale);
  const tipG = ctx2d.createRadialGradient(rodX - 1, rodTop.y - 1, 0.5, rodX, rodTop.y, tipR);
  tipG.addColorStop(0, "rgba(255, 255, 255, 0.95)");
  tipG.addColorStop(1, "rgba(140, 145, 155, 0.9)");
  ctx2d.fillStyle = tipG;
  ctx2d.beginPath();
  ctx2d.arc(rodX, rodTop.y, tipR, 0, Math.PI * 2);
  ctx2d.fill();

  // Soft stand shadow under cabinet
  ctx2d.fillStyle = "rgba(0, 0, 0, 0.18)";
  ctx2d.beginPath();
  ctx2d.ellipse(cabX + cabW / 2, cabY + cabH + 6, cabW * 0.28, 5, 0, 0, Math.PI * 2);
  ctx2d.fill();

  ctx2d.restore();
}

function drawPerformers(layout, bundled, assignment) {
  for (let i = 0; i < bundled.length; i++) {
    const performerId = assignment.get(i);
    const performer = performers.get(performerId);
    if (!performer) continue;

    const person = bundled[i];
    if (person.shoulderWidth) {
      performer.shoulderWidth = lerp(performer.shoulderWidth, person.shoulderWidth, 0.3);
    }
    updatePerformerAudio(performer, person);

    if (showThereminOverlay) {
      drawThereminAntennas(layout, performer.torso, performer.shoulderWidth);
    }

    if (!showPoseOverlay) continue;

    drawLandmarks(person.poseLm, layout, POSE_CONNECTIONS, "rgba(220, 220, 220, 0.7)");
    if (person.left) {
      drawLandmarks(person.left, layout, HAND_CONNECTIONS, "rgba(160, 210, 255, 0.75)");
    }
    if (person.right) {
      drawLandmarks(person.right, layout, HAND_CONNECTIONS, "rgba(255, 200, 150, 0.75)");
    }

    const torsoPt = toCanvasPoint(performer.torso, layout);
    ctx2d.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx2d.beginPath();
    ctx2d.arc(torsoPt.x, torsoPt.y, 4, 0, Math.PI * 2);
    ctx2d.fill();
  }
}

function processFrame() {
  if (!running) return;
  requestAnimationFrame(processFrame);

  if (audioCtx?.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }

  if (!poseLandmarker || !handLandmarker) return;
  if (video.readyState < 2) return;

  resizeCanvas();
  const layout = drawMirroredVideo();
  if (!layout.drawW) return;

  const timestamp = performance.now();

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    let poseResults;
    let handResults;
    try {
      poseResults = poseLandmarker.detectForVideo(video, timestamp);
      handResults = handLandmarker.detectForVideo(video, timestamp);
    } catch (err) {
      console.error(err);
      return;
    }

    // Mirror pose landmarks to match mirrored canvas / selfie interaction
    const mirroredPose = {
      landmarks: (poseResults.landmarks ?? []).map(mirrorLandmarks),
    };
    lastBundled = associateHandsToPoses(mirroredPose, handResults);
    const anchors = lastBundled.map((b) => b.torso);
    lastAssignment = matchPerformers(anchors, timestamp);
  } else {
    // Age out / soft-silence performers even between detection ticks
    const stale = [];
    for (const performer of performers.values()) {
      const missingFor = timestamp - performer.lastSeen;
      if (audioCtx && missingFor > PERFORMER_SILENCE_MS && performer.lastVolume > 0) {
        performer.gain.gain.setTargetAtTime(
          0,
          audioCtx.currentTime,
          GAIN_RELEASE_TIME_CONSTANT
        );
        performer.lastVolume = 0;
      }
      if (missingFor > PERFORMER_TIMEOUT_MS) {
        stale.push(performer);
      }
    }
    for (const performer of stale) destroyPerformer(performer);
  }

  drawPerformers(layout, lastBundled, lastAssignment);
}

async function start() {
  startBtn.disabled = true;
  setStatus("Starting audio and camera…");

  try {
    audioCtx = new AudioContext();
    if (audioCtx.state === "suspended") {
      await audioCtx.resume();
    }
  } catch (err) {
    console.error(err);
    setStatus("Could not start audio. Check browser permissions and try again.", true);
    startBtn.disabled = false;
    return;
  }

  try {
    setStatus("Loading vision models…");
    const models = await initMediaPipe();
    poseLandmarker = models.pose;
    handLandmarker = models.hands;
  } catch (err) {
    console.error(err);
    setStatus("Failed to load MediaPipe models. Check your network and reload.", true);
    startBtn.disabled = false;
    return;
  }

  try {
    setStatus("Requesting camera…");
    await initWebcam();
  } catch (err) {
    console.error(err);
    const msg =
      err?.name === "NotAllowedError"
        ? "Camera permission denied. Allow camera access and try again."
        : err?.name === "NotFoundError"
          ? "No camera found. Connect a webcam and try again."
          : "Could not access the camera. Check permissions and try again.";
    setStatus(msg, true);
    startBtn.disabled = false;
    return;
  }

  overlay.classList.add("hidden");
  hud.classList.remove("hidden");
  setStatus("");
  running = true;
  resizeCanvas();
  requestAnimationFrame(processFrame);
}

async function stop() {
  running = false;
  lastVideoTime = -1;
  lastBundled = [];
  lastAssignment = new Map();

  for (const performer of [...performers.values()]) {
    destroyPerformer(performer, { immediate: true });
  }

  stopMediaTracks();

  if (poseLandmarker) {
    try {
      poseLandmarker.close();
    } catch {
      /* ignore */
    }
    poseLandmarker = null;
  }
  if (handLandmarker) {
    try {
      handLandmarker.close();
    } catch {
      /* ignore */
    }
    handLandmarker = null;
  }

  if (audioCtx) {
    try {
      await audioCtx.close();
    } catch {
      /* ignore */
    }
    audioCtx = null;
  }

  ctx2d.setTransform(1, 0, 0, 1, 0, 0);
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);

  hud.classList.add("hidden");
  overlay.classList.remove("hidden");
  startBtn.disabled = false;
  setStatus("Stopped. Click Start Audio when you are ready again.");
}

function syncHudTip() {
  if (hudTip) {
    hudTip.textContent = reverseInstrument ? HUD_TIP_REVERSED : HUD_TIP_DEFAULT;
  }
}

window.addEventListener("resize", () => {
  if (running) resizeCanvas();
});

startBtn.addEventListener("click", () => {
  start();
});

stopBtn.addEventListener("click", () => {
  stop();
});

poseOverlayToggle.addEventListener("change", () => {
  showPoseOverlay = poseOverlayToggle.checked;
});

thereminOverlayToggle.addEventListener("change", () => {
  showThereminOverlay = thereminOverlayToggle.checked;
});

reverseToggle.addEventListener("change", () => {
  reverseInstrument = reverseToggle.checked;
  syncHudTip();
});

showPoseOverlay = poseOverlayToggle.checked;
showThereminOverlay = thereminOverlayToggle.checked;
reverseInstrument = reverseToggle.checked;
syncHudTip();
