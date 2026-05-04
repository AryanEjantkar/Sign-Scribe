let detector;
let video;
let canvas;
let ctx;
let lastDetected = null;
let currentSentence = "";
let isReady = false;
let predictionBuffer = []; // For temporal smoothing
const BUFFER_SIZE = 10;

const UI = {
  char: document.getElementById("detected-char"),
  sentence: document.getElementById("sentence-display"),
  confidence: document.getElementById("confidence-fill"),
  loading: document.getElementById("loading-overlay")
};

/* ---------- INITIALIZATION ---------- */
async function init() {
  video = document.getElementById("webcam");
  canvas = document.getElementById("output");
  ctx = canvas.getContext("2d");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await new Promise(resolve => video.onloadedmetadata = resolve);
    video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig = {
      runtime: 'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands',
      modelType: 'full'
    };
    detector = await handPoseDetection.createDetector(model, detectorConfig);

    UI.loading.style.display = "none";
    isReady = true;
    runDetection();
  } catch (err) {
    console.error("Error initializing:", err);
    alert("Camera access denied or hardware error.");
  }
}

/* ---------- GESTURE CLASSIFIER ---------- */
// Simple geometric classifier based on landmark relationships
function classifyGesture(landmarks) {
  // Helper: Vector math
  const getVector = (p1, p2) => ({ x: p2.x - p1.x, y: p2.y - p1.y, z: p2.z - p1.z });
  const getDotProduct = (v1, v2) => v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const getMagnitude = (v) => Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  const getAngle = (v1, v2) => Math.acos(getDotProduct(v1, v2) / (getMagnitude(v1) * getMagnitude(v2))) * (180 / Math.PI);
  const getDistance = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

  const handSize = getDistance(landmarks[0], landmarks[9]);
  const normDist = (p1, p2) => getDistance(p1, p2) / handSize;

  // Finger Curl Analysis (Angle-based)
  // Check angle between MCP->PIP and PIP->TIP vectors
  const getCurl = (mcp, pip, tip) => {
    const v1 = getVector(mcp, pip);
    const v2 = getVector(pip, tip);
    return getAngle(v1, v2);
  };

  const indexCurl = getCurl(landmarks[5], landmarks[6], landmarks[8]);
  const middleCurl = getCurl(landmarks[9], landmarks[10], landmarks[12]);
  const ringCurl = getCurl(landmarks[13], landmarks[14], landmarks[16]);
  const pinkyCurl = getCurl(landmarks[17], landmarks[18], landmarks[20]);
  
  const isStraight = (curl) => curl < 35;
  const isCurled = (curl) => curl > 90;

  const indexUp = isStraight(indexCurl);
  const middleUp = isStraight(middleCurl);
  const ringUp = isStraight(ringCurl);
  const pinkyUp = isStraight(pinkyCurl);

  const thumbTip = landmarks[4];
  const thumbOut = thumbTip.x < landmarks[2].x - (handSize * 0.15);
  const thumbAcross = thumbTip.x > landmarks[9].x;

  // --- ASL CLASSIFICATION (Advanced Vector Version) ---
  
  // B: All straight and touching
  if (indexUp && middleUp && ringUp && pinkyUp) {
    if (normDist(landmarks[8], landmarks[12]) < 0.15) return "B";
    return "Open Hand";
  }

  // D: Index up, others curled
  if (indexUp && !middleUp && !ringUp && !pinkyUp) return "D";

  // V, U, R, K: Index & Middle up variants
  if (indexUp && middleUp && !ringUp && !pinkyUp) {
    const tipDist = normDist(landmarks[8], landmarks[12]);
    if (tipDist < 0.08) return "R"; // Fingers crossed/touching
    if (normDist(thumbTip, landmarks[10]) < 0.25) return "K"; // Thumb at middle joint
    if (tipDist < 0.2) return "U";
    return "V";
  }

  // W: Index, Middle, Ring up
  if (indexUp && middleUp && ringUp && !pinkyUp) return "W";

  // L: Index up, thumb out
  if (indexUp && thumbOut && !middleUp && !ringUp && !pinkyUp) return "L";

  // Y: Thumb and Pinky out
  if (pinkyUp && thumbOut && !indexUp && !middleUp && !ringUp) return "Y";

  // I: Pinky up
  if (pinkyUp && !indexUp && !middleUp && !ringUp && !thumbOut) return "I";

  // F: Index and thumb touching, others up
  if (!indexUp && middleUp && ringUp && pinkyUp && normDist(thumbTip, landmarks[8]) < 0.2) return "F";

  // O and C (Circular shapes)
  const distThumbIndex = normDist(thumbTip, landmarks[8]);
  if (!indexUp && !middleUp && !ringUp && !pinkyUp) {
    if (distThumbIndex < 0.25) return "O";
    if (distThumbIndex < 0.5) return "C";
    
    // M, N, S, A (Fist variants)
    if (thumbTip.x > landmarks[13].x) return "M"; // Thumb tucked under 3
    if (thumbTip.x > landmarks[9].x) return "N";  // Thumb tucked under 2
    if (thumbTip.y < landmarks[6].y && thumbTip.x < landmarks[6].x) return "A";
    if (thumbAcross) return "S";
    return "E";
  }

  return "Detecting...";
}

/* ---------- DETECTION LOOP ---------- */
async function runDetection() {
  if (!isReady) return;

  const hands = await detector.estimateHands(video, { flipHorizontal: false });
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (hands.length > 0) {
    const hand = hands[0];
    drawHand(hand.keypoints);
    
    const gesture = classifyGesture(hand.keypoints);
    
    // Temporal Smoothing (Buffer voting)
    predictionBuffer.push(gesture);
    if (predictionBuffer.length > BUFFER_SIZE) predictionBuffer.shift();
    
    const counts = {};
    predictionBuffer.forEach(g => counts[g] = (counts[g] || 0) + 1);
    const stableGesture = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);

    UI.char.innerText = stableGesture;
    
    // Confidence based on buffer agreement
    const confidence = (counts[stableGesture] / predictionBuffer.length) * 100;
    UI.confidence.style.width = `${confidence}%`;
    
    if (stableGesture !== "Detecting..." && stableGesture !== "Open Hand" && stableGesture.length === 1) {
      if (lastDetected !== stableGesture && confidence > 80) {
        lastDetected = stableGesture;
      }
    }
  } else {
    UI.char.innerText = "—";
    UI.confidence.style.width = "0%";
    lastDetected = null;
  }

  requestAnimationFrame(runDetection);
}

function drawHand(keypoints) {
  ctx.fillStyle = "#6366f1";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
  ctx.lineWidth = 2;

  // Draw joints
  for (const kp of keypoints) {
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Draw connections (Simplified)
  const fingers = [
    [0, 1, 2, 3, 4], // thumb
    [0, 5, 6, 7, 8], // index
    [0, 9, 10, 11, 12], // middle
    [0, 13, 14, 15, 16], // ring
    [0, 17, 18, 19, 20] // pinky
  ];

  for (const finger of fingers) {
    ctx.beginPath();
    ctx.moveTo(keypoints[finger[0]].x, keypoints[finger[0]].y);
    for (let i = 1; i < finger.length; i++) {
      ctx.lineTo(keypoints[finger[i]].x, keypoints[finger[i]].y);
    }
    ctx.stroke();
  }
}

/* ---------- UI ACTIONS ---------- */
document.getElementById("btn-space").onclick = () => {
  currentSentence += " ";
  updateDisplay();
};

document.getElementById("btn-backspace").onclick = () => {
  currentSentence = currentSentence.slice(0, -1);
  updateDisplay();
};

document.getElementById("btn-clear").onclick = () => {
  currentSentence = "";
  updateDisplay();
};

document.getElementById("btn-speak").onclick = () => {
  if (!currentSentence) return;
  const utterance = new SpeechSynthesisUtterance(currentSentence);
  window.speechSynthesis.speak(utterance);
};

// Add detected character to sentence on click/hold logic
UI.char.onclick = () => {
  const char = UI.char.innerText;
  if (char && char.length === 1) {
    currentSentence += char;
    updateDisplay();
  }
};

function updateDisplay() {
  UI.sentence.innerText = currentSentence || "Start signing to see text...";
}

init();
