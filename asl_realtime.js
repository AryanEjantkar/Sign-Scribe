let model = null;
let video = null;
let isRunning = false;
let rafId = null;
let LABELS = [];

/* ---------- CAMERA ---------- */
async function setupCamera() {
  video = document.getElementById("webcam");

  if (video.srcObject) return;

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  return new Promise(resolve => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });
}

/* ---------- LOAD MODEL ---------- */
async function loadModel() {
  if (model) return;

  await tf.ready(); // 🔥 REQUIRED
  console.log("TensorFlow.js ready");

  console.log("Loading model...");
  model = await tf.loadLayersModel("./model/model.json");
  console.log("Model loaded successfully");

  model.summary();
}

/* ---------- LOAD LABELS ---------- */
async function loadLabels() {
  if (LABELS.length > 0) return;

  const res = await fetch("./model/labels.json");
  LABELS = await res.json();

  console.log("Labels loaded:", LABELS);
}

/* ---------- PREPROCESS ---------- */
function preprocessFrame(video) {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(video);

    // 🔥 CRITICAL CHECK
    if (img.shape[0] === 0 || img.shape[1] === 0) {
      img.dispose();
      return null;
    }

    return img
      .resizeBilinear([64, 64])
      .toFloat()
      .div(255.0)
      .expandDims(0);
  });
}

/* ---------- PREDICT LOOP ---------- */
function predictLoop() {
  if (!isRunning || !model || video.readyState !== 4) {
    rafId = requestAnimationFrame(predictLoop);
    return;
  }

  const input = preprocessFrame(video);
  if (!input) {
    rafId = requestAnimationFrame(predictLoop);
    return;
  }

  const prediction = model.predict(input);
  const probs = prediction.dataSync();

  const index = probs.indexOf(Math.max(...probs));
  const confidence = (probs[index] * 100).toFixed(2);

  document.getElementById("result").innerText =
    `Prediction: ${LABELS[index]} (${confidence}%)`;

  tf.dispose([input, prediction]);

  rafId = requestAnimationFrame(predictLoop);
}

/* ---------- START ---------- */
async function startASL() {
  if (isRunning) return;

  isRunning = true;

  await setupCamera();
  await loadModel();
  await loadLabels();

  predictLoop();
}

/* ---------- STOP ---------- */
function stopASL() {
  isRunning = false;

  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }

  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  document.getElementById("result").innerText = "Prediction: —";
}
