const FACE_ORDER = ["U", "R", "F", "D", "L", "B"];
const FACE_NAMES = {
  U: "WHITE center (U)",
  R: "RED center (R)",
  F: "GREEN center (F)",
  D: "YELLOW center (D)",
  L: "ORANGE center (L)",
  B: "BLUE center (B)",
};

const COLOR_HEX = {
  W: "#f3f4f6",
  R: "#ef4444",
  G: "#22c55e",
  Y: "#facc15",
  O: "#f97316",
  B: "#3b82f6",
};

const COLOR_CYCLE = ["W", "R", "G", "Y", "O", "B"];

const video = document.getElementById("video");
const canvas = document.getElementById("snapshotCanvas");
const ctx = canvas.getContext("2d");

const stepText = document.getElementById("stepText");
const statusText = document.getElementById("statusText");
const facesContainer = document.getElementById("facesContainer");
const resultBox = document.getElementById("resultBox");

const startBtn = document.getElementById("startBtn");
const captureBtn = document.getElementById("captureBtn");
const retakeBtn = document.getElementById("retakeBtn");
const finalizeBtn = document.getElementById("finalizeBtn");

let stream = null;
let currentFaceIndex = 0;
let faces = {}; // { U:[9], R:[9], ... }
let isBusy = false;

function setStatus(msg, kind = "") {
  statusText.textContent = msg;
  statusText.className = `status ${kind}`.trim();
}

function nextColor(current) {
  const idx = COLOR_CYCLE.indexOf(current);
  if (idx === -1) return "W";
  return COLOR_CYCLE[(idx + 1) % COLOR_CYCLE.length];
}

function currentFaceKey() {
  return FACE_ORDER[currentFaceIndex];
}

function updateStepInstruction() {
  if (currentFaceIndex >= FACE_ORDER.length) {
    stepText.textContent = "All faces captured. Click Validate Cube State.";
    return;
  }
  const key = currentFaceKey();
  stepText.textContent = `Face ${currentFaceIndex + 1}/6: Show ${FACE_NAMES[key]} and press any key to capture.`;
}

function createFaceCard(faceKey, grid) {
  const card = document.createElement("div");
  card.className = "face-card";

  const title = document.createElement("div");
  title.className = "face-title";
  title.textContent = `${faceKey} - ${FACE_NAMES[faceKey]}`;

  const gridEl = document.createElement("div");
  gridEl.className = "grid3";

  grid.forEach((c, i) => {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.style.background = COLOR_HEX[c] || "#6b7280";
    cell.title = `${c} (click to change)`;
    cell.style.cursor = "pointer";
    cell.addEventListener("click", () => {
      if (!faces[faceKey]) return;
      faces[faceKey][i] = nextColor(faces[faceKey][i]);
      renderFacePreviews();
      setStatus(`Edited ${faceKey} face sticker ${i + 1}. Click Validate Cube State after corrections.`);
    });
    gridEl.appendChild(cell);
  });

  card.appendChild(title);
  card.appendChild(gridEl);
  return card;
}

function renderFacePreviews() {
  facesContainer.innerHTML = "";
  FACE_ORDER.forEach((faceKey) => {
    if (faces[faceKey]) {
      facesContainer.appendChild(createFaceCard(faceKey, faces[faceKey]));
    }
  });
}

function updateButtons() {
  const started = !!stream;
  startBtn.disabled = started;
  captureBtn.disabled = !started || currentFaceIndex >= FACE_ORDER.length || isBusy;
  retakeBtn.disabled = !started || currentFaceIndex === 0 || isBusy;
  finalizeBtn.disabled = currentFaceIndex < FACE_ORDER.length || isBusy;
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    video.srcObject = stream;
    setStatus("Camera started. Keep one face centered inside white square and fill most of it.");
    updateButtons();
    updateStepInstruction();
  } catch (err) {
    setStatus(`Camera error: ${err.message}`, "bad");
  }
}

function getSnapshotBlob() {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  canvas.width = vw;
  canvas.height = vh;
  ctx.drawImage(video, 0, 0, vw, vh);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
}

async function captureCurrentFace() {
  if (isBusy || currentFaceIndex >= FACE_ORDER.length) return;
  isBusy = true;
  updateButtons();

  const faceKey = currentFaceKey();
  setStatus(`Capturing ${faceKey}...`);

  try {
    const blob = await getSnapshotBlob();
    if (!blob) throw new Error("Unable to capture frame");

    const form = new FormData();
    form.append("file", blob, `${faceKey}.jpg`);

    const res = await fetch("/api/detect-face", {
      method: "POST",
      body: form,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Detection failed");

    const grid = data.grid;
    if (!Array.isArray(grid) || grid.length !== 9) throw new Error("Invalid detection payload");

    faces[faceKey] = grid;
    renderFacePreviews();

    const center = grid[4];
    const expectedCenter = { U: "W", R: "R", F: "G", D: "Y", L: "O", B: "B" }[faceKey];
    if (center !== expectedCenter) {
      setStatus(
        `Captured ${faceKey}, but center detected ${center} (expected ${expectedCenter}). Move cube to center and retake.`,
        "bad"
      );
    } else {
      setStatus(`Captured ${faceKey} successfully.`, "good");
      currentFaceIndex += 1;
    }

    updateStepInstruction();
    resultBox.innerHTML = "";
  } catch (err) {
    setStatus(`Capture error: ${err.message}`, "bad");
  } finally {
    isBusy = false;
    updateButtons();
  }
}

function retakePrevious() {
  if (isBusy || currentFaceIndex === 0) return;
  const prevIndex = currentFaceIndex - 1;
  const key = FACE_ORDER[prevIndex];
  delete faces[key];
  currentFaceIndex = prevIndex;
  renderFacePreviews();
  updateStepInstruction();
  setStatus(`Retake enabled for ${key}.`);
  updateButtons();
}

async function finalizeState() {
  if (isBusy) return;
  isBusy = true;
  updateButtons();
  setStatus("Validating cube state...");

  try {
    const res = await fetch("/api/finalize-state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ faces }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Validation failed");

    if (data.is_solvable) {
      resultBox.innerHTML = `
        <div class="good"><strong>Cube is valid and solvable.</strong></div>
        <div>Cube state: <code>${data.cube_state}</code></div>
        <div>Preview solution (phase-2 will guide moves): <code>${data.solution}</code></div>
      `;
      setStatus("Validation complete.", "good");
    } else {
      resultBox.innerHTML = `
        <div class="bad"><strong>Cube is NOT solvable.</strong></div>
        <div>Reason: ${data.error || "Unknown"}</div>
      `;
      setStatus("Validation complete: unsolvable.", "bad");
    }
  } catch (err) {
    resultBox.innerHTML = `<div class="bad"><strong>Error:</strong> ${err.message}</div>`;
    setStatus(`Validation error: ${err.message}`, "bad");
  } finally {
    isBusy = false;
    updateButtons();
  }
}

startBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", captureCurrentFace);
retakeBtn.addEventListener("click", retakePrevious);
finalizeBtn.addEventListener("click", finalizeState);

window.addEventListener("keydown", (event) => {
  if (!stream) return;
  const tag = document.activeElement?.tagName?.toLowerCase();
  if (tag === "input" || tag === "textarea") return;
  event.preventDefault();
  captureCurrentFace();
});

updateStepInstruction();
updateButtons();
