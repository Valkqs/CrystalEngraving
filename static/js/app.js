import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { autoInvert, otsuThreshold } from "./image_preprocess.js";

// Redirect all console.log to server for debugging
const _origLog = console.log;
console.log = (...args) => {
  _origLog.apply(console, args);
  fetch("/api/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ msg: args.map(a => String(a)).join(" ") }),
  }).catch(() => {});
};

const canvasHost = document.getElementById("canvasHost");
const imageFrontInput = document.getElementById("imageFront");
const imageSideInput = document.getElementById("imageSide");
const imageTopInput = document.getElementById("imageTop");
const previewFront = document.getElementById("previewFront");
const previewSide = document.getElementById("previewSide");
const previewTop = document.getElementById("previewTop");
const topWeightsSection = document.getElementById("topWeightsSection");
const wF1FrontInput = document.getElementById("wF1Front");
const wF1TopInput = document.getElementById("wF1Top");
const wF1SideInput = document.getElementById("wF1Side");
const wF1FrontValue = document.getElementById("wF1FrontValue");
const wF1TopValue = document.getElementById("wF1TopValue");
const wF1SideValue = document.getElementById("wF1SideValue");
const projTopCanvas = document.getElementById("projTop");
const projTopContainer = document.getElementById("projTopContainer");
const generateBtn = document.getElementById("generateBtn");
const previewBtn = document.getElementById("previewBtn");
const statusEl = document.getElementById("status");
const sizeInput = document.getElementById("size");
const sizeValue = document.getElementById("sizeValue");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("thresholdValue");
const thresholdField = document.getElementById("thresholdField");
const autoThresholdInput = document.getElementById("autoThreshold");
const dilateInput = document.getElementById("dilate");
const dilateValue = document.getElementById("dilateValue");
const alignXInput = document.getElementById("alignX");
const cleanMaskInput = document.getElementById("cleanMask");
const overlapDilateInput = document.getElementById("overlapDilate");
const depthFaceBridgeInput = document.getElementById("depthFaceBridge");
const closeSideZGapInput = document.getElementById("closeSideZGap");
const detailModeInput = document.getElementById("detailMode");
const densityInput = document.getElementById("density");
const densityValue = document.getElementById("densityValue");
const uniformStrengthInput = document.getElementById("uniformStrength");
const uniformValue = document.getElementById("uniformValue");
const invertInput = document.getElementById("invert");
const chaosPenaltyInput = document.getElementById("chaosPenalty");
const chaosPenaltyValue = document.getElementById("chaosPenaltyValue");
const minF1Input = document.getElementById("minF1");
const minF1Value = document.getElementById("minF1Value");
const saStepsInput = document.getElementById("saSteps");
const saStepsValue = document.getElementById("saStepsValue");
const weightVolumeInput = document.getElementById("weightVolume");
const weightVolumeValue = document.getElementById("weightVolumeValue");
const rngSeedInput = document.getElementById("rngSeed");
const rngSeedValue = document.getElementById("rngSeedValue");
const optimizerAlgoInput = document.getElementById("optimizerAlgo");
const pointSizeInput = document.getElementById("pointSize");
const showWireframeInput = document.getElementById("showWireframe");
const autoRotateInput = document.getElementById("autoRotate");
const voxelCountEl = document.getElementById("voxelCount");
const f1ScoreEl = document.getElementById("f1Score");
const currentViewEl = document.getElementById("currentView");
const projFrontCanvas = document.getElementById("projFront");
const projSideCanvas = document.getElementById("projSide");
const maskFrontCanvas = document.getElementById("maskFront");
const maskSideCanvas = document.getElementById("maskSide");
const thresholdSummaryEl = document.getElementById("thresholdSummary");
const diagModeEl = document.getElementById("diagMode");
const diagFrontEl = document.getElementById("diagFront");
const diagSideEl = document.getElementById("diagSide");
const exportSection = document.getElementById("exportSection");
const downloadPlyBtn = document.getElementById("downloadPlyBtn");
const downloadStlBtn = document.getElementById("downloadStlBtn");
const downloadObjBtn = document.getElementById("downloadObjBtn");
const exportInfoEl = document.getElementById("exportInfo");

let frontFile = null;
let sideFile = null;
let topFile = null;
let currentJobId = null;
let lastPreviewParams = null;

// --- Three.js setup ---
console.log("[DEBUG] === Three.js initialization START ===");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x12151e);
console.log("[DEBUG] Scene created, background:", scene.background.getHexString());

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
canvasHost.appendChild(renderer.domElement);
console.log("[DEBUG] Renderer created, canvas:", renderer.domElement.tagName);

const gl = renderer.getContext();
console.log("[DEBUG] WebGL context:", gl ? gl.getParameter(gl.VERSION) : "FAILED");
console.log("[DEBUG] Renderer info:", renderer.info);

const controls = new OrbitControls(camera, renderer.domElement);
console.log("[DEBUG] OrbitControls created, camera position:", camera.position.x, camera.position.y, camera.position.z);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

function resizeCanvasHost() {
  const viewer = document.querySelector('.viewer-area');
  const stats = viewer?.querySelector('.stats-bar');
  const viewerH = viewer?.clientHeight ?? 0;
  const statsH = stats?.clientHeight ?? 24;
  const computedH = viewerH > 0 ? viewerH - statsH : 0;
  const rect = canvasHost.getBoundingClientRect();
  const w = Math.floor(rect.width);
  const h = computedH > 0 ? computedH : Math.floor(rect.height);
  console.log(`[DEBUG] resizeCanvasHost bounding:${w}x${h} | viewer:${viewerH} stats:${statsH} computed:${computedH}`);
  if (w > 0 && h > 0) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
}

const resizeObserver = new ResizeObserver(() => {
  resizeCanvasHost();
});
resizeObserver.observe(canvasHost);
resizeCanvasHost();
console.log("[DEBUG] ResizeObserver attached, watching:", canvasHost);

const ambient = new THREE.AmbientLight(0xffffff, 0.55);
scene.add(ambient);

const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
dirLight.position.set(1, 2, 1.5);
scene.add(dirLight);

const axesHelper = new THREE.AxesHelper(1.2);
scene.add(axesHelper);

let pointCloud = null;
let wireframe = null;
let cubeSize = 64;

function centerCamera(size) {
  const half = size / 2;
  controls.target.set(half, half, half);
  camera.position.set(half + size * 1.4, half + size * 0.9, half + size * 1.6);
  controls.update();
}

centerCamera(cubeSize);
console.log("[DEBUG] === Three.js initialization END ===");

function clearPointCloud() {
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
    pointCloud = null;
  }
}

function updateWireframe(size, visible) {
  if (wireframe) {
    scene.remove(wireframe);
    wireframe.geometry.dispose();
    wireframe.material.dispose();
    wireframe = null;
  }
  if (!visible) return;

  const geo = new THREE.BoxGeometry(size, size, size);
  const edges = new THREE.EdgesGeometry(geo);
  geo.dispose();
  const mat = new THREE.LineBasicMaterial({ color: 0x4da3ff, transparent: true, opacity: 0.35 });
  wireframe = new THREE.LineSegments(edges, mat);
  wireframe.position.set(size / 2, size / 2, size / 2);
  scene.add(wireframe);
}

updateWireframe(cubeSize, true);

function buildPointCloud(points, size) {
  console.log("[DEBUG] buildPointCloud called with", points?.length, "points, size:", size);
  clearPointCloud();
  cubeSize = size;
  updateWireframe(size, showWireframeInput.checked);
  centerCamera(size);

  if (!Array.isArray(points) || !points.length) return;

  const positions = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) {
    positions[i * 3] = points[i][0] + 0.5;
    positions[i * 3 + 1] = points[i][1] + 0.5;
    positions[i * 3 + 2] = points[i][2] + 0.5;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: 0x88ccff,
    size: parseFloat(pointSizeInput.value) || Math.max(0.04, size * 0.0006),
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.92,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);
  console.log("[DEBUG] pointCloud added to scene, scene children:", scene.children.length);

  updateWireframe(size, showWireframeInput.checked);
  centerCamera(size);
}

function setCameraView(viewName) {
  const half = cubeSize / 2;
  const d = cubeSize * 1.8;
  const target = new THREE.Vector3(half, half, half);

  const views = {
    front: { pos: [half, half, half + d], label: "正视图 (-Z)" },
    side: { pos: [half, half + d, half], label: "侧视图 (-Y)" },
    top: { pos: [half + d, half, half], label: "右视图 (+X)" },
    iso: { pos: [half + d, half + d * 0.7, half + d], label: "等轴测" },
  };

  const v = views[viewName];
  if (!v) return;

  camera.position.set(...v.pos);
  controls.target.copy(target);
  controls.update();
  currentViewEl.textContent = `当前视角：${v.label}`;
}

document.querySelectorAll("[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => setCameraView(btn.dataset.view));
});

function drawProjection(canvas, matrix) {
  const ctx = canvas.getContext("2d");
  if (!Array.isArray(matrix) || !matrix.length) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }

  const n = matrix.length;
  canvas.width = n;
  canvas.height = n;

  const imageData = ctx.createImageData(n, n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const v = matrix[y][x] ? 200 : 0;
      const i = (y * n + x) * 4;
      imageData.data[i] = v;
      imageData.data[i + 1] = v + 20;
      imageData.data[i + 2] = v + 55;
      imageData.data[i + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function formatThresholdDetail(label, threshold, invert) {
  if (threshold == null) return `${label}：—`;
  const fg = invert ? "深色为实体" : "浅色为实体";
  return `${label}：阈值 ${threshold} · ${fg}`;
}

function updateThresholdDiagnostics(data, autoThreshold) {
  const mode = autoThreshold ? "自动阈值 Otsu" : "手动阈值";
  const solver = data.solve_mode === "backend_sa" ? "后端模拟退火" : "后端未连接";
  const frontText = formatThresholdDetail("正视", data.threshold_front, data.invert_front);
  const sideText = formatThresholdDetail("侧视", data.threshold_side, data.invert_side);
  diagModeEl.textContent = solver;
  diagFrontEl.textContent = frontText;
  diagSideEl.textContent = sideText;
  thresholdSummaryEl.textContent = `${mode}｜${solver}：${frontText}；${sideText}`;
}

function setupUpload(input, preview, card, onFile) {
  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    onFile(file);
    preview.src = URL.createObjectURL(file);
    card.classList.add("has-image");
    updatePreviewBtnState();
    updateGenerateState();
  });
}

setupUpload(imageFrontInput, previewFront, imageFrontInput.closest(".upload-card"), (f) => {
  frontFile = f;
});
setupUpload(imageSideInput, previewSide, imageSideInput.closest(".upload-card"), (f) => {
  sideFile = f;
});
if (imageTopInput) {
  setupUpload(imageTopInput, previewTop, imageTopInput.closest(".upload-card"), (f) => {
    topFile = f;
    if (topWeightsSection) topWeightsSection.style.display = "";
    updateGenerateState();
  });
  imageTopInput.addEventListener("change", () => {
    if (!imageTopInput.files || imageTopInput.files.length === 0) {
      topFile = null;
      if (topWeightsSection) topWeightsSection.style.display = "none";
      if (projTopContainer) projTopContainer.style.display = "none";
      updateGenerateState();
    }
  });
}

function bindWeightSlider(input, valueEl, divider) {
  if (!input || !valueEl) return;
  input.addEventListener("input", () => {
    const raw = parseInt(input.value, 10);
    valueEl.textContent = String(divider ? Math.round(raw) : raw);
  });
}
bindWeightSlider(wF1FrontInput, wF1FrontValue);
bindWeightSlider(wF1TopInput, wF1TopValue);
bindWeightSlider(wF1SideInput, wF1SideValue);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollJobUntilDone(jobId, timeoutMs = 4 * 60 * 60 * 1000) {
  const startedAt = Date.now();

  while (true) {
    if (Date.now() - startedAt > timeoutMs) {
      throw new Error("后端生成超时（已等待 1 小时），请降低分辨率或减少模拟退火迭代数后重试");
    }

    const res = await fetch(`/api/generate/${jobId}`);
    if (!res.ok) {
      throw new Error(`读取生成进度失败 (${res.status})`);
    }

    const info = await res.json();
    const percent = Math.max(0, Math.min(100, Math.round((info.progress || 0) * 100)));
    const detail = info.detail || {};
    let extra = "";

    if (detail.step != null && detail.max_steps != null) {
      extra += ` · SA ${detail.step}/${detail.max_steps}`;
    }
    if (detail.best_f1 != null) {
      extra += ` · 最佳F1 ${(detail.best_f1 * 100).toFixed(1)}%`;
    }
    if (detail.best_count != null) {
      extra += ` · 体素 ${Number(detail.best_count).toLocaleString()}`;
    }

    statusEl.textContent = `[后端模拟退火] ${percent}% · ${info.stage || "生成中"}${extra}`;
    statusEl.className = "status";
    diagModeEl.textContent = "后端优化中";

    if (info.status === "completed") {
      return;
    }
    if (info.status === "failed") {
      throw new Error(info.error || "后端生成失败");
    }

    await sleep(800);
  }
}

function updateGenerateState() {
  generateBtn.disabled = !(frontFile && sideFile);
  if (frontFile && sideFile) {
    statusEl.textContent = "就绪，点击生成";
    statusEl.className = "status";
  }
}

function updatePreviewBtnState() {
  previewBtn.disabled = !(frontFile && sideFile);
}

previewBtn.addEventListener("click", () => {
  previewBtn.disabled = true;
  statusEl.textContent = "正在生成二值预览...";
  statusEl.className = "status";

  const formData = new FormData();
  formData.append("image_front", frontFile);
  formData.append("image_side", sideFile);
  formData.append("size", parseInt(sizeInput.value, 10));
  formData.append("threshold", thresholdInput.value);
  formData.append("invert", invertInput.checked ? "true" : "false");
  formData.append("auto_threshold", autoThresholdInput.checked ? "true" : "false");

  fetch("/api/preview", { method: "POST", body: formData })
    .then((r) => {
      if (!r.ok) throw new Error(`预览请求失败 ${r.status}`);
      return r.json();
    })
    .then((data) => {
      console.log("Preview result:", data);
      const frontImg = new Image();
      frontImg.onload = () => {
        const ctx = maskFrontCanvas.getContext("2d");
        ctx.clearRect(0, 0, maskFrontCanvas.width, maskFrontCanvas.height);
        maskFrontCanvas.width = 160;
        maskFrontCanvas.height = 160;
        ctx.drawImage(frontImg, 0, 0, 160, 160);
      };
      frontImg.src = "data:image/png;base64," + data.front_png;

      const sideImg = new Image();
      sideImg.onload = () => {
        const ctx = maskSideCanvas.getContext("2d");
        ctx.clearRect(0, 0, maskSideCanvas.width, maskSideCanvas.height);
        maskSideCanvas.width = 160;
        maskSideCanvas.height = 160;
        ctx.drawImage(sideImg, 0, 0, 160, 160);
      };
      sideImg.src = "data:image/png;base64," + data.side_png;

      diagFrontEl.textContent = `阈值${data.threshold_front} / ${data.invert_front ? "反转" : "正相"} / ${(data.mask_front_ratio * 100).toFixed(1)}%`;
      diagSideEl.textContent = `阈值${data.threshold_side} / ${data.invert_side ? "反转" : "正相"} / ${(data.mask_side_ratio * 100).toFixed(1)}%`;
      diagModeEl.textContent = data.auto_threshold ? "自动阈值" : "手动阈值";

      lastPreviewParams = {
        threshold_front: data.threshold_front,
        threshold_side: data.threshold_side,
        invert_front: data.invert_front,
        invert_side: data.invert_side,
      };

      previewBtn.disabled = false;
      statusEl.textContent = "二值预览已更新，请确认后点击生成";
      statusEl.className = "status";
    })
    .catch((err) => {
      console.error("Preview error:", err);
      previewBtn.disabled = false;
      statusEl.textContent = "预览失败：" + err.message;
      statusEl.className = "status error";
    });
});

sizeInput.addEventListener("input", () => {
  sizeValue.textContent = sizeInput.value;
});
thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = thresholdInput.value;
});
dilateInput.addEventListener("input", () => {
  dilateValue.textContent = dilateInput.value;
});
autoThresholdInput.dispatchEvent(new Event("change"));
autoThresholdInput.addEventListener("change", () => {
  thresholdField.style.opacity = autoThresholdInput.checked ? "0.45" : "1";
  thresholdField.style.pointerEvents = autoThresholdInput.checked ? "none" : "auto";
});
densityInput.addEventListener("input", () => {
  densityValue.textContent = densityInput.value;
});
detailModeInput?.addEventListener("change", () => {
  const on = detailModeInput.checked;
  if (on) {
    if (parseInt(dilateInput.value, 10) > 0) dilateInput.value = "0";
    dilateValue.textContent = dilateInput.value;
    cleanMaskInput.checked = false;
    if (parseInt(densityInput.value, 10) < 70) {
      densityInput.value = "75";
      densityValue.textContent = "75";
    }
    if (parseInt(uniformStrengthInput.value, 10) > 30) {
      uniformStrengthInput.value = "25";
      uniformValue.textContent = "25";
    }
    if (parseInt(sizeInput.value, 10) < 160) {
      sizeInput.value = "192";
      sizeValue.textContent = "192";
    }
  }
});
overlapDilateInput?.addEventListener("input", () => {
  const el = document.getElementById("overlapValue");
  if (el && overlapDilateInput) el.textContent = overlapDilateInput.value;
});
closeSideZGapInput?.addEventListener("input", () => {
  const el = document.getElementById("closeZGapValue");
  if (el && closeSideZGapInput) el.textContent = closeSideZGapInput.value;
});

chaosPenaltyInput.addEventListener("input", () => {
  chaosPenaltyValue.textContent = chaosPenaltyInput.value;
});
minF1Input.addEventListener("input", () => {
  minF1Value.textContent = minF1Input.value;
});
saStepsInput.addEventListener("input", () => {
  saStepsValue.textContent = saStepsInput.value;
});
weightVolumeInput.addEventListener("input", () => {
  weightVolumeValue.textContent = weightVolumeInput.value;
});
rngSeedInput.addEventListener("input", () => {
  rngSeedValue.textContent = rngSeedInput.value;
});

uniformStrengthInput.addEventListener("input", () => {
  uniformValue.textContent = uniformStrengthInput.value;
});

pointSizeInput.addEventListener("input", () => {
  if (pointCloud) {
    pointCloud.material.size = parseFloat(pointSizeInput.value);
  }
});

showWireframeInput.addEventListener("change", () => {
  updateWireframe(cubeSize, showWireframeInput.checked);
});

autoRotateInput.addEventListener("change", () => {
  controls.autoRotate = autoRotateInput.checked;
  controls.autoRotateSpeed = 1.2;
});

function setupDownloadButton(btn, format) {
  if (!btn) return;
  btn.addEventListener("click", async () => {
    if (!currentJobId) return;
    btn.disabled = true;
    const originalLabel = btn.textContent;
    btn.textContent = "正在生成…";
    try {
      const res = await fetch(`/api/export/${currentJobId}/${format}`);
      if (!res.ok) throw new Error(`/api/export returned ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `crystal_${currentJobId.slice(0, 8)}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error(e);
    } finally {
      btn.disabled = false;
      btn.textContent = originalLabel;
    }
  });
}

setupDownloadButton(downloadPlyBtn, "ply");
setupDownloadButton(downloadStlBtn, "stl");
setupDownloadButton(downloadObjBtn, "obj");

downloadPlyBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  downloadPlyBtn.disabled = true;
  try {
    const res = await fetch(`/api/export/${currentJobId}/ply`);
    if (!res.ok) throw new Error(`/api/export returned ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `crystal_${currentJobId.slice(0, 8)}.ply`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
  } finally {
    downloadPlyBtn.disabled = false;
  }
});

generateBtn.addEventListener("click", async () => {
  console.log("[DEBUG] Generate button clicked");
  if (!frontFile || !sideFile) return;

  generateBtn.disabled = true;
  if (exportSection) exportSection.style.display = "none";
  statusEl.textContent = "正在提交后端任务…";
  statusEl.className = "status";

  const size = parseInt(sizeInput.value, 10);
  const threshold = parseInt(thresholdInput.value, 10);
  const invert = invertInput.checked ? true : null;
  const detailMode = detailModeInput?.checked !== false;
  const autoThreshold = autoThresholdInput.checked;
  const dilate = parseInt(dilateInput.value, 10);
  const alignX = alignXInput.checked;
  const cleanMask = cleanMaskInput.checked;
  const overlapDilate = parseInt(overlapDilateInput?.value || "1", 10);
  const depthFaceBridge = depthFaceBridgeInput?.checked !== false;
  const closeSideZGaps = parseInt(closeSideZGapInput?.value || "2", 10);
  const density = parseInt(densityInput.value, 10) / 100;
  const uniformStrength = parseInt(uniformStrengthInput.value, 10) / 100;
  const optimize = true;
  const chaosPenalty = parseInt(chaosPenaltyInput.value, 10) / 100;
  const minF1 = parseInt(minF1Input.value, 10) / 100;
  const saSteps = parseInt(saStepsInput.value, 10);
  const weightVolume = parseInt(weightVolumeInput.value, 10) / 100;
  const rngSeed = parseInt(rngSeedInput.value, 10);
  const optimizerAlgo = optimizerAlgoInput ? optimizerAlgoInput.value : "fast";

  await new Promise((r) => setTimeout(r, 0));

  try {
    const form = new FormData();
    form.append("image_front", frontFile);
    form.append("image_side", sideFile);
    form.append("size", String(size));
    form.append("threshold", autoThreshold ? "" : String(threshold));
    form.append("invert", invert === true ? "true" : invert === false ? "false" : "");
    form.append("detail_mode", detailMode ? "true" : "false");
    form.append("auto_threshold", autoThreshold ? "true" : "false");
    form.append("dilate", String(dilate));
    form.append("align_x", alignX ? "true" : "false");
    form.append("clean_mask", cleanMask ? "true" : "false");
    form.append("overlap_dilate", String(overlapDilate));
    form.append("depth_face_bridge", depthFaceBridge ? "true" : "false");
    form.append("close_side_z_gaps", String(closeSideZGaps));
    form.append("density", String(density));
    form.append("uniform_strength", String(uniformStrength));
    form.append("optimize", optimize ? "true" : "false");
    form.append("chaos_penalty", String(chaosPenalty));
    form.append("min_f1", String(minF1));
    form.append("sa_steps", String(saSteps));
    form.append("weight_volume", String(weightVolume));
    form.append("rng_seed", String(rngSeed));
    form.append("optimizer_algo", optimizerAlgo);
    if (topFile) {
      form.append("image_top", topFile);
      form.append("w_f1_front", String((parseInt(wF1FrontInput.value, 10) / 100).toFixed(2)));
      form.append("w_f1_top", String((parseInt(wF1TopInput.value, 10) / 100).toFixed(2)));
      form.append("w_f1_side", String((parseInt(wF1SideInput.value, 10) / 100).toFixed(2)));
    }

    const submitRes = await fetch("/api/generate", { method: "POST", body: form });
    console.log("[DEBUG] /api/generate response status:", submitRes.status);
    if (!submitRes.ok) {
      throw new Error(`后端任务提交失败 (${submitRes.status})`);
    }

    const submitData = await submitRes.json();
    const jobId = submitData.job_id;
    currentJobId = jobId;
    if (!jobId) {
      throw new Error("后端未返回任务 ID");
    }

    await pollJobUntilDone(jobId, 60 * 60 * 1000);

    const resultRes = await fetch(`/api/generate/${jobId}/result`);
    if (!resultRes.ok) {
      throw new Error(`读取生成结果失败 (${resultRes.status})`);
    }

    const data = await resultRes.json();
    console.log("[DEBUG] result data: count=", data.count, "count_full=", data.count_full, "size=", data.size, "points_len=", data.points?.length, "projection_front_len=", data.projection_front?.length);
    data.solve_mode = "backend_sa";

    if (!Array.isArray(data.points)) {
      throw new Error("后端结果缺少 points 数据");
    }
    if (!Array.isArray(data.projection_front) || !Array.isArray(data.projection_side)) {
      throw new Error("后端结果缺少 projection 数据");
    }

    statusEl.textContent = "正在构建 3D 视图…";
    await new Promise((r) => setTimeout(r, 0));
    buildPointCloud(data.points, data.size);
    const fullLabel = data.count_full != null ? `（雕刻后 ${data.count_full.toLocaleString()} → 稀疏化 ${data.count.toLocaleString()}）` : "";
    voxelCountEl.textContent = `体素数：${data.count.toLocaleString()} / ${(data.size ** 3).toLocaleString()} ${fullLabel}`;

    if (data.f1_total != null) {
      f1ScoreEl.style.display = "";
      const f1 = (data.f1_total * 100).toFixed(1);
      const f1f = (data.f1_front * 100).toFixed(1);
      const f1s = (data.f1_side * 100).toFixed(1);
      const f1t = (data.f1_top || 0) * 100;
      const f1tStr = data.projection_top ? ` / 俯视${f1t.toFixed(1)}%` : "";
      const chaos = data.chaos != null ? ` 混乱度=${(data.chaos * 100).toFixed(1)}%` : "";
      const obj = data.objective != null ? ` obj=${(data.objective * 100).toFixed(1)}%` : "";
      const badge = data.f1_total >= 0.999 ? "✓" : data.f1_total >= 0.9 ? "△" : "✗";
      f1ScoreEl.textContent = `${badge} F1=${f1}%（正视${f1f}% / 侧视${f1s}%${f1tStr}）${obj}${chaos}`;
    } else {
      f1ScoreEl.style.display = "none";
    }

    if (data.mask_front) drawProjection(maskFrontCanvas, data.mask_front);
    if (data.mask_side) drawProjection(maskSideCanvas, data.mask_side);
    updateThresholdDiagnostics(data, autoThreshold);
    drawProjection(projFrontCanvas, data.projection_front);
    drawProjection(projSideCanvas, data.projection_side);
    if (data.projection_top && projTopCanvas && projTopContainer) {
      projTopContainer.style.display = "";
      drawProjection(projTopCanvas, data.projection_top);
    } else if (projTopContainer) {
      projTopContainer.style.display = "none";
    }

    const tInfo =
      data.threshold_front != null
        ? `（阈值: 正 ${data.threshold_front} / 侧 ${data.threshold_side}${
            data.invert_front != null ? `，正视${data.invert_front ? "深色" : "浅色"}为实体` : ""
          }${
            data.invert_side != null ? `，侧视${data.invert_side ? "深色" : "浅色"}为实体` : ""
          }）`
        : "";
    statusEl.textContent = `[后端模拟退火] 生成完成，共 ${data.count.toLocaleString()} 个体素 ${tInfo}`.trim();
    statusEl.className = "status success";
    currentViewEl.textContent = "当前视角：自由（可拖拽旋转）";
    if (exportSection) {
      exportSection.style.display = "";
      if (exportInfoEl) exportInfoEl.textContent = `PLY · ${data.size}³ · ${data.count.toLocaleString()} 点`;
    }
  } catch (e) {
    const message = e?.message || "请先启动 server.py 后端";
    statusEl.textContent = `生成失败：${message}`;
    statusEl.className = "status error";
    diagModeEl.textContent = "后端未连接";
  } finally {
    generateBtn.disabled = !(frontFile && sideFile);
  }
});

function onResize() {
  resizeCanvasHost();
}

window.addEventListener("resize", onResize);

let _frameCount = 0;
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  _frameCount++;
  if (_frameCount === 1 || _frameCount % 60 === 0) {
    console.log("[DEBUG] Frame", _frameCount, "| camera pos:", camera.position.x.toFixed(1), camera.position.y.toFixed(1), camera.position.z.toFixed(1));
  }
}
console.log("[DEBUG] animate loop started");
animate();

// Demo hint: load sample images if user opens without uploads
async function createSampleImages() {
  function drawShape(ctx, size, fn) {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = "#fff";
    fn(ctx, size);
  }

  function canvasToFile(canvas, name) {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(new File([blob], name, { type: "image/png" })), "image/png");
    });
  }

  const s = 128;
  const c1 = document.createElement("canvas");
  c1.width = c1.height = s;
  const ctx1 = c1.getContext("2d");
  drawShape(ctx1, s, (ctx, size) => {
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size * 0.35, 0, Math.PI * 2);
    ctx.fill();
  });

  const c2 = document.createElement("canvas");
  c2.width = c2.height = s;
  const ctx2 = c2.getContext("2d");
  drawShape(ctx2, s, (ctx, size) => {
    ctx.fillRect(size * 0.2, size * 0.25, size * 0.6, size * 0.5);
  });

  return {
    front: await canvasToFile(c1, "sample_front.png"),
    side: await canvasToFile(c2, "sample_side.png"),
  };
}

createSampleImages().then(({ front, side }) => {
  if (!frontFile && !sideFile) {
    frontFile = front;
    sideFile = side;
    previewFront.src = URL.createObjectURL(front);
    previewSide.src = URL.createObjectURL(side);
    imageFrontInput.closest(".upload-card").classList.add("has-image");
    imageSideInput.closest(".upload-card").classList.add("has-image");
    updateGenerateState();
    statusEl.textContent = "已加载示例图片（圆形 + 重叠矩形），可直接生成";
  }
});
