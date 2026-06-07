import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { generateFromFiles } from "./voxel.js";
import { exportPointsToObj, exportPointsToStl } from "./stl_export.js";

const canvasHost = document.getElementById("canvasHost");
const imageFrontInput = document.getElementById("imageFront");
const imageSideInput = document.getElementById("imageSide");
const previewFront = document.getElementById("previewFront");
const previewSide = document.getElementById("previewSide");
const generateBtn = document.getElementById("generateBtn");
const exportStlBtn = document.getElementById("exportStlBtn");
const exportObjBtn = document.getElementById("exportObjBtn");
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
const edgeStripFillInput = document.getElementById("edgeStripFill");
const edgeWallWrapInput = document.getElementById("edgeWallWrap");
const closeSideZGapInput = document.getElementById("closeSideZGap");
const detailModeInput = document.getElementById("detailMode");
const densityInput = document.getElementById("density");
const densityValue = document.getElementById("densityValue");
const uniformStrengthInput = document.getElementById("uniformStrength");
const uniformValue = document.getElementById("uniformValue");
const invertInput = document.getElementById("invert");
const pointSizeInput = document.getElementById("pointSize");
const showWireframeInput = document.getElementById("showWireframe");
const autoRotateInput = document.getElementById("autoRotate");
const voxelCountEl = document.getElementById("voxelCount");
const currentViewEl = document.getElementById("currentView");
const projFrontCanvas = document.getElementById("projFront");
const projSideCanvas = document.getElementById("projSide");

let frontFile = null;
let sideFile = null;
let latestResult = null;

// --- Three.js setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x12151e);

const camera = new THREE.PerspectiveCamera(
  45,
  canvasHost.clientWidth / canvasHost.clientHeight,
  0.1,
  1000
);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(canvasHost.clientWidth, canvasHost.clientHeight);
canvasHost.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

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

function updateCameraClipping(size) {
  camera.near = Math.max(0.1, size * 0.001);
  camera.far = Math.max(2000, size * 12);
  camera.updateProjectionMatrix();
}

function renderPointSize(size) {
  const base = parseFloat(pointSizeInput.value) || 0.08;
  return Math.max(0.02, base * (size / 192));
}

function centerCamera(size) {
  const half = size / 2;
  controls.target.set(half, half, half);
  camera.position.set(half + size * 1.4, half + size * 0.9, half + size * 1.6);
  updateCameraClipping(size);
  controls.update();
}

centerCamera(cubeSize);

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
  clearPointCloud();
  cubeSize = size;

  if (!points.length) return;

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
    size: renderPointSize(size),
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.92,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

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
  updateCameraClipping(cubeSize);
  controls.target.copy(target);
  controls.update();
  currentViewEl.textContent = `当前视角：${v.label}`;
}

document.querySelectorAll("[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => setCameraView(btn.dataset.view));
});

function drawProjection(canvas, matrix) {
  const ctx = canvas.getContext("2d");
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

function setupUpload(input, preview, card, onFile) {
  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    latestResult = null;
    exportStlBtn.disabled = true;
    exportObjBtn.disabled = true;
    onFile(file);
    preview.src = URL.createObjectURL(file);
    card.classList.add("has-image");
    updateGenerateState();
  });
}

setupUpload(imageFrontInput, previewFront, imageFrontInput.closest(".upload-card"), (f) => {
  frontFile = f;
});
setupUpload(imageSideInput, previewSide, imageSideInput.closest(".upload-card"), (f) => {
  sideFile = f;
});

function updateGenerateState() {
  generateBtn.disabled = !(frontFile && sideFile);
  if (!latestResult) {
    exportStlBtn.disabled = true;
    exportObjBtn.disabled = true;
  }
  if (frontFile && sideFile) {
    statusEl.textContent = "就绪，点击生成";
    statusEl.className = "status";
  }
}

exportStlBtn.addEventListener("click", async () => {
  if (!latestResult?.points?.length) return;
  exportStlBtn.disabled = true;
  const prev = statusEl.textContent;
  try {
    statusEl.textContent = "正在导出 STL…";
    await new Promise((r) => setTimeout(r, 0));
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const filename = `voxel_${latestResult.size}_${stamp}.stl`;
    const info = exportPointsToStl(latestResult.points, filename);
    statusEl.textContent = `STL 导出完成，三角面 ${info.triangleCount.toLocaleString()} 个`;
    statusEl.className = "status success";
  } catch (e) {
    statusEl.textContent = `STL 导出失败：${e.message}`;
    statusEl.className = "status error";
  } finally {
    exportStlBtn.disabled = !latestResult?.points?.length;
    if (statusEl.className === "status" && prev) statusEl.textContent = prev;
  }
});

exportObjBtn.addEventListener("click", async () => {
  if (!latestResult?.points?.length) return;
  exportObjBtn.disabled = true;
  const prev = statusEl.textContent;
  try {
    statusEl.textContent = "正在导出 OBJ…";
    await new Promise((r) => setTimeout(r, 0));
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const filename = `voxel_${latestResult.size}_${stamp}.obj`;
    const info = exportPointsToObj(latestResult.points, filename);
    statusEl.textContent = `OBJ 导出完成，顶点 ${info.vertexCount.toLocaleString()}，面 ${info.faceCount.toLocaleString()} 个`;
    statusEl.className = "status success";
  } catch (e) {
    statusEl.textContent = `OBJ 导出失败：${e.message}`;
    statusEl.className = "status error";
  } finally {
    exportObjBtn.disabled = !latestResult?.points?.length;
    if (statusEl.className === "status" && prev) statusEl.textContent = prev;
  }
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
autoThresholdInput.addEventListener("change", () => {
  thresholdField.style.opacity = autoThresholdInput.checked ? "0.45" : "1";
  thresholdField.style.pointerEvents = autoThresholdInput.checked ? "none" : "auto";
});
autoThresholdInput.dispatchEvent(new Event("change"));
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
uniformStrengthInput.addEventListener("input", () => {
  uniformValue.textContent = uniformStrengthInput.value;
});

pointSizeInput.addEventListener("input", () => {
  if (pointCloud) {
    pointCloud.material.size = renderPointSize(cubeSize);
  }
});

showWireframeInput.addEventListener("change", () => {
  updateWireframe(cubeSize, showWireframeInput.checked);
});

autoRotateInput.addEventListener("change", () => {
  controls.autoRotate = autoRotateInput.checked;
  controls.autoRotateSpeed = 1.2;
});

generateBtn.addEventListener("click", async () => {
  if (!frontFile || !sideFile) return;

  generateBtn.disabled = true;
  exportStlBtn.disabled = true;
  exportObjBtn.disabled = true;
  statusEl.textContent = "正在生成…";
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
  const edgeStripFill = edgeStripFillInput?.checked !== false;
  const edgeWallWrap = edgeWallWrapInput?.checked !== false;
  const closeSideZGaps = parseInt(closeSideZGapInput?.value || "2", 10);
  const density = parseInt(densityInput.value, 10) / 100;
  const uniformStrength = parseInt(uniformStrengthInput.value, 10) / 100;

  await new Promise((r) => setTimeout(r, 0));

  try {
    let data;
    const genOptions = {
      threshold,
      invert,
      autoThreshold,
      dilate,
      alignX,
      cleanMask,
      overlapDilate,
      depthFaceBridge,
      edgeStripFill,
      edgeWallWrap,
      closeSideZGaps,
      density,
      uniformStrength,
      detailMode,
    };

    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1500);
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
      form.append("edge_strip_fill", edgeStripFill ? "true" : "false");
      form.append("edge_wall_wrap", edgeWallWrap ? "true" : "false");
      form.append("close_side_z_gaps", String(closeSideZGaps));
      form.append("density", String(density));
      form.append("uniform_strength", String(uniformStrength));
      const res = await fetch("/api/generate", { method: "POST", body: form, signal: ctrl.signal });
      clearTimeout(timer);
      if (res.ok) data = await res.json();
    } catch {
      /* 无后端或超时：使用浏览器本地计算 */
    }
    if (!data) {
      statusEl.textContent = "正在计算体素…";
      await new Promise((r) => setTimeout(r, 0));
      data = await generateFromFiles(frontFile, sideFile, size, genOptions);
    }

    statusEl.textContent = "正在构建 3D 视图…";
    await new Promise((r) => setTimeout(r, 0));
    buildPointCloud(data.points, data.size);
    latestResult = data;
    exportStlBtn.disabled = !data.points?.length;
    exportObjBtn.disabled = !data.points?.length;
    const fullLabel = data.count_full != null ? `（雕刻后 ${data.count_full.toLocaleString()} → 稀疏化 ${data.count.toLocaleString()}）` : "";
    voxelCountEl.textContent = `体素数：${data.count.toLocaleString()} / ${(data.size ** 3).toLocaleString()} ${fullLabel}`;
    drawProjection(projFrontCanvas, data.projection_front);
    drawProjection(projSideCanvas, data.projection_side);

    const tInfo =
      data.threshold_front != null
        ? `（阈值: 正 ${data.threshold_front} / 侧 ${data.threshold_side}${
            data.invert_front != null ? `，自动${data.invert_front ? "深色" : "浅色"}为实体` : ""
          }）`
        : "";
    statusEl.textContent = `生成完成，共 ${data.count.toLocaleString()} 个体素 ${tInfo}`;
    statusEl.className = "status success";
    currentViewEl.textContent = "当前视角：自由（可拖拽旋转）";
  } catch (e) {
    latestResult = null;
    exportStlBtn.disabled = true;
    exportObjBtn.disabled = true;
    statusEl.textContent = `生成失败：${e.message}`;
    statusEl.className = "status error";
  } finally {
    generateBtn.disabled = !(frontFile && sideFile);
  }
});

function onResize() {
  const w = canvasHost.clientWidth;
  const h = canvasHost.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

window.addEventListener("resize", onResize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

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
