/**
 * High-quality 2D mask extraction (center-fit, luminance, auto-invert).
 */

export function loadImageBitmapFit(file, size) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const cropped = cropImageContent(img, 8);
      const rendered = renderCroppedToSquare(img, cropped, size, null);
      URL.revokeObjectURL(img.src);
      resolve(rendered);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function cropImageContent(img, whiteTol = 8) {
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = img.width;
  srcCanvas.height = img.height;
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.drawImage(img, 0, 0);
  const src = srcCtx.getImageData(0, 0, img.width, img.height);
  const bbox = contentBBox(src, whiteTol);
  return bbox || { x: 0, y: 0, w: img.width, h: img.height };
}

function renderCroppedToSquare(img, crop, size, targetWidth = null) {
  const { x: sx, y: sy, w: sw, h: sh } = crop;
  let dw;
  let dh;
  if (targetWidth == null) {
    const scale = Math.min(size / sw, size / sh);
    dw = Math.max(1, Math.round(sw * scale));
    dh = Math.max(1, Math.round(sh * scale));
  } else {
    dw = Math.max(1, Math.min(size, Math.round(targetWidth)));
    let scale = dw / Math.max(1, sw);
    dh = Math.max(1, Math.round(sh * scale));
    if (dh > size) {
      scale = size / Math.max(1, sh);
      dh = size;
      dw = Math.max(1, Math.round(sw * scale));
    }
  }

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);
  const ox = Math.floor((size - dw) / 2);
  const oy = Math.floor((size - dh) / 2);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, sx, sy, sw, sh, ox, oy, dw, dh);
  return ctx.getImageData(0, 0, size, size);
}

export async function loadImageBitmapPairFit(frontFile, sideFile, size) {
  const [frontImg, sideImg] = await Promise.all([
    loadImageFromFile(frontFile),
    loadImageFromFile(sideFile),
  ]);
  try {
    const frontCrop = cropImageContent(frontImg, 8);
    const sideCrop = cropImageContent(sideImg, 8);
    const maxSharedWidth = Math.min(
      size,
      Math.floor((size * frontCrop.w) / Math.max(1, frontCrop.h)),
      Math.floor((size * sideCrop.w) / Math.max(1, sideCrop.h))
    );
    const targetWidth = Math.max(1, Math.min(size, maxSharedWidth));
    return [
      renderCroppedToSquare(frontImg, frontCrop, size, targetWidth),
      renderCroppedToSquare(sideImg, sideCrop, size, targetWidth),
    ];
  } finally {
    URL.revokeObjectURL(frontImg.src);
    URL.revokeObjectURL(sideImg.src);
  }
}

function contentBBox(imageData, whiteTol = 8) {
  const { width, height, data } = imageData;
  const content = new Uint8Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];
      const isNonWhite = (255 - r) > whiteTol || (255 - g) > whiteTol || (255 - b) > whiteTol;
      const isContent = isNonWhite || a < 250;
      content[y * width + x] = isContent ? 1 : 0;
    }
  }

  const visited = new Uint8Array(width * height);
  const interiorBoxes = [];
  const edgeBoxes = [];
  const stack = [];

  const push = (x, y) => {
    stack.push(x, y);
  };

  for (let y0 = 0; y0 < height; y0++) {
    for (let x0 = 0; x0 < width; x0++) {
      const start = y0 * width + x0;
      if (!content[start] || visited[start]) continue;

      let minX = x0;
      let minY = y0;
      let maxX = x0;
      let maxY = y0;
      let area = 0;
      let touchTop = y0 === 0;
      let touchBottom = y0 === height - 1;
      let touchLeft = x0 === 0;
      let touchRight = x0 === width - 1;
      visited[start] = 1;
      stack.length = 0;
      push(x0, y0);

      while (stack.length) {
        const y = stack.pop();
        const x = stack.pop();
        area++;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;

        const neighbors = [
          [x - 1, y],
          [x + 1, y],
          [x, y - 1],
          [x, y + 1],
        ];
        for (const [nx, ny] of neighbors) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const ni = ny * width + nx;
          if (!content[ni] || visited[ni]) continue;
          visited[ni] = 1;
          if (ny === 0) touchTop = true;
          if (ny === height - 1) touchBottom = true;
          if (nx === 0) touchLeft = true;
          if (nx === width - 1) touchRight = true;
          push(nx, ny);
        }
      }

      const box = { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
      const bboxArea = Math.max(1, box.w * box.h);
      const touchesCount =
        (touchTop ? 1 : 0) +
        (touchBottom ? 1 : 0) +
        (touchLeft ? 1 : 0) +
        (touchRight ? 1 : 0);
      const coversMost = box.w >= Math.floor(width * 0.9) && box.h >= Math.floor(height * 0.9);
      const sparse = area <= Math.floor(bboxArea * 0.2);
      const isEdgeFrame = touchesCount >= 3 && coversMost && sparse;
      if (isEdgeFrame) continue;
      if (touchesCount === 0) interiorBoxes.push(box);
      else edgeBoxes.push({ area, ...box });
    }
  }

  let boxes = interiorBoxes;
  if (!boxes.length) {
    const areaFloor = Math.max(32, Math.floor(width * height * 0.005));
    boxes = edgeBoxes.filter((b) => b.area >= areaFloor);
  }
  if (!boxes.length) return null;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (const b of boxes) {
    if (b.x < minX) minX = b.x;
    if (b.y < minY) minY = b.y;
    if (b.x + b.w - 1 > maxX) maxX = b.x + b.w - 1;
    if (b.y + b.h - 1 > maxY) maxY = b.y + b.h - 1;
  }
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function luminance(data, i) {
  return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
}

export function autoInvert(imageData) {
  const { width, height, data } = imageData;
  const border = [];
  for (let x = 0; x < width; x++) {
    border.push(luminance(data, x * 4));
    border.push(luminance(data, ((height - 1) * width + x) * 4));
  }
  for (let y = 1; y < height - 1; y++) {
    border.push(luminance(data, (y * width) * 4));
    border.push(luminance(data, (y * width + width - 1) * 4));
  }
  let sumB = 0;
  for (const v of border) sumB += v;
  const meanB = sumB / border.length;
  let sumAll = 0;
  const n = width * height;
  for (let i = 0; i < n; i++) sumAll += luminance(data, i * 4);
  return meanB > sumAll / n + 8;
}

export function otsuThreshold(imageData) {
  const hist = new Array(256).fill(0);
  const { data } = imageData;
  const n = imageData.width * imageData.height;
  const gray = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const g = luminance(data, i * 4) | 0;
    gray[i] = g;
    hist[g]++;
  }
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0;
  let wB = 0;
  let bestVar = -1;
  let bestT = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = n - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) ** 2;
    if (v > bestVar) {
      bestVar = v;
      bestT = t;
    }
  }
  return { threshold: bestT, gray };
}

function morphClose(mask, size) {
  const out = mask.slice();
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let on = false;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < size && nx >= 0 && nx < size && mask[ny * size + nx]) on = true;
        }
      }
      if (on) out[y * size + x] = 1;
    }
  }
  const out2 = out.slice();
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let all = true;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny < 0 || ny >= size || nx < 0 || nx >= size || !out[ny * size + nx]) all = false;
        }
      }
      if (all) out2[y * size + x] = 1;
      else out2[y * size + x] = out[y * size + x];
    }
  }
  return out2;
}

export function extractMask(imageData, threshold, invertUser, autoThreshold) {
  const { width, height } = imageData;
  const size = width;
  const invAuto = autoInvert(imageData);
  const inv = invertUser != null ? invertUser : invAuto;
  const { threshold: t, gray } = autoThreshold
    ? otsuThreshold(imageData)
    : { threshold, gray: null };

  const mask = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const g = gray ? gray[i] : luminance(imageData.data, i * 4) | 0;
    const on = inv ? g <= t : g >= t;
    mask[i] = on ? 1 : 0;
  }
  return { mask, threshold: t, invert: inv, maskClosed: morphClose(mask, size) };
}

export function gentleClean(mask, size) {
  const out = mask.slice();
  const colOn = new Uint8Array(size);
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      if (out[y * size + x]) {
        colOn[x] = 1;
        break;
      }
    }
  }
  for (let x = 0; x < size; x++) {
    if (!colOn[x]) continue;
    const left = x > 0 && colOn[x - 1];
    const right = x < size - 1 && colOn[x + 1];
    if (left || right) continue;
    let fill = 0;
    for (let y = 0; y < size; y++) if (out[y * size + x]) fill++;
    if (fill >= Math.floor(size * 0.85)) {
      for (let y = 0; y < size; y++) out[y * size + x] = 0;
    }
  }
  return out;
}
