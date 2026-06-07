/**
 * High-quality 2D mask extraction (center-fit, luminance, auto-invert).
 */

export function loadImageBitmapFit(file, size) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, size, size);
      const scale = Math.min(size / img.width, size / img.height);
      const dw = Math.max(1, Math.round(img.width * scale));
      const dh = Math.max(1, Math.round(img.height * scale));
      const ox = Math.floor((size - dw) / 2);
      const oy = Math.floor((size - dh) / 2);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, ox, oy, dw, dh);
      resolve(ctx.getImageData(0, 0, size, size));
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
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
