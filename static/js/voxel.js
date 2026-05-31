import { sparsifyUniform, voxelsToPoints } from "./sparsify.js";
import {
  extractMask,
  gentleClean,
  loadImageBitmapFit,
  loadImageBitmapPairFit,
} from "./image_preprocess.js";
import { carveDualCover, closeSideZGaps } from "./carve_helpers.js";

export { loadImageBitmapFit as loadImageBitmap };

function dilateMask(mask, size, radius) {
  if (radius <= 0) return mask.slice();
  let out = mask.slice();
  for (let iter = 0; iter < radius; iter++) {
    const next = out.slice();
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (!out[y * size + x]) continue;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const ny = y + dy;
            const nx = x + dx;
            if (ny >= 0 && ny < size && nx >= 0 && nx < size) {
              next[ny * size + nx] = 1;
            }
          }
        }
      }
    }
    out = next;
  }
  return out;
}

function erodeHorizontal(mask, size, radius = 1) {
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let keep = true;
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= size || !mask[y * size + nx]) {
          keep = false;
          break;
        }
      }
      if (keep) out[y * size + x] = 1;
    }
  }
  return out;
}

function dilateHorizontal(mask, size, radius = 1) {
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < size && mask[y * size + nx]) {
          out[y * size + x] = 1;
          break;
        }
      }
    }
  }
  return out;
}

function pruneIsolatedColumns(mask, size) {
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
    if (fill >= Math.floor(size * 0.75) || fill <= 2) {
      for (let y = 0; y < size; y++) out[y * size + x] = 0;
    }
  }
  return out;
}

function cleanMask(mask, size) {
  let out = erodeHorizontal(mask, size, 1);
  out = dilateHorizontal(out, size, 1);
  return pruneIsolatedColumns(out, size);
}

function columnProfile(mask, size, axis) {
  const prof = new Uint8Array(size);
  if (axis === "x") {
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        if (mask[y * size + x]) {
          prof[x] = 1;
          break;
        }
      }
    }
  }
  return prof;
}

function alignSideX(front, side, size) {
  const pf = columnProfile(front, size, "x");
  let hasFront = false;
  for (let i = 0; i < size; i++) if (pf[i]) hasFront = true;
  if (!hasFront) return side;

  let bestShift = 0;
  let bestScore = -1;
  const ps0 = columnProfile(side, size, "x");
  for (let shift = 0; shift < size; shift++) {
    let score = 0;
    for (let x = 0; x < size; x++) {
      const xs = (x + shift) % size;
      if (pf[x] && ps0[xs]) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestShift = shift;
    }
  }
  if (bestShift === 0) return side;

  const out = new Uint8Array(side.length);
  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      const src = (x - bestShift + size) % size;
      out[z * size + x] = side[z * size + src];
    }
  }
  return out;
}

function dilateHorizontalFront(front, size, radius) {
  if (radius <= 0) return front.slice();
  const out = new Uint8Array(front.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < size && front[y * size + nx]) {
          out[y * size + x] = 1;
          break;
        }
      }
    }
  }
  return out;
}

function dilateHorizontalSide(side, size, radius) {
  if (radius <= 0) return side.slice();
  const out = new Uint8Array(side.length);
  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < size && side[z * size + nx]) {
          out[z * size + x] = 1;
          break;
        }
      }
    }
  }
  return out;
}

export function generateVoxels(frontMask, sideMask, size, options = {}) {
  const {
    dilate = 0,
    alignX = true,
    cleanMask: doClean = false,
    overlapDilate = 1,
    depthFaceBridge = true,
    edgeWallWrap = true,
    closeSideZGaps: closeZGaps = 2,
    density = 0.75,
    uniformStrength = 0.25,
    detailMode = true,
  } = options;
  let front = frontMask.maskClosed || frontMask.mask;
  let side = sideMask.maskClosed || sideMask.mask;

  let dil = dilate;
  if (detailMode) {
    if (dil === 1) dil = 0;
  }

  if (doClean && !detailMode) {
    front = gentleClean(front, size);
    side = gentleClean(side, size);
  }

  front = dilateMask(front, size, dil);
  side = dilateMask(side, size, dil);

  if (alignX) {
    side = alignSideX(front, side, size);
  }

  const od = Math.max(0, Math.min(3, overlapDilate | 0));
  if (od > 0) {
    front = dilateHorizontalFront(front, size, od);
    side = dilateHorizontalSide(side, size, od);
  }

  const gapClose = Math.max(0, Math.min(8, closeZGaps | 0));
  if (gapClose > 0) {
    side = closeSideZGaps(side, size, gapClose);
  }

  const voxels = carveDualCover(front, side, size, depthFaceBridge, edgeWallWrap);

  let countFull = 0;
  for (let i = 0; i < voxels.length; i++) countFull += voxels[i];

  const sparse = sparsifyUniform(
    voxels,
    size,
    density,
    uniformStrength,
    front,
    side,
    depthFaceBridge,
    edgeWallWrap
  );
  const { points, projectionFront, projectionSide } = voxelsToPoints(sparse, size);

  return {
    size,
    count: points.length,
    count_full: countFull,
    points,
    projection_front: projectionFront,
    projection_side: projectionSide,
    threshold_front: frontMask.threshold,
    threshold_side: sideMask.threshold,
    invert_front: frontMask.invert,
    invert_side: sideMask.invert,
  };
}

export async function generateFromFiles(frontFile, sideFile, size, options = {}) {
  const {
    threshold = 128,
    invert = null,
    autoThreshold = true,
    dilate = 0,
    alignX = true,
    cleanMask = false,
    overlapDilate = 1,
    depthFaceBridge = true,
    edgeWallWrap = true,
    closeSideZGaps: closeZGaps = 2,
    density = 0.75,
    uniformStrength = 0.25,
    detailMode = true,
  } = options;

  const invertUser = invert === null || invert === undefined ? null : !!invert;

  const [frontImg, sideImg] = await loadImageBitmapPairFit(frontFile, sideFile, size);
  const frontMask = extractMask(frontImg, threshold, invertUser, autoThreshold);
  const sideMask = extractMask(sideImg, threshold, invertUser, autoThreshold);
  return generateVoxels(frontMask, sideMask, size, {
    dilate,
    alignX,
    cleanMask,
    overlapDilate,
    density,
    uniformStrength,
    depthFaceBridge,
    edgeWallWrap,
    closeSideZGaps: closeZGaps,
    detailMode,
  });
}
