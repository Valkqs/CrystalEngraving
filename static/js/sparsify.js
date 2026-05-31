/**
 * Sparsify voxel clouds while preserving front (-Z) and side (-Y) projections.
 * Optimized for browser: O(voxels) memory, no full-cube scans in hot paths.
 */

import {
  applyEdgeWallWrap,
  boundaryFaces,
  pickYForSide,
  pickZForFront,
  yAtColumn,
  zPool,
} from "./carve_helpers.js";

function evenlyPick(indices, count) {
  if (count >= indices.length) return indices.slice();
  if (count <= 1) return [indices[Math.floor(indices.length / 2)]];
  const picked = new Set();
  for (let i = 0; i < count; i++) {
    const pos = Math.round((i * (indices.length - 1)) / Math.max(1, count - 1));
    picked.add(indices[pos]);
  }
  return [...picked].sort((a, b) => a - b);
}

function pickBestY(validY, x, z, points) {
  if (points.length === 0) return validY[Math.floor(validY.length / 2)];
  let bestY = validY[0];
  let bestD = -1;
  for (const y of validY) {
    let minD = Infinity;
    for (let i = 0; i < points.length; i++) {
      const py = points[i][0];
      const px = points[i][1];
      const pz = points[i][2];
      const d = (py - y) ** 2 + (px - x) ** 2 + (pz - z) ** 2;
      if (d < minD) minD = d;
    }
    if (minD > bestD) {
      bestD = minD;
      bestY = y;
    }
  }
  return bestY;
}

function repairCoverageMasks(
  selected,
  reqFront,
  reqSide,
  front,
  side,
  size,
  colCount,
  rowCount,
  points,
  depthFaceBridge,
  edgeStripFill,
  edgeWallWrap
) {
  const idx = (y, x, z) => y * size * size + x * size + z;
  let fallbackYFaces = null;
  let fallbackZFaces = null;
  if (edgeStripFill || edgeWallWrap) {
    const faces = boundaryFaces(front, side, size);
    fallbackYFaces = faces.yFaces;
    fallbackZFaces = faces.zFaces;
  }

  const add = (y, x, z) => {
    if (selected[idx(y, x, z)]) return;
    selected[idx(y, x, z)] = 1;
    colCount[y * size + x]++;
    rowCount[z * size + x]++;
    points.push([y, x, z]);
  };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!reqFront[y * size + x] || colCount[y * size + x] > 0) continue;
      const z = pickZForFront(y, x, side, size, depthFaceBridge, fallbackZFaces);
      if (z != null) add(y, x, z);
    }
  }

  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      if (!reqSide[z * size + x] || rowCount[z * size + x] > 0) continue;
      const y = pickYForSide(z, x, front, size, depthFaceBridge, fallbackYFaces);
      if (y != null) add(y, x, z);
    }
  }
}

export function sparsifyUniform(
  voxels,
  size,
  density = 0.4,
  uniformStrength = 0.6,
  targetFront = null,
  targetSide = null,
  depthFaceBridge = true,
  edgeStripFill = true,
  edgeWallWrap = true
) {
  density = Math.max(0.05, Math.min(1, density));
  uniformStrength = Math.max(0, Math.min(1, uniformStrength));

  const idx = (y, x, z) => y * size * size + x * size + z;
  let total = 0;
  for (let i = 0; i < voxels.length; i++) total += voxels[i];

  const reqFront = targetFront || new Uint8Array(size * size);
  const reqSide = targetSide || new Uint8Array(size * size);
  if (!targetFront) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        for (let z = 0; z < size; z++) {
          if (voxels[idx(y, x, z)]) {
            reqFront[y * size + x] = 1;
            break;
          }
        }
      }
    }
  }
  if (!targetSide) {
    for (let z = 0; z < size; z++) {
      for (let x = 0; x < size; x++) {
        for (let y = 0; y < size; y++) {
          if (voxels[idx(y, x, z)]) {
            reqSide[z * size + x] = 1;
            break;
          }
        }
      }
    }
  }

  const front = reqFront;
  const side = reqSide;

  if (density >= 0.999) {
    const out = new Uint8Array(voxels);
    const colCount = new Uint16Array(size * size);
    const rowCount = new Uint16Array(size * size);
    const points = [];
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        for (let z = 0; z < size; z++) {
          if (out[idx(y, x, z)]) {
            colCount[y * size + x]++;
            rowCount[z * size + x]++;
            points.push([y, x, z]);
          }
        }
      }
    }
    repairCoverageMasks(
      out,
      reqFront,
      reqSide,
      front,
      side,
      size,
      colCount,
      rowCount,
      points,
      depthFaceBridge,
      edgeStripFill,
      edgeWallWrap
    );
    if (edgeWallWrap) applyEdgeWallWrap(out, front, side, size);
    return out;
  }

  if (total === 0 && !reqFront.some?.((v) => v) && !reqSide.some?.((v) => v)) {
    let any = false;
    for (let i = 0; i < reqFront.length; i++) if (reqFront[i]) any = true;
    for (let i = 0; i < reqSide.length; i++) if (reqSide[i]) any = true;
    if (!any) return voxels;
  }

  const selected = new Uint8Array(voxels.length);
  const colCount = new Uint16Array(size * size);
  const rowCount = new Uint16Array(size * size);
  const points = [];

  const addPoint = (y, x, z) => {
    if (selected[idx(y, x, z)]) return;
    selected[idx(y, x, z)] = 1;
    colCount[y * size + x]++;
    rowCount[z * size + x]++;
    points.push([y, x, z]);
  };

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!reqFront[y * size + x]) continue;
      let zList = zPool(side, size, x, depthFaceBridge);
      if (!zList.length) continue;
      const n = Math.max(1, Math.ceil(zList.length * density));
      const keep =
        uniformStrength > 0.2
          ? evenlyPick(zList, n)
          : zList.filter((_, i) => i % Math.max(1, Math.floor(zList.length / n)) === 0).slice(0, n);
      for (const z of keep) addPoint(y, x, z);
    }
  }

  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      if (!reqSide[z * size + x] || rowCount[z * size + x] > 0) continue;
      let yList = yAtColumn(front, size, x);
      if (!yList.length) continue;
      const y =
        uniformStrength > 0.2 && points.length < 8000
          ? pickBestY(yList, x, z, points)
          : yList[(z * yList.length) / size % yList.length | 0];
      addPoint(y, x, z);
    }
  }

  // Phase 3: 3D grid dedup (iterate points list only)
  if (uniformStrength > 0.15 && points.length > 0) {
    const cell = Math.max(2, Math.round(size * (0.06 + 0.22 * uniformStrength * (1 - density * 0.5))));
    const buckets = new Map();
    for (const [y, x, z] of points) {
      const ky = Math.floor(y / cell);
      const kx = Math.floor(x / cell);
      const kz = Math.floor(z / cell);
      const k = `${ky},${kx},${kz}`;
      const cy = (ky + 0.5) * cell;
      const cx = (kx + 0.5) * cell;
      const cz = (kz + 0.5) * cell;
      const dist = (y - cy) ** 2 + (x - cx) ** 2 + (z - cz) ** 2;
      const prev = buckets.get(k);
      if (!prev || dist < prev.dist) buckets.set(k, { dist, y, x, z });
    }
    selected.fill(0);
    colCount.fill(0);
    rowCount.fill(0);
    points.length = 0;
    for (const { y, x, z } of buckets.values()) addPoint(y, x, z);
    repairCoverageMasks(
      selected,
      reqFront,
      reqSide,
      front,
      side,
      size,
      colCount,
      rowCount,
      points,
      depthFaceBridge,
      edgeStripFill,
      edgeWallWrap
    );
  }

  // Phase 4: batch greedy removal (max iterations capped)
  let reqFrontCount = 0;
  let reqSideCount = 0;
  for (let i = 0; i < reqFront.length; i++) reqFrontCount += reqFront[i];
  for (let i = 0; i < reqSide.length; i++) reqSideCount += reqSide[i];

  let target = Math.max(Math.floor((reqFrontCount + reqSideCount) / 2), Math.floor(total * density));
  target = Math.max(target, Math.floor(reqFrontCount * 0.3));

  const maxPasses = 64;
  let pass = 0;
  while (points.length > target && pass < maxPasses) {
    pass++;
    const batch = Math.min(256, points.length - target);
    const candidates = [];
    for (let i = 0; i < points.length; i++) {
      const [y, x, z] = points[i];
      const col = colCount[y * size + x];
      const row = rowCount[z * size + x];
      if (col <= 1 && row <= 1) continue;
      candidates.push({ score: col + row, i, y, x, z });
    }
    if (!candidates.length) break;
    candidates.sort((a, b) => b.score - a.score);
    const removeIdx = new Set();
    for (let k = 0; k < Math.min(batch, candidates.length); k++) {
      removeIdx.add(candidates[k].i);
    }
    const next = [];
    for (let i = 0; i < points.length; i++) {
      if (removeIdx.has(i)) {
        const [y, x, z] = points[i];
        selected[idx(y, x, z)] = 0;
        colCount[y * size + x]--;
        rowCount[z * size + x]--;
      } else {
        next.push(points[i]);
      }
    }
    points.length = 0;
    points.push(...next);
    repairCoverageMasks(
      selected,
      reqFront,
      reqSide,
      front,
      side,
      size,
      colCount,
      rowCount,
      points,
      depthFaceBridge,
      edgeStripFill,
      edgeWallWrap
    );
  }

  if (edgeWallWrap) applyEdgeWallWrap(selected, front, side, size);
  return selected;
}

export function voxelsToPoints(selected, size) {
  const idx = (y, x, z) => y * size * size + x * size + z;
  const points = [];
  const projectionFront = Array.from({ length: size }, () => Array(size).fill(0));
  const projectionSide = Array.from({ length: size }, () => Array(size).fill(0));

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      for (let z = 0; z < size; z++) {
        if (!selected[idx(y, x, z)]) continue;
        points.push([x, y, z]);
        projectionFront[y][x] = 1;
        projectionSide[z][x] = 1;
      }
    }
  }
  return { points, projectionFront, projectionSide };
}
