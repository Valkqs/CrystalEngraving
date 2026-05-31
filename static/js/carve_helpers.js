/** Dual-projection carving with depth face bridging. */

export function closeSideZGaps(side, size, maxGap) {
  if (maxGap <= 0) return side.slice();
  const out = side.slice();
  for (let x = 0; x < size; x++) {
    const zs = [];
    for (let z = 0; z < size; z++) {
      if (out[z * size + x]) zs.push(z);
    }
    for (let i = 0; i < zs.length - 1; i++) {
      const a = zs[i];
      const b = zs[i + 1];
      if (b - a > 1 && b - a <= maxGap + 1) {
        for (let z = a; z <= b; z++) out[z * size + x] = 1;
      }
    }
  }
  return out;
}

export function zAtColumn(side, size, x) {
  const zs = [];
  for (let z = 0; z < size; z++) {
    if (side[z * size + x]) zs.push(z);
  }
  return zs;
}

export function yAtColumn(front, size, x) {
  const ys = [];
  for (let y = 0; y < size; y++) {
    if (front[y * size + x]) ys.push(y);
  }
  return ys;
}

export function boundaryFaces(front, side, size) {
  let yLo = -1;
  let yHi = -1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!front[y * size + x]) continue;
      if (yLo < 0) yLo = y;
      yHi = y;
      break;
    }
  }
  const yFaces = yLo < 0 ? [] : yLo === yHi ? [yLo] : [yLo, yHi];

  let zLo = -1;
  let zHi = -1;
  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      if (!side[z * size + x]) continue;
      if (zLo < 0) zLo = z;
      zHi = z;
      break;
    }
  }
  const zFaces = zLo < 0 ? [] : zLo === zHi ? [zLo] : [zLo, zHi];
  return { yFaces, zFaces };
}

export function applyEdgeWallWrap(voxels, front, side, size) {
  const idx = (y, x, z) => y * size * size + x * size + z;
  const { yFaces, zFaces } = boundaryFaces(front, side, size);
  if (yFaces.length === 0 && zFaces.length === 0) return;

  for (const z of zFaces) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (front[y * size + x]) voxels[idx(y, x, z)] = 1;
      }
    }
  }
  for (const y of yFaces) {
    for (let z = 0; z < size; z++) {
      for (let x = 0; x < size; x++) {
        if (side[z * size + x]) voxels[idx(y, x, z)] = 1;
      }
    }
  }
}

export function pickZForFront(y, x, side, size, depthFaceBridge, fallbackZFaces = null) {
  const strict = zAtColumn(side, size, x);
  if (!strict.length) {
    if (fallbackZFaces && fallbackZFaces.length) {
      return fallbackZFaces[(y * fallbackZFaces.length) / size % fallbackZFaces.length | 0];
    }
    return null;
  }
  if (depthFaceBridge) {
    const zLo = strict[0];
    const zHi = strict[strict.length - 1];
    if (zHi > zLo + 1) {
      const pool = [zLo, zHi];
      return pool[(y * pool.length) / size % pool.length | 0];
    }
  }
  return strict[(y * strict.length) / size % strict.length | 0];
}

export function pickYForSide(z, x, front, size, depthFaceBridge, fallbackYFaces = null) {
  const strict = yAtColumn(front, size, x);
  if (!strict.length) {
    if (fallbackYFaces && fallbackYFaces.length) {
      return fallbackYFaces[(z * fallbackYFaces.length) / size % fallbackYFaces.length | 0];
    }
    return null;
  }
  if (depthFaceBridge) {
    const yLo = strict[0];
    const yHi = strict[strict.length - 1];
    if (yHi > yLo + 1) {
      const pool = [yLo, yHi];
      return pool[(z * pool.length) / size % pool.length | 0];
    }
  }
  return strict[(z * strict.length) / size % strict.length | 0];
}

export function zPool(side, size, x, depthFaceBridge) {
  const strict = zAtColumn(side, size, x);
  if (!strict.length) return strict;
  const zLo = strict[0];
  const zHi = strict[strict.length - 1];
  if (depthFaceBridge && zHi > zLo + 1) {
    const set = new Set(strict);
    set.add(zLo);
    set.add(zHi);
    return [...set].sort((a, b) => a - b);
  }
  return strict;
}

export function carveDualCover(
  front,
  side,
  size,
  depthFaceBridge = true,
  edgeStripFill = true,
  edgeWallWrap = true
) {
  const voxels = new Uint8Array(size * size * size);
  const idx = (y, x, z) => y * size * size + x * size + z;
  let fallbackYFaces = null;
  let fallbackZFaces = null;

  if (edgeStripFill || edgeWallWrap) {
    const faces = boundaryFaces(front, side, size);
    fallbackYFaces = faces.yFaces;
    fallbackZFaces = faces.zFaces;
  }

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!front[y * size + x]) continue;
      for (let z = 0; z < size; z++) {
        if (side[z * size + x]) voxels[idx(y, x, z)] = 1;
      }
    }
  }

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (!front[y * size + x]) continue;
      let any = false;
      for (let z = 0; z < size; z++) {
        if (voxels[idx(y, x, z)]) {
          any = true;
          break;
        }
      }
      if (any) continue;
      const z = pickZForFront(y, x, side, size, depthFaceBridge, fallbackZFaces);
      if (z != null) voxels[idx(y, x, z)] = 1;
    }
  }

  for (let z = 0; z < size; z++) {
    for (let x = 0; x < size; x++) {
      if (!side[z * size + x]) continue;
      let any = false;
      for (let y = 0; y < size; y++) {
        if (voxels[idx(y, x, z)]) {
          any = true;
          break;
        }
      }
      if (any) continue;
      const y = pickYForSide(z, x, front, size, depthFaceBridge, fallbackYFaces);
      if (y != null) voxels[idx(y, x, z)] = 1;
    }
  }

  if (edgeWallWrap) {
    applyEdgeWallWrap(voxels, front, side, size);
  }

  return voxels;
}
