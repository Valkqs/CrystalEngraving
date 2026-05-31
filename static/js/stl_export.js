/**
 * Export voxel points to STL/OBJ (surface faces only).
 */

function key(x, y, z) {
  return `${x}|${y}|${z}`;
}

function buildSurfaceQuads(points) {
  const occupied = new Set();
  for (const p of points) {
    occupied.add(key(p[0] | 0, p[1] | 0, p[2] | 0));
  }

  const faces = [
    {
      n: [1, 0, 0],
      d: [1, 0, 0],
      q: [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
      ],
    },
    {
      n: [-1, 0, 0],
      d: [-1, 0, 0],
      q: [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
      ],
    },
    {
      n: [0, 1, 0],
      d: [0, 1, 0],
      q: [
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
      ],
    },
    {
      n: [0, -1, 0],
      d: [0, -1, 0],
      q: [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
      ],
    },
    {
      n: [0, 0, 1],
      d: [0, 0, 1],
      q: [
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
      ],
    },
    {
      n: [0, 0, -1],
      d: [0, 0, -1],
      q: [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
      ],
    },
  ];

  const quads = [];
  for (const p of points) {
    const x = p[0] | 0;
    const y = p[1] | 0;
    const z = p[2] | 0;
    for (const f of faces) {
      const nx = x + f.d[0];
      const ny = y + f.d[1];
      const nz = z + f.d[2];
      if (occupied.has(key(nx, ny, nz))) continue;

      const v0 = [x + f.q[0][0], y + f.q[0][1], z + f.q[0][2]];
      const v1 = [x + f.q[1][0], y + f.q[1][1], z + f.q[1][2]];
      const v2 = [x + f.q[2][0], y + f.q[2][1], z + f.q[2][2]];
      const v3 = [x + f.q[3][0], y + f.q[3][1], z + f.q[3][2]];
      quads.push({ n: f.n, v: [v0, v1, v2, v3] });
    }
  }
  return quads;
}

function quadsToTriangles(quads) {
  const triangles = [];
  for (const q of quads) {
    triangles.push({ n: q.n, v: [q.v[0], q.v[1], q.v[2]] });
    triangles.push({ n: q.n, v: [q.v[0], q.v[2], q.v[3]] });
  }
  return triangles;
}

function trianglesToBinaryStl(triangles) {
  const TRI_SIZE = 50;
  const HEADER_SIZE = 84;
  const buffer = new ArrayBuffer(HEADER_SIZE + triangles.length * TRI_SIZE);
  const view = new DataView(buffer);

  const headerText = "EngineerDesign voxel STL";
  for (let i = 0; i < headerText.length && i < 80; i++) {
    view.setUint8(i, headerText.charCodeAt(i));
  }
  view.setUint32(80, triangles.length, true);

  let offset = 84;
  for (const tri of triangles) {
    view.setFloat32(offset, tri.n[0], true);
    view.setFloat32(offset + 4, tri.n[1], true);
    view.setFloat32(offset + 8, tri.n[2], true);
    offset += 12;
    for (let i = 0; i < 3; i++) {
      view.setFloat32(offset, tri.v[i][0], true);
      view.setFloat32(offset + 4, tri.v[i][1], true);
      view.setFloat32(offset + 8, tri.v[i][2], true);
      offset += 12;
    }
    view.setUint16(offset, 0, true);
    offset += 2;
  }

  return buffer;
}

export function exportPointsToStl(points, filename = "voxel_model.stl") {
  if (!points || !points.length) {
    throw new Error("当前没有可导出的点阵");
  }
  const triangles = quadsToTriangles(buildSurfaceQuads(points));
  if (!triangles.length) {
    throw new Error("未生成可导出的表面");
  }
  const stlBuffer = trianglesToBinaryStl(triangles);
  const blob = new Blob([stlBuffer], { type: "model/stl" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return { triangleCount: triangles.length };
}

function pointsKey(p) {
  return `${p[0]},${p[1]},${p[2]}`;
}

function quadsToObjText(quads) {
  const vertIndex = new Map();
  const vertices = [];
  const faces = [];

  const addVertex = (p) => {
    const k = pointsKey(p);
    const existing = vertIndex.get(k);
    if (existing != null) return existing;
    const idx = vertices.length + 1;
    vertices.push(p);
    vertIndex.set(k, idx);
    return idx;
  };

  for (const q of quads) {
    const i0 = addVertex(q.v[0]);
    const i1 = addVertex(q.v[1]);
    const i2 = addVertex(q.v[2]);
    const i3 = addVertex(q.v[3]);
    faces.push([i0, i1, i2, i3]);
  }

  const lines = ["# EngineerDesign voxel OBJ"];
  for (const v of vertices) {
    lines.push(`v ${v[0]} ${v[1]} ${v[2]}`);
  }
  for (const f of faces) {
    lines.push(`f ${f[0]} ${f[1]} ${f[2]} ${f[3]}`);
  }
  return { text: `${lines.join("\n")}\n`, vertexCount: vertices.length, faceCount: faces.length };
}

export function exportPointsToObj(points, filename = "voxel_model.obj") {
  if (!points || !points.length) {
    throw new Error("当前没有可导出的点阵");
  }
  const quads = buildSurfaceQuads(points);
  if (!quads.length) {
    throw new Error("未生成可导出的表面");
  }
  const obj = quadsToObjText(quads);
  const blob = new Blob([obj.text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return { vertexCount: obj.vertexCount, faceCount: obj.faceCount };
}
