"""Build triangulated mesh from voxel grid (cubes) for STL/OBJ export."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# Cube vertex offsets (8 corners of unit cube at origin)
_CUBE_VERTS = np.array(
    [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ],
    dtype=np.float32,
)

# 6 faces, each 4 vertex indices. Order chosen so the right-hand normal points OUTWARD.
# Face 0: -X (left),  Face 1: +X (right)
# Face 2: -Y (bottom), Face 3: +Y (top)
# Face 4: -Z (back),  Face 5: +Z (front)
_CUBE_FACES = [
    (0, 3, 7, 4),  # -X  (normal (-1, 0, 0))
    (1, 2, 6, 5),  # +X  (normal (+1, 0, 0))
    (0, 1, 5, 4),  # -Y  (normal (0, -1, 0))
    (3, 2, 6, 7),  # +Y  (normal (0, +1, 0))
    (0, 1, 2, 3),  # -Z  (normal (0, 0, -1))
    (4, 5, 6, 7),  # +Z  (normal (0, 0, +1))
]

# 6 neighbor offsets (one per face) — a face is exposed if no voxel in that direction.
_NEIGHBOR_OFFSETS = [
    (-1, 0, 0),  # -X
    (1, 0, 0),   # +X
    (0, -1, 0),  # -Y
    (0, 1, 0),   # +Y
    (0, 0, -1),  # -Z
    (0, 0, 1),   # +Z
]


def voxel_grid_from_points(points: List[List[float]], size: int) -> np.ndarray:
    """Reconstruct a boolean voxel grid (size x size x size) from point list."""
    grid = np.zeros((size, size, size), dtype=bool)
    for pt in points:
        x, y, z = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
        if 0 <= x < size and 0 <= y < size and 0 <= z < size:
            grid[y, x, z] = True
    return grid


def build_mesh(
    points: List[List[float]],
    size: int,
) -> Dict[str, np.ndarray]:
    """
    Build a triangulated mesh of exposed cube faces.

    Returns dict with:
        vertices: (N, 3) float32
        triangles: (M, 3) int32 (indices into vertices)
    """
    if not points:
        return {
            "vertices": np.zeros((0, 3), dtype=np.float32),
            "triangles": np.zeros((0, 3), dtype=np.int32),
        }

    grid = voxel_grid_from_points(points, size)
    return build_mesh_from_grid(grid)


def build_mesh_from_grid(grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Build mesh directly from boolean voxel grid."""
    size_y, size_x, size_z = grid.shape
    if grid.sum() == 0:
        return {
            "vertices": np.zeros((0, 3), dtype=np.float32),
            "triangles": np.zeros((0, 3), dtype=np.int32),
        }

    padded = np.pad(grid, 1, mode="constant", constant_values=False)

    vertices_list: List[np.ndarray] = []
    triangles_list: List[np.ndarray] = []
    base_index = 0

    # Iterate over voxel positions
    ys, xs, zs = np.where(grid)
    for y, x, z in zip(ys, xs, zs):
        py, px, pz = y + 1, x + 1, z + 1  # adjust for padding
        for face_idx, (dy, dx, dz) in enumerate(_NEIGHBOR_OFFSETS):
            if not padded[py + dy, px + dx, pz + dz]:
                face = _CUBE_FACES[face_idx]
                face_verts = _CUBE_VERTS[list(face)] + np.array([x, y, z], dtype=np.float32)
                vertices_list.append(face_verts)
                v0, v1, v2, v3 = base_index, base_index + 1, base_index + 2, base_index + 3
                triangles_list.append(np.array([v0, v1, v2, v0, v2, v3], dtype=np.int32).reshape(2, 3))
                base_index += 4

    if not vertices_list:
        return {
            "vertices": np.zeros((0, 3), dtype=np.float32),
            "triangles": np.zeros((0, 3), dtype=np.int32),
        }

    vertices = np.concatenate(vertices_list, axis=0).astype(np.float32)
    triangles = np.concatenate(triangles_list, axis=0).astype(np.int32)

    return {"vertices": vertices, "triangles": triangles}


def compute_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals (averaged from adjacent face normals)."""
    normals = np.zeros_like(vertices)
    if len(triangles) == 0:
        return normals

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    return (normals / lens).astype(np.float32)


def to_binary_stl(vertices: np.ndarray, triangles: np.ndarray) -> bytes:
    """
    Encode mesh as binary STL (little-endian).
    Header: 80 bytes. Per-triangle: normal(3*f4) + 3*vertex(3*f4) + u16 attr = 50 bytes.
    """
    n_tri = len(triangles)
    if n_tri == 0:
        header = b"\x00" * 80
        return header + (0).to_bytes(4, "little")

    normals = compute_normals(vertices, triangles)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    n0 = normals[triangles[:, 0]]

    # Triangles: 12 floats (normal + 3 verts) + 2 bytes attr = 50 bytes each
    body = bytearray()
    for i in range(n_tri):
        body += n0[i].tobytes()              # 12 bytes: face normal
        body += v0[i].tobytes()              # 12 bytes: v0
        body += v1[i].tobytes()              # 12 bytes: v1
        body += v2[i].tobytes()              # 12 bytes: v2
        body += (0).to_bytes(2, "little")    #  2 bytes: attribute byte count

    header = b"CrystalEngraving voxel mesh".ljust(80, b"\x00")
    count_bytes = n_tri.to_bytes(4, "little")
    return header + count_bytes + bytes(body)


def to_obj_text(vertices: np.ndarray, triangles: np.ndarray) -> str:
    """Encode mesh as ASCII OBJ."""
    if len(vertices) == 0:
        return "# CrystalEngraving voxel mesh\n# empty\n"

    # Center around origin: shift so mesh center is at (0, 0, 0)
    lines = ["# CrystalEngraving voxel mesh", f"# vertices: {len(vertices)}", f"# faces: {len(triangles)}"]
    for v in vertices:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    for tri in triangles:
        lines.append(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}")
    return "\n".join(lines) + "\n"
