"""Sparsify voxel clouds while preserving dual-view projections."""

from __future__ import annotations

import numpy as np

from carve_helpers import (
    apply_edge_wall_wrap,
    boundary_faces,
    pick_y_for_side,
    pick_z_for_front,
    y_at_column,
    z_at_column,
)


def _evenly_pick(indices: np.ndarray, count: int) -> np.ndarray:
    if count >= indices.size:
        return indices
    if count <= 1:
        return indices[indices.size // 2 : indices.size // 2 + 1]
    pos = np.linspace(0, indices.size - 1, count)
    return indices[np.unique(np.round(pos).astype(int))]


def _repair_coverage(
    selected: np.ndarray,
    target_front: np.ndarray,
    target_side: np.ndarray,
    front: np.ndarray,
    side: np.ndarray,
    *,
    depth_face_bridge: bool = True,
    edge_strip_fill: bool = True,
    edge_wall_wrap: bool = True,
) -> None:
    size = front.shape[0]
    fallback_y_faces: np.ndarray | None = None
    fallback_z_faces: np.ndarray | None = None
    if edge_strip_fill or edge_wall_wrap:
        fallback_y_faces, fallback_z_faces = boundary_faces(front, side)

    for y in range(size):
        for x in range(size):
            if not target_front[y, x] or np.any(selected[y, x, :]):
                continue
            z = pick_z_for_front(
                y,
                x,
                side,
                size,
                depth_face_bridge,
                fallback_z_faces=fallback_z_faces,
            )
            if z is not None:
                selected[y, x, z] = True

    for z in range(size):
        for x in range(size):
            if not target_side[z, x] or np.any(selected[:, x, z]):
                continue
            y = pick_y_for_side(
                z,
                x,
                front,
                size,
                depth_face_bridge,
                fallback_y_faces=fallback_y_faces,
            )
            if y is not None:
                selected[y, x, z] = True


def _z_pool(side: np.ndarray, x: int, depth_face_bridge: bool) -> np.ndarray:
    strict = z_at_column(side, x)
    if strict.size == 0:
        return strict
    z_lo, z_hi = int(strict[0]), int(strict[-1])
    if depth_face_bridge and z_hi > z_lo + 1:
        return np.unique(np.concatenate([strict, np.array([z_lo, z_hi], dtype=int)]))
    return strict


def _pick_best_y(valid_y: np.ndarray, x: int, z: int, existing: np.ndarray) -> int:
    if existing.size == 0:
        return int(valid_y[valid_y.size // 2])
    best_y = int(valid_y[0])
    best_d = -1.0
    for y in valid_y:
        d = float(np.min((existing[:, 0] - y) ** 2 + (existing[:, 1] - x) ** 2 + (existing[:, 2] - z) ** 2))
        if d > best_d:
            best_d = d
            best_y = int(y)
    return best_y


def sparsify_uniform(
    voxels: np.ndarray,
    density: float = 0.4,
    uniform_strength: float = 0.6,
    target_front: np.ndarray | None = None,
    target_side: np.ndarray | None = None,
    depth_face_bridge: bool = True,
    edge_strip_fill: bool = True,
    edge_wall_wrap: bool = True,
) -> np.ndarray:
    density = float(np.clip(density, 0.05, 1.0))
    uniform_strength = float(np.clip(uniform_strength, 0.0, 1.0))

    size = voxels.shape[0]
    req_front = target_front if target_front is not None else np.any(voxels, axis=2)
    req_side = target_side if target_side is not None else np.any(voxels, axis=0).T
    front = req_front
    side = req_side

    if density >= 0.999:
        out = voxels.copy()
        _repair_coverage(
            out,
            req_front,
            req_side,
            front,
            side,
            depth_face_bridge=depth_face_bridge,
            edge_strip_fill=edge_strip_fill,
            edge_wall_wrap=edge_wall_wrap,
        )
        if edge_wall_wrap:
            apply_edge_wall_wrap(out, front, side)
        return out

    total = int(voxels.sum())
    selected = np.zeros_like(voxels)

    for y in range(size):
        for x in range(size):
            if not req_front[y, x]:
                continue
            z_cands = _z_pool(side, x, depth_face_bridge)
            if z_cands.size == 0:
                continue
            n = max(1, int(np.ceil(z_cands.size * density)))
            if uniform_strength > 0.2:
                keep = _evenly_pick(z_cands, n)
            else:
                step = max(1, z_cands.size // n)
                keep = z_cands[::step][:n]
            selected[y, x, keep] = True

    existing = np.argwhere(selected)
    for z in range(size):
        for x in range(size):
            if not req_side[z, x] or np.any(selected[:, x, z]):
                continue
            y_cands = y_at_column(front, x)
            if y_cands.size == 0:
                continue
            if uniform_strength > 0.2 and existing.size < 8000:
                y = _pick_best_y(y_cands, x, z, existing)
            else:
                y = int(y_cands[(z * y_cands.size) // max(1, size) % y_cands.size])
            selected[y, x, z] = True
            existing = np.argwhere(selected)

    if uniform_strength > 0.15:
        cell = max(2, int(round(size * (0.06 + 0.22 * uniform_strength * (1.0 - density * 0.5)))))
        buckets: dict[tuple[int, int, int], tuple[float, int, int, int]] = {}
        for y, x, z in np.argwhere(selected):
            key = (y // cell, x // cell, z // cell)
            cy = (key[0] + 0.5) * cell
            cx = (key[1] + 0.5) * cell
            cz = (key[2] + 0.5) * cell
            dist = (y - cy) ** 2 + (x - cx) ** 2 + (z - cz) ** 2
            if key not in buckets or dist < buckets[key][0]:
                buckets[key] = (dist, int(y), int(x), int(z))
        selected.fill(False)
        for _, y, x, z in buckets.values():
            selected[y, x, z] = True
        _repair_coverage(
            selected,
            req_front,
            req_side,
            front,
            side,
            depth_face_bridge=depth_face_bridge,
            edge_strip_fill=edge_strip_fill,
            edge_wall_wrap=edge_wall_wrap,
        )

    target = max(int(req_front.sum() + req_side.sum()) // 2, int(total * density))
    target = max(target, int(req_front.sum() * 0.3))

    coords = np.argwhere(selected)
    max_passes = 64
    for _ in range(max_passes):
        if coords.shape[0] <= target:
            break
        col = selected.sum(axis=2)
        row = selected.sum(axis=0)
        removable = []
        for y, x, z in coords:
            if col[y, x] <= 1 and row[x, z] <= 1:
                continue
            removable.append((int(col[y, x] + row[x, z]), int(y), int(x), int(z)))
        if not removable:
            break
        removable.sort(reverse=True)
        batch = min(256, coords.shape[0] - target, len(removable))
        for _, y, x, z in removable[:batch]:
            selected[y, x, z] = False
        _repair_coverage(
            selected,
            req_front,
            req_side,
            front,
            side,
            depth_face_bridge=depth_face_bridge,
            edge_strip_fill=edge_strip_fill,
            edge_wall_wrap=edge_wall_wrap,
        )
        coords = np.argwhere(selected)

    _repair_coverage(
        selected,
        req_front,
        req_side,
        front,
        side,
        depth_face_bridge=depth_face_bridge,
        edge_strip_fill=edge_strip_fill,
        edge_wall_wrap=edge_wall_wrap,
    )
    if edge_wall_wrap:
        apply_edge_wall_wrap(selected, front, side)
    return selected
