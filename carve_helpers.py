"""Dual-projection voxel carving helpers."""

from typing import Optional

import numpy as np


def close_side_z_gaps(side: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """
    Per X column, fill short gaps along Z in the side silhouette (side view holes).
    """
    if max_gap <= 0:
        return side
    side = side.copy()
    size = side.shape[0]
    for x in range(size):
        zs = np.where(side[:, x])[0]
        if zs.size < 2:
            continue
        for i in range(len(zs) - 1):
            a, b = int(zs[i]), int(zs[i + 1])
            if 1 < b - a <= max_gap + 1:
                side[a : b + 1, x] = True
    return side


def z_at_column(side: np.ndarray, x: int) -> np.ndarray:
    return np.where(side[:, x])[0]


def y_at_column(front: np.ndarray, x: int) -> np.ndarray:
    return np.where(front[:, x])[0]


def pick_z_for_front(
    y: int,
    x: int,
    side: np.ndarray,
    size: int,
    depth_face_bridge: bool = True,
) -> Optional[int]:
    """
    Choose Z for voxel (y, x, z).

    When side view has depth gaps at column x, use the two boundary faces
    (z_min and z_max of side foreground) so front projection stays complete
  without filling the entire depth slab (preserves side view holes).
    """
    strict = z_at_column(side, x)
    if strict.size == 0:
        return None

    if depth_face_bridge and strict.size >= 1:
        z_lo, z_hi = int(strict[0]), int(strict[-1])
        if z_hi > z_lo + 1:
            pool = np.array([z_lo, z_hi], dtype=int)
            return int(pool[(y * len(pool)) // max(1, size) % len(pool)])

    return int(strict[(y * strict.size) // max(1, size) % strict.size])


def pick_y_for_side(
    z: int,
    x: int,
    front: np.ndarray,
    size: int,
    depth_face_bridge: bool = True,
) -> Optional[int]:
    """Symmetric: use front column Y extent faces when bridging."""
    strict = y_at_column(front, x)
    if strict.size == 0:
        return None

    if depth_face_bridge and strict.size >= 1:
        y_lo, y_hi = int(strict[0]), int(strict[-1])
        if y_hi > y_lo + 1:
            pool = np.array([y_lo, y_hi], dtype=int)
            return int(pool[(z * len(pool)) // max(1, size) % len(pool)])

    return int(strict[(z * strict.size) // max(1, size) % strict.size])


def carve_dual_cover(
    front: np.ndarray,
    side: np.ndarray,
    *,
    depth_face_bridge: bool = True,
) -> np.ndarray:
    """
    Build voxels: OR_z == front, OR_y == side (exact on side mask pixels).
    depth_face_bridge: place voxels at depth/Y faces when the other view has gaps.
    """
    size = front.shape[0]
    voxels = front[:, :, np.newaxis] & side.T[np.newaxis, :, :]

    for y in range(size):
        for x in range(size):
            if not front[y, x] or np.any(voxels[y, x, :]):
                continue
            z = pick_z_for_front(y, x, side, size, depth_face_bridge)
            if z is not None:
                voxels[y, x, z] = True

    for z in range(size):
        for x in range(size):
            if not side[z, x] or np.any(voxels[:, x, z]):
                continue
            y = pick_y_for_side(z, x, front, size, depth_face_bridge)
            if y is not None:
                voxels[y, x, z] = True

    return voxels


def projection_front(voxels: np.ndarray) -> np.ndarray:
    return np.any(voxels, axis=2)


def projection_side(voxels: np.ndarray) -> np.ndarray:
    return np.any(voxels, axis=0).T
