"""Crystal internal engraving: dual-projection voxel carving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carve_helpers import carve_dual_cover, close_side_z_gaps as fill_side_z_gaps
from image_preprocess import extract_mask, gentle_clean, load_grayscale_pair_fit
from sparsify import sparsify_uniform


@dataclass
class VoxelResult:
    size: int
    points: list[list[float]]
    count: int
    count_full: int
    projection_front: list[list[int]]
    projection_side: list[list[int]]
    threshold_front: int
    threshold_side: int
    invert_front: bool
    invert_side: bool


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        merged = out.copy()
        h, w = out.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                merged |= padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        out = merged
    return out


def _dilate_horizontal(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask)
    for row in range(h):
        for x in range(w):
            for dx in range(-radius, radius + 1):
                nx = x + dx
                if 0 <= nx < w and mask[row, nx]:
                    out[row, x] = True
                    break
    return out


def _align_side_x(front: np.ndarray, side: np.ndarray) -> np.ndarray:
    pf = np.any(front, axis=0)
    if not np.any(pf):
        return side
    best_shift = 0
    best_score = -1
    for shift in range(side.shape[1]):
        ps = np.roll(np.any(side, axis=0), shift)
        score = int(np.sum(pf & ps))
        if score > best_score:
            best_score = score
            best_shift = shift
    if best_shift == 0:
        return side
    return np.roll(side, best_shift, axis=1)


def generate_voxels(
    image_front_bytes: bytes,
    image_side_bytes: bytes,
    size: int = 192,
    threshold: int | None = None,
    invert: bool | None = None,
    dilate: int = 0,
    align_x: bool = True,
    clean_mask: bool = False,
    overlap_dilate: int = 1,
    depth_face_bridge: bool = True,
    edge_strip_fill: bool = True,
    edge_wall_wrap: bool = True,
    close_side_z_gaps: int = 2,
    density: float = 0.75,
    uniform_strength: float = 0.25,
    detail_mode: bool = True,
) -> VoxelResult:
    size = int(np.clip(size, 16, 1024))
    dilate = int(np.clip(dilate, 0, 5))

    if detail_mode:
        clean_mask = False
        if dilate == 1:
            dilate = 0

    front_gray, side_gray = load_grayscale_pair_fit(
        image_front_bytes, image_side_bytes, size
    )

    user_invert = None if invert is None else bool(invert)
    front, t_front, inv_front = extract_mask(
        front_gray, threshold, user_invert, denoise=True
    )
    side, t_side, inv_side = extract_mask(side_gray, threshold, user_invert, denoise=True)

    if clean_mask:
        front = gentle_clean(front)
        side = gentle_clean(side)

    front = _dilate(front, dilate)
    side = _dilate(side, dilate)

    if align_x:
        side = _align_side_x(front, side)

    overlap_dilate = int(np.clip(overlap_dilate, 0, 3))
    if overlap_dilate > 0:
        front = _dilate_horizontal(front, overlap_dilate)
        side = _dilate_horizontal(side, overlap_dilate)

    gap_close = int(np.clip(close_side_z_gaps, 0, 8))
    if gap_close > 0:
        side = fill_side_z_gaps(side, gap_close)

    voxels = carve_dual_cover(
        front,
        side,
        depth_face_bridge=depth_face_bridge,
        edge_strip_fill=edge_strip_fill,
        edge_wall_wrap=edge_wall_wrap,
    )
    count_full = int(voxels.sum())

    voxels = sparsify_uniform(
        voxels,
        density=float(np.clip(density, 0.05, 1.0)),
        uniform_strength=float(np.clip(uniform_strength, 0.0, 1.0)),
        target_front=front,
        target_side=side,
        depth_face_bridge=depth_face_bridge,
        edge_strip_fill=edge_strip_fill,
        edge_wall_wrap=edge_wall_wrap,
    )

    ys, xs, zs = np.where(voxels)
    points = [[float(x), float(y), float(z)] for y, x, z in zip(ys, xs, zs)]

    return VoxelResult(
        size=size,
        points=points,
        count=len(points),
        count_full=count_full,
        projection_front=np.any(voxels, axis=2).astype(np.int8).tolist(),
        projection_side=np.any(voxels, axis=0).T.astype(np.int8).tolist(),
        threshold_front=t_front,
        threshold_side=t_side,
        invert_front=inv_front,
        invert_side=inv_side,
    )
