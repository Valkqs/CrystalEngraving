"""Crystal internal engraving: dual-projection voxel carving."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from carve_helpers import carve_dual_cover, close_side_z_gaps as fill_side_z_gaps
from image_preprocess import extract_mask, gentle_clean, load_grayscale_fit
from sparsify import sparsify_uniform
from voxel_optimizer import optimize_voxels


class VoxelResult:
    __slots__ = ('size', 'points', 'count', 'count_full',
                 'projection_front', 'projection_side',
                 'threshold_front', 'threshold_side',
                 'invert_front', 'invert_side',
                 'mask_front', 'mask_side',
                 'f1_front', 'f1_side', 'f1_total',
                 'chaos', 'objective',
                 'optimize_params')

    def __init__(
        self,
        size: int,
        points: List[List[float]],
        count: int,
        count_full: int,
        projection_front: List[List[int]],
        projection_side: List[List[int]],
        threshold_front: int,
        threshold_side: int,
        invert_front: bool,
        invert_side: bool,
        mask_front: List[List[int]],
        mask_side: List[List[int]],
        f1_front: float = 1.0,
        f1_side: float = 1.0,
        f1_total: float = 1.0,
        chaos: float = 0.0,
        objective: float = 1.0,
        optimize_params: Optional[Dict] = None,
    ):
        self.size = size
        self.points = points
        self.count = count
        self.count_full = count_full
        self.projection_front = projection_front
        self.projection_side = projection_side
        self.threshold_front = threshold_front
        self.threshold_side = threshold_side
        self.invert_front = invert_front
        self.invert_side = invert_side
        self.mask_front = mask_front
        self.mask_side = mask_side
        self.f1_front = f1_front
        self.f1_side = f1_side
        self.f1_total = f1_total
        self.chaos = chaos
        self.objective = objective
        self.optimize_params = optimize_params if optimize_params is not None else {}


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
    threshold: Optional[int] = None,
    invert: Optional[bool] = None,
    dilate: int = 0,
    align_x: bool = True,
    clean_mask: bool = False,
    overlap_dilate: int = 1,
    depth_face_bridge: bool = True,
    close_side_z_gaps: int = 2,
    density: float = 0.75,
    uniform_strength: float = 0.25,
    detail_mode: bool = True,
    optimize: bool = True,
    chaos_penalty: float = 0.5,
    min_f1: float = 0.72,
    sa_steps: int = 12000,
    weight_volume: float = 0.10,
    rng_seed: int = 42,
    progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None,
) -> VoxelResult:
    size = int(np.clip(size, 16, 512))
    dilate = int(np.clip(dilate, 0, 5))

    def report(progress: float, stage: str, detail: Optional[Dict] = None) -> None:
        if progress_callback is not None:
            progress_callback(float(np.clip(progress, 0.0, 1.0)), stage, detail)

    report(0.02, "读取并缩放输入图片")

    if detail_mode:
        clean_mask = False
        if dilate == 1:
            dilate = 0

    front_gray = load_grayscale_fit(image_front_bytes, size)
    side_gray = load_grayscale_fit(image_side_bytes, size)
    report(0.10, "图像预处理完成，开始提取二值掩码")

    user_invert = None if invert is None else bool(invert)
    front, t_front, inv_front = extract_mask(
        front_gray, threshold, user_invert, denoise=True
    )
    side, t_side, inv_side = extract_mask(side_gray, threshold, user_invert, denoise=True)

    if clean_mask:
        front = gentle_clean(front)
        side = gentle_clean(side)

    report(0.18, "掩码提取完成，开始形态学处理")

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

    report(0.28, "掩码修正完成，开始构建立体点阵")

    f1_front_val = 1.0
    f1_side_val = 1.0
    f1_total_val = 1.0
    chaos_val = 0.0
    objective_val = 1.0
    opt_params = {}
    count_full = 0

    if optimize:
        (
            voxels,
            f1_front_val,
            f1_side_val,
            f1_total_val,
            chaos_val,
            objective_val,
            _,
            count_full,
            opt_params,
        ) = optimize_voxels(
            front,
            side,
            density=density,
            uniform_strength=uniform_strength,
            align_x=False,
            chaos_penalty=chaos_penalty,
            min_f1=float(np.clip(min_f1, 0.50, 0.99)),
            sa_steps=int(np.clip(sa_steps, 500, 50000)),
            rng_seed=int(rng_seed),
            weight_volume=float(np.clip(weight_volume, 0.0, 1.0)),
            verbose=False,
            progress_callback=lambda p, stage, detail=None: report(0.30 + 0.62 * p, stage, detail),
        )
    else:
        report(0.34, "构建初始双视角体素覆盖")
        voxels = carve_dual_cover(
            front, side, depth_face_bridge=depth_face_bridge
        )
        count_full = int(voxels.sum())

        report(0.58, "执行稀疏均匀化")
        voxels = sparsify_uniform(
            voxels,
            density=float(np.clip(density, 0.05, 1.0)),
            uniform_strength=float(np.clip(uniform_strength, 0.0, 1.0)),
            target_front=front,
            target_side=side,
            depth_face_bridge=depth_face_bridge,
        )
        report(0.82, "点阵优化完成，开始整理结果")

    ys, xs, zs = np.where(voxels)
    points = [[float(x), float(y), float(z)] for y, x, z in zip(ys, xs, zs)]
    report(0.94, "正在生成投影与点云数据", {"point_count": len(points)})

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
        mask_front=front.astype(np.int8).tolist(),
        mask_side=side.astype(np.int8).tolist(),
        f1_front=f1_front_val,
        f1_side=f1_side_val,
        f1_total=f1_total_val,
        chaos=chaos_val,
        objective=objective_val,
        optimize_params=opt_params,
    )
