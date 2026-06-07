"""Tests for projection-preserving sparsification."""

import io

import numpy as np
from PIL import Image

from sparsify import sparsify_uniform
from voxel import generate_voxels
from carve_helpers import carve_dual_cover


def _full_cube(size: int = 24) -> np.ndarray:
    front = np.ones((size, size), dtype=bool)
    side = np.ones((size, size), dtype=bool)
    return carve_dual_cover(front, side)


def test_sparsify_preserves_projections():
    voxels = _full_cube(20)
    full = int(voxels.sum())
    sparse = sparsify_uniform(voxels, density=0.2, uniform_strength=0.7)
    n = int(sparse.sum())
    assert n < full
    assert np.array_equal(np.any(sparse, axis=2), np.any(voxels, axis=2))
    assert np.array_equal(np.any(sparse, axis=0).T, np.any(voxels, axis=0).T)


def test_generate_with_density():
    img = np.full((32, 32), 255, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    b = buf.getvalue()
    dense = generate_voxels(b, b, size=32, threshold=128, density=1.0)
    sparse = generate_voxels(b, b, size=32, threshold=128, density=0.15, uniform_strength=0.8)
    assert sparse.count < dense.count
    assert sparse.count_full == dense.count


if __name__ == "__main__":
    test_sparsify_preserves_projections()
    test_generate_with_density()
    print("Sparsify tests passed.")
