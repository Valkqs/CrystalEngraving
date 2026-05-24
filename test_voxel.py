"""Tests for voxel generation."""

import io

import numpy as np
from PIL import Image

from carve_helpers import (
    carve_dual_cover,
    close_side_z_gaps,
    projection_front,
    projection_side,
)
from image_preprocess import extract_mask, gentle_clean, load_grayscale_fit
from voxel import _align_side_x, _dilate_horizontal, generate_voxels


def _make_image(pattern: str, size: int = 32) -> bytes:
    img = np.zeros((size, size), dtype=np.uint8)
    if pattern == "full":
        img[:, :] = 255
    elif pattern == "rect":
        img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _load_mask(pattern: str, size: int = 32) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(_make_image(pattern, size))).convert("L")) >= 128


def test_full_cube():
    result = generate_voxels(_make_image("full"), _make_image("full"), size=16, density=1.0)
    assert result.count == 16 ** 3


def test_dual_cover_complete():
    f = _load_mask("rect", 32)
    s = _load_mask("rect", 32)
    v = carve_dual_cover(f, s)
    assert np.array_equal(projection_front(v), f)
    assert np.array_equal(projection_side(v), s)


def test_auto_invert_dark_logo():
    """Simulate white bg + dark blue disk."""
    size = 64
    img = np.full((size, size), 255, dtype=np.uint8)
    img[16:48, 16:48] = 40
    mask, _, inv = extract_mask(img, None, None)
    assert inv is True
    assert mask[32, 32]
    assert not mask[0, 0]


def test_depth_face_bridge_uses_z_faces():
    """Depth bridge picks z_min/z_max faces; side projection keeps the Z gap."""
    from carve_helpers import pick_z_for_front

    size = 32
    front = np.zeros((size, size), dtype=bool)
    front[10, 16] = True

    side = np.zeros((size, size), dtype=bool)
    side[6, 16] = True
    side[26, 16] = True

    z = pick_z_for_front(10, 16, side, size, depth_face_bridge=True)
    assert z in (6, 26)

    v = carve_dual_cover(front, side, depth_face_bridge=True)
    ps = projection_side(v)
    assert ps[6, 16] and ps[26, 16]
    assert not ps[15, 16]


def test_close_side_z_gaps():
    side = np.zeros((16, 16), dtype=bool)
    side[4, 8] = True
    side[7, 8] = True
    closed = close_side_z_gaps(side, max_gap=2)
    assert closed[5, 8]
    assert closed[6, 8]


def test_gentle_clean_keeps_ring():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    mask[12:20, 12:20] = False
    cleaned = gentle_clean(mask)
    assert cleaned[10, 16]
    assert not cleaned[16, 16]


if __name__ == "__main__":
    test_full_cube()
    test_dual_cover_complete()
    test_auto_invert_dark_logo()
    test_depth_face_bridge_uses_z_faces()
    test_close_side_z_gaps()
    test_gentle_clean_keeps_ring()
    print("All tests passed.")
