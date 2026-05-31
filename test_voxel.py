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
from image_preprocess import (
    _content_bbox_rgba,
    extract_mask,
    gentle_clean,
    load_grayscale_fit,
    load_grayscale_pair_fit,
)
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


def _make_bordered_square_image(size: int = 64, border: int = 16) -> bytes:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[border : size - border, border : size - border, :] = 0
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _make_edge_frame_with_center_logo(size: int = 96) -> bytes:
    img = np.full((size, size, 4), 255, dtype=np.uint8)
    # Simulate screenshot frame touching all image edges.
    img[0, :, :3] = (20, 80, 180)
    img[-1, :, :3] = (20, 80, 180)
    img[:, 0, :3] = (20, 80, 180)
    img[:, -1, :3] = (20, 80, 180)
    # Real center content.
    img[28:68, 32:64, :3] = 0
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _make_rect_image(width: int, height: int, rect: tuple[int, int, int, int]) -> bytes:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = rect
    img[y0:y1, x0:x1, :] = 0
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _mask_width(gray: np.ndarray, thr: int = 245) -> int:
    mask = gray < thr
    xs = np.where(np.any(mask, axis=0))[0]
    if xs.size == 0:
        return 0
    return int(xs[-1] - xs[0] + 1)


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


def test_load_grayscale_fit_trims_white_border():
    gray = load_grayscale_fit(_make_bordered_square_image(), size=64)
    # After trimming + fit, the black square should fill almost the entire frame.
    assert float(np.mean(gray)) < 8.0


def test_content_bbox_ignores_edge_frame():
    img = Image.open(io.BytesIO(_make_edge_frame_with_center_logo())).convert("RGBA")
    bbox = _content_bbox_rgba(img)
    assert bbox == (32, 28, 64, 68)


def test_content_bbox_only_frame_returns_none():
    size = 80
    img = np.full((size, size, 4), 255, dtype=np.uint8)
    img[0, :, :3] = (30, 80, 190)
    img[-1, :, :3] = (30, 80, 190)
    img[:, 0, :3] = (30, 80, 190)
    img[:, -1, :3] = (30, 80, 190)
    pil = Image.fromarray(img)
    assert _content_bbox_rgba(pil.convert("RGBA")) is None


def test_load_grayscale_pair_fit_aligns_shared_width():
    front = _make_rect_image(width=120, height=60, rect=(20, 15, 100, 45))
    side = _make_rect_image(width=60, height=120, rect=(15, 20, 45, 100))
    f_gray, s_gray = load_grayscale_pair_fit(front, side, size=96)
    assert _mask_width(f_gray) == _mask_width(s_gray)


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


def test_edge_wall_wrap_fills_front_when_side_column_missing():
    size = 24
    front = np.zeros((size, size), dtype=bool)
    front[6:18, 12] = True

    side = np.zeros((size, size), dtype=bool)
    side[4, 11] = True
    side[19, 11] = True

    v = carve_dual_cover(front, side, depth_face_bridge=True, edge_wall_wrap=True)
    pf = projection_front(v)
    ps = projection_side(v)

    assert np.all(pf[front])
    # Front silhouette should appear on both z boundary faces (two enclosing faces).
    ys, xs = np.where(front)
    assert np.all(v[ys, xs, 4])
    assert np.all(v[ys, xs, 19])
    # Missing side x=12 is now covered at boundary faces.
    assert ps[4, 12] and ps[19, 12]


def test_edge_strip_fill_fills_stripe_without_full_wall():
    size = 24
    front = np.zeros((size, size), dtype=bool)
    front[6:18, 12] = True
    side = np.zeros((size, size), dtype=bool)
    side[4, 11] = True
    side[19, 11] = True

    v = carve_dual_cover(
        front,
        side,
        depth_face_bridge=True,
        edge_strip_fill=True,
        edge_wall_wrap=False,
    )
    pf = projection_front(v)
    ps = projection_side(v)
    ys, xs = np.where(front)

    assert np.all(pf[front])
    assert ps[4, 12] and ps[19, 12]
    # No full z face fill when wall-wrap is disabled.
    assert not np.all(v[ys, xs, 4])


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
    test_load_grayscale_fit_trims_white_border()
    test_content_bbox_ignores_edge_frame()
    test_content_bbox_only_frame_returns_none()
    test_load_grayscale_pair_fit_aligns_shared_width()
    test_depth_face_bridge_uses_z_faces()
    test_edge_wall_wrap_fills_front_when_side_column_missing()
    test_edge_strip_fill_fills_stripe_without_full_wall()
    test_close_side_z_gaps()
    test_gentle_clean_keeps_ring()
    print("All tests passed.")
