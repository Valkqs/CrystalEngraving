"""High-quality 2D mask extraction for logo / fine artwork."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _content_bbox_from_components(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find bbox of non-edge connected content; fallback to full content bbox."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    interior_boxes: list[tuple[int, int, int, int]] = []
    edge_boxes: list[tuple[int, int, int, int, int]] = []

    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            min_x = max_x = x0
            min_y = max_y = y0
            area = 0
            touch_top = y0 == 0
            touch_bottom = y0 == h - 1
            touch_left = x0 == 0
            touch_right = x0 == w - 1
            while stack:
                y, x = stack.pop()
                area += 1
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not mask[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    if ny == 0:
                        touch_top = True
                    if ny == h - 1:
                        touch_bottom = True
                    if nx == 0:
                        touch_left = True
                    if nx == w - 1:
                        touch_right = True
                    stack.append((ny, nx))
            box = (min_x, min_y, max_x + 1, max_y + 1)
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            bbox_area = max(1, bw * bh)
            touches_count = int(touch_top) + int(touch_bottom) + int(touch_left) + int(touch_right)
            covers_most = bw >= int(0.9 * w) and bh >= int(0.9 * h)
            sparse = area <= int(0.2 * bbox_area)
            # Ignore thin screenshot-like frame components hugging edges.
            is_edge_frame = touches_count >= 3 and covers_most and sparse
            if is_edge_frame:
                continue
            if touches_count == 0:
                interior_boxes.append(box)
            else:
                edge_boxes.append((area, *box))

    if interior_boxes:
        boxes = interior_boxes
    else:
        area_floor = max(32, int(0.005 * h * w))
        boxes = [(x0, y0, x1, y1) for a, x0, y0, x1, y1 in edge_boxes if a >= area_floor]

    if not boxes:
        return None
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    max_x = max(b[2] for b in boxes)
    max_y = max(b[3] for b in boxes)
    return int(min_x), int(min_y), int(max_x), int(max_y)


def _content_bbox_rgba(img: Image.Image, white_tol: int = 8) -> tuple[int, int, int, int] | None:
    """Return content bbox (left, upper, right, lower), excluding near-white border."""
    arr = np.asarray(img, dtype=np.uint8)
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]
    non_white = np.any((255 - rgb) > white_tol, axis=2)
    content = non_white | (alpha < 250)
    return _content_bbox_from_components(content)


def _crop_content_rgba(img: Image.Image) -> Image.Image:
    bbox = _content_bbox_rgba(img)
    return img.crop(bbox) if bbox is not None else img


def _render_cropped_to_gray(
    img: Image.Image,
    size: int,
    target_width: int | None = None,
) -> np.ndarray:
    w, h = img.size
    if target_width is None:
        scale = min(size / w, size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
    else:
        nw = int(np.clip(target_width, 1, size))
        scale = nw / max(1, w)
        nh = max(1, int(round(h * scale)))
        if nh > size:
            scale = size / max(1, h)
            nh = size
            nw = max(1, int(round(w * scale)))

    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(img, (ox, oy), img)
    arr = np.asarray(canvas, dtype=np.float32)
    rgb = arr[..., :3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.astype(np.uint8)


def load_grayscale_fit(data: bytes, size: int) -> np.ndarray:
    """Center-fit image into size×size canvas (no stretch), white padding."""
    img = _crop_content_rgba(Image.open(io.BytesIO(data)).convert("RGBA"))
    return _render_cropped_to_gray(img, size)


def load_grayscale_pair_fit(
    front_data: bytes,
    side_data: bytes,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop each image to content bbox, then align both by shared X-edge length.
    The two rendered images use the same target width before projection carving.
    """
    front = _crop_content_rgba(Image.open(io.BytesIO(front_data)).convert("RGBA"))
    side = _crop_content_rgba(Image.open(io.BytesIO(side_data)).convert("RGBA"))
    wf, hf = front.size
    ws, hs = side.size
    max_shared_width = min(
        size,
        int(np.floor(size * wf / max(1, hf))),
        int(np.floor(size * ws / max(1, hs))),
    )
    target_width = int(np.clip(max_shared_width, 1, size))
    return (
        _render_cropped_to_gray(front, size, target_width=target_width),
        _render_cropped_to_gray(side, size, target_width=target_width),
    )


def auto_invert(gray: np.ndarray) -> bool:
    """True when foreground is darker than background (typical logos)."""
    h, w = gray.shape
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, w - 1]]
    )
    if border.mean() > gray.mean() + 8:
        return True
    return False


def otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0
    best_var = -1.0
    best_t = 128
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


def extract_mask(
    gray: np.ndarray,
    threshold: int | None = None,
    invert: bool | None = None,
    *,
    denoise: bool = True,
) -> tuple[np.ndarray, int, bool]:
    """Binarize with optional light denoise (preserves thin strokes)."""
    work = gray
    if denoise:
        work = np.asarray(
            Image.fromarray(gray).filter(ImageFilter.MedianFilter(size=3)),
            dtype=np.uint8,
        )

    inv = auto_invert(work) if invert is None else invert
    t = otsu_threshold(work) if threshold is None else int(threshold)
    mask = work <= t if inv else work >= t

    # Fill tiny holes inside emblem (helps circular ring)
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    mask = np.asarray(mask_img) > 127
    return mask.astype(np.bool_), t, inv


def gentle_clean(mask: np.ndarray) -> np.ndarray:
    """Remove only full-height isolated columns; keep thin circular/text strokes."""
    mask = mask.copy()
    h, w = mask.shape
    col_on = np.any(mask, axis=0)
    for x in range(w):
        if not col_on[x]:
            continue
        left = col_on[x - 1] if x > 0 else False
        right = col_on[x + 1] if x < w - 1 else False
        if left or right:
            continue
        if int(mask[:, x].sum()) >= int(h * 0.85):
            mask[:, x] = False
    return mask
