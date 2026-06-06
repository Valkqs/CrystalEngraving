"""High-quality 2D mask extraction for logo / fine artwork."""

from typing import Optional, Tuple

import io

import numpy as np
from PIL import Image, ImageFilter


def load_grayscale_fit(data: bytes, size: int) -> np.ndarray:
    """Center-fit image into size×size canvas (no stretch), white padding."""
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(img, (ox, oy), img)
    arr = np.asarray(canvas, dtype=np.float32)
    rgb = arr[..., :3]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.astype(np.uint8)


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
    threshold: Optional[int] = None,
    invert: Optional[bool] = None,
    *,
    denoise: bool = True,
) -> Tuple[np.ndarray, int, bool]:
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
