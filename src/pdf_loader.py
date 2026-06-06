"""PDF -> list of A4-sized RGB page images using PyMuPDF (fitz).

"Native split" handling: if a page is already ~A4 portrait we render it as-is.
If a page is much larger / a different ratio (e.g. an A3 spread, a long scan),
we slice it into A4-height bands so downstream tearing always works on A4 tiles.
"""
from __future__ import annotations

import numpy as np

from .config import A4_ASPECT, MAX_PAGES_PER_PDF, a4_pixels


def _page_pixmap(page, dpi: int) -> np.ndarray:
    import fitz  # PyMuPDF

    zoom = dpi / 72.0
    pm = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)
    if pm.n == 4:           # RGBA -> RGB
        arr = arr[:, :, :3]
    elif pm.n == 1:         # gray -> RGB
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def _fit_to_a4(img: np.ndarray, dpi: int) -> list[np.ndarray]:
    """Return one or more A4-portrait tiles covering `img`.

    Tall pages are sliced into A4-height bands (each band padded to A4 width).
    """
    target_w, target_h = a4_pixels(dpi)
    H, W = img.shape[:2]
    aspect = H / W

    # Close enough to A4 portrait: letterbox-resize the whole page onto A4.
    if abs(aspect - A4_ASPECT) < 0.12:
        return [_letterbox(img, target_w, target_h)]

    # Otherwise scale to A4 width, then slice the (now-tall) image into bands.
    scale = target_w / W
    new_h = max(1, int(round(H * scale)))
    resized = _resize(img, target_w, new_h)
    tiles = []
    for top in range(0, new_h, target_h):
        band = resized[top:top + target_h]
        if band.shape[0] < target_h:
            band = _letterbox(band, target_w, target_h)
        tiles.append(band)
    return tiles


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))


def _letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize preserving aspect, pad with white onto a w*h canvas."""
    H, W = img.shape[:2]
    scale = min(w / W, h / H)
    nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
    resized = _resize(img, nw, nh)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    oy, ox = (h - nh) // 2, (w - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def load_pdf_pages(path: str, dpi: int) -> list[np.ndarray]:
    """Render `path` into a list of A4 RGB uint8 page images."""
    import fitz

    doc = fitz.open(path)
    try:
        out: list[np.ndarray] = []
        for page in doc:
            raw = _page_pixmap(page, dpi)
            out.extend(_fit_to_a4(raw, dpi))
            if len(out) >= MAX_PAGES_PER_PDF:
                return out[:MAX_PAGES_PER_PDF]
        return out
    finally:
        doc.close()
