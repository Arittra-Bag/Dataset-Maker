"""Image optimization for the exported pieces.

Torn pieces are mostly black background -> they compress extremely well as PNG.
We:
  * tight-crop to the fragment bbox (already done in tearing),
  * optionally quantize the foreground to <=256 colors (palette PNG) to shrink
    size further when `lossy` is on,
  * always write with PNG optimize + max zlib compression.

Palette quantization is Theta(P) over piece pixels (Pillow median-cut).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode_piece(rgb: np.ndarray, lossy: bool = False, colors: int = 64) -> bytes:
    """Encode an (h, w, 3) uint8 piece to optimized PNG bytes."""
    img = Image.fromarray(rgb, mode="RGB")
    if lossy:
        # Median-cut palette; black background collapses to one palette entry.
        img = img.quantize(colors=max(2, min(256, colors)), method=Image.MEDIANCUT)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def encode_preview(rgb: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Downscale a page/preview for fast UI display (keeps aspect)."""
    H, W = rgb.shape[:2]
    scale = min(1.0, max_side / max(H, W))
    if scale >= 1.0:
        return rgb
    nw, nh = int(W * scale), int(H * scale)
    return np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
