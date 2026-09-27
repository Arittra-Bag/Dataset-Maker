"""Torn page -> puzzle fragments with exact ground-truth poses.

Normative per-fragment order (see docs/BENCHMARK.md, leakage review):
    1. crop the CLEAN page with the piece mask (the page is never corrupted
       before tearing, so noise/JPEG never correlate across seams)
    2. erosion (paper loss) on torn borders only; lost RGB is zeroed
    3. rotation by an integer millidegree angle with an exact affine
    4. tight crop to alpha + random 0..PAD_MAX px margin per side, so the
       canvas adds no information beyond the fragment's own mask
    5. blur, integer noise, JPEG, all in the fragment's own frame
    6. RGB zeroed wherever alpha == 0; alpha is exactly {0, 255}

Pose convention: A (2x3) maps fragment pixel centres (integer x right, y
down) to page pixel centres: p = A @ [x, y, 1].
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_cdt, label as cc_label

from .canon import cos_sin_mdeg

PAD_MAX = 8


@dataclass(frozen=True)
class Corruption:
    """Per-tier corruption recipe. Integers only where platform-exactness matters."""
    rotate: bool = False
    blur_radius: float = 0.0          # Pillow GaussianBlur radius (0 = off)
    noise_halfwidth: int = 0          # Irwin-Hall: sum of 4 U{-a..a} integers (0 = off)
    jpeg_quality: int = 0             # off when zero
    erosion_px: tuple[int, int] = (0, 0)   # inclusive range of band width (0 = off)


@dataclass
class Fragment:
    rgba: np.ndarray                  # (h, w, 4) uint8
    affine: list[list[float]]         # 2x3, fragment px -> page px
    rot_mdeg: int
    erosion_px: int
    label: int                        # raw partition label (answers only)
    kept_px: int                      # opaque pixels after erosion
    meta: dict = field(default_factory=dict)


def _erode(mask: np.ndarray, x0: int, y0: int, page_w: int, page_h: int,
           width: int) -> np.ndarray:
    """Remove a `width`-px band along torn borders; page borders are kept.

    Chessboard distance (integer, platform-exact). Out-of-page neighbours
    count as paper so the page edge never erodes. Keeps the largest
    4-connected component so a fragment stays one piece of paper.
    """
    if width <= 0:
        return mask
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    # Neighbours outside the page are "paper", not "tear".
    if x0 == 0:
        padded[:, 0] = True
    if y0 == 0:
        padded[0, :] = True
    if x0 + w == page_w:
        padded[:, -1] = True
    if y0 + h == page_h:
        padded[-1, :] = True
    if padded.all():                  # no torn border anywhere: nothing to erode
        return mask                   # (cdt would return -1 with no background)
    dist = distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1]
    kept = mask & (dist > width)
    lab, n = cc_label(kept)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        kept = lab == (int(np.argmax(sizes)) + 1)
    return kept


def _rotate(rgb: np.ndarray, alpha: np.ndarray, mdeg: int):
    """Rotate crop by `mdeg` about its centre; return (rgb, alpha, A_cont).

    A_cont maps output continuous coords (pixel centres at +0.5) to crop
    continuous coords. RGB bilinear, alpha nearest so it stays binary.
    """
    h, w = alpha.shape
    c, s = cos_sin_mdeg(mdeg)
    # Output canvas holds the whole rotated crop (+2 px guard).
    W = int(np.ceil(abs(w * c) + abs(h * s))) + 2
    H = int(np.ceil(abs(w * s) + abs(h * c))) + 2
    # Output -> input: u = R^T (q - c_out) + c_in, with R rotating page->fragment.
    a, b, d, e = c, s, -s, c
    cx_out, cy_out, cx_in, cy_in = W / 2, H / 2, w / 2, h / 2
    tx = cx_in - (a * cx_out + b * cy_out)
    ty = cy_in - (d * cx_out + e * cy_out)
    data = (a, b, tx, d, e, ty)
    rgb_out = Image.fromarray(rgb).transform((W, H), Image.AFFINE, data, resample=Image.BILINEAR)
    a_out = Image.fromarray(alpha).transform((W, H), Image.AFFINE, data, resample=Image.NEAREST)
    A = np.array([[a, b, tx], [d, e, ty], [0.0, 0.0, 1.0]])
    return np.asarray(rgb_out), np.asarray(a_out), A


def _corrupt(rgb: np.ndarray, alpha: np.ndarray, spec: Corruption, rngs: dict) -> np.ndarray:
    img = rgb
    if spec.blur_radius > 0:
        img = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(spec.blur_radius)))
    if spec.noise_halfwidth > 0:
        a = spec.noise_halfwidth
        noise = rngs["noise"].integers(-a, a + 1, size=(4,) + img.shape, dtype=np.int16).sum(axis=0)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if spec.jpeg_quality > 0:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=spec.jpeg_quality, subsampling=2,
                                  optimize=False, progressive=False)
        with Image.open(io.BytesIO(buf.getvalue())) as im:
            img = np.asarray(im.convert("RGB"))
    img = img.copy()
    img[alpha == 0] = 0
    return img


def make_fragment(page_rgb: np.ndarray, piece, spec: Corruption, rngs: dict) -> Fragment:
    """One torn piece -> puzzle fragment + exact pose. `rngs` holds named Generators."""
    H, W = page_rgb.shape[:2]
    y0, x0 = piece.y, piece.x
    mask = piece.mask
    h, w = mask.shape
    width = 0
    if spec.erosion_px[1] > 0:
        width = int(rngs["erosion"].integers(spec.erosion_px[0], spec.erosion_px[1] + 1))
    mask = _erode(mask, x0, y0, W, H, width)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[mask] = page_rgb[y0:y0 + h, x0:x0 + w][mask]
    alpha = np.where(mask, 255, 0).astype(np.uint8)

    # Continuous-coordinate map: fragment canvas -> page.
    A = np.array([[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]])
    mdeg = int(rngs["rotation"].integers(0, 360000)) if spec.rotate else 0
    if mdeg:
        rgb, alpha, a_rot = _rotate(rgb, alpha, mdeg)
        A = A @ a_rot

    ys, xs = np.nonzero(alpha)
    if ys.size == 0:
        raise ValueError("fragment lost all pixels")
    pl, pt, pr, pb = (int(v) for v in rngs["pad"].integers(0, PAD_MAX + 1, 4))
    cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ch, cw = cy1 - cy0, cx1 - cx0
    out_rgb = np.zeros((ch + pt + pb, cw + pl + pr, 3), np.uint8)
    out_a = np.zeros((ch + pt + pb, cw + pl + pr), np.uint8)
    out_rgb[pt:pt + ch, pl:pl + cw] = rgb[cy0:cy1, cx0:cx1]
    out_a[pt:pt + ch, pl:pl + cw] = alpha[cy0:cy1, cx0:cx1]
    # final canvas coord q -> pre-crop canvas coord q + (cx0 - pl, cy0 - pt)
    A = A @ np.array([[1.0, 0.0, float(cx0 - pl)], [0.0, 1.0, float(cy0 - pt)], [0.0, 0.0, 1.0]])
    # Continuous (centres at +0.5) -> integer pixel-centre convention.
    a_int = (np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]]) @ A
             @ np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]]))

    out_rgb = _corrupt(out_rgb, out_a, spec, rngs)
    rgba = np.dstack([out_rgb, out_a])
    return Fragment(
        rgba=rgba,
        affine=[[float(v) for v in a_int[0]], [float(v) for v in a_int[1]]],
        rot_mdeg=mdeg,
        erosion_px=width,
        label=int(piece.label),
        kept_px=int((out_a > 0).sum()),
    )
