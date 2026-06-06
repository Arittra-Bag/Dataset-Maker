"""Vectorized value-noise field for domain warping.

We do NOT pull in a Perlin dependency: a tileable value-noise (random lattice +
bilinear upsample, summed across octaves) is enough to warp Voronoi boundaries
into organic "torn" edges, and it is pure NumPy.

Complexity: building an (H, W) field over `octaves` is
    Theta(H * W * octaves)  time,
    Theta(H * W)            space.
"""
from __future__ import annotations

import numpy as np


def _bilinear_upsample(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Upsample a small lattice to (out_h, out_w) with bilinear interpolation."""
    gh, gw = grid.shape
    # Sample positions in lattice space.
    ys = np.linspace(0, gh - 1, out_h)
    xs = np.linspace(0, gw - 1, out_w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bot = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def value_noise(
    h: int,
    w: int,
    scale: float,
    rng: np.random.Generator,
    octaves: int = 3,
    persistence: float = 0.5,
) -> np.ndarray:
    """Return an (h, w) float field in roughly [-1, 1].

    `scale` is the wavelength in pixels of the base octave: larger -> smoother.
    Octaves add finer detail (fractal/fBm), persistence weights their amplitude.
    """
    field = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    freq_scale = max(scale, 2.0)

    for _ in range(max(1, octaves)):
        gh = max(2, int(np.ceil(h / freq_scale)) + 1)
        gw = max(2, int(np.ceil(w / freq_scale)) + 1)
        lattice = rng.uniform(-1.0, 1.0, size=(gh, gw)).astype(np.float32)
        field += amplitude * _bilinear_upsample(lattice, h, w)
        total_amp += amplitude
        amplitude *= persistence
        freq_scale = max(freq_scale * 0.5, 2.0)

    return field / total_amp
