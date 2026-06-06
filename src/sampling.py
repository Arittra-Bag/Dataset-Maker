"""Seed-point sampling for the Voronoi partition.

Bridson Poisson-disk sampling gives blue-noise seeds: random but evenly spread,
so piece areas stay within sane bounds (no slivers, no giant blobs). We target
a piece count, derive the min-distance r from area, then top up / trim to hit
the exact count.

Bridson is Theta(k) in the number of accepted samples (grid-accelerated
neighbour test), Theta(grid cells) = Theta(W*H / r^2) space.
"""
from __future__ import annotations

import math

import numpy as np


def poisson_disk(
    width: int,
    height: int,
    radius: float,
    rng: np.random.Generator,
    k: int = 30,
) -> np.ndarray:
    """Bridson's algorithm. Returns (n, 2) float array of (x, y) samples."""
    cell = radius / math.sqrt(2)
    gw = int(math.ceil(width / cell))
    gh = int(math.ceil(height / cell))
    grid = -np.ones((gh, gw), dtype=np.int64)
    samples: list[tuple[float, float]] = []
    active: list[int] = []

    def grid_xy(p):
        return int(p[0] / cell), int(p[1] / cell)

    first = (rng.uniform(0, width), rng.uniform(0, height))
    samples.append(first)
    gx, gy = grid_xy(first)
    grid[gy, gx] = 0
    active.append(0)

    while active:
        idx = active[rng.integers(len(active))]
        base = samples[idx]
        placed = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(radius, 2 * radius)
            cand = (base[0] + rad * math.cos(ang), base[1] + rad * math.sin(ang))
            if not (0 <= cand[0] < width and 0 <= cand[1] < height):
                continue
            cgx, cgy = grid_xy(cand)
            ok = True
            for ny in range(max(0, cgy - 2), min(gh, cgy + 3)):
                for nx in range(max(0, cgx - 2), min(gw, cgx + 3)):
                    j = grid[ny, nx]
                    if j >= 0:
                        d = math.dist(cand, samples[j])
                        if d < radius:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                samples.append(cand)
                grid[cgy, cgx] = len(samples) - 1
                active.append(len(samples) - 1)
                placed = True
                break
        if not placed:
            active.remove(idx)

    return np.asarray(samples, dtype=np.float32)


def sample_seeds(
    width: int,
    height: int,
    n_pieces: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return exactly `n_pieces` blue-noise seeds (x, y) inside the page."""
    n_pieces = max(2, int(n_pieces))
    # r from target area per piece: area ~ W*H/n, packing factor ~0.7.
    area_per = (width * height) / n_pieces
    radius = math.sqrt(area_per) * 0.66

    pts = poisson_disk(width, height, radius, rng)

    # Top up with uniform random points if Poisson under-shot.
    if len(pts) < n_pieces:
        extra = rng.uniform([0, 0], [width, height], size=(n_pieces - len(pts), 2))
        pts = np.vstack([pts, extra.astype(np.float32)])
    # Trim deterministically if it over-shot.
    if len(pts) > n_pieces:
        keep = rng.choice(len(pts), size=n_pieces, replace=False)
        pts = pts[keep]

    return pts.astype(np.float32)
