"""Core: tear a page image into non-overlapping torn pieces.

Guarantee (the whole point of the dataset): the output is a PARTITION of the
page. Every pixel is assigned to exactly one piece via nearest-seed argmin, so
pieces never overlap and together cover the page exactly -> perfect ground
truth for image stitching.

The "torn" look comes from DOMAIN WARPING: before computing the nearest seed we
displace each pixel's coordinate by a value-noise vector. Warping the query
points (not the partition rule) keeps the result a strict partition while making
boundaries jagged/organic instead of straight Voronoi edges.

Complexity (H*W pixels, S seeds, kd-tree):
    nearest-seed query : O(H*W * log S)   time
    label / mask pass  : Theta(H*W)       time, Theta(H*W) space
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .noise import value_noise
from .sampling import sample_seeds


@dataclass
class Piece:
    """One torn fragment + its placement for reassembly ground truth."""
    label: int
    x: int          # left offset on the original page canvas
    y: int          # top offset
    rgb: np.ndarray      # (h, w, 3) uint8, black where outside the fragment
    mask: np.ndarray     # (h, w) bool, True inside the fragment


@dataclass
class TornPage:
    width: int
    height: int
    pieces: list[Piece]
    labels: np.ndarray   # (H, W) int32 partition map (for verification / GT)
    adjacency: list[tuple[int, int]]   # undirected (i, j) piece-index neighbor pairs


def _adjacency_pairs(labels: np.ndarray) -> np.ndarray:
    """Return unique unordered raw-label neighbor pairs from a partition map.

    4-connectivity: two pieces are neighbors iff they touch horizontally or
    vertically. Vectorized: compare each pixel to its right/down neighbor, keep
    label pairs that differ. Cost Theta(H*W) - a few ms even at 150 DPI, dwarfed
    by the kd-tree query, so no measurable pipeline slowdown.
    """
    h_a, h_b = labels[:, :-1], labels[:, 1:]
    v_a, v_b = labels[:-1, :], labels[1:, :]
    hd, vd = h_a != h_b, v_a != v_b
    pairs = np.concatenate(
        [
            np.stack([h_a[hd], h_b[hd]], axis=1),
            np.stack([v_a[vd], v_b[vd]], axis=1),
        ],
        axis=0,
    )
    if pairs.size == 0:                       # single-piece page
        return pairs.reshape(0, 2)
    pairs.sort(axis=1)                        # (min, max) -> undirected
    return np.unique(pairs, axis=0)


def compute_adjacency(
    labels: np.ndarray, label_to_idx: dict[int, int]
) -> list[tuple[int, int]]:
    """Map raw-label neighbor pairs to manifest piece indices, sorted."""
    out = []
    for a, b in _adjacency_pairs(labels):
        ia, ib = label_to_idx.get(int(a)), label_to_idx.get(int(b))
        if ia is not None and ib is not None and ia != ib:
            out.append((ia, ib) if ia < ib else (ib, ia))
    return sorted(set(out))


def tear_page(
    page_rgb: np.ndarray,
    n_pieces: int,
    seed: int,
    noise_strength: float,
    noise_scale: float,
) -> TornPage:
    """Partition `page_rgb` (H, W, 3 uint8) into `n_pieces` torn fragments.

    `seed` makes a page reproducible; pass a different seed per page so the
    randomness changes page by page.
    """
    if page_rgb.ndim != 3 or page_rgb.shape[2] != 3:
        raise ValueError("page_rgb must be (H, W, 3) uint8")
    H, W = page_rgb.shape[:2]
    rng = np.random.default_rng(seed)

    seeds = sample_seeds(W, H, n_pieces, rng)             # (S, 2) -> (x, y)

    # Domain-warp the pixel grid with two independent noise fields.
    ys, xs = np.mgrid[0:H, 0:W]
    wx = value_noise(H, W, noise_scale, rng) * noise_strength
    wy = value_noise(H, W, noise_scale, rng) * noise_strength
    qx = (xs + wx).ravel()
    qy = (ys + wy).ravel()
    query = np.stack([qx, qy], axis=1).astype(np.float32)

    tree = cKDTree(seeds)
    _, flat_labels = tree.query(query, k=1, workers=-1)
    labels = flat_labels.reshape(H, W).astype(np.int32)

    pieces: list[Piece] = []
    for lbl in np.unique(labels):
        mask = labels == lbl
        ys_idx, xs_idx = np.nonzero(mask)
        if ys_idx.size == 0:
            continue
        y0, y1 = int(ys_idx.min()), int(ys_idx.max()) + 1
        x0, x1 = int(xs_idx.min()), int(xs_idx.max()) + 1
        sub_mask = mask[y0:y1, x0:x1]
        rgb = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)   # black background
        rgb[sub_mask] = page_rgb[y0:y1, x0:x1][sub_mask]
        pieces.append(
            Piece(label=int(lbl), x=x0, y=y0, rgb=rgb, mask=sub_mask)
        )

    # Piece-index <-> raw-label map from the pieces we actually emitted, so
    # adjacency indices line up exactly with the manifest's piece ordering.
    label_to_idx = {p.label: i for i, p in enumerate(pieces)}
    adjacency = compute_adjacency(labels, label_to_idx)

    return TornPage(
        width=W, height=H, pieces=pieces, labels=labels, adjacency=adjacency
    )


def verify_partition(torn: TornPage) -> dict:
    """Assert the no-overlap / full-coverage invariants. Returns a report.

    Reassembles a coverage counter from piece masks at their offsets and checks
    every pixel is covered exactly once.
    """
    cover = np.zeros((torn.height, torn.width), dtype=np.int32)
    for p in torn.pieces:
        h, w = p.mask.shape
        cover[p.y:p.y + h, p.x:p.x + w] += p.mask
    max_cover = int(cover.max())
    min_cover = int(cover.min())
    return {
        "pieces": len(torn.pieces),
        "max_overlap": max_cover,       # must be 1 -> no overlap
        "uncovered_pixels": int((cover == 0).sum()),  # must be 0 -> full cover
        "is_partition": max_cover == 1 and min_cover == 1,
    }
