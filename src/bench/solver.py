"""edge-greedy v0.1: the dm-bench baseline reassembly solver.

Answer-free by construction: reads only a puzzle directory (puzzle.json +
RGBA fragments) and imports nothing from the generator (tests enforce this).

    1. contour: Moore tracing of each fragment's alpha, resampled every
       STEP px and lightly smoothed; cut into CHUNK-sample windows
    2. pair hypotheses: complementary chunks (one traversed in reverse) are
       aligned by closed-form 2D Kabsch; best TOPK residuals per pair
    3. refine: boundary ICP; score = seam length * fit * colour continuity
       * (1 - overlap)
    4. assemble: greedy spanning placement by score with an overlap check;
       stuck fragments seed new clusters (shifted apart)

Deterministic: float64, no SVD, scores quantised before ranking, ties broken
by fragment ids.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter1d
from scipy.spatial import cKDTree

SOLVER = "edge-greedy"
SOLVER_VERSION = "0.1"
SOLUTION_SCHEMA = "dm-bench/solution/1"

STEP = 2.0            # contour resampling (px)
CHUNK = 48            # samples per matching window (~96 px); tuned on val
STRIDE = 4            # window stride (samples)
TOPK = 12             # hypotheses refined per fragment pair
SEAM_TOL = 4.0        # px: boundary points this close count as seam
MIN_SEAM = 24.0       # px of seam needed to accept a join
MAX_OVERLAP = 0.04    # share of a fragment's pixels allowed on placed ones
CLUSTER_SHIFT = 5000.0
_N8 = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))   # (dy, dx), clockwise from W


# ---------------------------------------------------------------------------
# Fragment geometry
# ---------------------------------------------------------------------------
class Frag:
    def __init__(self, fid: str, rgba: np.ndarray):
        self.id = fid
        self.alpha = rgba[..., 3] > 0
        self.rgb = rgba[..., :3].astype(np.float32)
        self.contour = resample(moore_contour(self.alpha), STEP)
        self.smooth = smooth_closed(self.contour, 1.5)
        self.tree = cKDTree(self.smooth) if len(self.smooth) else None
        self.border = straight_runs(self.smooth)
        self.colour = inward_colours(self.smooth, self.alpha, self.rgb)
        n = len(self.smooth)
        starts = [s for s in range(0, n, STRIDE)
                  if n >= CHUNK and not self.border[[(s + k) % n for k in range(CHUNK)]].any()]
        self.windows = np.array([[(s + k) % n for k in range(CHUNK)] for s in starts], dtype=int) \
            .reshape(-1, CHUNK)
        ys, xs = np.nonzero(self.alpha)
        self.pixels = np.stack([xs, ys], axis=1)[::3].astype(float)      # subsample for overlap
        self.inner = binary_erosion(self.alpha, iterations=2)


def straight_runs(pts: np.ndarray, window: int = 48, tol: float = 0.2) -> np.ndarray:
    """Flag contour samples on long straight runs (page borders): they have
    no mate, and two of them would otherwise align as a perfect white seam."""
    n = len(pts)
    flags = np.zeros(n, dtype=bool)
    if n < window:
        return flags
    for s in range(0, n, 4):
        idx = [(s + k) % n for k in range(window)]
        seg = pts[idx] - pts[idx].mean(axis=0)
        # RMS distance to the best-fit line = smallest singular value / sqrt(n)
        cov = seg.T @ seg / window
        tr, det = cov[0, 0] + cov[1, 1], cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
        small = tr / 2 - math.sqrt(max(tr * tr / 4 - det, 0.0))
        if math.sqrt(max(small, 0.0)) < tol:
            flags[idx] = True
    return flags


def inward_colours(pts: np.ndarray, alpha: np.ndarray, rgb: np.ndarray, depth: float = 2.5) -> np.ndarray:
    """RGB sampled `depth` px inside the fragment along the contour normal
    (boundary pixels are darkened by rotation resampling)."""
    n = len(pts)
    if n < 3:
        return np.zeros((n, 3), np.float32)
    tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    h, w = alpha.shape
    out = np.zeros((n, 3), np.float32)
    for sign in (1.0, -1.0):
        q = np.rint(pts + sign * depth * normal).astype(int)
        qx, qy = np.clip(q[:, 0], 0, w - 1), np.clip(q[:, 1], 0, h - 1)
        inside = alpha[qy, qx] & ~out.any(axis=1)
        out[inside] = rgb[qy[inside], qx[inside]]
    return out


def moore_contour(mask: np.ndarray) -> np.ndarray:
    """Ordered outer boundary pixels (x, y), clockwise (Moore tracing).

    Stops when the step out of the start pixel repeats the first step, which
    also terminates on 1-px necks where the start is revisited.
    """
    m = np.pad(mask, 1)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return np.zeros((0, 2))
    start = (int(ys[0]), int(xs[0]))                  # top-most, then left-most
    cur, back = start, (start[0], start[1] - 1)       # its west neighbour is background
    pts = [start]
    first_step = None
    for _ in range(8 * int(m.sum()) + 8):
        d0 = _N8.index((back[0] - cur[0], back[1] - cur[1]))
        prev, found = back, None
        for k in range(1, 9):
            dy, dx = _N8[(d0 + k) % 8]
            cand = (cur[0] + dy, cur[1] + dx)
            if m[cand]:
                found = cand
                break
            prev = cand
        if found is None:                             # isolated pixel
            break
        if cur == start:
            if first_step is None:
                first_step = found
            elif found == first_step:
                break
        cur, back = found, prev
        if cur != start:
            pts.append(cur)
    arr = np.array(pts, dtype=float)
    return np.stack([arr[:, 1] - 1, arr[:, 0] - 1], axis=1)


def resample(pts: np.ndarray, step: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, cum[-1], step)
    return np.stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])], axis=1)


def smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    if len(pts) < 5:
        return pts
    return np.stack([gaussian_filter1d(pts[:, i], sigma, mode="wrap") for i in (0, 1)], axis=1)


# ---------------------------------------------------------------------------
# Rigid helpers (closed form, deterministic)
# ---------------------------------------------------------------------------
def rigid(angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:2, :2].T + m[:2, 2]


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - cs, dst - cd
    angle = math.atan2(float((s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum()), float((s * d).sum()))
    m = rigid(angle, 0.0, 0.0)
    m[:2, 2] = cd - m[:2, :2] @ cs
    return m


def batch_kabsch_residual(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """RMS residual of the best rigid fit for each (src[k], dst[k]) pair set."""
    s = src - src.mean(axis=1, keepdims=True)
    d = dst - dst.mean(axis=1, keepdims=True)
    cross = (s[..., 0] * d[..., 1] - s[..., 1] * d[..., 0]).sum(axis=1)
    dot = (s * d).sum(axis=(1, 2))
    ang = np.arctan2(cross, dot)
    c, n = np.cos(ang)[:, None], np.sin(ang)[:, None]
    rx = c * s[..., 0] - n * s[..., 1]
    ry = n * s[..., 0] + c * s[..., 1]
    return np.sqrt(((rx - d[..., 0]) ** 2 + (ry - d[..., 1]) ** 2).mean(axis=1))


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------
def _hypotheses(a: Frag, b: Frag) -> list[np.ndarray]:
    """Rigid maps b -> a from the best complementary chunk alignments."""
    if not len(a.windows) or not len(b.windows):
        return []
    pa = a.smooth[a.windows]                               # (na, CHUNK, 2)
    pb = b.smooth[b.windows][:, ::-1]                      # reversed traversal
    na, nb = len(pa), len(pb)
    src = np.repeat(pb[None], na, axis=0).reshape(na * nb, CHUNK, 2)
    dst = np.repeat(pa[:, None], nb, axis=1).reshape(na * nb, CHUNK, 2)
    res = np.round(batch_kabsch_residual(src, dst), 9)
    order = np.lexsort((np.arange(res.size), res))[:TOPK]
    return [kabsch(src[k], dst[k]) for k in order]


def _icp(a: Frag, b: Frag, t: np.ndarray, iters: int = 6) -> np.ndarray:
    for _ in range(iters):
        dist, idx = a.tree.query(apply(t, b.smooth))
        near = (dist < SEAM_TOL * 2) & ~b.border & ~a.border[idx]
        if near.sum() < 6:
            break
        t = kabsch(b.smooth[near], a.smooth[idx[near]])
    return t


def _colour_diff(a: Frag, b: Frag, seam_b: np.ndarray, idx_a: np.ndarray) -> float:
    """Mean abs RGB difference between matched inward samples of both sides."""
    if not len(seam_b):
        return 255.0
    return float(np.abs(a.colour[idx_a] - b.colour[seam_b]).mean())


def _overlap(a: Frag, b: Frag, t: np.ndarray) -> float:
    p = np.rint(apply(t, b.pixels)).astype(int)
    h, w = a.inner.shape
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    return float(a.inner[p[ok, 1], p[ok, 0]].sum()) / max(len(p), 1)


def score_pair(a: Frag, b: Frag) -> tuple[float, np.ndarray | None]:
    """Best (quality, map b -> a). Quality 0 = no plausible join."""
    best_q, best_t = 0.0, None
    for t in _hypotheses(a, b):
        t = _icp(a, b, t)
        dist, idx = a.tree.query(apply(t, b.smooth))
        seam = (dist < SEAM_TOL) & ~b.border & ~a.border[idx]
        seam_len = float(seam.sum()) * STEP
        if seam_len < MIN_SEAM:
            continue
        rms = float(np.sqrt((dist[seam] ** 2).mean()))
        ov = _overlap(a, b, t)
        col = _colour_diff(a, b, np.nonzero(seam)[0], idx[seam])
        q = seam_len * math.exp(-(rms / 1.5) ** 2) * math.exp(-col / 40.0) * max(0.0, 1.0 - 10.0 * ov)
        q = round(q, 9)
        if q > best_q:
            best_q, best_t = q, t
    return best_q, best_t


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
class _Occupancy:
    """Placed pixels of one cluster on a fixed grid around the cluster seed."""

    SIZE = 7000

    def __init__(self, origin: np.ndarray):
        self.grid = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self.off = np.array([self.SIZE / 2, self.SIZE / 2]) - origin[:2, 2]

    def _cells(self, pose: np.ndarray, pts: np.ndarray):
        p = np.rint(apply(pose, pts) + self.off).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < self.SIZE) & (p[:, 1] >= 0) & (p[:, 1] < self.SIZE)
        return p[ok, 1], p[ok, 0], max(len(p), 1)

    def overlap(self, frag: Frag, pose: np.ndarray) -> float:
        ys, xs, n = self._cells(pose, frag.pixels)
        return float(self.grid[ys, xs].sum()) / n

    def add(self, frag: Frag, pose: np.ndarray) -> None:
        ys, xs = np.nonzero(frag.inner)
        gy, gx, _ = self._cells(pose, np.stack([xs, ys], axis=1).astype(float))
        self.grid[gy, gx] = True


def assemble(frags: dict[str, Frag], pairs: dict[tuple[str, str], tuple[float, np.ndarray]]):
    """Greedy placement. Returns (poses, n_clusters)."""
    edges = []                                   # (-q, a, b, T_b->a)
    for (a, b), (q, t) in pairs.items():
        if t is not None and q > 0:
            edges.append((-q, a, b, t))
            edges.append((-q, b, a, np.linalg.inv(t)))
    edges.sort(key=lambda e: (e[0], e[1], e[2]))
    poses: dict[str, np.ndarray] = {}
    unplaced = set(frags)
    cluster = 0
    while unplaced:
        seed_edges = [e for e in edges if e[1] in unplaced and e[2] in unplaced]
        if not seed_edges:
            for fid in sorted(unplaced):                     # isolated fragments
                poses[fid] = rigid(0.0, CLUSTER_SHIFT * (cluster + 1), 0.0)
                cluster += 1
            break
        _, a, b, t = seed_edges[0]
        base = rigid(0.0, CLUSTER_SHIFT * cluster, 0.0)
        occ = _Occupancy(base)
        poses[a] = base
        occ.add(frags[a], base)
        unplaced.discard(a)
        members = {a}
        grew = True
        while grew:
            grew = False
            for q, p, u, tpu in edges:
                if p in members and u in unplaced:
                    pose = poses[p] @ tpu
                    if occ.overlap(frags[u], pose) <= MAX_OVERLAP:
                        poses[u] = pose
                        occ.add(frags[u], pose)
                        unplaced.discard(u)
                        members.add(u)
                        grew = True
                        break
        cluster += 1
    return poses, cluster


def load_puzzle(puzzle_dir: str) -> tuple[dict, dict[str, Frag]]:
    with open(os.path.join(puzzle_dir, "puzzle.json"), encoding="utf-8") as fh:
        puzzle = json.load(fh)
    frags = {}
    for f in puzzle["fragments"]:
        with Image.open(os.path.join(puzzle_dir, f["file"])) as im:
            frags[f["id"]] = Frag(f["id"], np.array(im.convert("RGBA")))
    return puzzle, frags


def solve(puzzle_dir: str) -> dict:
    puzzle, frags = load_puzzle(puzzle_dir)
    ids = sorted(frags)
    pairs = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pairs[(a, b)] = score_pair(frags[a], frags[b])
    poses, _ = assemble(frags, pairs)
    candidates = {}
    for fid in ids:
        ranked = [((-q), (b if a == fid else a)) for (a, b), (q, _) in pairs.items() if fid in (a, b) and q > 0]
        candidates[fid] = [other for _, other in sorted(ranked)]
    return {
        "schema": SOLUTION_SCHEMA,
        "page_id": puzzle["page_id"],
        "solver": f"{SOLVER}@{SOLVER_VERSION}",
        "fragments": {fid: {"affine": [[float(v) for v in poses[fid][r]] for r in (0, 1)]} for fid in ids},
        "candidates": candidates,
    }
