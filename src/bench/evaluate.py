"""dm-bench evaluation harness (eval_version 1.0).

Per page, for a submitted solution {fragment id -> 2x3 affine, fragment px
-> page px}:

    error e(G, i)   max over the fragment's convex-hull points p of
                    |G A_pred_i p - A_gt_i p|  (worst pixel displacement)
    direct_acc      best global rigid alignment G (hypotheses from each placed
                    fragment, refit by Kabsch on inliers until stable); share of
                    present fragments with e <= tau. Fewer than 2 inliers -> 0,
                    so random or single-fragment submissions score exactly 0.
    neighbor_acc    GT adjacent pairs (shared edge >= MIN_SHARED_PX) whose two
                    fragments fit one rigid G (pair Kabsch fit, either anchor, or
                    the page's global G) within tau. Symmetric; perfect => 1.
    perfect         direct_acc == 1
    adjacency P/R/F1  pairs derived from the placed masks (within ADJ_GAP_PX)
    Hit@1/Hit@5/MRR   only if the solution lists ranked partner candidates

Poses must be rigid (finite, R^T R ~ I, det > 0); anything else counts as
unplaced and is reported as invalid. Missing solution files score 0.
Tolerance tau = TAU_FRAC * page width; also reported at TAU_CURVE.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, QhullError

from . import canon

EVAL_VERSION = "1.0"
SOLUTION_SCHEMA = "dm-bench/solution/1"
TAU_FRAC = 0.01
TAU_CURVE = (0.0025, 0.005, 0.01, 0.02)
LO_ITERS = 5
ORTHO_TOL = 1e-3
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260927


def min_shared_px(page_w: int) -> int:
    """GT pairs sharing less edge than 2*tau carry no matchable seam."""
    return int(math.ceil(2 * TAU_FRAC * page_w))


def adj_gap_px(max_erosion: int) -> int:
    return 2 + 2 * max_erosion


# ---------------------------------------------------------------------------
# Rigid geometry
# ---------------------------------------------------------------------------
def to3(a) -> np.ndarray:
    m = np.eye(3)
    m[:2] = np.asarray(a, dtype=float)
    return m


def rigid_inverse(m: np.ndarray) -> np.ndarray:
    """Closed form (R^T, -R^T t); avoids BLAS kernel differences."""
    r, t = m[:2, :2], m[:2, 2]
    out = np.eye(3)
    out[:2, :2] = r.T
    out[:2, 2] = -r.T @ t
    return out


def is_rigid(a) -> bool:
    try:
        m = np.asarray(a, dtype=float)
    except (TypeError, ValueError):
        return False
    if m.shape != (2, 3) or not np.isfinite(m).all():
        return False
    r = m[:, :2]
    return bool(np.linalg.norm(r.T @ r - np.eye(2)) <= ORTHO_TOL and (r[0, 0] * r[1, 1] - r[0, 1] * r[1, 0]) > 0)


def kabsch2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best rigid (rotation + translation) map src -> dst, closed form, no SVD."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - cs, dst - cd
    angle = math.atan2(float((s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum()),
                       float((s * d).sum()))
    c, n = math.cos(angle), math.sin(angle)
    m = np.eye(3)
    m[:2, :2] = [[c, -n], [n, c]]
    m[:2, 2] = cd - m[:2, :2] @ cs
    return m


def hull_points(alpha: np.ndarray) -> np.ndarray:
    """Convex-hull vertices of opaque pixel centres, (k, 2) in fragment px."""
    ys, xs = np.nonzero(alpha)
    pts = np.stack([xs, ys], axis=1).astype(float)
    if len(pts) < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except QhullError:                  # degenerate (collinear) masks
        return pts[[0, -1]]


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:2, :2].T + m[:2, 2]


# ---------------------------------------------------------------------------
# Per-page scoring
# ---------------------------------------------------------------------------
@dataclass
class PageInputs:
    page_id: str
    doc: int
    canvas: tuple[int, int]
    gt: dict[str, np.ndarray]                 # id -> 3x3
    hulls: dict[str, np.ndarray]              # id -> (k, 2)
    alphas: dict[str, np.ndarray]             # id -> (h, w) uint8
    adjacency: list[tuple[str, str, int]]
    max_erosion: int = 0


@dataclass
class PageScore:
    page_id: str
    doc: int
    n_fragments: int
    placed: int = 0
    invalid: int = 0
    direct_acc: float = 0.0
    neighbor_acc: float = 0.0
    perfect: float = 0.0
    adj_precision: float = 0.0
    adj_recall: float = 0.0
    adj_f1: float = 0.0
    curve_direct: list[float] = field(default_factory=list)
    curve_neighbor: list[float] = field(default_factory=list)
    hit1: float | None = None
    hit5: float | None = None
    mrr: float | None = None


class _Points:
    """All hull points of the placed fragments, predicted and GT, stacked once."""

    def __init__(self, pred: dict, page: PageInputs):
        self.ids = sorted(pred)
        self.pred = {f: _apply(pred[f], page.hulls[f]) for f in self.ids}
        self.gt = {f: _apply(page.gt[f], page.hulls[f]) for f in self.ids}
        if self.ids:
            self.p = np.concatenate([self.pred[f] for f in self.ids])
            self.q = np.concatenate([self.gt[f] for f in self.ids])
            sizes = [len(self.pred[f]) for f in self.ids]
            self.starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)

    def errors(self, g: np.ndarray) -> np.ndarray:
        """Per-fragment worst displacement under global map g (order = self.ids)."""
        d = np.linalg.norm(_apply(g, self.p) - self.q, axis=1)
        return np.maximum.reduceat(d, self.starts)

    def inliers(self, g: np.ndarray, tau: float) -> list[str]:
        return [f for f, e in zip(self.ids, self.errors(g)) if e <= tau]

    def refit(self, ids: list[str]) -> np.ndarray:
        return kabsch2d(np.concatenate([self.pred[f] for f in ids]),
                        np.concatenate([self.gt[f] for f in ids]))


def best_alignment(pred: dict, page: PageInputs, tau: float) -> tuple[np.ndarray | None, int]:
    """Deterministic LO-RANSAC: every single-fragment hypothesis, refit by
    Kabsch on its inliers while the inlier set grows."""
    if len(pred) < 2:
        return None, 0
    pts = _Points(pred, page)
    best_g, best_n = None, 0
    for k in pts.ids:
        g = page.gt[k] @ rigid_inverse(pred[k])
        inl = pts.inliers(g, tau)
        for _ in range(LO_ITERS):
            if len(inl) < 2:
                break
            g2 = pts.refit(inl)
            inl2 = pts.inliers(g2, tau)
            if len(inl2) <= len(inl):
                if len(inl2) == len(inl):
                    g = g2
                break
            g, inl = g2, inl2
        if len(inl) > best_n:
            best_g, best_n = g, len(inl)
    return (best_g, best_n) if best_n >= 2 else (None, 0)


def _pair_error(pred: dict, page: PageInputs, g: np.ndarray, ids) -> float:
    return max(float(np.max(np.linalg.norm(_apply(g @ pred[f], page.hulls[f])
                                           - _apply(page.gt[f], page.hulls[f]), axis=1)))
               for f in ids)


def _pair_ok(pred: dict, page: PageInputs, i: str, j: str, g_global, tau: float) -> bool:
    fit = kabsch2d(np.concatenate([_apply(pred[f], page.hulls[f]) for f in (i, j)]),
                   np.concatenate([_apply(page.gt[f], page.hulls[f]) for f in (i, j)]))
    cands = [fit, page.gt[i] @ rigid_inverse(pred[i]), page.gt[j] @ rigid_inverse(pred[j])]
    if g_global is not None:
        cands.append(g_global)
    return any(_pair_error(pred, page, g, (i, j)) <= tau for g in cands)


def _placed_adjacency(pred: dict, page: PageInputs, g: np.ndarray) -> set[tuple[str, str]]:
    """Neighbour pairs of the placed masks, rasterised in the page frame."""
    w, h = page.canvas
    pad = 64
    canvas = np.full((h + 2 * pad, w + 2 * pad), -1, dtype=np.int32)
    ids = sorted(pred)
    for k, fid in enumerate(ids):
        ys, xs = np.nonzero(page.alphas[fid])
        pp = np.rint(_apply(g @ pred[fid], np.stack([xs, ys], axis=1).astype(float))).astype(int) + pad
        ok = (pp[:, 0] >= 0) & (pp[:, 0] < canvas.shape[1]) & (pp[:, 1] >= 0) & (pp[:, 1] < canvas.shape[0])
        canvas[pp[ok, 1], pp[ok, 0]] = k
    gap = adj_gap_px(page.max_erosion)
    struct = np.ones((2 * gap + 1, 2 * gap + 1), dtype=bool)
    pairs = set()
    for k, fid in enumerate(ids):
        ys, xs = np.nonzero(canvas == k)
        if ys.size == 0:
            continue
        # Dilate only a window around the fragment: whole-canvas dilation
        # per fragment dominated runtime.
        y0, y1 = max(ys.min() - gap, 0), min(ys.max() + gap + 1, canvas.shape[0])
        x0, x1 = max(xs.min() - gap, 0), min(xs.max() + gap + 1, canvas.shape[1])
        win = canvas[y0:y1, x0:x1]
        mask = win == k
        ring = binary_dilation(mask, structure=struct) & (win >= 0) & ~mask
        for other in np.unique(win[ring]):
            a, b = sorted((fid, ids[int(other)]))
            pairs.add((a, b))
    return pairs


def score_page(page: PageInputs, solution: dict | None) -> PageScore:
    n = len(page.gt)
    score = PageScore(page.page_id, page.doc, n)
    if not solution:
        score.curve_direct = [0.0] * len(TAU_CURVE)
        score.curve_neighbor = [0.0] * len(TAU_CURVE)
        return score
    pred, invalid = {}, 0
    entries = solution.get("fragments")
    for fid, entry in (entries if isinstance(entries, dict) else {}).items():
        if fid in page.gt and isinstance(entry, dict) and is_rigid(entry.get("affine")):
            pred[fid] = to3(entry["affine"])
        else:
            invalid += 1
    score.placed, score.invalid = len(pred), invalid
    w = page.canvas[0]
    min_shared = min_shared_px(w)
    gt_pairs = [(a, b) for a, b, s in page.adjacency if s >= min_shared]

    for t_frac in TAU_CURVE:
        tau = t_frac * w
        g, n_in = best_alignment(pred, page, tau)
        direct = n_in / n if n else 0.0
        good = sum(1 for a, b in gt_pairs if a in pred and b in pred and _pair_ok(pred, page, a, b, g, tau))
        neighbor = good / len(gt_pairs) if gt_pairs else 0.0
        score.curve_direct.append(round(direct, 6))
        score.curve_neighbor.append(round(neighbor, 6))
        if t_frac == TAU_FRAC:
            score.direct_acc, score.neighbor_acc = direct, neighbor
            score.perfect = 1.0 if n_in == n else 0.0
            _adjacency_scores(score, pred, page, g, gt_pairs)
    _candidate_scores(score, solution.get("candidates"), gt_pairs)
    return score


def _adjacency_scores(score: PageScore, pred, page, g, gt_pairs) -> None:
    # Without a valid alignment no two fragments sit correctly relative to each
    # other, so contacts between the placed masks are meaningless.
    if len(pred) < 2 or g is None:
        return
    placed = _placed_adjacency(pred, page, g)
    all_gt = {(a, b) for a, b, _ in page.adjacency}
    filt = set(gt_pairs)
    tp_all = len(placed & all_gt)
    score.adj_precision = tp_all / len(placed) if placed else 0.0
    score.adj_recall = len(placed & filt) / len(filt) if filt else 0.0
    p, r = score.adj_precision, score.adj_recall
    score.adj_f1 = 2 * p * r / (p + r) if p + r else 0.0


def _candidate_scores(score: PageScore, candidates, gt_pairs) -> None:
    if not isinstance(candidates, dict):
        return
    nbrs: dict[str, set[str]] = {}
    for a, b in gt_pairs:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    hits1 = hits5 = rr = 0.0
    for fid, true in nbrs.items():
        raw = candidates.get(fid)
        raw = raw if isinstance(raw, (list, tuple)) else []
        ranked = [c[0] if isinstance(c, (list, tuple)) and c else c for c in raw]
        ranked = [c for c in ranked if isinstance(c, str)]    # malformed entries never match
        rank = next((k + 1 for k, c in enumerate(ranked) if c in true), None)
        hits1 += rank == 1
        hits5 += rank is not None and rank <= 5
        rr += 1.0 / rank if rank else 0.0
    m = len(nbrs) or 1
    score.hit1, score.hit5, score.mrr = hits1 / m, hits5 / m, rr / m


# ---------------------------------------------------------------------------
# Loading + aggregation
# ---------------------------------------------------------------------------
def _read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return canon.loads(fh.read())


def load_page(puzzle_dir: str, answer: dict) -> PageInputs:
    from PIL import Image

    puzzle = _read_json(os.path.join(puzzle_dir, "puzzle.json"))
    alphas, hulls = {}, {}
    for f in puzzle["fragments"]:
        with Image.open(os.path.join(puzzle_dir, f["file"])) as im:
            alpha = np.array(im.getchannel("A"))
        alphas[f["id"]] = alpha
        hulls[f["id"]] = hull_points(alpha)
    gt = {fid: to3(v["affine"]) for fid, v in answer["fragments"].items()}
    max_ero = max((v.get("erosion_px", 0) for v in answer["fragments"].values()), default=0)
    return PageInputs(
        page_id=answer["page_id"], doc=int(answer["doc"]), canvas=tuple(puzzle["canvas"]),
        gt=gt, hulls=hulls, alphas=alphas,
        adjacency=[(a, b, int(s)) for a, b, s in answer["adjacency"]], max_erosion=max_ero,
    )


def load_solution(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        sol = _read_json(path)
    except (ValueError, json.JSONDecodeError):
        return None
    return sol if isinstance(sol, dict) else None


METRICS = ("direct_acc", "neighbor_acc", "perfect", "adj_precision", "adj_recall", "adj_f1")


def aggregate(scores: list[PageScore]) -> dict:
    """Mean per metric with a document-cluster bootstrap 95% CI (fixed seed)."""
    scores = sorted(scores, key=lambda s: s.page_id)
    docs = sorted({s.doc for s in scores})
    by_doc = {d: [s for s in scores if s.doc == d] for d in docs}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = rng.integers(0, len(docs), size=(BOOTSTRAP_B, len(docs))) if docs else None
    out: dict = {"pages": len(scores), "docs": len(docs)}
    for m in METRICS:
        per_doc = np.array([[sum(getattr(s, m) for s in by_doc[d]), len(by_doc[d])] for d in docs], dtype=float)
        mean = float(np.mean([getattr(s, m) for s in scores])) if scores else 0.0
        if draws is not None:
            sums = per_doc[draws, 0].sum(axis=1) / per_doc[draws, 1].sum(axis=1)
            lo, hi = np.percentile(sums, [2.5, 97.5])
        else:
            lo = hi = 0.0
        out[m] = {"mean": round(mean, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    for key in ("curve_direct", "curve_neighbor"):
        arr = np.array([getattr(s, key) for s in scores]) if scores else np.zeros((1, len(TAU_CURVE)))
        out[key] = {"tau_frac": list(TAU_CURVE), "mean": [round(float(v), 4) for v in arr.mean(axis=0)],
                    "auc": round(float(arr.mean()), 4)}
    cand = [s for s in scores if s.mrr is not None]
    if cand:
        out["hit1"] = round(float(np.mean([s.hit1 for s in cand])), 4)
        out["hit5"] = round(float(np.mean([s.hit5 for s in cand])), 4)
        out["mrr"] = round(float(np.mean([s.mrr for s in cand])), 4)
    out["invalid_poses"] = int(sum(s.invalid for s in scores))
    return out


def split_answers(release: str, tier: str, split: str, secret_hex: str | None = None) -> dict[str, dict]:
    """page_id -> answer. Held-out `test` answers are regenerated from the secret."""
    if split != "test":
        adir = os.path.join(release, "answers", tier, split)
        return {name[:-5]: _read_json(os.path.join(adir, name)) for name in sorted(os.listdir(adir))}
    if not secret_hex:
        raise ValueError("scoring the test split requires DM_BENCH_TEST_SECRET")
    from . import build

    spec = _read_json(os.path.join(release, "benchmark.json"))
    n_docs = spec["splits"][tier]["test"]["docs"]
    entropy = build.secret_entropy(secret_hex)
    out = {}
    for doc in range(n_docs):
        for built in build.build_doc(entropy, tier, "test", doc):
            out[built.page_id] = canon.loads(canon.dumps(built.answer))
    return out


def evaluate_split(release: str, solutions: str, tier: str, split: str,
                   secret_hex: str | None = None) -> tuple[list[PageScore], dict]:
    answers = split_answers(release, tier, split, secret_hex)
    scores = []
    for page_id in sorted(answers):
        page = load_page(os.path.join(release, "puzzles", tier, split, page_id), answers[page_id])
        sol = load_solution(os.path.join(solutions, tier, split, f"{page_id}.json"))
        scores.append(score_page(page, sol))
    return scores, aggregate(scores)
