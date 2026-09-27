"""Evaluation harness: metric properties demanded by the design review."""
import json
import math
import os

import numpy as np
import pytest

from src.bench import baselines, build
from src.bench import evaluate as E

DOCS = {"train": 1, "val": 1, "test-dev": 1, "test": 1}


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    rel = str(tmp_path_factory.mktemp("rel"))
    build.write_release(rel, ["easy", "hard"], DOCS, ["val"])
    out = []
    for tier in ("easy", "hard"):
        for pid, ans in E.split_answers(rel, tier, "val").items():
            pdir = os.path.join(rel, "puzzles", tier, "val", pid)
            with open(os.path.join(pdir, "puzzle.json")) as fh:
                puzzle = json.load(fh)
            out.append((E.load_page(pdir, ans), ans, puzzle))
    return out


def _moved(sol, g):
    return {"fragments": {f: {"affine": (g @ E.to3(v["affine"]))[:2].tolist()}
                          for f, v in sol["fragments"].items()}}


def _rot(theta, tx, ty):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.0]])


def test_oracle_scores_one(pages):
    for page, ans, _ in pages:
        s = E.score_page(page, baselines.oracle_solution(ans))
        assert s.direct_acc == 1.0
        assert s.neighbor_acc == 1.0
        assert s.perfect == 1.0
        assert s.adj_recall == 1.0
        assert s.adj_precision >= 0.95


def test_invariant_to_global_rigid_transform(pages):
    for page, ans, _ in pages:
        s = E.score_page(page, _moved(baselines.oracle_solution(ans), _rot(2.1, -400, 900)))
        assert s.direct_acc == 1.0
        assert s.neighbor_acc == 1.0


def test_random_and_single_fragment_score_exactly_zero(pages):
    for page, ans, puzzle in pages:
        r = E.score_page(page, baselines.random_solution(puzzle, 3))
        assert r.direct_acc == 0.0
        assert r.neighbor_acc == 0.0
        assert r.adj_f1 == 0.0
        orc = baselines.oracle_solution(ans)
        one = {"fragments": dict(list(orc["fragments"].items())[:1])}
        assert E.score_page(page, one).direct_acc == 0.0


def test_non_rigid_poses_are_invalid_not_crashes(pages):
    page, ans, _ = pages[0]
    orc = baselines.oracle_solution(ans)
    mirrored = _moved(orc, np.diag([-1.0, 1.0, 1.0]))
    sheared = _moved(orc, np.array([[1.0, 0.4, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    for sol in (mirrored, sheared):
        s = E.score_page(page, sol)
        assert s.direct_acc == 0.0
        assert s.invalid == len(page.gt)
    broken = json.loads(json.dumps(orc))
    fids = sorted(broken["fragments"])
    broken["fragments"][fids[0]]["affine"] = [[0, 0, 0], [0, 0, 0]]          # singular
    broken["fragments"][fids[1]]["affine"] = "nonsense"
    broken["fragments"]["not-a-fragment"] = {"affine": [[1, 0, 0], [0, 1, 0]]}
    s = E.score_page(page, broken)
    assert s.invalid == 3
    assert s.placed == len(fids) - 2


def test_missing_solution_scores_zero(pages):
    page, _, _ = pages[0]
    s = E.score_page(page, None)
    assert (s.direct_acc, s.neighbor_acc, s.perfect) == (0.0, 0.0, 0.0)


def test_one_misplaced_fragment_gives_partial_credit(pages):
    page, ans, _ = pages[0]
    orc = baselines.oracle_solution(ans)
    fid = sorted(orc["fragments"])[0]
    orc["fragments"][fid]["affine"] = (_rot(0, 250, 0) @ E.to3(orc["fragments"][fid]["affine"]))[:2].tolist()
    s = E.score_page(page, orc)
    n = len(page.gt)
    assert s.direct_acc == pytest.approx((n - 1) / n)
    assert 0 < s.neighbor_acc < 1
    assert s.perfect == 0.0


def test_neighbor_test_is_symmetric_and_perfect_implies_one(pages):
    page, ans, _ = pages[-1]
    rng = np.random.default_rng(0)
    noisy = {"fragments": {}}
    for f, v in ans["fragments"].items():             # sub-tolerance per-fragment jitter
        jitter = _rot(float(rng.normal(0, 0.0015)), float(rng.normal(0, 1.0)), float(rng.normal(0, 1.0)))
        noisy["fragments"][f] = {"affine": (E.to3(v["affine"]) @ jitter)[:2].tolist()}
    pred = {f: E.to3(v["affine"]) for f, v in noisy["fragments"].items()}
    tau = E.TAU_FRAC * page.canvas[0]
    g, _ = E.best_alignment(pred, page, tau)
    for a, b, _ in page.adjacency:
        assert E._pair_ok(pred, page, a, b, g, tau) == E._pair_ok(pred, page, b, a, g, tau)
    s = E.score_page(page, noisy)
    if s.perfect == 1.0:
        assert s.neighbor_acc == 1.0


def test_candidates_hit_at_k(pages):
    page, ans, _ = pages[0]
    min_shared = E.min_shared_px(page.canvas[0])
    cands = {}
    for a, b, shared in page.adjacency:
        if shared >= min_shared:
            cands.setdefault(a, []).append(b)
            cands.setdefault(b, []).append(a)
    sol = dict(baselines.oracle_solution(ans), candidates=cands)
    s = E.score_page(page, sol)
    assert (s.hit1, s.hit5, s.mrr) == (1.0, 1.0, 1.0)


def test_aggregate_is_order_independent(pages):
    scores = [E.score_page(p, baselines.random_solution(z, 1)) for p, _, z in pages]
    scores += [E.score_page(p, baselines.oracle_solution(a)) for p, a, _ in pages]
    assert E.aggregate(scores) == E.aggregate(list(reversed(scores)))


def test_geometry_helpers():
    g = _rot(0.8, 12.0, -3.0)
    src = np.random.default_rng(1).uniform(0, 100, (20, 2))
    fit = E.kabsch2d(src, src @ g[:2, :2].T + g[:2, 2])
    assert np.allclose(fit, g)
    assert np.allclose(E.rigid_inverse(g) @ g, np.eye(3))
    assert E.is_rigid(g[:2])
    assert not E.is_rigid([[1, 0, 0], [0, -1, 0]])                          # mirror
    assert not E.is_rigid([[float("nan"), 0, 0], [0, 1, 0]])
    assert not E.is_rigid("x")
