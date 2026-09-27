"""edge-greedy baseline: answer-free, deterministic, solves an easy page."""
import ast
import os
import shutil

import numpy as np
import pytest

from src.bench import build, solver
from src.bench import evaluate as E

DOCS = {"train": 1, "val": 1, "test-dev": 1, "test": 1}
FORBIDDEN = {"tearing", "provenance", "build", "docgen", "fragments", "evaluate", "baselines",
             "pipeline", "packager", "inspection"}


def test_solver_imports_nothing_that_knows_answers():
    path = os.path.join(os.path.dirname(solver.__file__), "solver.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        for name in names:
            assert not (set(name.split(".")) & FORBIDDEN), name


@pytest.fixture(scope="module")
def easy_page(tmp_path_factory):
    rel = str(tmp_path_factory.mktemp("rel"))
    build.write_release(rel, ["easy"], DOCS, ["val"])
    answers = E.split_answers(rel, "easy", "val")
    pid = sorted(answers)[0]
    return rel, pid, answers[pid]


def test_solves_easy_page_without_answers(easy_page, tmp_path):
    rel, pid, answer = easy_page
    isolated = tmp_path / "puzzle"
    shutil.copytree(os.path.join(rel, "puzzles", "easy", "val", pid), isolated)   # no answers/, no spec
    sol = solver.solve(str(isolated))
    score = E.score_page(E.load_page(str(isolated), answer), sol)
    assert score.direct_acc >= 0.9
    assert score.neighbor_acc >= 0.9
    assert sol["solver"] == f"{solver.SOLVER}@{solver.SOLVER_VERSION}"


def test_deterministic(easy_page):
    rel, pid, _ = easy_page
    pdir = os.path.join(rel, "puzzles", "easy", "val", pid)
    assert solver.solve(pdir) == solver.solve(pdir)


def test_moore_contour_terminates_and_orders():
    square = np.zeros((6, 7), bool)
    square[1:5, 2:6] = True
    assert len(solver.moore_contour(square)) == 12
    neck = np.zeros((8, 8), bool)
    neck[2:6, 1:3] = True
    neck[3, 3:6] = True                                   # 1-px neck revisits pixels
    assert len(solver.moore_contour(neck)) < 4 * neck.sum()


def test_straight_runs_flag_lines_not_waves():
    t = np.arange(0, 400, 2.0)
    line = np.stack([t, 0.3 * t], axis=1)
    wave = np.stack([t, 12 * np.sin(t / 15.0)], axis=1)
    assert solver.straight_runs(line).mean() > 0.9
    assert solver.straight_runs(wave).mean() < 0.1


def test_batch_kabsch_residual_zero_for_rigid_copies():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 50, (5, 20, 2))
    g = solver.rigid(1.1, 4.0, -7.0)
    dst = np.stack([solver.apply(g, s) for s in src])
    assert np.allclose(solver.batch_kabsch_residual(src, dst), 0.0, atol=1e-9)
