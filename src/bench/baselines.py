"""Sanity baselines. `oracle` reads answers (harness self-check only, never a
method); `random` places fragments uniformly at random (the floor)."""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

from .evaluate import SOLUTION_SCHEMA


def oracle_solution(answer: dict) -> dict:
    return {"schema": SOLUTION_SCHEMA, "page_id": answer["page_id"],
            "fragments": {fid: {"affine": v["affine"]} for fid, v in answer["fragments"].items()}}


def random_solution(puzzle: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    w, h = puzzle["canvas"]
    frags = {}
    for f in puzzle["fragments"]:
        a = float(rng.uniform(0, 2 * math.pi))
        c, s = math.cos(a), math.sin(a)
        frags[f["id"]] = {"affine": [[c, -s, float(rng.uniform(0, w))], [s, c, float(rng.uniform(0, h))]]}
    return {"schema": SOLUTION_SCHEMA, "page_id": puzzle["page_id"], "fragments": frags}


def run_method(job: tuple) -> tuple:
    """Pool worker for `solve`: (release, tier, split, page_id, method).

    Only the puzzle directory is read, except by `oracle` (self-check only).
    Lives here, not in __main__, so spawned workers can unpickle it.
    """
    release, tier, split, pid, method = job
    pdir = os.path.join(release, "puzzles", tier, split, pid)
    if method == "edge-greedy":
        from . import solver

        return tier, split, pid, solver.solve(pdir)
    if method == "random":
        with open(os.path.join(pdir, "puzzle.json"), encoding="utf-8") as fh:
            puzzle = json.load(fh)
        seed = int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], "big")
        return tier, split, pid, random_solution(puzzle, seed)
    if method == "oracle":
        with open(os.path.join(release, "answers", tier, split, f"{pid}.json"), encoding="utf-8") as fh:
            return tier, split, pid, oracle_solution(json.load(fh))
    raise ValueError(f"unknown method {method!r}")
