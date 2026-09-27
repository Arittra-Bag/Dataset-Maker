"""Sanity baselines. `oracle` reads answers (harness self-check only, never a
method); `random` places fragments uniformly at random (the floor)."""
from __future__ import annotations

import math

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
