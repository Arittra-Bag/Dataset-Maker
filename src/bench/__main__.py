"""dm-bench command line.

    python -m src.bench build  --out DIR [--tiers ...] [--splits ...] [--limit-docs N] [--workers N]
    python -m src.bench verify DIR
    python -m src.bench solve  --release DIR --out DIR [--method edge-greedy|random|oracle] [--workers N]
    python -m src.bench eval   --release DIR --solutions DIR [--out results.json]

The held-out `test` split needs DM_BENCH_TEST_SECRET in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import baselines, build, canon


def _cmd_build(args) -> int:
    secret = os.environ.get(build.SECRET_ENV)
    splits = args.splits or list(build.RELEASE_SPLITS)
    if "test" in splits and not secret:
        print(f"skipping split 'test': {build.SECRET_ENV} is not set", file=sys.stderr)
        splits = [s for s in splits if s != "test"]
    spec = build.write_release(
        args.out, args.tiers, dict(build.DEFAULT_DOCS), splits, secret,
        limit_docs=args.limit_docs, workers=args.workers,
    )
    for tier, per_split in spec["splits"].items():
        pages = ", ".join(f"{s}={v['pages']}" for s, v in per_split.items())
        print(f"{tier}: {pages}")
    print(f"wrote {args.out}")
    return 0


def _cmd_verify(args) -> int:
    bad = build.verify_checksums(args.root)
    if bad:
        print(f"{len(bad)} file(s) do not match SHA256SUMS:", *bad[:20], sep="\n  ")
        return 1
    print("all files match SHA256SUMS")
    return 0


def _pages(release: str, tiers, splits):
    for tier in tiers:
        for split in splits:
            d = os.path.join(release, "puzzles", tier, split)
            if os.path.isdir(d):
                for pid in sorted(os.listdir(d)):
                    yield tier, split, pid


def _cmd_solve(args) -> int:
    if args.method == "oracle" and "test" in args.splits:
        print("oracle reads answers; it cannot run on the held-out test split", file=sys.stderr)
        return 2
    jobs = [(args.release, t, s, p, args.method) for t, s, p in _pages(args.release, args.tiers, args.splits)]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(baselines.run_method, jobs, chunksize=1))
    else:
        results = [baselines.run_method(j) for j in jobs]
    for tier, split, pid, sol in results:
        path = os.path.join(args.out, tier, split, f"{pid}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii") as fh:
            fh.write(canon.dumps(sol))
    print(f"wrote {len(results)} solution(s) to {args.out}")
    return 0


def _cmd_eval(args) -> int:
    from . import evaluate

    secret = os.environ.get(build.SECRET_ENV)
    with open(os.path.join(args.release, "SHA256SUMS"), "rb") as fh:
        digest = canon.sha256_bytes(fh.read())
    with open(os.path.join(args.release, "benchmark.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    results = {"benchmark": f"{spec['name']}@{spec['version']}", "release_sha256": digest,
               "eval_version": evaluate.EVAL_VERSION, "tau_frac": evaluate.TAU_FRAC,
               "solvers": set(), "scores": {}}
    for tier in args.tiers:
        for split in args.splits:
            if not os.path.isdir(os.path.join(args.release, "puzzles", tier, split)):
                continue
            if split == "test" and not secret:
                print(f"skipping {tier}/test: {build.SECRET_ENV} is not set", file=sys.stderr)
                continue
            _, agg = evaluate.evaluate_split(args.release, args.solutions, tier, split, secret)
            results["scores"].setdefault(tier, {})[split] = agg
            sdir = os.path.join(args.solutions, tier, split)
            for name in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
                sol = evaluate.load_solution(os.path.join(sdir, name)) or {}
                results["solvers"].add(str(sol.get("solver", "unknown")))
            m = agg
            print(f"{tier:6} {split:8} pages {m['pages']:3}  direct {m['direct_acc']['mean']:.3f} "
                  f"{m['direct_acc']['ci95']}  neighbor {m['neighbor_acc']['mean']:.3f}  "
                  f"perfect {m['perfect']['mean']:.3f}  adjF1 {m['adj_f1']['mean']:.3f}")
    results["solvers"] = sorted(results["solvers"])
    if args.out:
        with open(args.out, "w", encoding="ascii") as fh:
            fh.write(canon.dumps(results))
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a benchmark release")
    b.add_argument("--out", required=True)
    b.add_argument("--tiers", nargs="+", default=list(build.TIER_SPECS), choices=list(build.TIER_SPECS))
    b.add_argument("--splits", nargs="+", choices=["train", *build.PUBLIC_SPLITS[1:], "test"])
    b.add_argument("--limit-docs", type=int, default=None, help="first N docs per split (CI slices)")
    b.add_argument("--workers", type=int, default=1)
    b.set_defaults(func=_cmd_build)
    v = sub.add_parser("verify", help="check files against SHA256SUMS")
    v.add_argument("root")
    v.set_defaults(func=_cmd_verify)
    all_splits = ["train", *build.PUBLIC_SPLITS[1:], "test"]
    so = sub.add_parser("solve", help="run a solver over puzzles")
    so.add_argument("--release", required=True)
    so.add_argument("--out", required=True)
    so.add_argument("--method", default="edge-greedy", choices=["edge-greedy", "random", "oracle"])
    so.add_argument("--tiers", nargs="+", default=list(build.TIER_SPECS), choices=list(build.TIER_SPECS))
    so.add_argument("--splits", nargs="+", default=list(build.RELEASE_SPLITS), choices=all_splits)
    so.add_argument("--workers", type=int, default=1)
    so.set_defaults(func=_cmd_solve)
    ev = sub.add_parser("eval", help="score solutions against the answers")
    ev.add_argument("--release", required=True)
    ev.add_argument("--solutions", required=True)
    ev.add_argument("--tiers", nargs="+", default=list(build.TIER_SPECS), choices=list(build.TIER_SPECS))
    ev.add_argument("--splits", nargs="+", default=list(build.RELEASE_SPLITS), choices=all_splits)
    ev.add_argument("--out", default=None)
    ev.set_defaults(func=_cmd_eval)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
