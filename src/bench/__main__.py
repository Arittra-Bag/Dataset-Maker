"""dm-bench command line.

    python -m src.bench build  --out DIR [--tiers ...] [--splits ...] [--limit-docs N] [--workers N]
    python -m src.bench verify DIR

The held-out `test` split needs DM_BENCH_TEST_SECRET in the environment.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import build


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
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
