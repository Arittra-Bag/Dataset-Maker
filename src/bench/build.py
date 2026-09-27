"""Deterministic dm-bench release builder.

Every page is a pure function of (entropy, tier, doc, page). Public splits
use PUBLIC_ENTROPY; the held-out `test` split uses a 128-bit secret
(DM_BENCH_TEST_SECRET) and ships puzzles only, plus a commitment.

Layout:
    benchmark.json                     spec, counts, versions (no timestamps)
    SHA256SUMS                         byte hashes: download integrity
    CONTENT.sha256                     decoded-content digests: regeneration checks
    puzzles/<tier>/<split>/<page_id>/puzzle.json, f000.png ...
    answers/<tier>/<split>/<page_id>.json          (public splits only)
"""
from __future__ import annotations

import os
import platform
import tempfile
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label as cc_label

from .. import __version__
from ..tearing import TornPage, tear_page
from . import canon
from .docgen import PAGE_PX, make_document, render_document
from .fragments import Corruption, make_fragment

BENCH_NAME = "dm-bench"
BENCH_VERSION = "0.1.0"
PUZZLE_SCHEMA = "dm-bench/puzzle/1"
ANSWER_SCHEMA = "dm-bench/answer/1"
PUBLIC_ENTROPY = 20260927
SECRET_ENV = "DM_BENCH_TEST_SECRET"
MAX_TEAR_ATTEMPTS = 25


@dataclass(frozen=True)
class TierSpec:
    name: str
    n_range: tuple[int, int]
    noise_strength: float
    noise_scale: float
    corruption: Corruption
    missing_p: float = 0.0


TIER_SPECS = {
    "easy": TierSpec("easy", (8, 12), 20.0, 96.0, Corruption()),
    "medium": TierSpec("medium", (12, 20), 28.0, 96.0,
                       Corruption(rotate=True, noise_halfwidth=2, jpeg_quality=85)),
    "hard": TierSpec("hard", (20, 32), 32.0, 80.0,
                     Corruption(rotate=True, blur_radius=0.6, noise_halfwidth=4,
                                jpeg_quality=70, erosion_px=(1, 3)),
                     missing_p=0.1),
}

# Documents per split (per tier). Public splits index docs from disjoint
# ranges under PUBLIC_ENTROPY; `test` indexes from 0 under the secret.
DEFAULT_DOCS = {"train": 120, "val": 20, "test-dev": 10, "test": 30}
BLANK_INK = 0.01                  # fragments with < 1% ink pixels count as blank
PUBLIC_SPLITS = ("train", "val", "test-dev")


def doc_range(split: str, docs: dict[str, int]) -> range:
    if split == "test":
        return range(docs["test"])
    start = 0
    for s in PUBLIC_SPLITS:
        if s == split:
            return range(start, start + docs[s])
        start += docs[s]
    raise ValueError(f"unknown split {split!r}")


def secret_entropy(secret_hex: str) -> int:
    try:
        value = int(secret_hex, 16)
    except (TypeError, ValueError):
        # `from None`: the original error message contains the secret verbatim.
        raise ValueError(f"{SECRET_ENV} is not a valid hex string") from None
    if value.bit_length() < 100:
        raise ValueError(f"{SECRET_ENV} must be a >=128-bit hex string")
    return value


# ---------------------------------------------------------------------------
# Page construction
# ---------------------------------------------------------------------------
def _single_component(torn: TornPage) -> bool:
    return all(cc_label(p.mask)[1] == 1 for p in torn.pieces)


def _tear(entropy: int, tier: TierSpec, tier_code: int, doc: int, page: int,
          page_rgb: np.ndarray) -> tuple[TornPage, int, int]:
    """Tear with deterministic re-draws until every piece is one component."""
    for attempt in range(MAX_TEAR_ATTEMPTS):
        n_rng = canon.stream(entropy, tier_code, doc, page, canon.PURPOSE["npieces"], attempt)
        n = int(n_rng.integers(tier.n_range[0], tier.n_range[1] + 1))
        seed = canon.stream_u64(entropy, tier_code, doc, page, canon.PURPOSE["tear"], attempt)
        torn = tear_page(page_rgb, n, seed, tier.noise_strength, tier.noise_scale)
        if _single_component(torn):
            return torn, seed, attempt
    raise RuntimeError(f"no single-component tear after {MAX_TEAR_ATTEMPTS} attempts")


def _shared_lengths(labels: np.ndarray) -> dict[tuple[int, int], int]:
    """4-neighbour boundary pixel-pair counts per unordered label pair."""
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        diff = a != b
        pairs.append(np.stack([np.minimum(a[diff], b[diff]), np.maximum(a[diff], b[diff])], axis=1))
    allp = np.concatenate(pairs)
    if not allp.size:
        return {}
    uniq, counts = np.unique(allp, axis=0, return_counts=True)
    return {(int(i), int(j)): int(c) for (i, j), c in zip(uniq, counts)}


@dataclass
class BuiltPage:
    page_id: str
    puzzle: dict
    answer: dict
    pngs: dict[str, bytes]


def build_page(entropy: int, tier_name: str, split: str, doc: int, page: int,
               page_rgb: np.ndarray) -> BuiltPage:
    tier = TIER_SPECS[tier_name]
    code = canon.TIERS[tier_name]
    torn, tear_seed, attempt = _tear(entropy, tier, code, doc, page, page_rgb)

    def rng(purpose: str) -> np.random.Generator:
        return canon.stream(entropy, code, doc, page, canon.PURPOSE[purpose], attempt)

    pieces = sorted(torn.pieces, key=lambda p: p.label)
    n = len(pieces)
    missing_idx: set[int] = set()
    if tier.missing_p > 0:
        m_rng = rng("missing")
        m = int(np.clip(m_rng.binomial(n, tier.missing_p), 1, n - 2))
        missing_idx = {int(i) for i in m_rng.choice(n, size=m, replace=False)}
    present = [p for k, p in enumerate(pieces) if k not in missing_idx]
    perm = rng("idperm").permutation(len(present))
    ids = {p.label: f"f{int(perm[k]):03d}" for k, p in enumerate(present)}

    frag_rngs = {name: rng(name) for name in ("erosion", "rotation", "pad", "noise")}
    frags, pngs, ink = {}, {}, {}
    for p in present:
        frag = make_fragment(page_rgb, p, tier.corruption, frag_rngs)
        fid = ids[p.label]
        frags[fid] = frag
        pngs[f"{fid}.png"] = _png_bytes(frag.rgba)
        # Measured on the clean page under the pre-erosion mask.
        clean = page_rgb[p.y:p.y + p.mask.shape[0], p.x:p.x + p.mask.shape[1]][p.mask]
        ink[fid] = float((clean.mean(axis=1) < 200).mean())

    page_id = canon.opaque_id(entropy, code, doc, page)
    puzzle = {
        "schema": PUZZLE_SCHEMA,
        "page_id": page_id,
        "tier": tier_name,
        "canvas": [PAGE_PX[0], PAGE_PX[1]],
        "fragments": [{"id": fid, "file": f"{fid}.png", "w": int(frags[fid].rgba.shape[1]),
                       "h": int(frags[fid].rgba.shape[0])} for fid in sorted(frags)],
    }
    shared = _shared_lengths(torn.labels)
    adjacency = sorted(
        [ids[a], ids[b], length] if ids[a] < ids[b] else [ids[b], ids[a], length]
        for (a, b), length in shared.items() if a in ids and b in ids
    )
    answer = {
        "schema": ANSWER_SCHEMA,
        "page_id": page_id,
        "tier": tier_name,
        "split": split,
        "doc": doc,
        "page": page,
        "tear_seed": str(tear_seed),
        "tear_attempts": attempt + 1,
        "n_pieces": n,
        "fragments": {fid: {"affine": f.affine, "rot_mdeg": f.rot_mdeg, "erosion_px": f.erosion_px,
                            "ink_frac": round(ink[fid], 4)}
                      for fid, f in sorted(frags.items())},
        "missing": [{"bbox": [pieces[k].x, pieces[k].y, int(pieces[k].mask.shape[1]),
                              int(pieces[k].mask.shape[0])], "area": int(pieces[k].mask.sum())}
                    for k in sorted(missing_idx)],
        "adjacency": adjacency,
    }
    return BuiltPage(page_id, puzzle, answer, pngs)


def _png_bytes(rgba: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", compress_level=6, optimize=False)
    return buf.getvalue()


def build_doc(entropy: int, tier_name: str, split: str, doc: int) -> list[BuiltPage]:
    code = canon.TIERS[tier_name]
    doc_seed = canon.stream_u64(entropy, code, doc, 0, canon.PURPOSE["layout"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "doc.pdf")
        make_document(doc_seed, path)
        pages = render_document(path)
    return [build_page(entropy, tier_name, split, doc, i, rgb) for i, rgb in enumerate(pages)]


# ---------------------------------------------------------------------------
# Release writing
# ---------------------------------------------------------------------------
def build_env() -> dict:
    import fitz
    import PIL
    import scipy
    from PIL import features

    return {
        "python": platform.python_version(),
        "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "pillow_zlib": str(features.version("zlib")),
        "pillow_libjpeg_turbo": str(features.version_feature("libjpeg_turbo")),
        "pymupdf": fitz.VersionBind,
        "mupdf": fitz.VersionFitz,
    }


def _write(root: str, rel: str, data: bytes) -> None:
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


RELEASE_SPLITS = ("val", "test-dev", "test")   # train is regenerable on demand


def _job(args: tuple) -> tuple:
    """Pool worker: build one document. Top-level so it pickles."""
    tier_name, split, doc, entropy = args
    return tier_name, split, doc, build_doc(entropy, tier_name, split, doc)


def _jobs(tiers, splits, docs, secret_hex, limit_docs):
    for tier_name in tiers:
        for split in splits:
            entropy = secret_entropy(secret_hex) if split == "test" else PUBLIC_ENTROPY
            rng_docs = doc_range(split, docs)
            if limit_docs is not None:
                rng_docs = rng_docs[:limit_docs]
            for doc in rng_docs:
                yield tier_name, split, doc, entropy


_TALLY_KEYS = ("docs", "pages", "fragments", "blank_fragments", "missing", "extra_tear_attempts")


def _tally(c: dict, pages: list[BuiltPage]) -> None:
    c["docs"] += 1
    for built in pages:
        a = built.answer
        c["pages"] += 1
        c["fragments"] += len(a["fragments"])
        c["blank_fragments"] += sum(f["ink_frac"] < BLANK_INK for f in a["fragments"].values())
        c["missing"] += len(a["missing"])
        c["extra_tear_attempts"] += a["tear_attempts"] - 1


def _finish_tally(c: dict) -> dict:
    out = dict(c)
    out["blank_fraction"] = round(c["blank_fragments"] / c["fragments"], 4) if c["fragments"] else 0.0
    return out


def _write_pages(out: str, tier_name: str, split: str, pages: list[BuiltPage],
                 test_answers: list[str]) -> None:
    for built in pages:
        base = f"puzzles/{tier_name}/{split}/{built.page_id}"
        _write(out, f"{base}/puzzle.json", canon.dumps(built.puzzle).encode())
        for name, data in sorted(built.pngs.items()):
            _write(out, f"{base}/{name}", data)
        answer_bytes = canon.dumps(built.answer).encode()
        if split == "test":                 # held out: commit, never write
            test_answers.append(canon.sha256_bytes(answer_bytes))
        else:
            _write(out, f"answers/{tier_name}/{split}/{built.page_id}.json", answer_bytes)


def write_release(out: str, tiers: list[str], docs: dict[str, int],
                  splits: list[str], secret_hex: str | None = None,
                  limit_docs: int | None = None, workers: int = 1, progress=None) -> dict:
    """Build the requested tiers/splits into `out`; return benchmark.json dict.

    Output bytes do not depend on `workers`: pages are pure functions of
    their key and results are consumed in job order.
    """
    if "test" in splits and not secret_hex:
        raise ValueError(f"building the test split requires {SECRET_ENV}")
    if os.path.isdir(out) and os.listdir(out):
        # Leftover pages would be hashed into SHA256SUMS but not counted in
        # benchmark.json, so the release would disagree with itself.
        raise ValueError(f"output directory {out!r} is not empty")
    jobs = list(_jobs(tiers, splits, docs, secret_hex, limit_docs))
    counts: dict = {}
    test_answers: list[str] = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_job, jobs, chunksize=1)
    else:
        pool, results = None, map(_job, jobs)
    try:
        for tier_name, split, doc, pages in results:
            _write_pages(out, tier_name, split, pages, test_answers)
            _tally(counts.setdefault(tier_name, {}).setdefault(split, dict.fromkeys(_TALLY_KEYS, 0)),
                   pages)
            if progress:
                progress(tier_name, split, doc)
    finally:
        if pool:
            pool.shutdown()
    spec = {
        "name": BENCH_NAME,
        "version": BENCH_VERSION,
        "generator_version": __version__,
        "schemas": {"puzzle": PUZZLE_SCHEMA, "answer": ANSWER_SCHEMA},
        "public_entropy": PUBLIC_ENTROPY,
        "page_px": list(PAGE_PX),
        "tiers": {t: _tier_json(TIER_SPECS[t]) for t in tiers},
        "splits": {t: {s: _finish_tally(c) for s, c in per.items()} for t, per in counts.items()},
        "build_env": build_env(),
    }
    if "test" in splits:
        prefix = f"{BENCH_NAME}/{BENCH_VERSION}/test/"
        spec["test_commitment"] = {
            # Reveal the secret when the version is retired; anyone can then
            # check it against this digest.
            "secret_sha256": canon.sha256_bytes((prefix + secret_hex.strip().lower()).encode()),
            "answers_sha256": canon.sha256_bytes("".join(sorted(test_answers)).encode()),
        }
    _write(out, "benchmark.json", canon.dumps(spec).encode())
    write_checksums(out)
    return spec


def _tier_json(t: TierSpec) -> dict:
    d = asdict(t)
    d["n_range"] = list(t.n_range)
    d["corruption"]["erosion_px"] = list(t.corruption.erosion_px)
    return d


def _release_files(root: str) -> list[str]:
    rels = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/")
            if rel not in ("SHA256SUMS", "CONTENT.sha256"):
                rels.append(rel)
    return sorted(rels, key=lambda r: r.encode())      # byte order, LC_ALL=C


def write_checksums(root: str) -> None:
    byte_lines, content_lines = [], []
    for rel in _release_files(root):
        with open(os.path.join(root, *rel.split("/")), "rb") as fh:
            data = fh.read()
        byte_lines.append(f"{canon.sha256_bytes(data)}  {rel}\n")
        if rel.endswith(".png"):
            digest = canon.content_digest_png(data)
        elif rel.endswith(".json"):
            digest = canon.content_digest_json(data)
        else:
            digest = canon.sha256_bytes(data)
        content_lines.append(f"{digest}  {rel}\n")
    _write(root, "SHA256SUMS", "".join(byte_lines).encode())
    _write(root, "CONTENT.sha256", "".join(content_lines).encode())


def verify_checksums(root: str) -> list[str]:
    """Return relpaths whose bytes no longer match SHA256SUMS (empty = intact)."""
    bad = []
    with open(os.path.join(root, "SHA256SUMS"), encoding="ascii") as fh:
        for line in fh:
            digest, rel = line.rstrip("\n").split("  ", 1)
            path = os.path.join(root, *rel.split("/"))
            if not os.path.exists(path):
                bad.append(rel)
                continue
            with open(path, "rb") as f2:
                if canon.sha256_bytes(f2.read()) != digest:
                    bad.append(rel)
    return bad
