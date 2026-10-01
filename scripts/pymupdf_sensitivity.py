"""How much does the PyMuPDF version change dm-bench? (issue #10)

Measures, on the public splits only, what a different PyMuPDF version does to
rendered pixels, fragment geometry, fragment pixels and baseline scores. It
never changes the pinned benchmark: the canonical release stays the one built
with the versions in requirements-bench.txt.

Two parts:

  renderer only  PDFs are made once in the canonical environment, then
                 rendered and built under each PyMuPDF version
  full rebuild   `python -m src.bench build` under each version (PDF writing
                 and rendering both change)

Steps, each run with the Python of the environment being measured:

  python scripts/pymupdf_sensitivity.py make-pdfs PDFS             (canonical env)
  python scripts/pymupdf_sensitivity.py build-from-pdfs PDFS REL
  python scripts/pymupdf_sensitivity.py render-pages PDFS RENDERS
  python scripts/pymupdf_sensitivity.py compare --releases A B [--renders A B]
      [--results A B] [--pdfs A B] [--solutions A B] [--overlap-only]
      [--label NAME] --out SUMMARY

Release A and B are then solved and scored with the normal CLI
(`python -m src.bench solve` / `eval`). No secret is needed or read.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.bench import build, canon, docgen, evaluate  # noqa: E402

SPLITS = ("val", "test-dev")
INDEX = "index.json"
SPEC = "benchmark.json"
SCORE_KEYS = ("hit1", "hit5", "mrr", "perfect", "direct_acc", "neighbor_acc", "adj_f1")


# ---------------------------------------------------------------------------
# Source PDFs
# ---------------------------------------------------------------------------
def documents():
    """(tier, split, doc, doc_seed) for every public document, as build_doc keys them."""
    for tier in build.TIER_SPECS:
        for split in SPLITS:
            for doc in build.doc_range(split, build.DEFAULT_DOCS):
                seed = canon.stream_u64(build.PUBLIC_ENTROPY, canon.TIERS[tier], doc, 0,
                                        canon.PURPOSE["layout"])
                yield tier, split, doc, seed


def make_pdfs(out: str) -> dict:
    """Write every public source PDF plus an index keyed by doc_seed."""
    index = {}
    for tier, split, doc, seed in documents():
        rel = f"{tier}/{split}/{doc}.pdf"
        if str(seed) in index:
            raise RuntimeError(f"doc_seed {seed} is not unique ({index[str(seed)]}, {rel})")
        os.makedirs(os.path.join(out, tier, split), exist_ok=True)
        docgen.make_document(seed, os.path.join(out, rel))
        index[str(seed)] = rel
    meta = {"documents": index, "build_env": build.build_env()}
    with open(os.path.join(out, INDEX), "w") as fh:
        fh.write(canon.dumps(meta))
    return meta


def load_index(pdfs: str) -> dict:
    with open(os.path.join(pdfs, INDEX)) as fh:
        return json.load(fh)["documents"]


@contextlib.contextmanager
def pdfs_as_source(pdfs: str):
    """Make build_doc copy the saved PDF for a seed instead of writing a new one."""
    index = load_index(pdfs)

    def copy_pdf(seed: int, path: str) -> None:
        rel = index.get(str(seed))
        if rel is None:
            raise KeyError(f"no saved PDF for doc_seed {seed}")
        shutil.copyfile(os.path.join(pdfs, rel), path)

    original = build.make_document
    build.make_document = copy_pdf
    try:
        yield index
    finally:
        build.make_document = original


def build_from_pdfs(pdfs: str, out: str) -> dict:
    """Normal release writer, sequential so the in-process patch applies."""
    with pdfs_as_source(pdfs):
        return build.write_release(out, list(build.TIER_SPECS), dict(build.DEFAULT_DOCS),
                                   list(SPLITS), workers=1)


def render_pages(pdfs: str, out: str) -> int:
    """Lossless page renders (.npy) for page-level pixel comparison."""
    n = 0
    for rel in load_index(pdfs).values():
        stem = os.path.join(out, rel[:-len(".pdf")])
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        for i, page in enumerate(docgen.render_document(os.path.join(pdfs, rel))):
            np.save(f"{stem}_p{i}.npy", page)
            n += 1
    meta = {"build_env": build.build_env(), "pages": n}
    with open(os.path.join(out, INDEX), "w") as fh:
        fh.write(canon.dumps(meta))
    return n


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _npy_files(root: str) -> list[str]:
    found = []
    for dirpath, _, files in os.walk(root):
        found += [os.path.relpath(os.path.join(dirpath, f), root) for f in files if f.endswith(".npy")]
    return sorted(found)


def compare_renders(a: str, b: str) -> dict:
    files = _npy_files(a)
    if files != _npy_files(b):
        raise ValueError("render folders hold different pages")
    identical = lighter_pages = 0
    changed_frac, mean_abs, max_abs = [], [], 0
    n_changed = n_lighter = 0
    sum_signed = sum_abs = 0.0
    for rel in files:
        pa, pb = np.load(os.path.join(a, rel)), np.load(os.path.join(b, rel))
        if pa.shape != pb.shape:
            raise ValueError(f"{rel}: shape {pa.shape} vs {pb.shape}")
        signed = pb.astype(np.int16) - pa.astype(np.int16)        # > 0: lighter in B
        diff = np.abs(signed)
        identical += int(not diff.any())
        changed = diff.any(axis=2)
        changed_frac.append(float(changed.mean()))
        mean_abs.append(float(diff.mean()))
        max_abs = max(max_abs, int(diff.max()))
        per_pixel = signed.mean(axis=2)[changed]                   # channel mean, changed pixels
        n_changed += per_pixel.size
        n_lighter += int((per_pixel > 0).sum())
        sum_signed += float(per_pixel.sum())
        sum_abs += float(np.abs(per_pixel).sum())
        lighter_pages += int(signed.mean() > 0)
    return {
        "build_env": {"a": _read_json(os.path.join(a, INDEX))["build_env"],
                      "b": _read_json(os.path.join(b, INDEX))["build_env"]},
        "pages": len(files),
        "pages_identical": identical,
        "changed_pixel_fraction": {"mean": float(np.mean(changed_frac)), "max": float(np.max(changed_frac))},
        "mean_abs_diff": {"mean": float(np.mean(mean_abs)), "max": float(np.max(mean_abs))},
        "max_abs_diff": max_abs,
        "changed_pixels": n_changed,
        "changed_pixels_lighter_in_b": n_lighter,
        "changed_pixel_mean_signed_diff": sum_signed / n_changed if n_changed else 0.0,
        "changed_pixel_mean_abs_diff": sum_abs / n_changed if n_changed else 0.0,
        "pages_lighter_in_b": lighter_pages,
    }


def _read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def _geometry(answer: dict) -> str:
    """Canonical answer with only the pixel-derived ink_frac removed."""
    stripped = dict(answer)
    stripped["fragments"] = {fid: {k: v for k, v in f.items() if k != "ink_frac"}
                             for fid, f in answer["fragments"].items()}
    return canon.dumps(stripped)


def _digests(release: str, name: str) -> dict[str, str]:
    """rel path -> digest from SHA256SUMS (raw bytes) or CONTENT.sha256 (decoded)."""
    out = {}
    with open(os.path.join(release, name)) as fh:
        for line in fh:
            digest, rel = line.rstrip("\n").split("  ", 1)
            out[rel] = digest
    return out


def _blank_counts(a: str, b: str) -> dict:
    """blank_fragments per tier/split from both benchmark.json files."""
    sa = _read_json(os.path.join(a, SPEC))["splits"]
    sb = _read_json(os.path.join(b, SPEC))["splits"]
    out = {f"{t}/{s}": {"a": sa[t][s]["blank_fragments"], "b": sb[t][s]["blank_fragments"]}
           for t in sorted(sa) for s in sorted(sa[t])}
    out["total"] = {"a": sum(v["a"] for v in out.values()), "b": sum(v["b"] for v in out.values())}
    return out


def compare_releases(a: str, b: str, overlap_only: bool = False) -> dict:
    """Content (decoded) and raw-byte identity of two releases.

    overlap_only compares just the files both hold, for a subset build
    against a full release. Release-level files are then not compared.
    """
    da, db = _digests(a, "CONTENT.sha256"), _digests(b, "CONTENT.sha256")
    ba, bb = _digests(a, "SHA256SUMS"), _digests(b, "SHA256SUMS")
    keys = sorted(k for k in da if k != SPEC)
    if overlap_only:
        missing = [k for k in keys if k not in db]
        if missing or not keys:                         # A must be a subset of B
            raise ValueError(f"{len(missing)} file(s) of {a} are not in {b}")
    elif keys != sorted(k for k in db if k != SPEC):
        raise ValueError("releases hold different files")
    answers = [k for k in keys if k.startswith("answers/")]
    pngs = [k for k in keys if k.endswith(".png")]
    puzzles = [k for k in keys if k.endswith("puzzle.json")]
    geometry_equal, ink_delta = 0, 0.0
    ink_changed = ink_down = to_blank = to_ink = 0
    for rel in answers:
        aa, ab = _read_json(os.path.join(a, rel)), _read_json(os.path.join(b, rel))
        geometry_equal += _geometry(aa) == _geometry(ab)
        for fid, frag in aa["fragments"].items():
            ia, ib = frag["ink_frac"], ab["fragments"][fid]["ink_frac"]
            ink_delta = max(ink_delta, abs(ia - ib))
            ink_changed += ia != ib
            ink_down += ib < ia
            to_blank += ia >= build.BLANK_INK > ib
            to_ink += ib >= build.BLANK_INK > ia
    return {
        "pages": len(answers),
        "geometry_identical": geometry_equal,
        "answers_identical": sum(da[k] == db[k] for k in answers),
        "ink_frac_max_abs_delta": round(ink_delta, 4),
        "ink_frac_changed": ink_changed,
        "ink_frac_lower_in_b": ink_down,
        "fragments_became_blank": to_blank,
        "fragments_became_inked": to_ink,
        "puzzle_json_identical": sum(da[k] == db[k] for k in puzzles),
        "puzzles": len(puzzles),
        "fragments": len(pngs),
        "fragment_pixels_identical": sum(da[k] == db[k] for k in pngs),
        "files": len(keys),
        "files_byte_identical": sum(ba[k] == bb[k] for k in keys),
        "benchmark_json_byte_identical": None if overlap_only else ba[SPEC] == bb[SPEC],
        "blank_fragments": None if overlap_only else _blank_counts(a, b),
        "build_env": {"a": _read_json(os.path.join(a, SPEC))["build_env"],
                      "b": _read_json(os.path.join(b, SPEC))["build_env"]},
    }


def compare_pdfs(a: str, b: str) -> dict:
    """Byte identity of two sets of source PDFs made by make-pdfs."""
    ia, ib = load_index(a), load_index(b)
    if ia != ib:
        raise ValueError("PDF folders index different documents")
    import fitz

    same = same_objects = same_meta = 0
    for rel in ia.values():
        pa, pb = os.path.join(a, rel), os.path.join(b, rel)
        with open(pa, "rb") as fa, open(pb, "rb") as fb:
            same += fa.read() == fb.read()
        same_objects += _pdf_objects(fitz, pa) == _pdf_objects(fitz, pb)
        same_meta += _pdf_metadata(fitz, pa) == _pdf_metadata(fitz, pb)
    return {"build_env": {"a": _read_json(os.path.join(a, INDEX))["build_env"],
                          "b": _read_json(os.path.join(b, INDEX))["build_env"]},
            "documents": len(ia), "pdf_bytes_identical": same,
            "pdf_objects_identical_except_info": same_objects,
            "pdf_metadata_identical_except_producer": same_meta}


def _pdf_metadata(fitz, path: str) -> dict:
    """Document Info metadata (dates, title, format, ...) without the producer."""
    doc = fitz.open(path)
    try:
        return {k: v for k, v in doc.metadata.items() if k != "producer"}
    finally:
        doc.close()


def _strip_info(text: str) -> str:
    """Drop an inline Info dictionary (no nested dicts) from one object's text."""
    return re.sub(r"/Info<<[^<>]*>>", "", text)


def _pdf_objects(fitz, path: str) -> list:
    """Every PDF object (dictionary and raw stream) with the Info metadata removed.

    MuPDF writes the Info dictionary either as a separate object or inline in
    the catalog (`/Info<</Producer(MuPDF 1.28.2)>>`). Both forms are dropped
    here, and _pdf_metadata compares the Info fields other than the producer.
    """
    doc = fitz.open(path)
    try:
        objects = []
        for xref in range(1, doc.xref_length()):
            text = _strip_info(doc.xref_object(xref, compressed=True))
            if "/Producer" in text:
                continue
            raw = doc.xref_stream_raw(xref) if doc.xref_is_stream(xref) else None
            objects.append((text, raw))
        return objects
    finally:
        doc.close()


def compare_pages(release_a: str, sols_a: str, release_b: str, sols_b: str) -> dict:
    """Per-page paired score changes (same pages, deterministic solver)."""
    out = {}
    for tier in build.TIER_SPECS:
        for split in SPLITS:
            for release, sols in ((release_a, sols_a), (release_b, sols_b)):
                pages = os.listdir(os.path.join(release, "puzzles", tier, split))
                bad = [p for p in pages
                       if evaluate.load_solution(os.path.join(sols, tier, split, f"{p}.json")) is None]
                if bad:
                    raise ValueError(f"{sols}: {len(bad)} {tier}/{split} solution(s) missing or unreadable")
            pa, _ = evaluate.evaluate_split(release_a, sols_a, tier, split)
            pb, _ = evaluate.evaluate_split(release_b, sols_b, tier, split)
            if [p.page_id for p in pa] != [p.page_id for p in pb]:
                raise ValueError(f"{tier}/{split}: different pages")
            for key in SCORE_KEYS:
                row = out.setdefault(key, {"pages": 0, "changed": 0, "up": 0, "down": 0, "max_abs": 0.0})
                for sa, sb in zip(pa, pb):
                    va, vb = getattr(sa, key), getattr(sb, key)
                    if (va is None) != (vb is None):
                        raise ValueError(f"{sa.page_id}: {key} scored on one side only")
                    if va is None:
                        continue
                    row["pages"] += 1
                    if abs(vb - va) > 1e-12:
                        row["changed"] += 1
                        row["up" if vb > va else "down"] += 1
                        row["max_abs"] = max(row["max_abs"], abs(vb - va))
    return out


def _metric(entry):
    return entry["mean"] if isinstance(entry, dict) else entry


def compare_results(a: str, b: str) -> dict:
    ra, rb = _read_json(a), _read_json(b)
    out = {}
    for tier in sorted(ra["scores"]):
        for split in sorted(ra["scores"][tier]):
            sa, sb = ra["scores"][tier][split], rb["scores"][tier][split]
            row = {}
            for key in SCORE_KEYS:
                if key not in sa or key not in sb:      # Hit@k absent without candidates
                    continue
                va, vb = _metric(sa[key]), _metric(sb[key])
                row[key] = {"a": va, "b": vb, "delta": round(vb - va, 4)}
                if isinstance(sa[key], dict):
                    row[key]["ci95_a"], row[key]["ci95_b"] = sa[key]["ci95"], sb[key]["ci95"]
                    half = (sa[key]["ci95"][1] - sa[key]["ci95"][0]) / 2
                    if half:
                        ratio = abs(vb - va) / half
                    elif vb == va:
                        ratio = 0.0
                    else:
                        ratio = None                        # changed against a zero-width interval
                    row[key]["delta_over_ci_half_width_a"] = ratio
                    row[key]["changed_vs_zero_width_ci"] = ratio is None
            out[f"{tier}/{split}"] = row
    return {"release_sha256": {"a": ra["release_sha256"], "b": rb["release_sha256"]},
            "solvers": {"a": ra["solvers"], "b": rb["solvers"]},
            "eval_version": {"a": ra["eval_version"], "b": rb["eval_version"]},
            "scores": out}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("make-pdfs")
    s.add_argument("out")
    s = sub.add_parser("build-from-pdfs")
    s.add_argument("pdfs")
    s.add_argument("out")
    s = sub.add_parser("render-pages")
    s.add_argument("pdfs")
    s.add_argument("out")
    s = sub.add_parser("compare")
    s.add_argument("--releases", nargs=2, required=True)
    s.add_argument("--renders", nargs=2)
    s.add_argument("--results", nargs=2)
    s.add_argument("--pdfs", nargs=2)
    s.add_argument("--solutions", nargs=2, help="solution dirs for --releases, enables per-page pairs")
    s.add_argument("--overlap-only", action="store_true")
    s.add_argument("--out", required=True)
    s.add_argument("--label", help="store under this key, keeping other labels already in --out")
    args = p.parse_args(argv)

    if args.cmd == "make-pdfs":
        print(f"wrote {len(make_pdfs(args.out)['documents'])} PDFs to {args.out}")
    elif args.cmd == "build-from-pdfs":
        build_from_pdfs(args.pdfs, args.out)
        print(f"wrote {args.out}")
    elif args.cmd == "render-pages":
        print(f"rendered {render_pages(args.pdfs, args.out)} pages to {args.out}")
    else:
        merged = _read_json(args.out) if args.label and os.path.exists(args.out) else {}
        if "releases" in merged:                         # checked before any work
            raise ValueError(f"{args.out} holds an unlabeled summary, refusing to merge")
        summary = {"releases": compare_releases(*args.releases, overlap_only=args.overlap_only)}
        if args.renders:
            summary["renders"] = compare_renders(*args.renders)
        if args.results:
            summary["results"] = compare_results(*args.results)
        if args.pdfs:
            summary["pdfs"] = compare_pdfs(*args.pdfs)
        if args.solutions:
            summary["pages"] = compare_pages(args.releases[0], args.solutions[0],
                                             args.releases[1], args.solutions[1])
        if args.label:
            merged[args.label] = summary
            summary = merged
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(canon.dumps(summary))
        print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
