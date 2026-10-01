"""scripts/pymupdf_sensitivity.py: the from-PDF path must not change the build.

The experiment swaps how source PDFs reach build_doc. If that swap alone
changed any byte, every difference it reports later would be suspect.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

from src.bench import build, canon, docgen

_SPEC = importlib.util.spec_from_file_location(
    "pymupdf_sensitivity",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "pymupdf_sensitivity.py"))
sens = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sens)


def test_documents_match_build_doc_keys():
    docs = list(sens.documents())
    assert len({seed for *_, seed in docs}) == len(docs)              # seeds unique
    tier, split, doc, seed = docs[0]
    assert seed == canon.stream_u64(build.PUBLIC_ENTROPY, canon.TIERS[tier], doc, 0,
                                    canon.PURPOSE["layout"])
    assert {s for _, s, _, _ in docs} == {"val", "test-dev"}           # public only


def test_from_pdf_path_reproduces_build_doc(tmp_path):
    tier, split = "hard", "val"
    doc = build.doc_range(split, build.DEFAULT_DOCS).start
    seed = canon.stream_u64(build.PUBLIC_ENTROPY, canon.TIERS[tier], doc, 0, canon.PURPOSE["layout"])
    pdfs = tmp_path / "pdfs"
    (pdfs / tier / split).mkdir(parents=True)
    docgen.make_document(seed, str(pdfs / tier / split / f"{doc}.pdf"))
    (pdfs / sens.INDEX).write_text(json.dumps({"documents": {str(seed): f"{tier}/{split}/{doc}.pdf"}}))

    expected = build.build_doc(build.PUBLIC_ENTROPY, tier, split, doc)
    with sens.pdfs_as_source(str(pdfs)):
        got = build.build_doc(build.PUBLIC_ENTROPY, tier, split, doc)
    assert build.make_document is docgen.make_document                 # patch undone
    assert [p.page_id for p in got] == [p.page_id for p in expected]
    assert [canon.dumps(p.answer) for p in got] == [canon.dumps(p.answer) for p in expected]
    assert [p.pngs for p in got] == [p.pngs for p in expected]


def test_from_pdf_path_really_reads_the_saved_pdf(tmp_path):
    """Guards against a vacuous pass: a different saved PDF must change the build."""
    tier, split = "easy", "val"
    doc = build.doc_range(split, build.DEFAULT_DOCS).start
    seed = canon.stream_u64(build.PUBLIC_ENTROPY, canon.TIERS[tier], doc, 0, canon.PURPOSE["layout"])
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    docgen.make_document(seed + 1, str(pdfs / "other.pdf"))          # not this doc's PDF
    (pdfs / sens.INDEX).write_text(json.dumps({"documents": {str(seed): "other.pdf"}}))

    expected = build.build_doc(build.PUBLIC_ENTROPY, tier, split, doc)
    with sens.pdfs_as_source(str(pdfs)):
        got = build.build_doc(build.PUBLIC_ENTROPY, tier, split, doc)
    assert [p.pngs for p in got] != [p.pngs for p in expected]


def test_missing_pdf_fails_loudly(tmp_path):
    (tmp_path / sens.INDEX).write_text(json.dumps({"documents": {}}))
    with sens.pdfs_as_source(str(tmp_path)):
        with pytest.raises(KeyError, match="no saved PDF"):
            build.build_doc(build.PUBLIC_ENTROPY, "easy", "val",
                            build.doc_range("val", build.DEFAULT_DOCS).start)
    assert build.make_document is docgen.make_document


def test_geometry_ignores_only_ink_frac():
    answer = {"page_id": "x", "fragments": {"f000": {"affine": [[1, 0, 0], [0, 1, 0]], "ink_frac": 0.5}}}
    other = json.loads(json.dumps(answer))
    other["fragments"]["f000"]["ink_frac"] = 0.25
    assert sens._geometry(answer) == sens._geometry(other)
    other["fragments"]["f000"]["affine"][0][2] = 1
    assert sens._geometry(answer) != sens._geometry(other)


def _renders(root, pages):
    root.mkdir()
    for name, arr in pages.items():
        np.save(root / f"{name}.npy", arr)
    (root / sens.INDEX).write_text(json.dumps({"build_env": {}, "pages": len(pages)}))


def test_compare_renders_reports_size_and_direction(tmp_path):
    a = np.full((4, 5, 3), 100, np.uint8)
    b = a.copy()
    b[0, 0] = (130, 100, 100)                           # lighter: channel mean +10
    b[0, 1] = (106, 106, 106)                           # lighter: +6
    b[1, 1] = (85, 100, 100)                            # darker: channel mean -5
    _renders(tmp_path / "a", {"p": a, "q": a})
    _renders(tmp_path / "b", {"p": b, "q": a})
    r = sens.compare_renders(str(tmp_path / "a"), str(tmp_path / "b"))
    assert (r["pages"], r["pages_identical"], r["max_abs_diff"]) == (2, 1, 30)
    assert (r["changed_pixels"], r["changed_pixels_lighter_in_b"], r["pages_lighter_in_b"]) == (3, 2, 1)
    assert r["changed_pixel_mean_signed_diff"] == (10 + 6 - 5) / 3
    assert r["changed_pixel_mean_abs_diff"] == (10 + 6 + 5) / 3
    assert r["changed_pixel_fraction"]["max"] == 3 / 20
    swapped = sens.compare_renders(str(tmp_path / "b"), str(tmp_path / "a"))
    assert swapped["changed_pixels_lighter_in_b"] == 1 and swapped["pages_lighter_in_b"] == 0


def test_pdf_objects_ignore_only_the_producer_stamp(tmp_path):
    import fitz

    path = str(tmp_path / "d.pdf")
    docgen.make_document(7, path)
    base = sens._pdf_objects(fitz, path)
    assert base == sens._pdf_objects(fitz, path)
    doc = fitz.open(path)
    doc.set_metadata({"producer": "something else"})
    doc.save(str(tmp_path / "e.pdf"), garbage=4, deflate=True, no_new_id=True)
    doc.close()
    assert sens._pdf_objects(fitz, str(tmp_path / "e.pdf")) == base
    assert sens._pdf_metadata(fitz, str(tmp_path / "e.pdf")) == sens._pdf_metadata(fitz, path)
    doc = fitz.open(path)
    doc.set_metadata({"producer": "dm-bench docgen", "creationDate": "D:20250101000000Z",
                      "modDate": "D:20240101000000Z"})
    doc.save(str(tmp_path / "f.pdf"), garbage=4, deflate=True, no_new_id=True)
    doc.close()
    assert sens._pdf_metadata(fitz, str(tmp_path / "f.pdf")) != sens._pdf_metadata(fitz, path)
    other = str(tmp_path / "o.pdf")
    docgen.make_document(8, other)
    assert sens._pdf_objects(fitz, other) != base


def test_strip_info_removes_only_a_flat_info_dict():
    catalog = "<</Type/Catalog/Pages 2 0 R/Info<</Producer(MuPDF 1.28.2)>>>>"
    assert sens._strip_info(catalog) == "<</Type/Catalog/Pages 2 0 R>>"
    nested = "<</Info<</A<</B 1>>>>>>"                    # nested dicts are left alone
    assert sens._strip_info(nested) == nested


def test_label_merge_refuses_an_unlabeled_summary(tmp_path):
    out = tmp_path / "s.json"
    out.write_text(json.dumps({"releases": {}}))
    with pytest.raises(ValueError, match="unlabeled"):
        sens.main(["compare", "--releases", "x", "y", "--label", "z", "--out", str(out)])


@pytest.fixture(scope="module")
def tiny_release(tmp_path_factory):
    """Every tier, both public splits, one document each, with oracle and random solutions."""
    from src.bench.__main__ import main as bench

    root = tmp_path_factory.mktemp("tiny")
    rel = str(root / "rel")
    build.write_release(rel, list(build.TIER_SPECS), dict(build.DEFAULT_DOCS), list(sens.SPLITS),
                        limit_docs=1)
    for method in ("oracle", "random"):
        assert bench(["solve", "--release", rel, "--out", str(root / method), "--method", method,
                      "--splits", *sens.SPLITS]) == 0
    return root, rel


def _copy_with_blanked_fragment(rel: str, dst: str) -> tuple[str, str]:
    """Copy a release and push one inked fragment's ink_frac below BLANK_INK."""
    import shutil

    shutil.copytree(rel, dst)
    adir = os.path.join(dst, "answers", "easy", "val")
    name = sorted(os.listdir(adir))[0]
    path = os.path.join(adir, name)
    with open(path) as fh:
        answer = json.load(fh)
    fid = next(f for f, v in sorted(answer["fragments"].items()) if v["ink_frac"] >= build.BLANK_INK)
    answer["fragments"][fid]["ink_frac"] = 0.0
    with open(path, "w") as fh:
        fh.write(canon.dumps(answer))
    build.write_checksums(dst)
    return f"answers/easy/val/{name}", fid


def test_compare_releases_counts_ink_and_blank_changes(tiny_release, tmp_path):
    _, rel = tiny_release
    same = sens.compare_releases(rel, rel)
    assert same["files_byte_identical"] == same["files"] and same["benchmark_json_byte_identical"]
    assert same["blank_fragments"]["total"]["a"] == same["blank_fragments"]["total"]["b"]

    changed_file, _ = _copy_with_blanked_fragment(rel, str(tmp_path / "b"))
    r = sens.compare_releases(rel, str(tmp_path / "b"))
    assert r["geometry_identical"] == r["pages"]                   # ink_frac is not geometry
    assert r["answers_identical"] == r["pages"] - 1
    assert r["files_byte_identical"] == r["files"] - 1
    assert (r["ink_frac_changed"], r["ink_frac_lower_in_b"]) == (1, 1)
    assert (r["fragments_became_blank"], r["fragments_became_inked"]) == (1, 0)
    back = sens.compare_releases(str(tmp_path / "b"), rel)
    assert (back["fragments_became_blank"], back["fragments_became_inked"]) == (0, 1)


def test_compare_releases_overlap_requires_a_subset(tiny_release, tmp_path):
    import shutil

    _, rel = tiny_release
    sup = str(tmp_path / "sup")
    shutil.copytree(rel, sup)
    for name in ("CONTENT.sha256", "SHA256SUMS"):
        with open(os.path.join(sup, name), "a") as fh:
            fh.write("0" * 64 + "  extra/file.json\n")
    with pytest.raises(ValueError, match="different files"):
        sens.compare_releases(rel, sup)
    r = sens.compare_releases(rel, sup, overlap_only=True)
    assert r["files_byte_identical"] == r["files"] and r["blank_fragments"] is None
    with pytest.raises(ValueError, match="not in"):
        sens.compare_releases(sup, rel, overlap_only=True)


def test_compare_pages_counts_paired_changes(tiny_release, tmp_path):
    root, rel = tiny_release
    same = sens.compare_pages(rel, str(root / "oracle"), rel, str(root / "oracle"))
    assert all(v["changed"] == 0 for v in same.values())
    worse = sens.compare_pages(rel, str(root / "oracle"), rel, str(root / "random"))
    assert worse["direct_acc"]["changed"] == worse["direct_acc"]["pages"] > 0
    assert worse["direct_acc"]["down"] == worse["direct_acc"]["changed"] and worse["direct_acc"]["up"] == 0
    with pytest.raises(ValueError, match="missing"):
        sens.compare_pages(rel, str(root / "oracle"), rel, str(tmp_path / "nowhere"))


def _copy_solutions(src, dst):
    import shutil

    shutil.copytree(src, dst)
    sdir = os.path.join(dst, "easy", "val")
    return os.path.join(sdir, sorted(os.listdir(sdir))[0])


def test_compare_pages_refuses_one_bad_page(tiny_release, tmp_path):
    root, rel = tiny_release
    broken = _copy_solutions(str(root / "oracle"), str(tmp_path / "broken"))
    with open(broken, "w") as fh:
        fh.write("not json")
    with pytest.raises(ValueError, match="unreadable"):
        sens.compare_pages(rel, str(root / "oracle"), rel, str(tmp_path / "broken"))


def test_compare_pages_refuses_one_sided_hit_at_k(tiny_release, tmp_path):
    root, rel = tiny_release
    sol_path = _copy_solutions(str(root / "oracle"), str(tmp_path / "cands"))
    with open(sol_path) as fh:
        sol = json.load(fh)
    sol["candidates"] = {}                                # Hit@k now scored on this side only
    with open(sol_path, "w") as fh:
        json.dump(sol, fh)
    with pytest.raises(ValueError, match="one side only"):
        sens.compare_pages(rel, str(root / "oracle"), rel, str(tmp_path / "cands"))


def test_compare_pdfs_counts_bytes_objects_and_metadata(tmp_path):
    import shutil

    import fitz

    a = tmp_path / "a"
    a.mkdir()
    index = {}
    for seed in (3, 4):
        docgen.make_document(seed, str(a / f"{seed}.pdf"))
        index[str(seed)] = f"{seed}.pdf"
    (a / sens.INDEX).write_text(json.dumps({"documents": index, "build_env": {}}))
    b = tmp_path / "b"
    shutil.copytree(a, b)
    doc = fitz.open(str(a / "4.pdf"))
    doc.set_metadata({"producer": "other", "creationDate": "D:20240101000000Z",
                      "modDate": "D:20240101000000Z"})
    doc.save(str(b / "4.pdf"), garbage=4, deflate=True, no_new_id=True)
    doc.close()
    r = sens.compare_pdfs(str(a), str(b))
    assert (r["documents"], r["pdf_bytes_identical"]) == (2, 1)
    assert r["pdf_objects_identical_except_info"] == 2
    assert r["pdf_metadata_identical_except_producer"] == 2


def test_compare_results_ratio_and_zero_width_interval(tmp_path):
    def results(direct, perfect, ci_perfect):
        return {"release_sha256": "x", "solvers": ["s@1"], "eval_version": "1.0",
                "scores": {"easy": {"val": {"direct_acc": {"mean": direct, "ci95": [0.4, 0.6]},
                                            "perfect": {"mean": perfect, "ci95": ci_perfect},
                                            "hit1": 0.9}}}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(results(0.5, 0.0, [0.0, 0.0])))
    b.write_text(json.dumps(results(0.55, 0.1, [0.0, 0.2])))
    row = sens.compare_results(str(a), str(b))["scores"]["easy/val"]
    assert abs(row["direct_acc"]["delta_over_ci_half_width_a"] - 0.5) < 1e-9
    assert row["perfect"]["delta_over_ci_half_width_a"] is None and row["perfect"]["changed_vs_zero_width_ci"]
    assert row["hit1"]["delta"] == 0 and "ci95_a" not in row["hit1"]
