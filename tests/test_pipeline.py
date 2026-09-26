"""End-to-end: generate_dataset determinism, manifest provenance, ZIP round-trip."""
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from src import inspection
from src.packager import REASSEMBLY_SNIPPET
from src.pdf_loader import load_pdf_pages
from src.pipeline import PartitionError, generate_dataset
from src.provenance import page_seed, sha256_file

PARAMS = dict(dpi=72, n_pieces=10, noise_strength=20.0, noise_scale=60.0, lossy=False)


def _run(pdf, seed=3, **overrides):
    kw = {**PARAMS, **overrides}
    return generate_dataset(pdf, source_name="small.pdf", master_seed=seed, **kw)


def _strip_volatile(manifest):
    m = dict(manifest)
    m.pop("created_utc")
    return m


def test_same_seed_same_dataset(small_pdf):
    a, b = _run(small_pdf), _run(small_pdf)
    assert _strip_volatile(a.manifest) == _strip_volatile(b.manifest)
    for pa, pb in zip(a.pages, b.pages):
        assert np.array_equal(pa.labels, pb.labels)
    za, zb = zipfile.ZipFile(io.BytesIO(a.zip_bytes)), zipfile.ZipFile(io.BytesIO(b.zip_bytes))
    for name in za.namelist():
        if name.endswith(".png"):
            assert za.read(name) == zb.read(name), name


def test_different_seed_different_tears(small_pdf):
    a, b = _run(small_pdf, seed=3), _run(small_pdf, seed=4)
    assert not np.array_equal(a.pages[0].labels, b.pages[0].labels)


def test_every_page_verified(small_pdf):
    run = _run(small_pdf)
    assert len(run.reports) == len(run.pages) == 2
    assert all(r.is_partition for r in run.reports)
    assert set(run.timings) == {"render_and_tear", "verify", "package"}
    assert all(t >= 0 for t in run.timings.values())


def test_partition_failure_aborts(small_pdf, monkeypatch):
    bad = {"pieces": 1, "max_overlap": 2, "uncovered_pixels": 0, "is_partition": False}
    monkeypatch.setattr(inspection, "verify_partition", lambda torn: bad)
    with pytest.raises(PartitionError):
        _run(small_pdf)


def test_manifest_provenance_and_v1_fields(small_pdf):
    m = _run(small_pdf, seed=11).manifest
    # 1.0 fields keep names and types.
    for key, typ in [("generator", str), ("created_utc", str), ("source", str),
                     ("dpi", int), ("noise_strength", float), ("noise_scale", float),
                     ("lossy", bool), ("pages", list), ("total_pieces", int)]:
        assert isinstance(m[key], typ), key
    # 1.1 provenance.
    assert m["schema_version"] == "1.1"
    assert m["master_seed"] == 11
    assert m["n_pieces_requested"] == PARAMS["n_pieces"]
    assert m["source_sha256"] == sha256_file(small_pdf)
    assert set(m["environment"]) == {"python", "numpy", "scipy", "pillow", "pymupdf"}
    for i, page in enumerate(m["pages"]):
        assert page["index"] == i
        assert page["seed"] == page_seed(11, i)
        assert set(page["pieces"][0]) == {"file", "x", "y", "w", "h"}
    assert m["total_pieces"] == sum(len(p["pieces"]) for p in m["pages"])


def test_zip_contents_match_manifest(small_pdf):
    run = _run(small_pdf)
    zf = zipfile.ZipFile(io.BytesIO(run.zip_bytes))
    files = {p["file"] for page in run.manifest["pages"] for p in page["pieces"]}
    assert set(zf.namelist()) == files | {"manifest.json", "README.txt"}
    assert json.loads(zf.read("manifest.json")) == run.manifest
    for page in run.manifest["pages"]:
        for p in page["pieces"]:
            with Image.open(io.BytesIO(zf.read(p["file"]))) as im:
                assert im.size == (p["w"], p["h"])


def test_published_reassembly_snippet_is_pixel_exact(small_pdf, tmp_path, monkeypatch):
    """The snippet shipped in README.txt + UI must rebuild the rendered pages."""
    run = _run(small_pdf)
    zipfile.ZipFile(io.BytesIO(run.zip_bytes)).extractall(tmp_path)
    monkeypatch.chdir(tmp_path)
    exec(compile(REASSEMBLY_SNIPPET, "REASSEMBLY_SNIPPET", "exec"), {})
    rendered = load_pdf_pages(small_pdf, PARAMS["dpi"])
    assert len(rendered) == len(run.manifest["pages"])
    for i, original in enumerate(rendered):
        rebuilt = np.asarray(Image.open(tmp_path / f"reassembled_{i:04d}.png").convert("RGB"))
        assert np.array_equal(rebuilt, original), f"page {i}"
    # The fixture's black block makes the black-ink caveat real, not theoretical.
    assert any((p.rgb[p.mask].sum(axis=1) == 0).any() for p in run.pages[0].pieces)
