"""Ground-truth inspection: every number the UI shows is computed here."""
import io
import json
import zipfile

import numpy as np

from src.inspection import (
    MAX_CAPTIONED_PIECES,
    format_json,
    fragment_thumbnails,
    graph_is_connected,
    make_preview,
    manifest_excerpt,
    page_report,
    reassemble,
    render_adjacency,
    render_partition,
    zip_listing,
)
from src.tearing import tear_page


def _page(h=400, w=300, seed=0):
    return np.random.default_rng(seed).integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_reassemble_is_pixel_exact():
    img = _page()
    torn = tear_page(img, 14, seed=4, noise_strength=25, noise_scale=70)
    assert np.array_equal(reassemble(torn), img)


def test_page_report_consistency():
    torn = tear_page(_page(), 16, seed=3, noise_strength=25, noise_scale=80)
    r = page_report(torn, index=2, n_requested=16)
    assert r.is_partition and r.max_overlap == 1 and r.uncovered_pixels == 0
    assert r.n_pieces == len(torn.pieces) == len(r.rows)
    assert sum(row["degree"] for row in r.rows) == 2 * r.n_edges
    assert sum(row["area_px"] for row in r.rows) == torn.width * torn.height
    assert r.multi_component == sum(row["components"] > 1 for row in r.rows)
    assert r.graph_connected
    assert r.rows[0]["file"] == "pieces/page_0003/piece_000.png"
    assert r.seed == 3


def test_multi_component_metric_detects_folded_warp():
    img = _page()
    straight = page_report(tear_page(img, 12, 1, noise_strength=0, noise_scale=60), 0, 12)
    folded = page_report(tear_page(img, 12, 1, noise_strength=80, noise_scale=8), 0, 12)
    assert straight.multi_component == 0      # Voronoi cells are convex
    assert folded.multi_component > 0


def test_graph_is_connected():
    assert graph_is_connected(1, [])
    assert graph_is_connected(3, [(0, 1), (1, 2)])
    assert not graph_is_connected(3, [(0, 1)])


def test_preview_roundtrip_and_renders():
    torn = tear_page(_page(1000, 700), 20, seed=5, noise_strength=25, noise_scale=80)
    pv = make_preview(torn, max_side=300)
    assert max(pv.shape) <= 300
    lab = pv.labels()
    assert lab.shape == pv.shape
    lut = {p.label: k for k, p in enumerate(torn.pieces)}
    expected = np.vectorize(lut.get)(torn.labels[::pv.factor, ::pv.factor])
    assert np.array_equal(lab, expected)
    assert pv.page().shape == pv.shape + (3,)

    n = len(torn.pieces)
    for img in (render_partition(pv, n), render_adjacency(pv, n, torn.adjacency)):
        assert img.dtype == np.uint8 and img.shape == pv.shape + (3,)


def test_renders_skip_captions_for_many_pieces():
    n_req = MAX_CAPTIONED_PIECES + 20
    torn = tear_page(_page(900, 640), n_req, seed=2, noise_strength=10, noise_scale=40)
    pv = make_preview(torn, max_side=400)
    n = len(torn.pieces)
    assert render_partition(pv, n).shape == pv.shape + (3,)
    assert render_adjacency(pv, n, torn.adjacency).shape == pv.shape + (3,)


def test_fragment_thumbnails():
    torn = tear_page(_page(), 9, seed=1, noise_strength=20, noise_scale=60)
    r = page_report(torn, 0, 9)
    thumbs = fragment_thumbnails(make_preview(torn, 200), r.rows, limit=5)
    assert len(thumbs) == min(5, r.n_pieces)
    img, caption = thumbs[0]
    assert img.ndim == 3 and img.shape[2] == 3
    assert caption == f"#0 ({r.rows[0]['x']},{r.rows[0]['y']})"


def test_font_honours_size_with_freetype():
    from PIL import ImageFont

    from src.inspection import _font

    font = _font(17)
    if isinstance(ImageFont.load_default(), ImageFont.FreeTypeFont):
        assert font.size == 17


def test_manifest_excerpt_truncates_but_keeps_order():
    manifest = {
        "generator": "Dataset-Maker",
        "pages": [
            {"index": 0, "adjacency": [[0, 1]] * 20, "pieces": [{"x": 0}] * 10},
            {"index": 1, "adjacency": [], "pieces": []},
        ],
        "total_pieces": 10,
    }
    ex = manifest_excerpt(manifest, max_pieces=3, max_pairs=4)
    assert list(ex) == ["generator", "pages", "total_pieces"]
    first = ex["pages"][0]
    assert first["adjacency"][:4] == [[0, 1]] * 4 and first["adjacency"][4] == "... 16 more"
    assert len(first["pieces"]) == 4 and first["pieces"][-1] == "... 7 more"
    assert ex["pages"][1] == "... 1 more page(s)"
    assert len(manifest["pages"][0]["pieces"]) == 10      # input untouched


def test_format_json_roundtrips():
    obj = {"a": [[0, 1], [2, 3]], "b": {"x": 1, "y": [1, 2]}, "c": [{"f": "p.png", "x": 3}], "d": []}
    text = format_json(obj)
    assert json.loads(text) == obj
    assert "[0, 1]" in text and '{"f": "p.png", "x": 3}' in text


def test_zip_listing_groups_pages():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pieces/page_0001/piece_000.png", b"a" * 10)
        zf.writestr("pieces/page_0001/piece_001.png", b"b" * 5)
        zf.writestr("pieces/page_0002/piece_000.png", b"c")
        zf.writestr("manifest.json", b"{}")
    rows = zip_listing(buf.getvalue())
    assert rows == [
        {"path": "pieces/page_0001/", "files": 2, "bytes": 15},
        {"path": "pieces/page_0002/", "files": 1, "bytes": 1},
        {"path": "manifest.json", "files": 1, "bytes": 2},
    ]
