"""UI wiring smoke test: build the Blocks and drive the handlers directly."""
import os

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

from ui import build_ui, handlers  # noqa: E402


def _noop(*args, **kwargs):
    return None


def test_build_ui_constructs():
    demo = build_ui()
    assert isinstance(demo, gr.Blocks)


def test_sample_pdf_is_bundled():
    assert os.path.isfile(handlers.load_sample())


def test_generate_then_inspect_pages():
    view, source, partition, thumbs, zip_path = handlers.generate(
        handlers.load_sample(), 72, 12, 20.0, 60.0, False, 5, progress=_noop,
    )
    assert os.path.isfile(zip_path) and zip_path.endswith("_dataset.zip")
    assert source.shape == partition.shape and source.dtype == np.uint8
    assert 0 < len(thumbs) <= handlers.MAX_THUMBNAILS
    assert len(view.reports) == len(view.previews) >= 2
    assert all(r.is_partition for r in view.reports)

    summary, dd, adj, rows, meta, pairs, manifest_json, zip_rows = handlers.after_generate(view)
    assert "partition verified" in summary and f"{len(view.reports)}/{len(view.reports)}" in summary
    assert dd["choices"][0][1] == 0 and dd["value"] == 0
    assert len(rows) == view.reports[0].n_pieces
    assert '"schema_version": "1.1"' in manifest_json
    assert {r[0] for r in zip_rows} >= {"manifest.json", "README.txt"}

    page2 = handlers.show_page(view, 1)
    expected_rows = [[row[h] for h in handlers.GT_HEADERS] for row in view.reports[1].rows]
    assert len(page2) == 7 and page2[4] == expected_rows

    cleared = handlers.clear_all()
    assert cleared[0] is None and cleared[1] is None
    assert not os.path.exists(zip_path)


def test_generate_without_pdf_errors():
    with pytest.raises(gr.Error):
        handlers.generate(None, 72, 12, 20.0, 60.0, False, 0, progress=_noop)


def test_dpi_note_matches_config():
    from src.config import a4_pixels

    w, h = a4_pixels(150)
    assert f"{w} × {h}" in handlers.dpi_note(150)
