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
    assert os.path.isfile(zip_path)
    assert zip_path.endswith("_dataset.zip")
    assert source.shape == partition.shape
    assert source.dtype == np.uint8
    assert 0 < len(thumbs) <= handlers.MAX_THUMBNAILS
    assert len(view.reports) == len(view.previews) >= 2
    assert all(r.is_partition for r in view.reports)

    summary, dd, adj, rows, meta, pairs, manifest_json, zip_rows = handlers.after_generate(view)
    assert "partition verified" in summary
    assert f"{len(view.reports)}/{len(view.reports)}" in summary
    assert dd["choices"][0][1] == 0
    assert dd["value"] == 0
    assert len(rows) == view.reports[0].n_pieces
    assert '"schema_version": "1.1"' in manifest_json
    assert {r[0] for r in zip_rows} >= {"manifest.json", "README.txt"}

    page2 = handlers.show_page(view, 1)
    expected_rows = [[row[h] for h in handlers.GT_HEADERS] for row in view.reports[1].rows]
    assert len(page2) == 7
    assert page2[4] == expected_rows

    cleared = handlers.clear_session(view)
    assert cleared[0] is None
    assert cleared[1] is None
    assert not os.path.exists(zip_path)


def test_cleanup_is_scoped_to_the_session():
    from src import workspace

    other = workspace.new_temp(suffix="_dataset.zip")      # another session, mid-run
    first, *_ = handlers.generate(
        handlers.load_sample(), 72, 6, 20.0, 60.0, False, 1, progress=_noop,
    )
    second, *_ = handlers.generate(
        handlers.load_sample(), 72, 6, 20.0, 60.0, False, 2, first, progress=_noop,
    )
    assert not os.path.exists(first.zip_path)             # own previous export dropped
    assert os.path.exists(second.zip_path)
    handlers.clear_session(second)
    assert not os.path.exists(second.zip_path)
    assert os.path.exists(other)                          # never touched
    workspace.discard(other)


@pytest.mark.parametrize("n_pieces", [1, 257, 70000])
def test_generate_rejects_out_of_range_piece_count(n_pieces):
    # API clients bypass the slider; must fail fast with a readable gr.Error.
    with pytest.raises(gr.Error, match="between 2 and 256"):
        handlers.generate(handlers.load_sample(), 72, n_pieces, 20.0, 60.0, False, 0,
                          progress=_noop)


@pytest.mark.parametrize(
    ("dpi", "strength", "scale", "message"),
    [
        (3000, 20.0, 60.0, "Render DPI must be between 72 and 300"),
        (10, 20.0, 60.0, "Render DPI must be between 72 and 300"),
        (72, -1.0, 60.0, r"Edge displacement \(px\) must be between 0 and 80"),
        (72, 500.0, 60.0, r"Edge displacement \(px\) must be between 0 and 80"),
        (72, 20.0, 1.0, r"Edge wavelength \(px\) must be between 8 and 200"),
        (72, 20.0, float("nan"), r"Edge wavelength \(px\) must be between 8 and 200"),
        (None, 20.0, 60.0, "Render DPI must be between 72 and 300"),
    ],
)
def test_generate_rejects_out_of_range_render_inputs(dpi, strength, scale, message):
    with pytest.raises(gr.Error, match=message):
        handlers.generate(handlers.load_sample(), dpi, 12, strength, scale, False, 0,
                          progress=_noop)


def test_session_state_expiry_frees_export():
    import datetime

    from gradio.state_holder import StateHolder

    from src import config

    demo = build_ui()
    (state,) = [b for b in demo.blocks.values() if isinstance(b, gr.State)]
    assert state.time_to_live == config.TEMP_FILE_TTL_S
    view, *_ = handlers.generate(
        handlers.load_sample(), 72, 6, 20.0, 60.0, False, 3, progress=_noop,
    )
    holder = StateHolder()
    holder.set_blocks(demo)
    session = holder["session"]
    session[state._id] = view
    holder.delete_all_expired_state()
    assert os.path.exists(view.zip_path)                  # fresh: kept
    long_ago = datetime.datetime.now() - datetime.timedelta(seconds=config.TEMP_FILE_TTL_S + 5)
    session._state_ttl[state._id] = (state.time_to_live, long_ago)
    holder.delete_all_expired_state()
    assert not os.path.exists(view.zip_path)              # expired: callback freed it


def test_generate_without_pdf_errors():
    with pytest.raises(gr.Error):
        handlers.generate(None, 72, 12, 20.0, 60.0, False, 0, progress=_noop)


def test_dpi_note_matches_config():
    from src.config import a4_pixels

    w, h = a4_pixels(150)
    assert f"{w} × {h}" in handlers.dpi_note(150)
