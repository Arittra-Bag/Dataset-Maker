"""Gradio event handlers: thin glue between widgets and `src/`.

No generation or measurement logic lives here; handlers call
`src.pipeline.generate_dataset` and format what `src.inspection` computes.
Session state (`RunView`) holds compressed previews only, not full-res pages.
"""
from __future__ import annotations

import html
import math
import os
import re
import time
from dataclasses import dataclass

import fitz
import gradio as gr

from src import config, workspace
from src.inspection import (
    PagePreview,
    PageReport,
    format_json,
    fragment_thumbnails,
    make_preview,
    manifest_excerpt,
    render_adjacency,
    render_partition,
    zip_listing,
)
from src.pipeline import PartitionError, generate_dataset, save_temp_pdf

from .content import EMPTY_SUMMARY_HTML

SAMPLE_PDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sample.pdf"
)
MAX_THUMBNAILS = 64

GT_HEADERS = ["piece", "x", "y", "w", "h", "area_px", "degree", "components", "file"]


@dataclass
class RunView:
    """What the UI keeps per session after a run (a few MB for 60 pages)."""
    dpi: int
    reports: list[PageReport]
    adjacency: list[list[tuple[int, int]]]
    previews: list[PagePreview]
    summary_html: str
    manifest_json: str
    zip_rows: list[list]
    zip_path: str          # this session's export; freed by its next Generate/Clear or the TTL sweep


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def _mark(ok: bool, yes: str = "pass", no: str = "FAIL") -> str:
    return f'<span class="dm-pass">{yes}</span>' if ok else f'<span class="dm-fail">{no}</span>'


def _summary_html(reports: list[PageReport], timings: dict[str, float],
                  n_requested: int, zip_size: int) -> str:
    n_pages = len(reports)
    produced = sum(r.n_pieces for r in reports)
    verified = sum(r.is_partition for r in reports)
    multi = sum(r.multi_component for r in reports)
    edges = sum(r.n_edges for r in reports)
    total = sum(timings.values())
    stages = " · ".join(f"{k.replace('_', ' ')} {v:.2f}s" for k, v in timings.items())
    multi_cls = "dm-warn" if multi else "dm-pass"
    return f"""
<div class="dm-summary">
  <div class="dm-stats">
    <span><b>{n_pages}</b> page{'s' if n_pages != 1 else ''}</span>
    <span><b>{produced}</b> fragments ({n_requested * n_pages} requested)</span>
    <span><b>{edges}</b> adjacency pairs</span>
    <span>partition verified <b class="{'dm-pass' if verified == n_pages else 'dm-fail'}">{verified}/{n_pages}</b></span>
    <span>multi-component <b class="{multi_cls}">{multi}</b></span>
    <span>zip <b>{zip_size / 1024:.0f} KiB</b></span>
  </div>
  <div class="dm-timing">measured wall time {total:.2f}s ({stages})</div>
</div>"""


def _page_meta_html(r: PageReport, n_pages: int, dpi_hint: str) -> str:
    missing = r.n_requested - r.n_pieces
    count_note = "" if missing <= 0 else f' <span class="dm-warn">({missing} seed cell(s) vanished)</span>'
    multi = ('<span class="dm-pass">0</span>' if r.multi_component == 0
             else f'<span class="dm-warn">{r.multi_component}</span> fragment(s) with &gt;1 region')
    return f"""
<dl class="dm-kv">
  <dt>page</dt><dd>{r.index + 1} / {n_pages} &nbsp;·&nbsp; index {r.index}</dd>
  <dt>seed</dt><dd>{r.seed}</dd>
  <dt>canvas</dt><dd>{r.width} × {r.height} px {html.escape(dpi_hint)}</dd>
  <dt>fragments</dt><dd>{r.n_pieces} produced / {r.n_requested} requested{count_note}</dd>
  <dt>adjacency</dt><dd>{r.n_edges} pairs · mean degree {r.mean_degree:.2f} ·
    graph connected {_mark(r.graph_connected, 'yes', 'no')}</dd>
  <dt>multi-component</dt><dd>{multi}</dd>
  <dt>partition</dt><dd>max_overlap={r.max_overlap} · uncovered={r.uncovered_pixels} → {_mark(r.is_partition)}</dd>
</dl>"""


def _pairs_text(pairs: list[tuple[int, int]], per_line: int = 5) -> str:
    items = [f"[{i}, {j}]" for i, j in pairs]
    lines = [", ".join(items[k:k + per_line]) for k in range(0, len(items), per_line)]
    return "[\n  " + ",\n  ".join(lines) + "\n]" if lines else "[]"


def _require_range(label: str, value, lo: float, hi: float) -> None:
    """Raise a readable gr.Error unless `value` is a number in [lo, hi].

    Gradio 4.44 does not enforce slider bounds server-side, so API callers
    can send anything (huge DPI would exhaust memory on the free tier).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan                      # NaN fails every comparison below
    if not lo <= v <= hi:
        raise gr.Error(f"{label} must be between {lo:g} and {hi:g}.")


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem)[:40] or "document"


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
def _page_views(view: RunView, i: int):
    """Per-page outputs: source, partition, adjacency, thumbnails, table, meta, pairs."""
    r = view.reports[i]
    pv = view.previews[i]
    dpi_hint = f"(A4 @ {view.dpi} DPI)"
    rows = [[row[h] for h in GT_HEADERS] for row in r.rows]
    return (
        pv.page(),
        render_partition(pv, r.n_pieces),
        render_adjacency(pv, r.n_pieces, view.adjacency[i]),
        fragment_thumbnails(pv, r.rows, limit=MAX_THUMBNAILS),
        rows,
        _page_meta_html(r, len(view.reports), dpi_hint),
        _pairs_text(view.adjacency[i]),
    )


def _page_choices(view: RunView):
    return [(f"Page {r.index + 1} · seed {r.seed} · {r.n_pieces} fragments", r.index)
            for r in view.reports]


# --------------------------------------------------------------------------
# Event handlers
# --------------------------------------------------------------------------
def generate(pdf_file, dpi, n_pieces, noise_strength, noise_scale, lossy, seed,
             view=None, progress=gr.Progress()):
    """Run the pipeline. Returns (state, source, partition, gallery, zip)."""
    if pdf_file is None:
        raise gr.Error("Upload a PDF first (or click 'Load sample PDF').")
    # Reject out-of-range API input before spending minutes rendering/tearing.
    _require_range("Fragments per page", n_pieces, config.MIN_PIECES, config.MAX_PIECES)
    _require_range("Render DPI", dpi, config.MIN_DPI, config.MAX_DPI)
    _require_range("Edge displacement (px)", noise_strength,
                   config.MIN_NOISE_STRENGTH, config.MAX_NOISE_STRENGTH)
    _require_range("Edge wavelength (px)", noise_scale,
                   config.MIN_NOISE_SCALE, config.MAX_NOISE_SCALE)

    # Keep disk bounded on the shared server without touching other sessions'
    # in-flight files: drop this session's previous export, then anything
    # abandoned for a full TTL. HF free-tier disk is small.
    release_view(view)
    workspace.clear_stale(config.TEMP_FILE_TTL_S)

    progress(0.02, desc="Reading PDF…")
    with open(pdf_file, "rb") as fh:
        pdf_bytes = fh.read()
    if len(pdf_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise gr.Error(f"PDF exceeds {config.MAX_UPLOAD_MB} MB limit.")

    tmp_pdf = save_temp_pdf(pdf_bytes)
    try:
        run = generate_dataset(
            tmp_pdf,
            source_name=os.path.basename(pdf_file),
            dpi=int(dpi),
            n_pieces=int(n_pieces),
            noise_strength=float(noise_strength),
            noise_scale=float(noise_scale),
            master_seed=int(seed or 0),
            lossy=bool(lossy),
            progress=lambda f, m: progress(0.05 + 0.85 * f, desc=m),
        )
    except PartitionError as exc:          # should be unreachable; never hide it
        raise gr.Error(f"Partition check failed: {exc}") from exc
    except ValueError as exc:              # e.g. PDF with no renderable pages
        raise gr.Error(str(exc)) from exc
    except fitz.FileDataError as exc:      # corrupt / non-PDF upload
        raise gr.Error(f"Could not read PDF: {exc}") from exc
    finally:
        # Input PDF is fully rendered into `run.pages` now; free it immediately.
        workspace.discard(tmp_pdf)

    progress(0.93, desc="Building previews…")
    t = time.perf_counter()
    previews = [make_preview(p) for p in run.pages]
    timings = dict(run.timings)
    timings["preview"] = time.perf_counter() - t

    out_path = workspace.new_temp(
        suffix=f"_{_safe_stem(pdf_file)}_seed{int(seed or 0)}_dataset.zip"
    )
    with open(out_path, "wb") as fh:
        fh.write(run.zip_bytes)

    view = RunView(
        dpi=int(dpi),
        reports=run.reports,
        adjacency=[p.adjacency for p in run.pages],
        previews=previews,
        summary_html=_summary_html(run.reports, timings, int(n_pieces), len(run.zip_bytes)),
        manifest_json=format_json(manifest_excerpt(run.manifest)),
        zip_rows=[[g["path"], g["files"], g["bytes"]] for g in zip_listing(run.zip_bytes)],
        zip_path=out_path,
    )
    source, partition, _, thumbs, *_ = _page_views(view, 0)
    progress(1.0, desc="Done")
    return view, source, partition, thumbs, out_path


def after_generate(view: RunView | None):
    """Fill the lightweight views (no progress overlay on thin components)."""
    if view is None:
        return (gr.update(),) * 8
    _, _, adjacency, _, rows, meta, pairs = _page_views(view, 0)
    return (
        view.summary_html,
        gr.update(choices=_page_choices(view), value=0, interactive=True),
        adjacency,
        rows,
        meta,
        pairs,
        view.manifest_json,
        view.zip_rows,
    )


def show_page(view: RunView | None, page_index):
    if view is None or page_index is None:
        return (gr.update(),) * 7
    return _page_views(view, int(page_index))


def release_view(view: RunView | None) -> None:
    """Free a session's export. Also the State delete_callback, so it runs
    when an idle or closed session's state expires."""
    if view is not None:
        workspace.discard(view.zip_path)


def clear_session(view: RunView | None):
    """Delete this session's export and reset every output.

    Never calls workspace.clear_all(): the registry is process-wide and would
    unlink other sessions' in-flight files.
    """
    release_view(view)
    return (
        None,                  # pdf_in
        None,                  # state
        None, None, None,      # source, partition, adjacency
        None,                  # gallery
        None,                  # gt table
        "",                    # page meta
        "",                    # pairs
        "",                    # manifest
        None,                  # zip file
        None,                  # zip listing
        EMPTY_SUMMARY_HTML,
        gr.update(choices=[], value=None, interactive=False),
    )


def load_sample():
    return SAMPLE_PDF


def dpi_note(dpi) -> str:
    w, h = config.a4_pixels(int(dpi))
    return (f'<div class="dm-note">A4 at {int(dpi)} DPI → <code>{w} × {h}</code> px '
            f"per page. Non-A4 pages are letterboxed or sliced into A4 bands; at most "
            f"{config.MAX_PAGES_PER_PDF} A4 pages per PDF, {config.MAX_UPLOAD_MB} MB upload.</div>")
