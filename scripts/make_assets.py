"""Regenerate the bundled sample PDF and the README figure.

    python scripts/make_assets.py

Writes:
  assets/sample.pdf          2-page synthetic A4 document (text, table, chart)
  assets/figure_overview.png source | partition | adjacency | exploded fragments

The figure is rendered from a real `generate_dataset` run on the sample with
the parameters below, so it shows exactly what the tool produces.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inspection import (  # noqa: E402
    _font,
    make_preview,
    reassemble,
    render_adjacency,
    render_partition,
)
from src.pipeline import generate_dataset  # noqa: E402

SAMPLE = os.path.join(ROOT, "assets", "sample.pdf")
FIGURE = os.path.join(ROOT, "assets", "figure_overview.png")
PARAMS = dict(dpi=100, n_pieces=14, noise_strength=28.0, noise_scale=96.0,
              master_seed=7, lossy=False)

PARAGRAPH = (
    "Fragments recovered from the archive box were numbered in the order they "
    "were lifted, not in reading order. Each was photographed on a neutral "
    "background before any attempt at reconstruction. Edge profiles, ruling "
    "lines and ink density were logged per fragment so candidate joins can be "
    "scored independently of content."
)


def make_sample_pdf(path: str) -> None:
    import fitz

    doc = fitz.open()
    # Page 1: prose + table.
    page = doc.new_page(width=595, height=842)
    page.insert_text((56, 72), "Sample document - Dataset-Maker", fontsize=17, fontname="helv")
    page.insert_text((56, 90), "Synthetic test page. Contents are placeholder text.",
                     fontsize=9, fontname="helv", color=(0.35, 0.35, 0.35))
    page.draw_line((56, 100), (539, 100), color=(0, 0, 0), width=0.7)
    y = 116
    for n in range(5):
        page.insert_text((56, y + 10), f"{n + 1}. Observation", fontsize=10.5, fontname="hebo")
        page.insert_textbox(fitz.Rect(56, y + 16, 539, y + 86), PARAGRAPH,
                            fontsize=9.5, fontname="tiro")
        y += 92
    top = 600
    cols = ["Fragment", "Width (mm)", "Height (mm)", "Edge type"]
    rows = [["F-01", "62", "88", "torn"], ["F-02", "45", "71", "torn / cut"],
            ["F-03", "80", "39", "torn"], ["F-04", "58", "66", "folded"],
            ["F-05", "33", "52", "torn"]]
    for r, row in enumerate([cols] + rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(56 + c * 121, top + r * 22, 56 + (c + 1) * 121, top + (r + 1) * 22)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5,
                           fill=(0.92, 0.92, 0.9) if r == 0 else None)
            page.insert_textbox(rect + (5, 6, 0, 0), cell, fontsize=8.5,
                                fontname="hebo" if r == 0 else "cour")
    page.insert_text((56, 760), "Table 1. Fragment dimensions (placeholder values).",
                     fontsize=8.5, fontname="tiro")

    # Page 2: line chart + ruled notes area.
    page = doc.new_page(width=595, height=842)
    page.insert_text((56, 72), "Figure page", fontsize=15, fontname="helv")
    x0, y0, x1, y1 = 80, 110, 520, 400
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=0.8)
    for k in range(1, 6):
        yy = y0 + k * (y1 - y0) / 6
        page.draw_line((x0, yy), (x1, yy), color=(0.8, 0.8, 0.8), width=0.4)
    rng = np.random.default_rng(3)
    for series, colour in ((0, (0.1, 0.3, 0.8)), (1, (0.8, 0.2, 0.1))):
        vals = np.cumsum(rng.normal(0, 1, 30)) * 12 + (y0 + y1) / 2 + series * 40
        pts = [(x0 + i * (x1 - x0) / 29, float(np.clip(v, y0 + 5, y1 - 5)))
               for i, v in enumerate(vals)]
        for a, b in zip(pts, pts[1:]):
            page.draw_line(a, b, color=colour, width=1.4)
    page.insert_text((80, 420), "Figure 1. Two synthetic series (random walk).",
                     fontsize=8.5, fontname="tiro")
    for k in range(14):
        yy = 470 + k * 24
        page.draw_line((56, yy), (539, yy), color=(0.55, 0.65, 0.85), width=0.5)
    page.insert_text((60, 490), "Notes:", fontsize=10, fontname="hebo")

    doc.set_metadata({"producer": "Dataset-Maker scripts/make_assets.py",
                      "creationDate": "D:20240101000000Z", "modDate": "D:20240101000000Z"})
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    doc.close()


def exploded(torn, spread: float = 0.22) -> np.ndarray:
    """Fragments pushed outward from the page centre, on a dark canvas."""
    H, W = torn.height, torn.width
    pad = int(max(H, W) * spread)
    canvas = np.full((H + 2 * pad, W + 2 * pad, 3), 40, dtype=np.uint8)
    cx, cy = W / 2, H / 2
    for p in torn.pieces:
        h, w = p.mask.shape
        px, py = p.x + w / 2, p.y + h / 2
        dx = int((px - cx) * spread * 1.6) + pad
        dy = int((py - cy) * spread * 1.6) + pad
        region = canvas[p.y + dy:p.y + dy + h, p.x + dx:p.x + dx + w]
        region[p.mask] = p.rgb[p.mask]
    return canvas


def panel(img: np.ndarray, title: str, height: int) -> Image.Image:
    im = Image.fromarray(img)
    im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
    out = Image.new("RGB", (im.width, height + 34), (255, 255, 255))
    out.paste(im, (0, 34))
    ImageDraw.Draw(out).rectangle((0, 34, im.width - 1, height + 33), outline=(200, 200, 204))
    ImageDraw.Draw(out).text((2, 8), title, fill=(24, 24, 27), font=_font(17))
    return out


def main() -> None:
    make_sample_pdf(SAMPLE)
    run = generate_dataset(SAMPLE, source_name="sample.pdf", **PARAMS)
    torn = run.pages[0]
    pv = make_preview(torn, max_side=900)
    n = len(torn.pieces)
    panels = [
        panel(reassemble(torn), "1  source page (A4 render)", 620),
        panel(render_partition(pv, n), "2  partition + piece index", 620),
        panel(render_adjacency(pv, n, torn.adjacency), "3  adjacency ground truth", 620),
        panel(exploded(torn), "4  fragments, pulled apart from their offsets", 620),
    ]
    gap = 18
    fig = Image.new("RGB", (sum(p.width for p in panels) + gap * 3, panels[0].height),
                    (255, 255, 255))
    x = 0
    for p in panels:
        fig.paste(p, (x, 0))
        x += p.width + gap
    fig.save(FIGURE, optimize=True)
    r = run.reports[0]
    print(f"wrote {SAMPLE} and {FIGURE}")
    print(f"page 0: {r.n_pieces} fragments, {r.n_edges} adjacency pairs, "
          f"partition={'ok' if r.is_partition else 'FAIL'}; params={PARAMS}")


if __name__ == "__main__":
    main()
