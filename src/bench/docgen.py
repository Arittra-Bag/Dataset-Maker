"""Seeded synthetic source documents for the benchmark.

Why synthetic: every page is licence-clean, needs no download, and is fully
determined by one integer seed, so a benchmark release can be regenerated
instead of mirrored. Limitation (stated in the benchmark card): layouts are
simpler than real documents.

A document is 1..3 A4 pages built from a small layout grammar: title block,
headings, paragraphs of pseudo-words, bullet lists, tables, line/bar charts,
ruled note areas and page numbers. Only the PDF Base-14 fonts are used, so no
font files are embedded or fetched.
"""
from __future__ import annotations

import numpy as np

# A4 sized so 100 DPI renders to exactly 827 x 1169 px with no resampling
# (595.44 = 827 * 0.72, 841.68 = 1169 * 0.72).
BENCH_DPI = 100
PAGE_PX = (827, 1169)
A4_PT = (PAGE_PX[0] * 72 / BENCH_DPI, PAGE_PX[1] * 72 / BENCH_DPI)
AA_LEVEL = 8                      # MuPDF anti-aliasing bits, set explicitly

# (regular, bold) Base-14 families by PyMuPDF short name.
FONTS = (("helv", "hebo"), ("tiro", "tibo"), ("cour", "cobo"))
_SYLLABLES = (
    "ka", "ri", "to", "men", "sa", "lo", "ver", "di", "an", "po", "tel", "ru",
    "ex", "mo", "ni", "qua", "fe", "lis", "or", "tu", "ba", "cen", "gro", "vi",
)


def _word(rng: np.random.Generator) -> str:
    return "".join(rng.choice(_SYLLABLES, size=int(rng.integers(1, 4))))


def _sentence(rng: np.random.Generator, lo: int = 6, hi: int = 16) -> str:
    words = [_word(rng) for _ in range(int(rng.integers(lo, hi)))]
    words[0] = words[0].capitalize()
    return " ".join(words) + rng.choice([".", ".", ".", ";", ":"])


def _paragraph(rng: np.random.Generator) -> str:
    return " ".join(_sentence(rng) for _ in range(int(rng.integers(2, 6))))


class _Page:
    """Vertical layout cursor over one or two text columns."""

    def __init__(self, page, rng: np.random.Generator):
        self.page, self.rng = page, rng
        m = float(rng.uniform(40, 72))
        self.left, self.right = m, A4_PT[0] - m
        self.top, self.bottom = float(rng.uniform(44, 70)), A4_PT[1] - float(rng.uniform(50, 72))
        self.y = self.top
        self.font, self.bold = FONTS[int(rng.integers(len(FONTS)))]
        self.size = float(rng.choice([8.5, 9.0, 9.5, 10.0, 11.0]))
        self.columns = 2 if rng.random() < 0.3 else 1

    def room(self, h: float) -> bool:
        return self.y + h <= self.bottom

    def text_block(self, text: str, size: float | None = None, bold: bool = False,
                   indent: float = 0.0) -> bool:
        """Flow `text` into the remaining space; False once the page is full."""
        import fitz

        size = size or self.size
        font = self.bold if bold else self.font
        lines = max(1, int(len(text) * size * 0.52 / (self.right - self.left - indent)) + 1)
        h = lines * size * 1.35 + size
        if not self.room(h):
            return False
        rect = fitz.Rect(self.left + indent, self.y, self.right, self.y + h)
        if self.columns == 2 and not bold:
            mid = (self.left + self.right) / 2
            half = h / 2 + size * 1.4
            if not self.room(half):
                return False
            self.page.insert_textbox(fitz.Rect(self.left, self.y, mid - 8, self.y + half), text,
                                     fontsize=size, fontname=font)
            self.page.insert_textbox(fitz.Rect(mid + 8, self.y, self.right, self.y + half),
                                     _paragraph(self.rng), fontsize=size, fontname=font)
            self.y += half + size * 0.6
            return True
        self.page.insert_textbox(rect, text, fontsize=size, fontname=font)
        self.y += h
        return True

    def rule(self, width: float = 0.6) -> None:
        self.page.draw_line((self.left, self.y), (self.right, self.y), color=(0, 0, 0), width=width)
        self.y += 8

    def table(self) -> bool:
        import fitz

        rows, cols = int(self.rng.integers(3, 8)), int(self.rng.integers(2, 6))
        row_h = self.size * 2.1
        if not self.room(rows * row_h + 20):
            return False
        col_w = (self.right - self.left) / cols
        shade = float(self.rng.uniform(0.82, 0.95))
        for r in range(rows):
            for c in range(cols):
                cell = fitz.Rect(self.left + c * col_w, self.y + r * row_h,
                                 self.left + (c + 1) * col_w, self.y + (r + 1) * row_h)
                self.page.draw_rect(cell, color=(0, 0, 0), width=0.5,
                                    fill=(shade, shade, shade) if r == 0 else None)
                label = _word(self.rng).capitalize() if r == 0 else f"{self.rng.integers(0, 999)}"
                self.page.insert_textbox(cell + (4, 4, -2, 0), label, fontsize=self.size * 0.9,
                                         fontname=self.bold if r == 0 else "cour")
        self.y += rows * row_h + 12
        return True

    def chart(self) -> bool:
        import fitz

        h = float(self.rng.uniform(120, 200))
        if not self.room(h + 24):
            return False
        x0, y0, x1, y1 = self.left + 20, self.y, self.right - 20, self.y + h
        self.page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=0.8)
        for k in range(1, 5):
            yy = y0 + k * h / 5
            self.page.draw_line((x0, yy), (x1, yy), color=(0.8, 0.8, 0.8), width=0.4)
        n = int(self.rng.integers(6, 18))
        if self.rng.random() < 0.5:                       # bar chart
            w = (x1 - x0) / n
            colour = tuple(float(v) for v in self.rng.uniform(0.1, 0.8, 3))
            for i in range(n):
                bh = float(self.rng.uniform(0.1, 0.9)) * h
                self.page.draw_rect(fitz.Rect(x0 + i * w + 2, y1 - bh, x0 + (i + 1) * w - 2, y1),
                                    color=None, fill=colour)
        else:                                             # line chart, 1-3 series
            for _ in range(int(self.rng.integers(1, 4))):
                colour = tuple(float(v) for v in self.rng.uniform(0.0, 0.8, 3))
                vals = np.cumsum(self.rng.normal(0, 1, n))
                vals = (vals - vals.min()) / (np.ptp(vals) or 1.0)
                pts = [(x0 + i * (x1 - x0) / (n - 1), y1 - 6 - v * (h - 12)) for i, v in enumerate(vals)]
                for a, b in zip(pts, pts[1:]):
                    self.page.draw_line(a, b, color=colour, width=1.3)
        self.y = y1 + 6
        return self.text_block(f"Figure {self.rng.integers(1, 9)}. {_sentence(self.rng, 4, 9)}",
                               size=self.size * 0.85)

    def bullets(self) -> bool:
        for _ in range(int(self.rng.integers(2, 6))):
            # ASCII bullet: Base-14 text insertion does not map U+2022.
            if not self.text_block("- " + _sentence(self.rng, 4, 12), indent=12):
                return False
        return True

    def notes(self) -> bool:
        n = int(self.rng.integers(4, 10))
        if not self.room(n * 20 + 10):
            return False
        colour = (0.55, 0.65, 0.85)
        for k in range(n):
            yy = self.y + 10 + k * 20
            self.page.draw_line((self.left, yy), (self.right, yy), color=colour, width=0.5)
        self.y += n * 20 + 14
        return True


def make_document(seed: int, path: str) -> int:
    """Write a 1..3 page synthetic A4 PDF for `seed` to `path`; return page count."""
    import fitz

    rng = np.random.default_rng(seed)
    n_pages = int(rng.choice([1, 1, 2, 3]))
    doc = fitz.open()
    for pno in range(n_pages):
        page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
        lay = _Page(page, rng)
        if pno == 0:
            lay.text_block(" ".join(_word(rng).capitalize() for _ in range(int(rng.integers(2, 6)))),
                           size=lay.size * 1.8, bold=True)
            lay.rule(0.8)
        blocks = (lay.table, lay.chart, lay.bullets, lay.notes)
        misses = 0
        while misses < 3:                     # keep filling with smaller blocks
            k = rng.random()
            if k < 0.12:
                ok = lay.text_block(f"{rng.integers(1, 9)}. " + _word(rng).capitalize(),
                                    size=lay.size * 1.2, bold=True)
            elif k < 0.62:
                ok = lay.text_block(_paragraph(rng))
            else:
                ok = blocks[int(rng.integers(len(blocks)))]()
            misses = 0 if ok else misses + 1
        page.insert_text((A4_PT[0] / 2 - 6, A4_PT[1] - 28), str(pno + 1),
                         fontsize=lay.size * 0.9, fontname=lay.font)
    doc.set_metadata({"producer": "dm-bench docgen", "creationDate": "D:20240101000000Z",
                      "modDate": "D:20240101000000Z"})
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    doc.close()
    return n_pages


def render_document(path: str) -> list[np.ndarray]:
    """Rasterise every page at BENCH_DPI to (1169, 827, 3) uint8 RGB.

    Direct get_pixmap, deliberately not pdf_loader.load_pdf_pages: that path
    letterboxes with a Lanczos resample, which would tie benchmark pixels to
    app heuristics and a float resampler.
    """
    import fitz

    fitz.TOOLS.set_aa_level(AA_LEVEL)
    zoom = BENCH_DPI / 72
    doc = fitz.open(path)
    try:
        pages = []
        for page in doc:
            pm = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)
            if arr.shape != (PAGE_PX[1], PAGE_PX[0], 3):
                raise RuntimeError(f"unexpected render size {arr.shape}")
            pages.append(arr.copy())
        return pages
    finally:
        doc.close()
