"""Ground-truth inspection: per-page stats, reassembly, and preview renders.

UI-free (NumPy / SciPy / Pillow only) so every number the UI shows is computed
here and unit-tested. Nothing in this module changes generated data; it only
reads `TornPage` objects and the packaged ZIP.

Previews are stored compactly (JPEG page + zlib'd label map at <= `max_side`
px) so a 60-page run costs a few MB of session state instead of hundreds.
"""
from __future__ import annotations

import colorsys
import io
import json
import math
import re
import zipfile
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, find_objects
from scipy.ndimage import label as connected_components

from .packager import piece_path
from .tearing import TornPage, verify_partition

# Above this many pieces, index captions on the renders overlap into noise;
# the per-fragment table still lists every index.
MAX_CAPTIONED_PIECES = 120


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
@dataclass
class PageReport:
    """Measured ground-truth summary for one torn page."""
    index: int
    seed: int | None
    width: int
    height: int
    n_requested: int
    n_pieces: int
    n_edges: int
    mean_degree: float
    graph_connected: bool
    multi_component: int          # pieces made of >1 4-connected region
    max_overlap: int
    uncovered_pixels: int
    is_partition: bool
    rows: list[dict] = field(default_factory=list)   # one per piece


def graph_is_connected(n: int, adjacency: list[tuple[int, int]]) -> bool:
    """True iff the undirected adjacency graph over n nodes is connected."""
    if n <= 1:
        return True
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for i, j in adjacency:
        nbrs[i].append(j)
        nbrs[j].append(i)
    seen = {0}
    stack = [0]
    while stack:
        for v in nbrs[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def page_report(torn: TornPage, index: int, n_requested: int) -> PageReport:
    """Verify the partition and tabulate per-piece ground truth for a page."""
    check = verify_partition(torn)
    n = len(torn.pieces)
    degree = [0] * n
    for i, j in torn.adjacency:
        degree[i] += 1
        degree[j] += 1

    rows, multi = [], 0
    for k, p in enumerate(torn.pieces):
        _, n_cc = connected_components(p.mask)   # default structure = 4-conn
        multi += n_cc > 1
        h, w = p.mask.shape
        rows.append({
            "piece": k,
            "file": piece_path(index, k),
            "x": p.x, "y": p.y, "w": w, "h": h,
            "area_px": int(p.mask.sum()),
            "degree": degree[k],
            "components": int(n_cc),
        })

    return PageReport(
        index=index,
        seed=torn.seed,
        width=torn.width,
        height=torn.height,
        n_requested=int(n_requested),
        n_pieces=n,
        n_edges=len(torn.adjacency),
        mean_degree=(2 * len(torn.adjacency) / n) if n else 0.0,
        graph_connected=graph_is_connected(n, torn.adjacency),
        multi_component=int(multi),
        max_overlap=check["max_overlap"],
        uncovered_pixels=check["uncovered_pixels"],
        is_partition=check["is_partition"],
        rows=rows,
    )


def reassemble(torn: TornPage) -> np.ndarray:
    """Paste every piece at its (x, y) offset using its exact mask.

    For a valid partition this reproduces the source page pixel-exactly.
    """
    canvas = np.zeros((torn.height, torn.width, 3), dtype=np.uint8)
    for p in torn.pieces:
        h, w = p.mask.shape
        # copyto(where=) is ~5x faster than boolean gather/scatter here.
        np.copyto(canvas[p.y:p.y + h, p.x:p.x + w], p.rgb, where=p.mask[..., None])
    return canvas


# --------------------------------------------------------------------------
# Compact previews
# --------------------------------------------------------------------------
@dataclass
class PagePreview:
    """Downscaled page + piece-index map, compressed for session storage."""
    factor: int                   # integer downscale: preview px = page px / factor
    shape: tuple[int, int]        # preview (h, w)
    page_jpeg: bytes
    labels_z: bytes               # zlib'd uint16 piece-index map

    def page(self) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(self.page_jpeg)).convert("RGB"))

    def labels(self) -> np.ndarray:
        flat = np.frombuffer(zlib.decompress(self.labels_z), dtype=np.uint16)
        return flat.reshape(self.shape)


def make_preview(torn: TornPage, max_side: int = 900) -> PagePreview:
    """Downscale a torn page (nearest for labels, box filter for pixels)."""
    f = max(1, math.ceil(max(torn.height, torn.width) / max_side))
    # Raw labels -> manifest piece index (labels can have gaps when a seed's
    # warped cell vanished).
    lut = np.zeros(int(torn.labels.max()) + 1, dtype=np.uint16)
    for k, p in enumerate(torn.pieces):
        lut[p.label] = k
    small = lut[torn.labels[::f, ::f]]
    h, w = small.shape

    page = Image.fromarray(reassemble(torn)).resize((w, h), Image.BOX)
    buf = io.BytesIO()
    page.save(buf, format="JPEG", quality=88)
    return PagePreview(
        factor=f,
        shape=(h, w),
        page_jpeg=buf.getvalue(),
        labels_z=zlib.compress(np.ascontiguousarray(small).tobytes(), 6),
    )


def _palette(n: int) -> np.ndarray:
    """n muted, well-separated colours (golden-angle hue walk), uint8 (n, 3)."""
    out = np.empty((max(n, 1), 3), dtype=np.float32)
    for i in range(max(n, 1)):
        h = (i * 0.618033988749895) % 1.0
        s = 0.55 if i % 2 else 0.40
        v = 0.92 if i % 3 else 0.80
        out[i] = colorsys.hsv_to_rgb(h, s, v)
    return (out * 255).astype(np.uint8)


def _boundaries(lab: np.ndarray) -> np.ndarray:
    """2-px-wide mask of pixels whose 4-neighbour has a different label."""
    b = np.zeros(lab.shape, dtype=bool)
    dx = lab[:, 1:] != lab[:, :-1]
    dy = lab[1:, :] != lab[:-1, :]
    b[:, 1:] |= dx
    b[:, :-1] |= dx
    b[1:, :] |= dy
    b[:-1, :] |= dy
    return b


def _anchors(lab: np.ndarray, n: int) -> np.ndarray:
    """Per piece, the interior point farthest from its border (label spot)."""
    pts = np.zeros((n, 2), dtype=np.float32)
    for k, sl in enumerate(find_objects(lab.astype(np.int32) + 1)[:n]):
        if sl is None:
            continue
        inside = np.pad(lab[sl] == k, 1)          # pad: page edge counts as border
        d = distance_transform_edt(inside)
        yy, xx = np.unravel_index(int(np.argmax(d)), d.shape)
        pts[k] = (sl[1].start + xx - 1, sl[0].start + yy - 1)
    return pts


def _font(size: int):
    """Bundled default font at `size` px; fixed-size bitmap without FreeType."""
    base = ImageFont.load_default()
    if isinstance(base, ImageFont.FreeTypeFont):
        return base.font_variant(size=size)
    return base


def _caption(draw: ImageDraw.ImageDraw, xy, text: str, font) -> None:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    cx, cy = xy
    box = (cx - tw / 2 - 3, cy - th / 2 - 2, cx + tw / 2 + 3, cy + th / 2 + 3)
    draw.rectangle(box, fill=(255, 255, 255), outline=(24, 24, 27))
    draw.text((cx - tw / 2 - x0, cy - th / 2 - y0), text, fill=(24, 24, 27), font=font)


def render_partition(preview: PagePreview, n_pieces: int) -> np.ndarray:
    """Page tinted per fragment, tear lines drawn, piece indices captioned."""
    lab = preview.labels()
    page = preview.page().astype(np.float32)
    tint = _palette(n_pieces)[lab].astype(np.float32)
    out = page * 0.55 + tint * 0.45
    out[_boundaries(lab)] = (24, 24, 27)
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    if n_pieces <= MAX_CAPTIONED_PIECES:
        draw = ImageDraw.Draw(img)
        font = _font(max(10, preview.shape[0] // 70))
        for k, (x, y) in enumerate(_anchors(lab, n_pieces)):
            _caption(draw, (float(x), float(y)), str(k), font)
    return np.asarray(img)


def render_adjacency(
    preview: PagePreview, n_pieces: int, adjacency: list[tuple[int, int]]
) -> np.ndarray:
    """Adjacency graph (nodes = fragments, edges = shared tear) over the page."""
    lab = preview.labels()
    page = preview.page().astype(np.float32)
    out = page * 0.18 + 255 * 0.82                  # faded page for context
    out[_boundaries(lab)] = (170, 170, 176)
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    pts = _anchors(lab, n_pieces)
    lw = max(1, preview.shape[0] // 400)
    for i, j in adjacency:
        draw.line([tuple(pts[i]), tuple(pts[j])], fill=(37, 99, 235), width=lw + 1)
    r = max(3, preview.shape[0] // 180)
    font = _font(max(10, preview.shape[0] // 70))
    for k, (x, y) in enumerate(pts):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(24, 24, 27))
        if n_pieces <= MAX_CAPTIONED_PIECES:
            _caption(draw, (float(x), float(y) - r - 9), str(k), font)
    return np.asarray(img)


def fragment_thumbnails(
    preview: PagePreview, rows: list[dict], limit: int = 64
) -> list[tuple[np.ndarray, str]]:
    """Preview-resolution crops of each fragment on black, with GT captions."""
    lab = preview.labels()
    page = preview.page()
    f = preview.factor
    out = []
    for row in rows[:limit]:
        k = row["piece"]
        y0, x0 = row["y"] // f, row["x"] // f
        y1 = max(y0 + 1, math.ceil((row["y"] + row["h"]) / f))
        x1 = max(x0 + 1, math.ceil((row["x"] + row["w"]) / f))
        inside = lab[y0:y1, x0:x1] == k
        crop = np.zeros(inside.shape + (3,), dtype=np.uint8)
        crop[inside] = page[y0:y1, x0:x1][inside]
        caption = f"#{k} ({row['x']},{row['y']})"
        out.append((crop, caption))
    return out


# --------------------------------------------------------------------------
# Manifest / archive views
# --------------------------------------------------------------------------
def manifest_excerpt(
    manifest: dict, max_pieces: int = 3, max_pairs: int = 8
) -> dict:
    """First page of the manifest with long arrays truncated, keys in order."""
    out: "OrderedDict[str, object]" = OrderedDict()
    for key, value in manifest.items():
        if key != "pages":
            out[key] = value
            continue
        pages = list(value)
        if not pages:
            out[key] = []
            continue
        first = dict(pages[0])
        for arr_key, cap in (("adjacency", max_pairs), ("pieces", max_pieces)):
            arr = list(first[arr_key])
            if len(arr) > cap:
                arr = arr[:cap] + [f"... {len(first[arr_key]) - cap} more"]
            first[arr_key] = arr
        out[key] = [first] + (
            [f"... {len(pages) - 1} more page(s)"] if len(pages) > 1 else []
        )
    return out


def zip_listing(zip_bytes: bytes) -> list[dict]:
    """Group archive entries by top-level file / per-page piece folder."""
    groups: "OrderedDict[str, dict]" = OrderedDict()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            parts = info.filename.split("/")
            key = "/".join(parts[:2]) + "/" if len(parts) > 2 else info.filename
            g = groups.setdefault(key, {"path": key, "files": 0, "bytes": 0})
            g["files"] += 1
            g["bytes"] += info.file_size
    return list(groups.values())


_INNERMOST = re.compile(r"([\[{])([^\[\]{}]*?)([\]}])")


def format_json(obj) -> str:
    """indent=2 JSON with innermost arrays/objects collapsed onto one line.

    Keeps `[i, j]` pairs and per-piece `{file, x, y, w, h}` records readable.
    Assumes no string value contains brackets or braces (true for manifests).
    """
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    return _INNERMOST.sub(
        lambda m: m.group(1) + re.sub(r"\s*\n\s*", " ", m.group(2)).strip()
        + m.group(3),
        text,
    )
