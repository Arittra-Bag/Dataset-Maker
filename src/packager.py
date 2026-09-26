"""Package torn pieces + stitching ground-truth into a downloadable ZIP.

Layout inside the archive:
    pieces/page_0001/piece_000.png ...
    manifest.json          # global summary + per-piece placement (x, y, w, h)
    README.txt             # how to reassemble

The manifest IS the dataset label: each piece's (x, y) offset on its page is the
exact stitching target. Reassembling = paste every piece at its offset.

Schema 1.1 adds provenance only (seed, requested count, input hash, library
versions); every 1.0 field keeps its name, type and meaning.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

from . import __version__
from .optimizer import encode_piece
from .provenance import MANIFEST_SCHEMA_VERSION, PAGE_SEED_RULE, environment
from .tearing import TornPage


def piece_path(page_index: int, piece_index: int) -> str:
    """ZIP path of a fragment (0-based indices, 1-based page folder)."""
    return f"pieces/page_{page_index + 1:04d}/piece_{piece_index:03d}.png"


def build_zip(
    pages: list[TornPage],
    *,
    source_name: str,
    dpi: int,
    noise_strength: float,
    noise_scale: float,
    lossy: bool,
    master_seed: int,
    n_pieces_requested: int,
    source_sha256: str | None = None,
) -> tuple[bytes, dict]:
    """Return (zip_bytes, manifest_dict) for a list of torn pages."""
    manifest = {
        "generator": "Dataset-Maker",
        "generator_version": __version__,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        "source_sha256": source_sha256,
        "dpi": dpi,
        "n_pieces_requested": int(n_pieces_requested),
        "master_seed": int(master_seed),
        "page_seed_rule": PAGE_SEED_RULE,
        "noise_strength": noise_strength,
        "noise_scale": noise_scale,
        "lossy": lossy,
        "environment": environment(),
        "pages": [],
        "total_pieces": 0,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pi, page in enumerate(pages):
            page_entry = {
                "index": pi,
                "seed": page.seed,
                "width": page.width,
                "height": page.height,
                # Undirected neighbor pairs (piece-index i, j) = which fragments
                # share a torn border. Positive pairs for pairwise/graph stitching
                # models; non-listed pairs are negatives.
                "adjacency": [[int(i), int(j)] for i, j in page.adjacency],
                "pieces": [],
            }
            for k, piece in enumerate(page.pieces):
                fname = piece_path(pi, k)
                zf.writestr(fname, encode_piece(piece.rgb, lossy=lossy))
                h, w = piece.mask.shape
                page_entry["pieces"].append(
                    {"file": fname, "x": piece.x, "y": piece.y, "w": w, "h": h}
                )
            manifest["total_pieces"] += len(page.pieces)
            manifest["pages"].append(page_entry)

        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("README.txt", _README)

    return buf.getvalue(), manifest


REASSEMBLY_SNIPPET = """\
import json
import numpy as np
from PIL import Image

m = json.load(open("manifest.json"))
for page in m["pages"]:
    canvas = np.zeros((page["height"], page["width"], 3), np.uint8)
    for p in page["pieces"]:
        rgb = np.asarray(Image.open(p["file"]).convert("RGB"))
        inside = rgb.any(axis=2)          # non-black = fragment pixel
        region = canvas[p["y"]:p["y"] + p["h"], p["x"]:p["x"] + p["w"]]
        region[inside] = rgb[inside]
    Image.fromarray(canvas).save(f"reassembled_{page['index']:04d}.png")
"""

_README = """Dataset-Maker export
=====================
Each page was torn into NON-OVERLAPPING fragments (a strict partition: every
pixel belongs to exactly one piece). Fragments sit on a black background.

manifest.json
  top level : generator_version, schema_version, source, source_sha256, dpi,
              n_pieces_requested, master_seed, page_seed_rule,
              noise_strength, noise_scale, lossy, environment, total_pieces
  pages[]   : index, seed, width, height, adjacency, pieces[]
  pieces[]  : file, x, y, w, h   (x, y = top-left offset on the page canvas)

Each page carries `adjacency`: a list of [i, j] piece-index pairs that share a
torn border (4-connectivity, undirected, i < j). Use as positive pairs for
pairwise/graph-based stitching models; any pair not listed is a negative.

Reproducing: the partition is a pure function of (dpi, n_pieces_requested,
page seed, noise_strength, noise_scale). Page seeds follow `page_seed_rule`.
Fragment pixels also depend on the PDF renderer; see `environment`.

To reassemble a page (stitching ground truth):

""" + "\n".join("    " + ln if ln else "" for ln in REASSEMBLY_SNIPPET.splitlines()) + """

With lossless export (lossy=false) this reproduces the rendered page
pixel-exactly. Caveat: fragment pixels that are pure black (0, 0, 0) cannot be
told apart from the background, so `rgb.any(axis=2)` slightly under-estimates a
fragment's true mask where black ink touches a tear. Reassembly is unaffected
(those pixels are black on the page too).
"""
