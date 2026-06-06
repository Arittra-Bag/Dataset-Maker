"""Package torn pieces + stitching ground-truth into a downloadable ZIP.

Layout inside the archive:
    pieces/page_0001/piece_000.png ...
    manifest.json          # global summary + per-piece placement (x, y, w, h)
    README.txt             # how to reassemble

The manifest IS the dataset label: each piece's (x, y) offset on its page is the
exact stitching target. Reassembling = paste every piece at its offset.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

from .optimizer import encode_piece
from .tearing import TornPage


def build_zip(
    pages: list[TornPage],
    *,
    source_name: str,
    dpi: int,
    noise_strength: float,
    noise_scale: float,
    lossy: bool,
) -> tuple[bytes, dict]:
    """Return (zip_bytes, manifest_dict) for a list of torn pages."""
    manifest = {
        "generator": "Dataset-Maker",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_name,
        "dpi": dpi,
        "noise_strength": noise_strength,
        "noise_scale": noise_scale,
        "lossy": lossy,
        "pages": [],
        "total_pieces": 0,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pi, page in enumerate(pages):
            pdir = f"pieces/page_{pi + 1:04d}"
            page_entry = {
                "index": pi,
                "width": page.width,
                "height": page.height,
                "pieces": [],
            }
            for k, piece in enumerate(page.pieces):
                fname = f"{pdir}/piece_{k:03d}.png"
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


_README = """Dataset-Maker export
=====================
Each page was torn into NON-OVERLAPPING fragments (a strict partition: every
pixel belongs to exactly one piece). Fragments sit on a black background.

To reassemble a page (stitching ground truth):
    import json
    from PIL import Image
    m = json.load(open("manifest.json"))
    for page in m["pages"]:
        canvas = Image.new("RGB", (page["width"], page["height"]))
        for p in page["pieces"]:
            piece = Image.open(p["file"])
            canvas.paste(piece, (p["x"], p["y"]), mask=...)  # non-black pixels
        canvas.save(f"reassembled_{page['index']}.png")
"""
