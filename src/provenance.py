"""Reproducibility bookkeeping: per-page seed rule, input hash, environment.

What is (and is not) deterministic:
  * The partition of a page (labels, offsets, adjacency) is a pure function of
    (page size, n_pieces, page seed, noise_strength, noise_scale). Page size is
    fixed by DPI (A4), so PDF *content* never influences where the tears go.
  * Fragment *pixels* additionally depend on the PDF renderer (PyMuPDF
    version), so they are reproducible only within the recorded environment.
  * ZIP bytes are not reproducible: `created_utc` and zip entry timestamps
    change on every run.

Everything needed to regenerate a dataset is written into the manifest.
"""
from __future__ import annotations

import hashlib
import platform

# Bump when manifest fields are added/changed. 1.0 = original schema
# (implicit, unversioned). 1.1 adds provenance fields only; every 1.0 field is
# unchanged.
MANIFEST_SCHEMA_VERSION = "1.1"

PAGE_SEED_RULE = "(master_seed * 1000003 + page_index) & 0x7FFFFFFF"


def page_seed(master_seed: int, page_index: int) -> int:
    """Per-page RNG seed: changes page to page, reproducible for a master seed."""
    return (int(master_seed) * 1_000_003 + int(page_index)) & 0x7FFFFFFF


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, streamed so large PDFs aren't held twice."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict[str, str]:
    """Library versions that can change generated output."""
    import fitz
    import numpy
    import PIL
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "pymupdf": fitz.VersionBind,
    }
