"""Orchestration: PDF bytes -> torn pages, scheduled through the priority queue.

Kept UI-free so it is unit-testable and reusable from a CLI or batch worker.
`generate_dataset` is the single entry point a front-end should call: it tears,
verifies every page, packages, and reports measured stage timings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from . import workspace
from .inspection import PageReport, page_report
from .packager import build_zip
from .pdf_loader import load_pdf_pages
from .provenance import page_seed, sha256_file
from .queue_manager import PriorityJobQueue, page_priority
from .tearing import TornPage, tear_page


class PartitionError(RuntimeError):
    """A page failed the no-overlap / full-coverage check. Should never fire."""


def process_pdf(
    pdf_path: str,
    *,
    dpi: int,
    n_pieces: int,
    noise_strength: float,
    noise_scale: float,
    master_seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
) -> list[TornPage]:
    """Render + tear every page, ordered by the priority queue (cheap first)."""
    pages = load_pdf_pages(pdf_path, dpi)
    if not pages:
        raise ValueError("No renderable pages found in PDF.")

    queue = PriorityJobQueue()
    for idx, page_img in enumerate(pages):
        queue.push(page_priority(n_pieces, idx), payload=(idx, page_img))

    total = len(pages)
    results: dict[int, TornPage] = {}
    done = 0
    while True:
        job = queue.pop()
        if job is None:
            break
        idx, page_img = job.payload
        # Per-page seed -> randomness changes page by page, yet reproducible.
        results[idx] = tear_page(
            page_img,
            n_pieces=n_pieces,
            seed=page_seed(master_seed, idx),
            noise_strength=noise_strength,
            noise_scale=noise_scale,
        )
        done += 1
        if progress:
            progress(done / total, f"Torn page {done}/{total}")

    # Return in document order for a coherent manifest.
    return [results[i] for i in sorted(results)]


@dataclass
class DatasetRun:
    """Everything one generation produced, plus how long each stage took."""
    pages: list[TornPage]
    reports: list[PageReport]
    manifest: dict
    zip_bytes: bytes
    timings: dict[str, float]      # stage -> seconds (time.perf_counter)


def generate_dataset(
    pdf_path: str,
    *,
    source_name: str,
    dpi: int,
    n_pieces: int,
    noise_strength: float,
    noise_scale: float,
    master_seed: int,
    lossy: bool,
    progress: Callable[[float, str], None] | None = None,
) -> DatasetRun:
    """PDF -> torn pages -> per-page verification -> ZIP + manifest.

    Raises PartitionError if any page is not a strict partition.
    """
    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    timings: dict[str, float] = {}

    t = time.perf_counter()
    source_sha256 = sha256_file(pdf_path)
    pages = process_pdf(
        pdf_path,
        dpi=dpi,
        n_pieces=n_pieces,
        noise_strength=noise_strength,
        noise_scale=noise_scale,
        master_seed=master_seed,
        progress=lambda f, m: report(0.85 * f, m),
    )
    timings["render_and_tear"] = time.perf_counter() - t

    t = time.perf_counter()
    report(0.86, "Verifying partition on every page…")
    reports = [page_report(p, i, n_pieces) for i, p in enumerate(pages)]
    timings["verify"] = time.perf_counter() - t
    bad = [r for r in reports if not r.is_partition]
    if bad:
        r = bad[0]
        raise PartitionError(
            f"page {r.index}: max_overlap={r.max_overlap}, "
            f"uncovered={r.uncovered_pixels}"
        )

    t = time.perf_counter()
    report(0.9, "Packaging ZIP…")
    zip_bytes, manifest = build_zip(
        pages,
        source_name=source_name,
        dpi=dpi,
        noise_strength=noise_strength,
        noise_scale=noise_scale,
        lossy=lossy,
        master_seed=master_seed,
        n_pieces_requested=n_pieces,
        source_sha256=source_sha256,
    )
    timings["package"] = time.perf_counter() - t

    return DatasetRun(
        pages=pages,
        reports=reports,
        manifest=manifest,
        zip_bytes=zip_bytes,
        timings=timings,
    )


def save_temp_pdf(file_bytes: bytes) -> str:
    """Persist uploaded bytes to a tracked temp file PyMuPDF can open."""
    path = workspace.new_temp(suffix=".pdf")
    with open(path, "wb") as fh:
        fh.write(file_bytes)
    return path
