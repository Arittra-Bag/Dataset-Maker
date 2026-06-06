"""Orchestration: PDF bytes -> torn pages, scheduled through the priority queue.

Kept UI-free so it is unit-testable and reusable from a CLI or batch worker.
"""
from __future__ import annotations

from typing import Callable

from . import config, workspace
from .pdf_loader import load_pdf_pages
from .queue_manager import PriorityJobQueue, page_priority
from .tearing import TornPage, tear_page


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
        seed = (master_seed * 1_000_003 + idx) & 0x7FFFFFFF
        results[idx] = tear_page(
            page_img,
            n_pieces=n_pieces,
            seed=seed,
            noise_strength=noise_strength,
            noise_scale=noise_scale,
        )
        done += 1
        if progress:
            progress(done / total, f"Torn page {done}/{total}")

    # Return in document order for a coherent manifest.
    return [results[i] for i in sorted(results)]


def save_temp_pdf(file_bytes: bytes) -> str:
    """Persist uploaded bytes to a tracked temp file PyMuPDF can open."""
    path = workspace.new_temp(suffix=".pdf")
    with open(path, "wb") as fh:
        fh.write(file_bytes)
    return path
