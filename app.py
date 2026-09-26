"""Dataset-Maker - Gradio entry point (HuggingFace Spaces `app_file`).

Upload a PDF -> each page is rendered to A4, torn into NON-OVERLAPPING fragments
on a black background, and packaged as a ZIP with stitching ground truth.

Layout / handlers live in `ui/`; generation and measurement in `src/`.

Performance:
  * Gradio `.queue()` caps concurrent requests for the 2-vCPU free tier.
  * A priority queue (src/queue_manager.py) orders page jobs.
  * NumPy/SciPy vectorized partition; PNG-optimized export.
"""
from __future__ import annotations

from src import config
from ui import build_ui

demo = build_ui()
demo.queue(
    max_size=config.QUEUE_MAX_SIZE,
    default_concurrency_limit=config.WORKER_CONCURRENCY,
)

if __name__ == "__main__":
    # No public share link: uploads stay on this machine. Spaces ignores `share`.
    demo.launch()
