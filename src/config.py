"""Central configuration: A4 geometry, defaults, limits.

All tunables live here so the UI, workers and tests share one source of truth.
"""
from __future__ import annotations

# --- A4 geometry -----------------------------------------------------------
# A4 = 210 x 297 mm. Pixel size scales with DPI: px = mm / 25.4 * dpi.
A4_MM = (210.0, 297.0)
A4_ASPECT = A4_MM[1] / A4_MM[0]  # height / width ~= 1.4142

DEFAULT_DPI = 150           # 150 DPI A4 -> 1240 x 1754 px. Good detail/size trade-off.
MIN_DPI, MAX_DPI = 72, 300

# --- Tearing defaults ------------------------------------------------------
DEFAULT_PIECES = 16         # pieces per page (n^2 grid feel -> here a flat count)
MIN_PIECES, MAX_PIECES = 2, 256

DEFAULT_NOISE_STRENGTH = 28.0   # px of boundary displacement (the "tear" jaggedness)
DEFAULT_NOISE_SCALE = 96.0      # px wavelength of the noise (bigger = smoother tears)

# --- Performance / limits --------------------------------------------------
MAX_PAGES_PER_PDF = 60      # guardrail for the HF free tier (CPU/RAM bound)
MAX_UPLOAD_MB = 50
QUEUE_MAX_SIZE = 32         # Gradio request queue cap
WORKER_CONCURRENCY = 1      # HF free tier = 2 vCPU; keep 1 heavy job at a time


def a4_pixels(dpi: int) -> tuple[int, int]:
    """Return (width, height) in px for an A4 page at the given DPI."""
    w = round(A4_MM[0] / 25.4 * dpi)
    h = round(A4_MM[1] / 25.4 * dpi)
    return w, h
