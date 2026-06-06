"""Dataset-Maker — Gradio web app (HuggingFace Spaces ready).

Upload a PDF -> each page is rendered to A4, torn into NON-OVERLAPPING fragments
on a black background, and packaged as a ZIP with stitching ground truth.

Performance:
  * Gradio `.queue()` caps concurrent requests for the 2-vCPU free tier.
  * A priority queue (src/queue_manager.py) orders page jobs cheap-first.
  * NumPy/SciPy vectorized partition; PNG-optimized export.
"""
from __future__ import annotations

import gradio as gr
import numpy as np

from src import config, workspace
from src.optimizer import encode_preview
from src.packager import build_zip
from src.pipeline import process_pdf, save_temp_pdf
from src.tearing import verify_partition


def _resolve_theme(name: str):
    """Resolve a registry theme, falling back gracefully across Gradio versions.

    Some themes (Ocean, Citrus) only exist in Gradio 5+. On older Gradio we fall
    back to Default rather than crashing at startup.
    """
    cls_name, kwargs = config.THEME_REGISTRY.get(
        name, config.THEME_REGISTRY[config.DEFAULT_THEME]
    )
    cls = getattr(gr.themes, cls_name, None) or getattr(gr.themes, "Default")
    return cls(**kwargs)


def available_themes() -> list[str]:
    """Registry themes actually present in the installed Gradio build."""
    return [
        name for name, (cls, _) in config.THEME_REGISTRY.items()
        if getattr(gr.themes, cls, None) is not None
    ]


def _pieces_gallery(pages, max_pieces: int = 60):
    """Flatten a few torn pieces for the preview gallery (downscaled)."""
    out = []
    for pi, page in enumerate(pages):
        for k, piece in enumerate(page.pieces):
            out.append((encode_preview(piece.rgb, 256), f"p{pi+1}·{k}"))
            if len(out) >= max_pieces:
                return out
    return out


def generate(
    pdf_file,
    dpi: int,
    n_pieces: int,
    noise_strength: float,
    noise_scale: float,
    lossy: bool,
    seed: int,
    progress=gr.Progress(),
):
    """Main event handler: PDF -> (status, gallery, zip path)."""
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")

    progress(0.02, desc="Reading PDF…")
    with open(pdf_file, "rb") as fh:
        pdf_bytes = fh.read()
    if len(pdf_bytes) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise gr.Error(f"PDF exceeds {config.MAX_UPLOAD_MB} MB limit.")

    tmp_pdf = save_temp_pdf(pdf_bytes)
    pages = process_pdf(
        tmp_pdf,
        dpi=int(dpi),
        n_pieces=int(n_pieces),
        noise_strength=float(noise_strength),
        noise_scale=float(noise_scale),
        master_seed=int(seed),
        progress=lambda f, m: progress(0.05 + 0.8 * f, desc=m),
    )

    # Verify the no-overlap invariant on the first page (sanity gate).
    report = verify_partition(pages[0])
    if not report["is_partition"]:
        raise gr.Error(
            f"Partition check failed: overlap={report['max_overlap']}, "
            f"uncovered={report['uncovered_pixels']}"
        )

    progress(0.9, desc="Packaging ZIP…")
    zip_bytes, manifest = build_zip(
        pages,
        source_name="upload.pdf",
        dpi=int(dpi),
        noise_strength=float(noise_strength),
        noise_scale=float(noise_scale),
        lossy=lossy,
    )
    out_path = workspace.new_temp(suffix="_dataset.zip")
    with open(out_path, "wb") as fh:
        fh.write(zip_bytes)

    status = (
        f"✅ {len(pages)} pages · {manifest['total_pieces']} pieces · "
        f"no-overlap verified (max_overlap={report['max_overlap']}, "
        f"uncovered={report['uncovered_pixels']})"
    )
    progress(1.0, desc="Done")
    return status, _pieces_gallery(pages), out_path


def clear_all():
    """Delete tracked temp files (PDFs + ZIPs) and reset the UI outputs."""
    removed = workspace.clear_all()
    status = f"🧹 Cleared {removed} temp file(s). Upload a PDF and hit **Generate**."
    # outputs order: pdf_in, status, gallery, zip_out
    return None, status, None, None


def build_ui(theme_name: str = config.DEFAULT_THEME) -> gr.Blocks:
    with gr.Blocks(
        theme=_resolve_theme(theme_name),
        title="Dataset-Maker · Torn-page stitching dataset",
    ) as demo:
        gr.Markdown(
            "# 🧩 Dataset-Maker\n"
            "Tear PDF pages into **non-overlapping** torn fragments for "
            "image-stitching datasets. Every pixel lands in exactly one piece — "
            "guaranteed by a domain-warped Voronoi partition."
        )
        with gr.Row():
            with gr.Column(scale=1):
                pdf_in = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                n_pieces = gr.Slider(
                    config.MIN_PIECES, config.MAX_PIECES, config.DEFAULT_PIECES,
                    step=1, label="Pieces per page",
                )
                with gr.Accordion("Tearing controls", open=False):
                    noise_strength = gr.Slider(
                        0, 80, config.DEFAULT_NOISE_STRENGTH, step=1,
                        label="Tear jaggedness (px)",
                    )
                    noise_scale = gr.Slider(
                        8, 200, config.DEFAULT_NOISE_SCALE, step=1,
                        label="Tear smoothness (wavelength px)",
                    )
                    dpi = gr.Slider(
                        config.MIN_DPI, config.MAX_DPI, config.DEFAULT_DPI, step=1,
                        label="Render DPI",
                    )
                    seed = gr.Number(value=0, precision=0, label="Master seed")
                    lossy = gr.Checkbox(
                        value=False, label="Lossy palette PNG (smaller ZIP)"
                    )
                with gr.Row():
                    run = gr.Button("Generate dataset", variant="primary")
                    clear = gr.Button("Clear all", variant="secondary")
            with gr.Column(scale=2):
                status = gr.Markdown("Upload a PDF and hit **Generate**.")
                gallery = gr.Gallery(
                    label="Torn pieces (preview)", columns=6, height=420,
                    object_fit="contain",
                )
                zip_out = gr.File(label="Download dataset (.zip)")

        run.click(
            generate,
            inputs=[pdf_in, dpi, n_pieces, noise_strength, noise_scale, lossy, seed],
            outputs=[status, gallery, zip_out],
            concurrency_limit=config.WORKER_CONCURRENCY,  # heavy job throttle
        )
        clear.click(
            clear_all,
            inputs=None,
            outputs=[pdf_in, status, gallery, zip_out],
        )
        gr.Markdown(
            "Pieces sit on black backgrounds; `manifest.json` carries each "
            "piece's `(x, y)` offset = the stitching label."
        )
    return demo


demo = build_ui()
demo.queue(
    max_size=config.QUEUE_MAX_SIZE,
    default_concurrency_limit=config.WORKER_CONCURRENCY,
)

if __name__ == "__main__":
    demo.launch()
