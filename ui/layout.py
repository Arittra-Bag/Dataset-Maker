"""Page structure. Mirrors the workflow: source → configure → generate →
inspect ground truth → export. Widgets only; behaviour is in `handlers`."""
from __future__ import annotations

import gradio as gr

from src import config
from src.packager import REASSEMBLY_SNIPPET

from . import content, handlers
from .theme import CSS, build_theme


def _section(label: str) -> None:
    gr.HTML(f'<div class="dm-section">{label}</div>')


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        theme=build_theme(),
        title="Dataset-Maker",
        css=CSS,
        # Gradio copies every returned file/image into its cache and never
        # frees it by default; sweep it on the same TTL as our temp files.
        delete_cache=(config.TEMP_FILE_TTL_S, config.TEMP_FILE_TTL_S),
    ) as demo:
        gr.HTML(content.HEADER_HTML)
        with gr.Accordion("How each guarantee is enforced", open=False):
            gr.HTML(content.GUARANTEES_HTML)

        # Previews of uploaded pages live in session state; expire it (and the
        # session's export) on the same TTL instead of keeping idle tabs forever.
        state = gr.State(None, time_to_live=config.TEMP_FILE_TTL_S,
                         delete_callback=handlers.release_view)

        with gr.Row(equal_height=False):
            # ---------------- left: source + configuration ----------------
            with gr.Column(scale=4, min_width=320):
                _section("1 · Source")
                pdf_in = gr.File(label="PDF", file_types=[".pdf"], type="filepath", height=120)
                sample_btn = gr.Button("Load sample PDF", size="sm", variant="secondary")

                _section("2 · Tear configuration")
                n_pieces = gr.Slider(
                    config.MIN_PIECES, config.MAX_PIECES, config.DEFAULT_PIECES, step=1,
                    label="Fragments per page",
                    info="Voronoi seeds (Poisson-disk). Produced count is reported and can be lower.",
                )
                seed = gr.Number(
                    value=0, precision=0, label="Master seed",
                    info="Page seed = (seed × 1000003 + page_index) & 0x7FFFFFFF. Recorded in the manifest.",
                )
                with gr.Accordion("Tear geometry", open=True):
                    noise_strength = gr.Slider(
                        config.MIN_NOISE_STRENGTH, config.MAX_NOISE_STRENGTH,
                        config.DEFAULT_NOISE_STRENGTH, step=1,
                        label="Edge displacement (px) · noise_strength",
                        info="Domain-warp amplitude. 0 = straight Voronoi edges.",
                    )
                    noise_scale = gr.Slider(
                        config.MIN_NOISE_SCALE, config.MAX_NOISE_SCALE,
                        config.DEFAULT_NOISE_SCALE, step=1,
                        label="Edge wavelength (px) · noise_scale",
                        info="Base wavelength of the warp noise. Displacement far above "
                             "wavelength folds the warp → multi-component fragments (reported).",
                    )
                with gr.Accordion("Rendering and output", open=False):
                    dpi = gr.Slider(
                        config.MIN_DPI, config.MAX_DPI, config.DEFAULT_DPI, step=1,
                        label="Render DPI",
                    )
                    dpi_md = gr.HTML(handlers.dpi_note(config.DEFAULT_DPI))
                    lossy = gr.Checkbox(
                        value=False, label="Lossy palette PNG (64 colours)",
                        info="Smaller ZIP. Offsets stay exact; pixel values become approximate.",
                    )

                _section("3 · Generate")
                with gr.Row():
                    run_btn = gr.Button("Generate dataset", variant="primary",
                                        elem_id="dm-generate", scale=3)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            # ---------------- right: inspect + export ----------------
            with gr.Column(scale=9, min_width=480):
                _section("4 · Inspect ground truth  ·  5 · Export")
                with gr.Row(equal_height=True):
                    summary = gr.HTML(content.EMPTY_SUMMARY_HTML, elem_id="dm-summary")
                    page_dd = gr.Dropdown(
                        choices=[], value=None, label="Page (Preview, Ground truth)",
                        interactive=False, scale=0, min_width=300,
                    )
                with gr.Tabs():
                    with gr.Tab("Preview"):
                        with gr.Row():
                            src_img = gr.Image(label="Source page (A4 render)", type="numpy",
                                               interactive=False, height=560)
                            part_img = gr.Image(label="Partition map · tear lines + piece index",
                                                type="numpy", interactive=False, height=560)
                        gallery = gr.Gallery(
                            label=f"Fragments (preview resolution, first {handlers.MAX_THUMBNAILS})",
                            columns=6, height=300, object_fit="contain", elem_id="dm-gallery",
                        )
                    with gr.Tab("Ground truth"):
                        with gr.Row():
                            adj_img = gr.Image(label="Adjacency graph · node = fragment, edge = shared tear",
                                               type="numpy", interactive=False, height=560, scale=5)
                            with gr.Column(scale=4):
                                page_meta = gr.HTML("")
                                adj_code = gr.Code(label="pages[i].adjacency", language="json",
                                                   interactive=False, lines=10)
                        gt_table = gr.Dataframe(
                            headers=handlers.GT_HEADERS,
                            datatype=["number"] * 8 + ["str"],
                            label="Per-fragment ground truth (full-resolution px; x, y = top-left offset on page)",
                            interactive=False, wrap=False, height=320, elem_id="dm-gt-table",
                        )
                    with gr.Tab("manifest.json"):
                        with gr.Row():
                            with gr.Column(scale=5):
                                manifest_code = gr.Code(
                                    label="manifest.json of this run (arrays truncated for display)",
                                    language="json", interactive=False, lines=28,
                                )
                            with gr.Column(scale=4):
                                gr.HTML(content.MANIFEST_FIELDS_HTML)
                    with gr.Tab("Export"):
                        with gr.Row():
                            with gr.Column(scale=5):
                                zip_out = gr.File(label="Download dataset (.zip)", interactive=False)
                                zip_table = gr.Dataframe(
                                    headers=["path", "files", "bytes"],
                                    datatype=["str", "number", "number"],
                                    label="Archive contents of this run", interactive=False,
                                    height=220,
                                )
                                gr.HTML(content.EXPORT_LAYOUT_HTML)
                            with gr.Column(scale=4):
                                gr.Code(
                                    value=REASSEMBLY_SNIPPET, language="python",
                                    label="Reassemble pages from the export (also in README.txt)",
                                    interactive=False,
                                )
                                gr.HTML('<div class="dm-section">Known limitations</div>'
                                        + content.LIMITATIONS_HTML)

        gr.HTML(content.SCOPE_HTML)

        # ---------------- events ----------------
        sample_btn.click(handlers.load_sample, None, pdf_in, show_progress="hidden")
        dpi.change(handlers.dpi_note, dpi, dpi_md, show_progress="hidden")

        run_btn.click(
            handlers.generate,
            inputs=[pdf_in, dpi, n_pieces, noise_strength, noise_scale, lossy, seed, state],
            outputs=[state, src_img, part_img, gallery, zip_out],
            concurrency_limit=config.WORKER_CONCURRENCY,  # heavy job throttle
            api_name="generate",
        ).then(
            handlers.after_generate,
            inputs=state,
            outputs=[summary, page_dd, adj_img, gt_table, page_meta, adj_code,
                     manifest_code, zip_table],
            show_progress="hidden",
        )
        page_dd.input(
            handlers.show_page,
            inputs=[state, page_dd],
            outputs=[src_img, part_img, adj_img, gallery, gt_table, page_meta, adj_code],
            show_progress="minimal",
        )
        clear_btn.click(
            handlers.clear_session,
            inputs=state,
            outputs=[pdf_in, state, src_img, part_img, adj_img, gallery, gt_table,
                     page_meta, adj_code, manifest_code, zip_out, zip_table, summary, page_dd],
            show_progress="hidden",
        )
    return demo
