# Dataset-Maker — agent guide

Gradio web app (HuggingFace Spaces, free tier) that tears PDF pages into
**non-overlapping** torn fragments for image-stitching datasets.

## Run / test
```bash
pip install -r requirements.txt
python app.py        # serves on :7860
pytest -q            # tests/ : partition invariants + queue ordering
```
`requirements.txt` is pinned for HF Spaces' Python 3.10. Locally on newer
Pythons you may need unpinned numpy/scipy/pillow/pymupdf.

## Architecture (data flow)
`app.py` → `src/pipeline.py` → `pdf_loader` → priority `queue_manager` →
`tearing` → `packager`.

- **`src/tearing.py` is the core.** Invariant: output is a strict PARTITION of
  the page (every pixel in exactly one piece → no overlap). Implemented as
  nearest-seed `argmin` (Voronoi) over a value-noise **domain-warped** pixel
  grid. Warp the *query coords*, never the partition rule, or you break the
  no-overlap guarantee. `verify_partition()` gates this at runtime.
- **`src/queue_manager.py`** — binary min-heap priority queue. Documented Big-O
  in the module docstring; keep `push`/`pop` at `O(log n)`.
- **`src/noise.py` / `src/sampling.py`** — pure NumPy, no Perlin/extra deps.
- Per-page seed = `master_seed*1_000_003 + page_index` → randomness changes per
  page yet stays reproducible. Don't make it global.

## Conventions
- Keep `src/` UI-free (no `gradio` imports) so it stays testable. Gradio lives
  only in `app.py`.
- Images are `(H, W, 3)` uint8 RGB end-to-end; pieces use black background.
- `manifest.json` `(x, y)` offsets ARE the stitching labels — don't change the
  schema without updating `packager.py` README.txt + tests.

## Deploy
HF Spaces reads the YAML header in `README.md` (`sdk: gradio`, `app_file`).
Push to the Space repo; Spaces installs `requirements.txt` and runs `app.py`.
