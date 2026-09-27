# Dataset-Maker - agent guide

Gradio web app (HuggingFace Spaces, free tier) that tears PDF pages into
**non-overlapping** torn fragments with exact ground truth, for
fragment-reassembly / image-stitching datasets.

## Run / test
```bash
pip install -r requirements.txt
python app.py        # serves on :7860, local only (no share link)
pytest -q            # partition, determinism, manifest, ZIP round-trip, UI smoke
python scripts/make_assets.py   # regenerate assets/sample.pdf + README figure
```
`requirements.txt` is pinned for HF Spaces' Python 3.10. Locally on newer
Pythons you may need unpinned numpy/scipy/pillow/pymupdf.

## Architecture (data flow)
`app.py` → `ui/` (layout, handlers) → `src/pipeline.generate_dataset` →
`pdf_loader` → priority `queue_manager` → `tearing` → `inspection.page_report`
(verifies EVERY page) → `packager`.

- **`src/tearing.py` is the core.** Invariant: output is a strict PARTITION of
  the page (every pixel in exactly one piece → no overlap). Implemented as
  nearest-seed `argmin` (Voronoi) over a value-noise **domain-warped** pixel
  grid. Warp the *query coords*, never the partition rule, or you break the
  no-overlap guarantee. `verify_partition()` gates this at runtime on every
  page; `generate_dataset` raises `PartitionError` on failure.
- **`src/pipeline.generate_dataset`** is the single entry point for any
  front-end (UI today, CLI/benchmark later). Returns pages, reports, manifest,
  zip bytes and measured stage timings.
- **`src/inspection.py`** computes everything the UI displays (per-page stats,
  reassembly, previews, manifest excerpt, zip listing). Read-only over
  generated data.
- **`src/provenance.py`** owns the page-seed rule, input hash, environment and
  `MANIFEST_SCHEMA_VERSION`.
- **`src/queue_manager.py`** - binary min-heap priority queue. Documented Big-O
  in the module docstring; keep `push`/`pop` at `O(log n)`.
- **`src/noise.py` / `src/sampling.py`** - pure NumPy, no Perlin/extra deps.
- Per-page seed = `(master_seed*1_000_003 + page_index) & 0x7FFFFFFF`
  (`provenance.page_seed`) → randomness changes per page yet stays
  reproducible. Don't make it global.

## Conventions
- Keep `src/` UI-free (no `gradio` imports) so it stays testable. Gradio lives
  only in `ui/` and `app.py`. Handlers format; they don't compute.
- Every claim in UI copy (`ui/content.py`) or README must be enforced by code
  or a test. Any number shown must come from a measured run. The app never
  shows reconstruction accuracy; benchmark numbers in docs must come from an
  `eval` results file (cite release_sha256, eval_version, solver@version).
- Images are `(H, W, 3)` uint8 RGB end-to-end; pieces use black background.
- `manifest.json` `(x, y)` offsets ARE the stitching labels. Schema changes:
  bump `MANIFEST_SCHEMA_VERSION`, update `packager._README`, README.md and
  `tests/test_pipeline.py`. Additive only unless deliberately versioned.
- `tests/test_partition.py::test_partition_golden_hash` pins exact output. If
  a change to noise/sampling/tearing or a dependency bump breaks it, published
  datasets can no longer be regenerated: update deliberately, never silently.
- No em/en dashes or emojis in copy, docs, commits or PR text.

## dm-bench (`src/bench/`, see docs/BENCHMARK.md)
`python -m src.bench build|verify|solve|eval`. docgen -> fragments -> build
-> evaluate; `solver.py` is the edge-greedy baseline.
- Every page is a pure function of (entropy, tier, doc, page) via keyed
  `canon.stream`; never `hash()`, `spawn()`, host trig in stored numbers, or
  non-canonical JSON. `tests/test_bench_golden.py` pins output: bump
  `BENCH_VERSION` instead of updating digests.
- The `test` split comes from `DM_BENCH_TEST_SECRET`; never commit, log or
  hard-code a real secret. Tests use a throwaway value.
- `solver.py` must stay answer-free (no imports of build/docgen/fragments/
  tearing/evaluate); a test enforces it. Tune only on `val`.

## Deploy
HF Spaces reads the YAML header in `README.md` (`sdk: gradio`, `app_file`).
Push to the Space repo; Spaces installs `requirements.txt` and runs `app.py`.
`assets/sample.pdf` is needed at runtime (Load sample PDF button).
