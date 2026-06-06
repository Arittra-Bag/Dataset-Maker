---
title: Dataset Maker
emoji: 🧩
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🧩 Dataset-Maker

Turn any PDF into a **non-overlapping torn-fragment dataset** for image
stitching / fragment-reassembly research. Each page is rendered to A4, torn into
irregular organic pieces on a black background, and exported as a ZIP with
stitching ground truth.

Built to run on the **HuggingFace Spaces free tier** (Gradio SDK).

## The guarantee: zero overlap

The pieces form a strict **partition** of the page — every pixel belongs to
**exactly one** fragment, so pieces never overlap and together cover the page
exactly. That makes the `(x, y)` offset of each piece an exact stitching label.

How: instead of drawing blobs (which overlap), we assign each pixel to its
**nearest seed point** (`argmin`). That is a Voronoi tessellation — already a
perfect partition. To make edges look *torn* instead of straight, we
**domain-warp** the pixel grid with value noise *before* the nearest-seed test.
Warping the query points (not the rule) keeps the result a partition while
boundaries become jagged and organic.

Inspired by Voronoi-jigsaw / eroded-boundary fragment literature:
[Eroded Boundaries](https://arxiv.org/pdf/1912.00755),
[Pairwise Irregular Fragments](https://arxiv.org/pdf/2507.09767),
[Deepzzle](https://arxiv.org/abs/2005.12548).

## Pipeline

```
PDF ──▶ render @DPI ──▶ fit/slice to A4 ──▶ priority queue (cheap pages first)
                                                   │
                            per-page seed ─────────▼
                          domain-warped Voronoi partition (no overlap)
                                                   │
                            crop to bbox, black bg │
                                                   ▼
                       PNG-optimize ──▶ ZIP + manifest.json (x,y labels)
```

## Performance notes

* **Gradio queue** — `demo.queue(max_size, default_concurrency_limit)` caps load
  for the 2-vCPU free tier; the heavy event has its own `concurrency_limit`.
* **Priority queue** (`src/queue_manager.py`) — binary min-heap, `push`/`pop`
  `O(log n)`, `peek` `Θ(1)`, space `Θ(n)`. Orders page jobs cheap-first so the
  user sees early progress and tail latency drops. Best case `Ω(1)` per op; no
  comparison heap beats `Ω(log n)` amortized under interleaved ops.
* **Vectorized partition** — SciPy `cKDTree` nearest-seed query is
  `O(H·W·log S)`; mask extraction is `Θ(H·W)`.
* **Export** — tight bbox crop + PNG `optimize`/`compress_level=9`; optional
  median-cut palette for smaller archives.

## Output layout

```
pieces/page_0001/piece_000.png   # fragment on black bg
manifest.json                    # per-piece {file, x, y, w, h} = stitching GT
README.txt                       # reassembly snippet
```

## Run locally

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:7860
pytest -q                # invariant + queue tests
```

## Layout

```
app.py                 Gradio UI, queue config, theme
src/pdf_loader.py      PDF -> A4 RGB pages (PyMuPDF), tall-page slicing
src/sampling.py        Bridson Poisson-disk seed sampling
src/noise.py           Vectorized value noise (domain warp)
src/tearing.py         Voronoi partition + warp + piece extraction  ← core
src/optimizer.py       PNG optimization / quantization
src/queue_manager.py   Priority job queue (min-heap)
src/packager.py        ZIP + manifest builder
src/pipeline.py        PDF -> torn pages orchestration
tests/                 partition no-overlap + queue ordering
```

## License

MIT
