# dm-bench v0.1

A versioned, reproducible benchmark for torn-document reassembly, built on
Dataset-Maker's strict-partition tearing. It ships puzzles (fragments without
answers), an evaluation harness, and a deterministic baseline solver.

Everything here runs on a laptop and in CI: NumPy, SciPy, Pillow and PyMuPDF
only (the versions pinned in `requirements.txt`), no GPU.

## Quick start

```bash
python -m src.bench build --out dm-bench-v0.1 --workers 8        # val, test-dev (+ test with the secret)
python -m src.bench verify dm-bench-v0.1
python -m src.bench solve --release dm-bench-v0.1 --out solutions --workers 8
python -m src.bench eval  --release dm-bench-v0.1 --solutions solutions --out results.json
```

`train` is not shipped: it is regenerable on demand
(`build --splits train`), so learned methods can use as much as they need.

## What a puzzle is

One A4 page (827 x 1169 px, 100 DPI) of a synthetic document, torn into
fragments. The solver receives only the RGBA fragments and the canvas size;
it must output a rigid pose for every fragment.

### Documents

Synthetic, seeded, licence-clean (`src/bench/docgen.py`): Base-14 fonts, a
layout grammar of headings, paragraphs, lists, tables, bar/line charts and
ruled notes; 1 to 3 pages per document. No document is downloaded. They are
simpler than real documents; see Limitations.

### Tiers

| tier   | fragments | warp (strength/scale px) | rotation     | pixel corruption                         | missing        | erosion |
|--------|-----------|--------------------------|--------------|------------------------------------------|----------------|---------|
| easy   | 8 to 12   | 20 / 96                  | none         | none                                     | none           | none    |
| medium | 12 to 20  | 28 / 96                  | uniform 0-360 | integer noise (4 x U{-2..2}), JPEG q85   | none           | none    |
| hard   | 20 to 32  | 32 / 80                  | uniform 0-360 | blur r0.6, noise (4 x U{-4..4}), JPEG q70 | Binomial(n, 0.1), at least 1 | 1 to 3 px |

Every fragment is a single connected piece: a tear producing a
multi-component piece is re-drawn deterministically (the attempt count is
recorded). Tiers use disjoint document pools.

Per-fragment order is normative (`src/bench/fragments.py`), so no stage leaks
pose information:

1. crop the clean page with the piece mask (the page is never corrupted
   before tearing, so noise never correlates across seams);
2. erosion (paper loss) on torn borders only, never the page border; lost RGB
   is zeroed; the largest component is kept;
3. rotation by an integer millidegree angle with an exact affine (RGB
   bilinear, alpha nearest);
4. tight crop to alpha plus a random 0 to 8 px margin per side, so the canvas
   size says nothing about the angle;
5. blur, noise, JPEG in the fragment's own frame;
6. RGB zeroed wherever alpha is 0; alpha is exactly {0, 255}.

### Splits and the held-out test

| split    | docs per tier | seeded by | answers shipped |
|----------|---------------|-----------|-----------------|
| train    | 120 (not shipped) | public entropy | built on demand |
| val      | 20            | public entropy | yes (tuning split) |
| test-dev | 10            | public entropy | yes |
| test     | 30            | 128-bit secret | no |

With public seeds anyone could regenerate answers, so the `test` split
derives every random choice (document, tear, rotation, noise, ids) from a
secret held by the maintainer (`DM_BENCH_TEST_SECRET`). The release ships
test puzzles only, plus a commitment in `benchmark.json`:
`secret_sha256 = sha256("dm-bench/0.1.0/test/" + secret)` and the sha256 of the
canonical test answers. `eval` regenerates test answers from the secret. The
secret is revealed when v0.1 is retired, so the commitment can be checked.

## Release layout

```
benchmark.json                       spec, per-split counts and stats, build env; no timestamps
SHA256SUMS                           byte hashes (download integrity; sha256sum -c compatible)
CONTENT.sha256                       decoded-content digests (regeneration checks)
puzzles/<tier>/<split>/<page_id>/
    puzzle.json                      {schema, page_id, tier, canvas, fragments: [{id, file, w, h}]}
    f000.png ...                     RGBA fragments
answers/<tier>/<split>/<page_id>.json   public splits only
```

- `page_id` is an opaque token; fragment ids `f000..` are a random permutation
  assigned after missing pieces are dropped, so neither order nor gaps leak.
- Answer: `{doc, page, tear_seed, tear_attempts, n_pieces, fragments: {id:
  {affine, rot_mdeg, erosion_px, ink_frac}}, missing: [{bbox, area}],
  adjacency: [[id_a, id_b, shared_edge_px], ...]}`.
- Release id = sha256 of `SHA256SUMS`; `eval` cites it in every result file.

### Pose convention

A pose is a 2x3 affine `A` mapping fragment pixel centres (x right, y down,
integer coordinates) to page pixel centres: `p = A @ [x, y, 1]`. Ground truth
is exact: mapping every opaque pixel of every fragment through its answer
affine lands on its own partition label for more than 99.99% of pixels
(boundary rounding only; 100% on easy), enforced by
`tests/test_bench_fragments.py`.

## Solutions

`solutions/<tier>/<split>/<page_id>.json`:

```json
{"schema": "dm-bench/solution/1", "page_id": "...", "solver": "name@version",
 "fragments": {"f000": {"affine": [[a, b, tx], [c, d, ty]]}},
 "candidates": {"f000": ["f007", "f003"]}}
```

Poses may be expressed in any global frame. `candidates` (ranked partner ids)
is optional and only feeds Hit@k/MRR.

## Metrics (eval_version 1.0)

tau = 1% of page width (8.27 px). For a fragment, the error under a global map
G is the worst displacement of any convex-hull point of its mask:
`max_p |G A_pred p - A_gt p|`.

- **direct_acc**: share of present fragments within tau under the best global
  rigid alignment. Alignment: every placed fragment's pose gives a hypothesis,
  refit by Kabsch on its inliers until stable (deterministic LO-RANSAC). Fewer
  than 2 inliers scores 0, so random or single-fragment submissions score
  exactly 0.
- **neighbor_acc**: share of ground-truth adjacent pairs (shared edge at least
  2*tau) whose two fragments fit one rigid map within tau. Symmetric, and a
  perfect page always scores 1.
- **perfect**: direct_acc = 1.
- **adjacency P/R/F1**: contacts between the placed masks (within 2 + 2 x
  erosion px) against ground truth; computed for every solver.
- **Hit@1, Hit@5, MRR**: from `candidates`, if given.
- **tau curve**: direct and neighbor accuracy at 0.25, 0.5, 1 and 2% of page
  width, plus their mean (AUC).

Poses must be rigid (finite, R^T R = I within 1e-3, det > 0); mirrored,
sheared, singular or unknown entries count as unplaced and are reported as
`invalid_poses`. A page without a solution file scores 0. Means come with a
95% bootstrap CI that resamples documents (B = 10000, fixed seed), and pages
are sorted before aggregation, so results do not depend on file order.

## Baselines

| method | what it is |
|--------|-----------|
| `oracle` | reads the answers; harness self-check only (must score 1.0), refused on test |
| `random` | uniform random rigid poses; the floor (0 by construction) |
| `edge-greedy@0.1` | the baseline (`src/bench/solver.py`) |

edge-greedy: Moore-traced contours cut into ~96 px windows; complementary
windows (one traversed in reverse) aligned by closed-form Kabsch; the best 12
refined by boundary ICP and scored by seam length, fit, colour continuity just
inside each side and an overlap penalty; straight page-border runs excluded;
greedy placement with an occupancy check, stuck fragments seed new clusters.
It reads only puzzle directories; a test forbids imports of the generator or
harness. Window length, hypothesis count and border tolerance were tuned on
`val` only.

### Results

`edge-greedy@0.1`, eval_version 1.0, public splits of dm-bench 0.1.0 built at
commit e2faa55 on macOS arm64 (release sha256 `877b489cfd8e8ad8...` for that
local build; the canonical release is built in CI and will have its own
digest, at least because `benchmark.json` records the build environment).
157 pages solved in 6.6 min
with 8 workers on an Apple M4.

| tier | split | pages (docs) | direct_acc [95% CI] | neighbor_acc [95% CI] | perfect | adjacency F1 | Hit@1 |
|------|-------|--------------|---------------------|-----------------------|---------|--------------|-------|
| easy | test-dev | 13 (10) | 0.973 [0.927, 1.000] | 0.979 [0.942, 1.000] | 0.846 | 0.984 | 1.000 |
| medium | test-dev | 16 (10) | 0.673 [0.555, 0.765] | 0.687 [0.570, 0.782] | 0.125 | 0.743 | 0.945 |
| hard | test-dev | 19 (10) | 0.221 [0.203, 0.243] | 0.626 [0.564, 0.686] | 0.000 | 0.695 | 0.913 |
| easy | val | 39 (20) | 0.989 [0.980, 0.998] | 0.993 [0.986, 1.000] | 0.897 | 0.985 | 1.000 |
| medium | val | 37 (20) | 0.824 [0.764, 0.888] | 0.823 [0.764, 0.883] | 0.432 | 0.861 | 0.975 |
| hard | val | 33 (20) | 0.255 [0.228, 0.286] | 0.707 [0.659, 0.761] | 0.000 | 0.800 | 0.940 |

- `test-dev` is the number to cite: solver parameters were tuned on `val`, and
  the medium-tier gap (0.824 on val vs 0.673 on test-dev) shows why.
- `random` scores 0.000 on every metric, tier and split (the floor).
- Held-out `test` scores are published once the canonical release is built
  with the secret; none are claimed here.
- Reading the table: easy is close to solved; hard is not. On hard, pairwise
  matching is still good (Hit@1 above 0.9) but greedy global assembly breaks
  down, which is where better methods have room.

## Reproducibility

- Every page is a pure function of (entropy, tier, doc, page) through keyed
  `SeedSequence` streams; no Python `hash()`, no call-order dependence. A
  `--limit-docs` slice is a byte subset of the full build.
- Stored angles are integers; cos/sin are correctly rounded (Decimal series),
  because host libm differs by 1 ulp on some angles. Noise is an integer
  Irwin-Hall sum; blur is Pillow's box-based Gaussian; JPEG uses fixed
  settings. JSON is canonical (sorted keys, 9-decimal floats, no NaN).
- Rendering bypasses the app's letterbox resample: pages are sized to render
  at exactly 827 x 1169 px.
- `SHA256SUMS` guards download integrity; `CONTENT.sha256` hashes decoded
  pixels (RGB zeroed under alpha) and canonical JSON, so an encoder or zlib
  change does not look like a data change.
- CI builds the same slice in two processes with different hash seeds, thread
  counts, worker counts and output paths and requires identical files;
  `tests/test_bench_golden.py` pins one page per tier.
- Output `workers > 1` is byte-identical to sequential builds.

Checked locally on macOS arm64 (Python 3.11). The golden digests were computed
there, so the CI run on Linux x86_64 (Python 3.10) is the cross-platform check
for everything the golden test and the determinism job cover.

## What the numbers do and do not mean

Defensible:
- ground truth is exact for these synthetic tears;
- the benchmark regenerates bit-for-bit within the pinned environment;
- the baseline's scores above, with their CIs, on these tiers.

Not defensible (yet):
- that synthetic tears or scans look like real ones: the tear model is a
  warped Voronoi partition, masks are given exactly (segmentation is assumed
  solved), and documents are synthetic;
- that methods trained or tuned here transfer to real torn documents: no
  real-scan set exists yet;
- comparison with numbers from other papers, which use different data and
  metrics.

## Limitations and known stats

- A share of fragments is nearly blank (< 1% ink); `benchmark.json` reports
  it per tier and split. Such fragments can only be placed by shape.
- Tears are planar and complementary except for hard-tier erosion.
- No mixed-pages tier yet: a future version can add a `bag` puzzle unit
  (fragments from several pages, grouping metrics) without changing v0.1
  files.
- Real-scan validation set: planned, requires printing, tearing and scanning
  pages.

## Versions

| identifier | meaning |
|-----------|---------|
| `benchmark.json` `version` (0.1.0) | the data; any content change needs a new version |
| `generator_version` | Dataset-Maker code version, informational |
| `schemas` | puzzle/answer/solution JSON formats |
| `eval_version` (1.0) | metric definitions, tau, bootstrap |
| `edge-greedy@0.1` | baseline version |
