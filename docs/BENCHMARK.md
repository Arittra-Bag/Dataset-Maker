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
   is a function of the fragment's own mask plus independent padding and adds
   no information beyond the mask (an uncropped rotation canvas revealed the
   angle modulo 90 degrees exactly);
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
`secret_sha256 = sha256("dm-bench/0.1.0/test/" + secret.strip().lower())` and the sha256 of the
canonical test answers. `eval` regenerates test answers from the secret.

**Reveal.** The v0.1 test secret is published in the dm-bench dataset
repository on Hugging Face when the v0.2 held-out test split is published,
or on 2027-09-30, whichever comes first. It is published verbatim, because
the commitment is over the exact text. Until then nobody outside can verify
the commitment, which is why the date is fixed.

Canonical 0.1.0 commitment (release_sha256 `6fe00ba8...b185ed`):

- `secret_sha256`: `74d2bc987213c9e796bbf4de54dfad09caf02e40cfee842ce05533643f2adc5d`
- `answers_sha256`: `bdc3a674a6ca064cf029f8d6dc37825b951daf9703c1394afc289d5a8360ec9c`

#### Commitment recipe

Both digests are lowercase hex SHA-256. Byte for byte:

1. **Canonical JSON** of one answer record: keys sorted, separators `,` and
   `:` with no spaces, ASCII only, floats rounded to 9 decimals, `-0.0`
   written as `0.0`, NaN and infinity rejected, then exactly one trailing
   newline. Encoded as UTF-8. Floats are written the way Python's
   `repr(float)` writes them: shortest round-trip digits, `1.0` keeps its
   `.0`, exponent form below 1e-4 (`1.7453e-05`). Integers stay integers.
2. **Per page:** SHA-256 of those bytes.
3. **`answers_sha256`:** the per-page digests of every `test` page in all
   three tiers, sorted as text, joined with no separator, then SHA-256 of
   that string.
4. **`secret_sha256`:** SHA-256 of `"dm-bench/0.1.0/test/"` followed by the
   secret text, stripped and lowercased. The secret is a hex string and
   generation uses `int(secret, 16)`, but the commitment is over the text,
   so it must be checked against the revealed string as published.

Each answer record carries its `page_id`, tier, split, document and page
index and fragment ids, so adding or dropping a test page changes
`answers_sha256`. One field, `ink_frac` (the share of a fragment's pixels
whose RGB mean is below 200 on the clean page, under the pre-erosion mask),
is measured on the rendered page, so `answers_sha256` only reproduces under
the PyMuPDF version recorded in `build_env`. The hashing itself needs only
the Python standard library:

```python
import hashlib, json

def canonical_bytes(record):
    # Records as returned by src.bench.evaluate.split_answers: floats are
    # already rounded, so plain json reproduces the canonical bytes.
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()

def answers_sha256(records):
    # records: every test answer record of all three tiers in one list.
    pages = sorted(hashlib.sha256(canonical_bytes(r)).hexdigest() for r in records)
    return hashlib.sha256("".join(pages).encode()).hexdigest()

def secret_sha256(secret):
    return hashlib.sha256(("dm-bench/0.1.0/test/" + secret.strip().lower()).encode()).hexdigest()
```

After the reveal there are two ways to check, both in the environment
recorded in `build_env`. For the canonical release that is numpy 1.26.4,
scipy 1.13.1, Pillow 10.4.0, PyMuPDF 1.24.10 (MuPDF 1.24.9), built on
macOS arm64 with Python 3.11, and `requirements-bench.txt` pins the same
library versions. Rebuild the test split with `DM_BENCH_TEST_SECRET` set and compare `test_commitment` in the
new `benchmark.json`, and the test puzzle files against `CONTENT.sha256`
(decoded content, so an encoder change does not count as a difference). Or
call `split_answers(release, tier, "test", secret)` for each tier, pool the
`.values()` of all three into one list and apply the recipe above.

#### How to use the splits

- Fit learned methods on `train` (built on demand), tune on `val`.
- Use `test-dev` for public development checks.
- The held-out `test` split is scored through the maintainer, since scoring
  needs the secret. There is no submission server.

## Release layout

Canonical 0.1.0 release: [huggingface.co/datasets/arittrabag/dm-bench](https://huggingface.co/datasets/arittrabag/dm-bench).

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
  than 2 inliers scores 0, so a submission with fewer than 2 aligned
  fragments scores exactly 0. Random submissions score 0 unless two fragments
  line up within tau by chance.
- **neighbor_acc**: share of ground-truth adjacent pairs (shared edge at least
  2*tau) whose two fragments fit one rigid map within tau. Symmetric, and a
  perfect page always scores 1.
- **perfect**: direct_acc = 1.
- **adjacency P/R/F1**: contacts between the placed masks (within 2 + 2 x
  erosion px) against ground truth; computed for every solver.
- **Hit@1, Hit@5, MRR**: from `candidates`, if given. A ground-truth
  neighbour is a pair sharing at least 2*tau of edge, the same rule as
  neighbor_acc. Per page, the share of fragments with such a neighbour that
  get one ranked first (Hit@1) or in the top five (Hit@5), and the mean
  reciprocal rank (MRR). Reported as the mean over pages that submit
  candidates: pages without `candidates` are left out, not scored 0, so
  compare Hit@k only between solutions covering the same pages. A page that
  submits candidates but has no qualifying neighbour pair scores 0 and is
  counted. No confidence interval is computed for Hit@k.
- **tau curve**: direct and neighbor accuracy at 0.25, 0.5, 1 and 2% of page
  width, plus their mean (AUC).

Poses must be rigid (finite, R^T R = I within 1e-3, det > 0); mirrored,
sheared, singular or unknown entries count as unplaced and are reported as
`invalid_poses`. A page without a solution file scores 0. Means come with a
95% bootstrap CI that resamples documents (B = 10000, fixed seed), and pages
are sorted before aggregation, so results do not depend on file order.

The percentile bootstrap undercovers with few documents. Measured in
simulation (1 to 3 pages per document, intra-document correlation, 300 runs
each), the nominal 95% interval covered the true mean in 91% of runs with 10
documents, 93% with 20 and 96% with 30. Treat test-dev intervals (10 documents
per tier) as roughly 90% intervals.

## Baselines

| method | what it is |
|--------|-----------|
| `oracle` | reads the answers; harness self-check only (must score 1.0), refused on test |
| `random@0.1` | uniform random rigid poses; the floor (not 0 by construction: two fragments can line up by chance) |
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

`edge-greedy@0.1`, eval_version 1.0, canonical dm-bench 0.1.0 release
(val, test-dev, test), release_sha256
`6fe00ba8fac1e39b62cfa78a266095ba7e24c594d0fb2b5442b1148266b185ed`, built by
the maintainer on macOS arm64 at commit 4847932 (CI on Linux reproduces the
pinned golden digests). Source:
[`results/dm-bench-0.1.0_edge-greedy-0.1_eval-1.0.json`](results/dm-bench-0.1.0_edge-greedy-0.1_eval-1.0.json).
307 pages in total.

| tier | split | pages (docs) | direct_acc [95% CI] | neighbor_acc [95% CI] | perfect | adjacency F1 | Hit@1 |
|------|-------|--------------|---------------------|-----------------------|---------|--------------|-------|
| easy | test | 48 (30) | 0.993 [0.981, 1.000] | 0.994 [0.978, 1.000] | 0.958 | 0.980 | 0.997 |
| medium | test | 50 (30) | 0.865 [0.820, 0.907] | 0.851 [0.800, 0.900] | 0.300 | 0.893 | 0.959 |
| hard | test | 52 (30) | 0.246 [0.224, 0.267] | 0.640 [0.591, 0.686] | 0.000 | 0.733 | 0.928 |
| easy | test-dev | 13 (10) | 0.973 [0.927, 1.000] | 0.979 [0.942, 1.000] | 0.846 | 0.984 | 1.000 |
| medium | test-dev | 16 (10) | 0.673 [0.555, 0.765] | 0.687 [0.570, 0.782] | 0.125 | 0.743 | 0.945 |
| hard | test-dev | 19 (10) | 0.221 [0.203, 0.243] | 0.626 [0.564, 0.686] | 0.000 | 0.695 | 0.913 |
| easy | val | 39 (20) | 0.989 [0.980, 0.998] | 0.993 [0.986, 1.000] | 0.897 | 0.985 | 1.000 |
| medium | val | 37 (20) | 0.824 [0.764, 0.888] | 0.823 [0.764, 0.883] | 0.432 | 0.861 | 0.975 |
| hard | val | 33 (20) | 0.255 [0.228, 0.286] | 0.707 [0.659, 0.761] | 0.000 | 0.800 | 0.940 |

- `test` is the number to cite: it is held out, never used for tuning, and
  has the most documents.
- test-dev is small (10 documents per tier) and noisy: on medium it scores
  0.673 while val and test score 0.824 and 0.865. Test is within val's
  interval on every tier (hard 0.246 vs 0.255), so that dip is consistent
  with sampling variation; overlap alone cannot rule out some tuning effect.
  Its intervals cover roughly 90% (see the bootstrap note above).
- The val and test-dev rows are identical to an earlier independent build of
  the public splits, as the determinism guarantee requires.
- `random@0.1` scores 0.000 on every headline metric (direct, neighbor,
  perfect, adjacency) of val and test-dev on this release
  ([results file](results/dm-bench-0.1.0_random_val-testdev_eval-1.0.json)).
  Only the loosest tau-curve point is nonzero: medium val direct 0.0042 and
  neighbor 0.0011 at tau = 2% (curve AUC 0.001 and 0.0003). It was not run
  on test.
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
- **Geometry deterministic, pixels renderer-dependent.** Tears, poses and
  adjacency depend only on the seed, the tier parameters and the page size
  (given the pinned NumPy, SciPy and Pillow), never on rendered pixels.
  Fragment pixels, the per-fragment `ink_frac` in each answer and the blank
  counts in `benchmark.json` come from the PyMuPDF render, so another
  PyMuPDF version can change them. `benchmark.json` records the versions
  used (`build_env`). How much this moves scores is measured below
  ([#10](https://github.com/Arittra-Bag/Dataset-Maker/issues/10)).
- **PyMuPDF 1.24.10 (MuPDF 1.24.9) vs 1.28.2 (MuPDF 1.28.2), measured.**
  Public splits val and test-dev (157 pages, 2568 fragments), all other
  libraries identical, macOS arm64, Python 3.10. Solving and scoring run in
  the pinned environment for both. Script: `scripts/pymupdf_sensitivity.py`.
  Every number below is in
  [`results/pymupdf-sensitivity_1.24.10-vs-1.28.2.json`](results/pymupdf-sensitivity_1.24.10-vs-1.28.2.json).
  - Control: building from saved PDFs is byte-identical to a normal build.
    All 2882 val and test-dev files, and the `edge-greedy@0.1` val and
    test-dev results, equal the published release.
  - Geometry: identical on all 157 pages (answers compared with only
    `ink_frac` removed).
  - Pixels: 156 of 157 rendered pages differ, in 1.0% of pixels on average
    (at most 2.1%). The change has a direction: 98.8% of changed pixels are
    lighter under 1.28.2. Over all changed pixels the mean change is +10.4 of
    255 levels (mean absolute 10.6), the largest single channel change is 33,
    and averaged over all pixels it is 0.11. 762 of 2568 fragments keep
    identical pixels.
  - Ink and blank counts: `ink_frac` changes in 783 fragments and drops in
    779 of them (by at most 0.019). 83 fragments fall below the 1% blank
    threshold and none rise above it, so blank fragments go from 687 to 770
    of 2568 (+12%), up in every tier and split.
  - PDF writing: the 90 PDFs 1.28.2 writes have the same objects and the
    same Info metadata (dates, format) as the 1.24.10 ones, apart from the
    producer stamp (trailer and file layout not compared). The full rebuild
    is byte-identical to the renderer-only build. For this pair every change
    comes from rendering.
  - Scores, `edge-greedy@0.1`, eval_version 1.0, slice release_sha256
    `37da2a8a...` (1.24.10) vs `b5b67ea0...` (1.28.2), not the canonical
    release. Same pages and a deterministic solver, so every difference is
    caused by the pixel change. Perfect pages: 0 of 157 pages change. Direct
    accuracy changes on 2 pages (largest tier and split change 0.0019, hard
    val), neighbour accuracy on 8 (0.0113), adjacency F1 on 11 (0.0157),
    Hit@1 on 3 (0.0045), in both directions. The largest single-page change
    is 0.29 (adjacency F1). Each tier and split change is at most 0.41 of the
    half width of the pinned run's document-bootstrap 95% interval, so the
    effect is small next to between-document uncertainty. Hit@k has no
    interval.
  - Scope: one solver, one synthetic corpus, one version pair, one platform.
    Treat score differences up to about 0.016 between systems evaluated
    under different PyMuPDF versions as unresolved, report `build_env` with
    any score, and re-measure for other solvers.

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
