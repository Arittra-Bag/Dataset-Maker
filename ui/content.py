"""Static copy for the UI. Every claim here must be backed by code or a test.

Kept out of the layout so wording can be reviewed in one place; the table of
guarantees names the function/test that enforces each one.
"""
from __future__ import annotations

from src import __version__
from src.provenance import MANIFEST_SCHEMA_VERSION, PAGE_SEED_RULE

REPO_URL = "https://github.com/Arittra-Bag/Dataset-Maker"

HEADER_HTML = f"""
<div id="dm-header">
  <div class="dm-head-main">
    <div class="dm-title-row">
      <h1>Dataset-Maker</h1>
      <span class="dm-meta">v{__version__} · manifest schema {MANIFEST_SCHEMA_VERSION} · MIT ·
        <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a></span>
    </div>
    <p class="dm-sub">Reproducible torn-document dataset generation</p>
    <p class="dm-desc">
      Tears PDF pages into non-overlapping fragments and exports the answer key
      (exact placement and adjacency for every fragment) for training and
      evaluating fragment-reassembly methods.
    </p>
    <p class="dm-why"><b>Why this exists.</b> Reassembly research needs torn
      documents with a known solution. Physical tears come without one, and
      ad-hoc generators rarely publish theirs. Here a tear is a deterministic
      function of (page, seed, parameters), and the ground truth ships with the
      fragments.</p>
  </div>
  <ul class="dm-tags">
    <li>strict partition, verified per page</li>
    <li>seeded, deterministic tears</li>
    <li>exact (x, y, w, h) per fragment</li>
    <li>adjacency pairs</li>
    <li>self-describing manifest.json</li>
  </ul>
  <ol class="dm-steps">
    <li><span>1</span>Upload PDF</li>
    <li><span>2</span>Configure tear</li>
    <li><span>3</span>Generate fragments</li>
    <li><span>4</span>Inspect ground truth</li>
    <li><span>5</span>Export dataset</li>
  </ol>
</div>
"""

GUARANTEES_HTML = f"""
<div class="dm-scroll"><table class="dm-table dm-guarantees">
  <thead><tr><th>Guarantee</th><th>What it means</th><th>Enforced by</th></tr></thead>
  <tbody>
    <tr><td>Strict partition</td>
        <td>Every pixel belongs to exactly one fragment: no overlap, no gaps.
            Nearest-seed argmin over domain-warped coordinates.</td>
        <td><code>verify_partition()</code> on every page of every run;
            generation aborts on failure</td></tr>
    <tr><td>Deterministic tears</td>
        <td>Same seed, parameters and DPI give the same partition, offsets and
            adjacency. Page seed <code>{PAGE_SEED_RULE}</code>. Fragment pixels
            also depend on the PyMuPDF version (recorded).</td>
        <td><code>tests/test_pipeline.py</code>, golden hash in
            <code>tests/test_partition.py</code></td></tr>
    <tr><td>Exact placement</td>
        <td>Each fragment's <code>(x, y, w, h)</code> on the page canvas.
            Lossless exports reassemble the page pixel-exactly.</td>
        <td>ZIP round-trip test</td></tr>
    <tr><td>Adjacency</td>
        <td>Undirected <code>[i, j]</code> pairs of fragments sharing a torn
            border (4-connectivity).</td>
        <td>Known-grid + graph-connectivity tests</td></tr>
    <tr><td>Self-describing export</td>
        <td><code>manifest.json</code> records seed, parameters, page seeds,
            input SHA-256 and library versions.</td>
        <td><code>tests/test_pipeline.py</code></td></tr>
  </tbody>
</table></div>
"""

EMPTY_SUMMARY_HTML = (
    '<div class="dm-summary dm-empty">No run yet. Upload a PDF (or load the '
    "sample), then <b>Generate dataset</b>. Every number shown after a run is "
    "measured from that run.</div>"
)

MANIFEST_FIELDS_HTML = """
<div class="dm-scroll"><table class="dm-table">
  <thead><tr><th>Field</th><th>Meaning</th></tr></thead>
  <tbody>
    <tr><td><code>pages[].pieces[].x, y</code></td>
        <td>Top-left offset of the fragment's bounding box on the page canvas (px).
            The stitching label.</td></tr>
    <tr><td><code>pages[].pieces[].w, h</code></td><td>Bounding-box size = PNG size (px).</td></tr>
    <tr><td><code>pages[].pieces[].file</code></td>
        <td>Fragment PNG inside the ZIP. Pixels outside the fragment are black.</td></tr>
    <tr><td><code>pages[].adjacency</code></td>
        <td>Sorted, deduplicated <code>[i, j]</code> piece-index pairs (i &lt; j) sharing a
            border. Unlisted pairs are negatives.</td></tr>
    <tr><td><code>pages[].seed</code></td>
        <td>RNG seed the page was torn with (from <code>page_seed_rule</code>).</td></tr>
    <tr><td><code>pages[].width, height</code></td><td>A4 canvas size at the chosen DPI.</td></tr>
    <tr><td><code>master_seed</code>, <code>n_pieces_requested</code>,<br><code>noise_strength</code>,
            <code>noise_scale</code>, <code>dpi</code></td>
        <td>Everything the partition depends on.</td></tr>
    <tr><td><code>source</code>, <code>source_sha256</code></td><td>Input file name and hash.</td></tr>
    <tr><td><code>environment</code></td>
        <td>Python / NumPy / SciPy / Pillow / PyMuPDF versions used.</td></tr>
    <tr><td><code>schema_version</code></td>
        <td>1.1 adds provenance fields; all 1.0 fields are unchanged.</td></tr>
  </tbody>
</table></div>
"""

EXPORT_LAYOUT_HTML = """
<pre class="dm-pre">dataset.zip
├── manifest.json        ground truth + provenance
├── README.txt           schema + reassembly snippet
└── pieces/
    ├── page_0001/
    │   ├── piece_000.png  RGB crop, black outside
    │   └── ...
    └── page_0002/ ...</pre>
"""

LIMITATIONS_HTML = """
<ul class="dm-list">
  <li><b>Masks are implicit.</b> A fragment's shape is its non-black pixels. Pure-black
      ink touching a tear is indistinguishable from background (reassembly is still
      exact). No alpha/mask export yet.</li>
  <li><b>Folded warps.</b> Large displacement relative to wavelength can split a
      fragment into several disconnected regions. Counted per page as
      <i>multi-component</i>.</li>
  <li><b>Fragment count</b> can be lower than requested if a seed's warped cell
      vanishes. Produced counts are reported.</li>
  <li><b>Planar tears only.</b> No rotation, missing pieces, paper texture, fibre
      edges or scan noise yet.</li>
  <li><b>Lossy palette PNG</b> keeps offsets exact but pixel values approximate.</li>
  <li><b>ZIP bytes</b> are not reproducible (timestamps); its contents are.</li>
</ul>
"""

SCOPE_HTML = f"""
<div class="dm-footer">
  <b>Scope:</b> this page is the dataset generator. The benchmark (versioned
  splits, evaluation harness, baseline solver) is the command-line tool
  <code>python -m src.bench</code>, documented in docs/BENCHMARK.md; no number on
  this page comes from a solver. Source, tests and roadmap:
  <a href="{REPO_URL}" target="_blank" rel="noopener">{REPO_URL.removeprefix("https://")}</a>
</div>
"""
