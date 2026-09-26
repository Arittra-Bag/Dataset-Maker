"""No-overlap / full-coverage invariant tests - the dataset's core guarantee."""
import numpy as np

from src.tearing import compute_adjacency, tear_page, verify_partition


def _page(h=400, w=300):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_partition_no_overlap_full_cover():
    torn = tear_page(_page(), n_pieces=12, seed=1,
                     noise_strength=20, noise_scale=60)
    rep = verify_partition(torn)
    assert rep["is_partition"], rep
    assert rep["max_overlap"] == 1
    assert rep["uncovered_pixels"] == 0


def test_piece_count_close_to_request():
    torn = tear_page(_page(), n_pieces=16, seed=2,
                     noise_strength=25, noise_scale=80)
    # Some seeds can lose their cell after warping; allow a small shortfall.
    assert 10 <= len(torn.pieces) <= 16


def test_per_page_randomness_changes():
    p = _page()
    a = tear_page(p, 12, seed=1, noise_strength=20, noise_scale=60).labels
    b = tear_page(p, 12, seed=2, noise_strength=20, noise_scale=60).labels
    assert not np.array_equal(a, b)


def test_reproducible_same_seed():
    p = _page()
    a = tear_page(p, 12, seed=5, noise_strength=20, noise_scale=60).labels
    b = tear_page(p, 12, seed=5, noise_strength=20, noise_scale=60).labels
    assert np.array_equal(a, b)


def test_adjacency_known_grid():
    # 3x3 label map: pieces 0,1,2 all mutually touch.
    labels = np.array([[0, 0, 1],
                       [0, 2, 1],
                       [2, 2, 1]], dtype=np.int32)
    adj = compute_adjacency(labels, {0: 0, 1: 1, 2: 2})
    assert adj == [(0, 1), (0, 2), (1, 2)]


def test_adjacency_invariants_on_real_page():
    torn = tear_page(_page(), n_pieces=16, seed=3,
                     noise_strength=25, noise_scale=80)
    n = len(torn.pieces)
    adj = torn.adjacency
    # undirected, deduped, sorted, in-range, no self-loops
    assert adj == sorted(set(adj))
    for i, j in adj:
        assert 0 <= i < j < n
    # partition over a connected page -> adjacency graph is connected
    seen = set()
    stack = [0]
    nbrs = {k: set() for k in range(n)}
    for i, j in adj:
        nbrs[i].add(j)
        nbrs[j].add(i)
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(nbrs[u] - seen)
    assert len(seen) == n          # every piece reachable


def test_partition_golden_hash():
    # Pins the exact partition for fixed inputs under the pinned numpy/scipy.
    # If this fails after a dependency bump or a change to noise/sampling/
    # tearing, previously published datasets can no longer be regenerated
    # bit-for-bit: bump MANIFEST_SCHEMA_VERSION / generator version and update
    # the hash deliberately, never silently.
    import hashlib

    img = np.zeros((240, 170, 3), np.uint8)
    torn = tear_page(img, 9, seed=7, noise_strength=18, noise_scale=40)
    digest = hashlib.sha256(torn.labels.astype("<i4").tobytes()).hexdigest()
    assert digest == (
        "b608c92d28943ae462fabd7c5a331ae56319cafa40984582654846e57e51dd9c"
    )
    assert torn.seed == 7
