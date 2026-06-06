"""No-overlap / full-coverage invariant tests — the dataset's core guarantee."""
import numpy as np

from src.tearing import tear_page, verify_partition


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
