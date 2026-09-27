"""Fragment pipeline: exact poses, no leaks through canvas, alpha or RGB."""
import numpy as np
import pytest

from src.bench.fragments import PAD_MAX, Corruption, make_fragment
from src.tearing import tear_page

EASY = Corruption()
MEDIUM = Corruption(rotate=True, noise_halfwidth=2, jpeg_quality=85)
HARD = Corruption(rotate=True, blur_radius=0.6, noise_halfwidth=4, jpeg_quality=70, erosion_px=(1, 3))


def _page(seed=0):
    rng = np.random.default_rng(seed)
    page = np.full((420, 300, 3), 255, np.uint8)
    page[rng.integers(0, 420, 3000), rng.integers(0, 300, 3000)] = 0       # ink specks
    return page


def _rngs(seed=1):
    names = ("erosion", "rotation", "pad", "noise")
    return {k: np.random.default_rng(seed + i) for i, k in enumerate(names)}


def _fragments(spec, seed=3):
    page = _page()
    torn = tear_page(page, 12, seed=seed, noise_strength=20, noise_scale=60)
    rngs = _rngs()
    return page, torn, [(p, make_fragment(page, p, spec, rngs)) for p in torn.pieces]


def _mapped(frag):
    alpha = frag.rgba[..., 3]
    ys, xs = np.nonzero(alpha)
    pts = np.stack([xs, ys, np.ones_like(xs)]).astype(float)
    return xs, ys, np.rint(np.array(frag.affine) @ pts).astype(int)


@pytest.mark.parametrize("spec", [EASY, MEDIUM, HARD], ids=["easy", "medium", "hard"])
def test_gt_pose_lands_on_own_label(spec):
    page, torn, frags = _fragments(spec)
    wrong = total = 0
    for piece, frag in frags:
        _, _, pp = _mapped(frag)
        inside = (pp[0] >= 0) & (pp[0] < page.shape[1]) & (pp[1] >= 0) & (pp[1] < page.shape[0])
        wrong += int((~inside).sum())
        wrong += int((torn.labels[pp[1][inside], pp[0][inside]] != piece.label).sum())
        total += pp.shape[1]
    assert wrong / total < 1e-4                  # boundary rounding only


def test_easy_fragments_are_pixel_exact():
    page, _, frags = _fragments(EASY)
    for _, frag in frags:
        xs, ys, pp = _mapped(frag)
        assert frag.rot_mdeg == 0
        assert np.array_equal(frag.rgba[ys, xs, :3], page[pp[1], pp[0]])


@pytest.mark.parametrize("spec", [EASY, MEDIUM, HARD], ids=["easy", "medium", "hard"])
def test_no_leak_through_alpha_rgb_or_canvas(spec):
    _, _, frags = _fragments(spec)
    for _, frag in frags:
        alpha = frag.rgba[..., 3]
        assert set(np.unique(alpha).tolist()) <= {0, 255}
        assert not frag.rgba[alpha == 0, :3].any()            # hidden RGB zeroed
        ys, xs = np.nonzero(alpha)
        margins = (xs.min(), ys.min(), alpha.shape[1] - 1 - xs.max(), alpha.shape[0] - 1 - ys.max())
        assert all(0 <= m <= PAD_MAX for m in margins)         # tight crop + small pad


def test_erosion_spares_page_border_and_removes_torn_edge():
    page, torn, frags = _fragments(HARD)
    for piece, frag in frags:
        assert 1 <= frag.erosion_px <= 3
        assert frag.kept_px < int(piece.mask.sum())
    # A piece touching the page's left edge keeps pixels at x == 0.
    left = [(p, f) for p, f in frags if p.x == 0]
    assert left
    for _, frag in left:
        _, _, pp = _mapped(frag)
        assert (pp[0] == 0).any()


def test_same_rngs_same_fragment():
    page = _page()
    torn = tear_page(page, 8, seed=5, noise_strength=20, noise_scale=60)
    a = make_fragment(page, torn.pieces[0], HARD, _rngs(9))
    b = make_fragment(page, torn.pieces[0], HARD, _rngs(9))
    assert np.array_equal(a.rgba, b.rgba)
    assert a.affine == b.affine
