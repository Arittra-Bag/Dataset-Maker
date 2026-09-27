"""Golden digests for dm-bench v0.1: published data must stay regenerable.

If one of these fails after a change to docgen, tearing, noise, sampling,
fragments, build or a dependency bump, previously published releases can no
longer be rebuilt bit-for-bit. Do not update the digests silently: bump
BENCH_VERSION and document the change instead.

Answer JSON (poses, adjacency, ids) is checked on every tier. Fragment pixels
are checked where no lossy stage runs (easy); JPEG/blur tiers are covered by
the cross-platform determinism job in CI.
"""
import pytest

from src.bench import build, canon

GOLDEN = {
    # tier: (page_id, answer sha256, fragment content digest | None)
    "easy": ("1d1e7796a767",
             "a47baf1663f66c8f65c1da13934a255528691517c238dab714af0588beeb382a",
             "87357a9ef8cdf255eb2864be253249eea6c1d441decc11f8af0222973c183f95"),
    "medium": ("6a2d1de386b3",
               "2e826ae02b82a44439cbaafeab7da40b9b4ecc9f501517f7a98ca9c80b900083", None),
    "hard": ("ca80ed390756",
             "291bd06a9f270c2d73481f8cc0751681810c7663a97b8aeccd5ff35a0a089df6", None),
}


@pytest.mark.parametrize("tier", sorted(GOLDEN))
def test_golden_page(tier):
    page_id, answer_digest, pixel_digest = GOLDEN[tier]
    doc = build.doc_range("val", build.DEFAULT_DOCS).start
    page = build.build_doc(build.PUBLIC_ENTROPY, tier, "val", doc)[0]
    assert page.page_id == page_id
    assert canon.sha256_bytes(canon.dumps(page.answer).encode()) == answer_digest
    if pixel_digest:
        joined = "".join(canon.content_digest_png(page.pngs[k]) for k in sorted(page.pngs))
        assert canon.sha256_bytes(joined.encode()) == pixel_digest
