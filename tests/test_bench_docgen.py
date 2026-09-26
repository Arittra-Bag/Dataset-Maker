"""Synthetic benchmark documents: deterministic, A4, varied, non-blank."""
import hashlib

from src.bench.docgen import make_document, render_document


def _digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_same_seed_same_pdf_bytes(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    assert make_document(11, str(a)) == make_document(11, str(b))
    assert _digest(a) == _digest(b)


def test_different_seeds_differ(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    make_document(1, str(a))
    make_document(2, str(b))
    assert _digest(a) != _digest(b)


def test_pages_render_to_a4_with_content(tmp_path):
    counts = set()
    for seed in range(6):
        path = tmp_path / f"d{seed}.pdf"
        n = make_document(seed, str(path))
        pages = render_document(str(path))
        counts.add(n)
        assert len(pages) == n
        for page in pages:
            assert page.shape == (1169, 827, 3)
            assert (page.mean(axis=2) < 200).mean() > 0.01     # visible ink
    assert counts <= {1, 2, 3}
    assert len(counts) > 1                                      # page count varies
