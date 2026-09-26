"""Shared fixtures. PDFs are generated on the fly so tests need no binaries."""
import pytest


@pytest.fixture(scope="session")
def small_pdf(tmp_path_factory):
    """2-page A4 PDF with text, grey fill and a pure-black block (the black
    block exercises the black-ink vs black-background ambiguity)."""
    import fitz

    path = tmp_path_factory.mktemp("pdf") / "small.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 80), f"Test page {i + 1}", fontsize=20)
        page.insert_textbox(fitz.Rect(60, 110, 530, 400),
                            "Lorem ipsum dolor sit amet. " * 40, fontsize=10)
        page.draw_rect(fitz.Rect(60, 450, 300, 600), color=None, fill=(0, 0, 0))
        page.draw_rect(fitz.Rect(320, 450, 530, 600), color=None, fill=(0.6, 0.6, 0.6))
    doc.save(str(path))
    doc.close()
    return str(path)
