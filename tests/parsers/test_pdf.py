"""BT06: PDF parser. Per-page p.N locators for native text; image-only pages detected as
needs_ocr (not silently indexed as empty); no crash on a corrupt PDF.

Fixtures are generated in-test with PyMuPDF so they're deterministic and need no committed binary.
"""

import fitz  # PyMuPDF

from docqa.parsers.pdf import PdfParser


def _born_digital_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=14)
    doc.save(str(path))
    doc.close()


def _image_only_page(doc):
    """A page with a raster image and no text layer (an OCR candidate)."""
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 80))
    pix.clear_with(220)  # light-gray filled rectangle -> an image, no glyphs
    page.insert_image(fitz.Rect(10, 10, 190, 70), pixmap=pix)


def test_native_pdf_page_locators(tmp_path):
    f = tmp_path / "paper.pdf"
    _born_digital_pdf(f, ["Page one intro.", "The Q3 pilot achieved a 92% completion rate."])
    out = PdfParser().parse_file(str(f))
    assert out.usable
    p2 = next(s for s in out.segments if s.locator == "p.2")
    assert "92%" in p2.text
    assert all(s.locator.startswith("p.") for s in out.segments)


def test_image_only_pdf_flagged_needs_ocr_not_garbage(tmp_path):
    f = tmp_path / "scan.pdf"
    doc = fitz.open()
    _image_only_page(doc)
    doc.save(str(f))
    doc.close()
    out = PdfParser().parse_file(str(f))
    # No native text -> whole doc skipped-with-reason, page listed for OCR, zero garbage segments.
    assert out.skipped
    assert out.segments == []
    assert out.needs_ocr == ["p.1"]
    assert "OCR" in out.skip_reason


def test_mixed_pdf_native_and_image_only(tmp_path):
    f = tmp_path / "mixed.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Native page with the budget figure $4,250.", fontsize=14)
    _image_only_page(doc)  # page 2 is image-only
    doc.save(str(f))
    doc.close()
    out = PdfParser().parse_file(str(f))
    # Native page contributes a segment; image-only page is an OCR candidate. Both coexist.
    assert any(s.locator == "p.1" and "$4,250" in s.text for s in out.segments)
    assert out.needs_ocr == ["p.2"]


def test_corrupt_pdf_no_crash(tmp_path):
    f = tmp_path / "corrupt.pdf"
    f.write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    out = PdfParser().parse_file(str(f))
    assert out.skipped and out.segments == []
    assert out.skip_reason


def test_can_parse_extension():
    assert PdfParser().can_parse("a.pdf")
    assert not PdfParser().can_parse("a.md")
