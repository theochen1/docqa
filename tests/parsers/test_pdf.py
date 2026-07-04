"""PDF handling is a DELIBERATE SKIP (scope cut). A .pdf is recognized, skipped with a logged
reason, and never indexed — no crash, no silent garbage, no pymupdf dependency."""

from docqa.parsers.pdf import PdfParser


def test_pdf_recognized_by_extension():
    assert PdfParser().can_parse("report.pdf")
    assert not PdfParser().can_parse("notes.md")


def test_pdf_is_skipped_with_reason(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF-1.4 whatever")  # content is irrelevant; we don't parse it
    out = PdfParser().parse_file(str(f))
    assert out.skipped
    assert out.segments == []
    assert "PDF" in out.skip_reason and "skip" in out.skip_reason.lower()


def test_pdf_never_raises_on_any_bytes(tmp_path):
    for content in (b"", b"\x00\x01\x02", b"not a pdf at all"):
        f = tmp_path / "x.pdf"
        f.write_bytes(content)
        out = PdfParser().parse_file(str(f))
        assert out.skipped and out.segments == []


def test_no_pymupdf_dependency():
    # The skip path must not import a PDF library.
    import importlib

    import docqa.parsers.pdf as pdfmod
    importlib.reload(pdfmod)
    assert "fitz" not in dir(pdfmod)
