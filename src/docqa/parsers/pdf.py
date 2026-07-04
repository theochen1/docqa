"""PDF parser: per-page native-text extraction with "p.N" locators (1-indexed).

Classification is PER PAGE (not per file): a page with extractable native text becomes a segment;
a page that has an image but < MIN_CHARS native text is flagged image-only (needs_ocr) — at BT06
it is logged + skipped, at BT22 OCR recovers it. A file can thus contribute native pages AND
image-only pages (the mixed-PDF case).
"""

from __future__ import annotations

from pathlib import Path

from docqa.parsers import MIN_USABLE_CHARS, ParseOutcome
from docqa.types import TextSegment

# A page needs OCR when it carries an image but yields near-zero native text.
_MIN_NATIVE_CHARS = 10


class PdfParser:
    def can_parse(self, path: str) -> bool:
        return path.lower().endswith(".pdf")

    def parse(self, path: str) -> list[TextSegment]:
        return self.parse_file(path).segments

    def parse_file(self, path: str) -> ParseOutcome:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ParseOutcome(skipped=True, skip_reason="PyMuPDF not installed")

        p = Path(path)
        try:
            doc = fitz.open(str(p))
        except Exception as e:  # noqa: BLE001 - never crash on a corrupt/truncated PDF
            return ParseOutcome(skipped=True, skip_reason=f"unopenable PDF: {e}")

        segments: list[TextSegment] = []
        needs_ocr: list[str] = []
        try:
            for i, page in enumerate(doc, start=1):
                locator = f"p.{i}"
                try:
                    text = page.get_text("text") or ""
                    has_image = bool(page.get_images(full=False))
                except Exception:  # noqa: BLE001 - a bad page shouldn't kill the file
                    continue
                if len(text.strip()) >= _MIN_NATIVE_CHARS:
                    segments.append(
                        TextSegment(filename=p.name, locator=locator, text=text.strip())
                    )
                elif has_image:
                    # image-only page: no native text but pixels present -> OCR candidate
                    needs_ocr.append(locator)
                # else: genuinely blank page -> nothing to index, not an OCR candidate
        finally:
            doc.close()

        if not segments and not needs_ocr:
            return ParseOutcome(skipped=True, skip_reason="no usable text or images in PDF")
        if not segments and needs_ocr:
            # Whole doc is image-only: skip-with-warning at BT06 (OCR arrives at BT22).
            return ParseOutcome(
                skipped=True,
                skip_reason=f"image-only PDF (needs OCR): pages {needs_ocr}",
                needs_ocr=needs_ocr,
            )
        # Mixed or fully-native: keep native segments; note any image-only pages for OCR.
        reason = ""
        if len(" ".join(s.text for s in segments).strip()) < MIN_USABLE_CHARS:
            reason = "extracted text below usable threshold"
        return ParseOutcome(segments=segments, needs_ocr=needs_ocr, skip_reason=reason)
