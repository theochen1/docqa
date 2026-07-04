"""PDF handling: DELIBERATE SKIP (documented cut).

PDFs (text-based and scanned) are skipped with a logged warning rather than parsed. This is a
conscious scope decision, not an oversight: robust PDF text extraction and OCR are their own
projects, and the tool's focus is answer accuracy + reliability on the formats it does handle
(Markdown, .eml, plain text). The assessment explicitly accepts "skip them with a logged warning"
as deliberate handling; the no-crash / no-silent-garbage floor is preserved because a PDF produces
zero indexed claims and a clear skip reason, never garbage.

If PDF support is wanted later, this is the seam to implement (parse -> TextSegments); nothing else
in the pipeline assumes PDFs are absent.
"""

from __future__ import annotations

from docqa.parsers import ParseOutcome


class PdfParser:
    def can_parse(self, path: str) -> bool:
        return path.lower().endswith(".pdf")

    def parse(self, path: str):
        return self.parse_file(path).segments

    def parse_file(self, path: str) -> ParseOutcome:
        # Deliberate skip: recognized, logged, never indexed.
        return ParseOutcome(
            skipped=True,
            skip_reason="PDF skipped (deliberate scope cut — PDFs/OCR not handled in this version)",
        )
