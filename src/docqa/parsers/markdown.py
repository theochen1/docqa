"""Markdown parser: segments keyed by their nearest heading. Locator = "#Heading".

Splits the document at ATX headings (`#`..`######`). Each segment's text is the body under a
heading; the locator is that heading. Content before the first heading is attributed to a
synthetic "#(top)" locator so no text is dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

from docqa.parsers import MIN_USABLE_CHARS, ParseOutcome
from docqa.types import TextSegment

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


class MarkdownParser:
    def can_parse(self, path: str) -> bool:
        return path.lower().endswith((".md", ".markdown"))

    def parse_text(self, filename: str, content: str) -> ParseOutcome:
        """Core logic, decoupled from disk for testability."""
        if len(content.strip()) < MIN_USABLE_CHARS:
            return ParseOutcome(skipped=True, skip_reason="empty or whitespace-only markdown")

        segments: list[TextSegment] = []
        current_heading = "#(top)"
        buf: list[str] = []

        def flush() -> None:
            body = "\n".join(buf).strip()
            if len(body) >= MIN_USABLE_CHARS:
                segments.append(
                    TextSegment(filename=filename, locator=current_heading, text=body)
                )
            buf.clear()

        for line in content.splitlines():
            m = _HEADING.match(line)
            if m:
                flush()
                current_heading = "#" + m.group(2).strip()
            else:
                buf.append(line)
        flush()

        if not segments:
            return ParseOutcome(skipped=True, skip_reason="no usable text under any heading")
        return ParseOutcome(segments=segments)

    def parse(self, path: str) -> list[TextSegment]:
        outcome = self.parse_file(path)
        return outcome.segments

    def parse_file(self, path: str) -> ParseOutcome:
        p = Path(path)
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = p.read_text(encoding="latin-1")
            except Exception as e:  # noqa: BLE001 - never crash on a bad file
                return ParseOutcome(skipped=True, skip_reason=f"unreadable markdown: {e}")
        except Exception as e:  # noqa: BLE001
            return ParseOutcome(skipped=True, skip_reason=f"unreadable markdown: {e}")
        return self.parse_text(p.name, content)
