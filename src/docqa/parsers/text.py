"""Plain-text parser: segments carry a line-range locator "L{start}-L{end}" (1-indexed).

Chunks the file into contiguous blocks separated by blank lines, so a citation resolves to the
exact lines it came from. Deterministic: same bytes -> same line ranges (stable citations).
"""

from __future__ import annotations

from pathlib import Path

from docqa.parsers import MIN_USABLE_CHARS, ParseOutcome
from docqa.types import TextSegment


class TextParser:
    def can_parse(self, path: str) -> bool:
        return path.lower().endswith((".txt", ".text"))

    def parse_text(self, filename: str, content: str) -> ParseOutcome:
        if len(content.strip()) < MIN_USABLE_CHARS:
            return ParseOutcome(skipped=True, skip_reason="empty or whitespace-only text file")

        lines = content.splitlines()
        segments: list[TextSegment] = []
        block: list[str] = []
        block_start = 1  # 1-indexed line number where the current block began

        def flush(end_line: int) -> None:
            body = "\n".join(block).strip()
            if len(body) >= MIN_USABLE_CHARS:
                locator = f"L{block_start}-L{end_line}"
                segments.append(TextSegment(filename=filename, locator=locator, text=body))
            block.clear()

        for i, line in enumerate(lines, start=1):
            if line.strip() == "":
                if block:
                    flush(i - 1)
                block_start = i + 1
            else:
                if not block:
                    block_start = i
                block.append(line)
        if block:
            flush(len(lines))

        if not segments:
            return ParseOutcome(skipped=True, skip_reason="no usable text blocks")
        return ParseOutcome(segments=segments)

    def parse(self, path: str) -> list[TextSegment]:
        return self.parse_file(path).segments

    def parse_file(self, path: str) -> ParseOutcome:
        p = Path(path)
        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = p.read_text(encoding="latin-1")
            except Exception as e:  # noqa: BLE001 - never crash on a bad file
                return ParseOutcome(skipped=True, skip_reason=f"unreadable text: {e}")
        except Exception as e:  # noqa: BLE001
            return ParseOutcome(skipped=True, skip_reason=f"unreadable text: {e}")
        return self.parse_text(p.name, content)
