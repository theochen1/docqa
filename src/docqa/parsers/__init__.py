"""Format parsers: file -> TextSegments with provenance. Each never raises on a bad file.

A parser's contract (the no-crash / no-silent-garbage floor):
- return [] with a logged skip reason rather than raising, on any malformed input
- attach a format-appropriate locator to every segment (#Heading / L-range / p.N / .eml)
- be content-aware where it matters (do not silently emit garbage)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from docqa.types import TextSegment


@dataclass
class ParseOutcome:
    """Result of parsing one file: segments plus a skip reason if nothing usable came out."""

    segments: list[TextSegment] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.segments) and not self.skipped


# Minimum non-whitespace characters for an extraction to count as usable content
# (the no-silent-garbage gate; documented + tunable). Below this -> skip-with-reason.
MIN_USABLE_CHARS = 3
