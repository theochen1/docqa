"""Conflict detection — surface contradictions, never silently pick a side.

Primary rule is DETERMINISTIC value-mismatch (not NLI): within a proposition-cluster
(subject+predicate), if >=2 distinct canonical values come from >=2 distinct files, that's a
conflict. This is reliable for numbers/dates/currency, where small NLI models score "15 days" vs
"20 days" as merely 'neutral'. Agreeing/duplicate values (canonicalized equal) never fire — the
over-conflict guard. The two-track retriever (BT17) guarantees both sides are present to compare.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from docqa.types import ClaimRecord


def _proposition_key(c: ClaimRecord) -> str:
    return f"{c.subject_norm}|{c.predicate_norm}".strip("|")


@dataclass
class ConflictSide:
    value_canon: str
    filename: str
    locator: str
    text: str


@dataclass
class Conflict:
    proposition: str
    sides: list[ConflictSide]


def detect_conflicts(claims: list[ClaimRecord]) -> list[Conflict]:
    """Find value-mismatch conflicts among the given claims. Deterministic + order-stable."""
    clusters: dict[str, list[ClaimRecord]] = defaultdict(list)
    for c in claims:
        key = _proposition_key(c)
        if key and c.value_canon:
            clusters[key].append(c)

    conflicts: list[Conflict] = []
    for key in sorted(clusters):
        members = clusters[key]
        # Group by distinct canonical value; keep the first (top-ranked) claim per value.
        by_value: dict[str, ClaimRecord] = {}
        for c in members:  # input order == retrieval rank
            by_value.setdefault(c.value_canon, c)
        distinct_files = {c.filename for c in members}
        if len(by_value) >= 2 and len(distinct_files) >= 2:
            sides = [
                ConflictSide(value_canon=v, filename=c.filename, locator=c.locator, text=c.text)
                for v, c in sorted(by_value.items())
            ]
            conflicts.append(Conflict(proposition=key, sides=sides))
    return conflicts
