"""The pluggable seams — Protocols every implementation plugs into.

These are the interface contracts the eval harness asserts against, not any implementation. A
retriever/parser/generator can be swapped or reverted without touching this file — which is what
lets the retrieval algorithm be measured against a fixed contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from docqa.types import ClaimRecord


@runtime_checkable
class Parser(Protocol):
    """Turns one file into text + within-document locators. Pure: file -> spans.

    Must never raise on a bad file — returns an empty result with a skip reason instead
    (the no-crash / no-silent-garbage floor).
    """

    def can_parse(self, path: str) -> bool:
        """Content-aware, not extension-trusting."""
        ...

    def parse(self, path: str) -> list[ClaimRecord]:
        """Return sourced claims (or []); each carries filename + a format-appropriate locator."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Selects candidate claims for a query. Selects — never decides correctness or conflict.

    Returns claims ordered by fused score; deterministic for a fixed index + query.
    """

    def retrieve(self, query: str, k: int) -> list[ClaimRecord]:
        ...


@runtime_checkable
class Generator(Protocol):
    """Proposes an answer as claims-with-citations from candidate claims passed as DATA.

    The proposer sees claim id + text only (never filename/locator) so it cannot fabricate a
    plausible citation. Provider-agnostic (Anthropic / OpenAI / local).
    """

    def propose(self, question: str, claims: list[ClaimRecord]) -> dict:
        """Return a structured proposal: {"claims": [{"text", "cite_ids"}], "refusal_token"}."""
        ...
