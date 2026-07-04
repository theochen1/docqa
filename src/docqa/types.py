"""Core data types — the frozen shapes the whole pipeline shares.

These are pure definitions (no behavior). `ClaimRecord` is the unit of ingestion + retrieval;
`AnswerResult` is what `core.answer_question` returns and what the CLI, web layer, and eval harness
all assert against. Shapes match algorithm-design.md §2 (ClaimRecord) and §5.6 (AnswerResult).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ValueType(StrEnum):
    NUMBER = "number"
    DATE = "date"
    CURRENCY = "currency"
    DURATION = "duration"
    STRING = "string"
    ENTITY = "entity"


class SourceStatus(StrEnum):
    """How a claim's text was obtained — asserted by the eval harness (e.g. OCR provenance)."""

    PARSED = "parsed"   # born-digital / native text
    OCR = "ocr"         # recovered by OCR from an image-only page


class TextSegment(BaseModel):
    """A parser's output unit: a span of readable text with its source provenance.

    Parsers produce TextSegments (extract text + locator, per format). The claimizer (BT07)
    consumes them and produces atomic ClaimRecords (split + canonicalize value). This separates
    format-specific extraction from format-agnostic claim-making.
    """

    filename: str
    locator: str = Field(description='"#Heading" | "p.N" | "L12-L18" | structured .eml')
    text: str
    source_status: SourceStatus = Field(default=SourceStatus.PARSED)


class ClaimRecord(BaseModel):
    """One atomic sourced factual statement — the assumption-sheet cell.

    Its 'formula' is =extract(filename, locator). subject/predicate are embedded SEPARATELY from
    the value so claims about the same proposition cluster together regardless of the value
    (this is what makes conflict-surfacing structural — see algorithm-design.md §4.3).
    """

    claim_id: str = Field(description="Stable content-derived id: sha1(filename+locator+text)[:12]")
    filename: str
    locator: str = Field(description='"#Heading" | "p.N" | "L12-L18" | structured .eml')
    text: str = Field(description="The atomic statement, verbatim-enough to round-trip to the span")
    subject_norm: str = Field(default="", description="Normalized subject (proposition-key part)")
    predicate_norm: str = Field(default="", description="Normalized predicate (proposition key)")
    value_span: str = Field(default="", description="The asserted value, raw")
    value_type: ValueType = Field(default=ValueType.STRING)
    value_canon: str = Field(default="", description="Deterministic canonicalization of value_span")
    source_status: SourceStatus = Field(default=SourceStatus.PARSED)


class Citation(BaseModel):
    filename: str
    locator: str
    span: str = Field(description="The resolved source text the locator round-trips to")


class Claim(BaseModel):
    """One claim in an answer, with its citation and entailment verdict."""

    text: str
    citation: Citation
    entailed: bool = Field(description="R-ENTAIL verdict; always true for emitted claims")
    entail_score: float = Field(default=1.0, description="NLI P(entailment) or HHEM score")


RefusalToken = Literal["INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"]


class Markers(BaseModel):
    refused: bool = False
    refusal_token: RefusalToken | None = None
    conflict: bool = False
    warning: str | None = None


class AnswerResult(BaseModel):
    """The single shape core.answer_question returns; CLI/web/eval all assert on it.

    answer_text is ASSEMBLED by the verifier from surviving verified claims — never free LLM prose —
    so no uncited sentence can appear (algorithm-design.md §5.2).
    """

    answer_text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    markers: Markers = Field(default_factory=Markers)
    latency_ms: int = 0
    meta: dict = Field(default_factory=dict, description="Determinism/audit stamp (S03/S04)")


# Reserved process exit codes for degenerate inputs (BT20b) — machine-detectable, not free text.
EXIT_OK = 0
EXIT_EMPTY_CORPUS = 3
EXIT_EMPTY_QUERY = 4
