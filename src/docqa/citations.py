"""Citation verification — the DISPOSE half of propose/verify.

BT13 ships the RESOLVE check (referential integrity): a cited claim's text must round-trip to the
real source span at its locator. The ENTAILMENT gate (inductive-step validity) lands at BT18;
until then every resolved claim is stamped entailed=True as the pre-gate default, so BT18 is a
pure narrowing (true -> false), never a shape change.
"""

from __future__ import annotations

from docqa.types import Citation, Claim, ClaimRecord


def resolve(claim_record: ClaimRecord) -> Citation:
    """Build the citation for a claim. The span is the claim's own stored text (it round-trips to
    the source by construction, since claims are extracted from parsed segments with provenance)."""
    return Citation(
        filename=claim_record.filename,
        locator=claim_record.locator,
        span=claim_record.text,
    )


# A proposed claim must carry at least this many content characters that are actually present in
# the source span, so a trivial fragment ("the") can't resolve against a long span.
_MIN_OVERLAP_CHARS = 12


def span_resolves(claim_text: str, record: ClaimRecord) -> bool:
    """Referential integrity: the proposed claim text must itself be supported WITHIN the source
    span (proposal ⊆ source), never the reverse.

    Directionality matters (M3-review HIGH): accepting `source ⊆ proposal` let a proposer append
    fabricated prose after one real span and have the whole sentence round-trip as cited. We accept
    ONLY `proposal ⊆ source`, and require a minimum real overlap so a tiny fragment can't match a
    long span. The entailment gate (BT18) tightens 'supported' further to genuine entailment.
    """
    a = " ".join(claim_text.lower().split())
    b = " ".join(record.text.lower().split())
    if not a or not b:
        return False
    if a not in b:
        return False
    # Guard trivial short-fragment matches: the claim must be a substantial part of the span,
    # OR (for genuinely short source spans) essentially the whole span.
    return len(a) >= _MIN_OVERLAP_CHARS or a == b


def verify_claim(proposed_text: str, record: ClaimRecord | None) -> Claim | None:
    """Return a verified Claim, or None if the citation doesn't resolve to a real record.

    BT13 guarantee = REFERENTIAL INTEGRITY: the cite must resolve to a real retrieved record.
    Acceptance is that resolution (record is not None) — NOT a substring match against the
    proposer's paraphrase (that was over-strict and refused every rephrased answer).

    The emitted claim text is the SOURCE span (record.text), never the proposer's free text, so
    answer_text can never contain a character absent from a cited source — the no-fabrication
    guarantee holds by construction, independent of how the proposer phrased its draft.

    Support-verification (does the span actually ENTAIL the claim?) is the BT18 entailment gate.
    entailed=True here is the pre-gate default that BT18 narrows to false when a span fails NLI.
    `proposed_text` is retained in the signature for that future gate.
    """
    if record is None:
        return None
    return Claim(
        text=record.text,        # anchor to the source, never the proposer's free text
        citation=resolve(record),
        entailed=True,           # pre-gate default; BT18 narrows this via entailment
        entail_score=1.0,
    )
