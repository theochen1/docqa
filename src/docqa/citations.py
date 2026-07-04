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


def span_resolves(claim_text: str, record: ClaimRecord) -> bool:
    """Referential integrity: the proposed claim text must be supported by the record's span.

    A generous containment check at BT13 (normalized substring either direction) — the entailment
    gate at BT18 tightens 'supported' to genuine entailment.
    """
    a = " ".join(claim_text.lower().split())
    b = " ".join(record.text.lower().split())
    if not a or not b:
        return False
    return a in b or b in a


def verify_claim(proposed_text: str, record: ClaimRecord) -> Claim | None:
    """Return a verified Claim (entailed=True default) if the citation resolves, else None."""
    if not span_resolves(proposed_text, record):
        return None
    return Claim(
        text=proposed_text,
        citation=resolve(record),
        entailed=True,       # pre-gate default; BT18 narrows this
        entail_score=1.0,
    )
