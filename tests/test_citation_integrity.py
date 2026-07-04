"""The citation guarantee (M3 review HIGH): fabricated prose must never reach answer_text.

The guarantee is now enforced BY CONSTRUCTION: verify_claim emits the retrieved record's own
text (record.text), never the proposer's free text. So even when the proposer pads its draft with
fabricated content, only the grounded source span is emitted. And a cite that doesn't resolve to a
retrieved record is dropped (referential integrity). Support-checking — does the span actually
entail the claim — is the BT18 entailment gate, tested there.
"""

from docqa.citations import verify_claim
from docqa.core import answer_from_proposal
from docqa.types import ClaimRecord


def _rec(text, cid="c1"):
    return ClaimRecord(claim_id=cid, filename="f.md", locator="#h", text=text)


# --- the core guarantee: answer_text contains only grounded source spans, never proposer prose ---

def test_fabricated_padding_never_reaches_answer_text():
    rec = _rec("Employees accrue 15 days of PTO", cid="c_pto")
    proposal = {
        "claims": [{"text": "Employees accrue 15 days of PTO and the CEO earns 2M",
                    "cite_ids": ["c_pto"]}],
        "refusal_token": None,
    }
    res = answer_from_proposal(proposal, [rec])
    # Claim resolves (real record) so we answer — with the SOURCE span, not the padded draft.
    assert not res.markers.refused
    assert res.answer_text == "Employees accrue 15 days of PTO"
    assert "2M" not in res.answer_text and "CEO" not in res.answer_text


def test_answer_text_is_exactly_source_spans():
    r1 = _rec("The rate is 5%.", cid="c_rate")
    proposal = {"claims": [{"text": "The rate is 5%. Additionally the fee is 500 dollars.",
                            "cite_ids": ["c_rate"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, [r1])
    assert res.answer_text == "The rate is 5%."
    assert "500" not in res.answer_text


def test_unresolvable_citation_is_dropped():
    rec = _rec("A real fact.", cid="c_real")
    # Proposer cites an id that was never retrieved -> referential integrity fails -> refuse.
    proposal = {"claims": [{"text": "A fabricated fact.", "cite_ids": ["c_ghost"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, [rec])
    assert res.markers.refused
    assert "fabricated" not in res.answer_text


def test_resolving_claim_emits_grounded_span():
    rec = _rec("Full-time employees accrue 15 days of paid time off per year.", cid="c_pto")
    # The proposer paraphrases — this MUST still resolve (referential integrity), and emit the span.
    proposal = {"claims": [{"text": "employees get 15 days off", "cite_ids": ["c_pto"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, [rec])
    assert not res.markers.refused
    assert res.answer_text == rec.text
    assert res.claims[0].citation.filename == "f.md"


def test_verify_claim_none_record_returns_none():
    assert verify_claim("anything", None) is None


def test_duplicate_source_not_emitted_twice():
    rec = _rec("The only fact.", cid="c_dup")
    proposal = {"claims": [
        {"text": "the only fact", "cite_ids": ["c_dup"]},
        {"text": "restating the only fact", "cite_ids": ["c_dup"]},
    ], "refusal_token": None}
    res = answer_from_proposal(proposal, [rec])
    assert res.answer_text == "The only fact."  # emitted once, not doubled
    assert len(res.claims) == 1
