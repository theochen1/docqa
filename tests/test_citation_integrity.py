"""Follow-up to the M3 review (HIGH): the citation guarantee must not admit fabricated prose.

The bug: span_resolves accepted `b in a`, so a proposer could append uncited/fabricated content
after one real span and have the whole sentence round-trip as 'cited'. answer_text is emitted from
the proposer's text, so the fabrication reached the user. These tests reproduce it, then pin the
fix: a proposed claim resolves ONLY when it is itself supported within the source span (a in b),
with a minimum-overlap guard against trivial short-fragment matches.
"""

from docqa.citations import span_resolves, verify_claim
from docqa.core import answer_from_proposal
from docqa.types import ClaimRecord


def _rec(text, cid="c1"):
    return ClaimRecord(claim_id=cid, filename="f.md", locator="#h", text=text)


# --- the reproduced leak: fabricated content appended to a real span must NOT resolve ---

def test_fabricated_suffix_does_not_resolve():
    rec = _rec("Employees accrue 15 days of PTO")
    fabricated = "Employees accrue 15 days of PTO and the CEO is Dana Reed who earns 2M"
    assert span_resolves(fabricated, rec) is False


def test_source_engulfed_by_proposal_does_not_resolve():
    rec = _rec("The rate is 5%.")
    proposal = "The rate is 5%. Additionally the fee is 500 dollars."
    assert span_resolves(proposal, rec) is False


def test_fabricated_answer_is_refused_not_emitted():
    rec = _rec("Employees accrue 15 days of PTO", cid="c_pto")
    proposal = {
        "claims": [{"text": "Employees accrue 15 days of PTO and the CEO earns 2M",
                    "cite_ids": ["c_pto"]}],
        "refusal_token": None,
    }
    res = answer_from_proposal(proposal, [rec])
    assert res.markers.refused
    assert "2M" not in res.answer_text
    assert "CEO" not in res.answer_text


# --- what SHOULD still resolve: a faithful sub-span of the source ---

def test_faithful_subspan_resolves():
    rec = _rec("Full-time employees accrue 15 days of paid time off per year.")
    assert span_resolves("employees accrue 15 days of paid time off", rec) is True
    vc = verify_claim("employees accrue 15 days of paid time off", rec)
    assert vc is not None and vc.entailed is True


def test_exact_match_resolves():
    rec = _rec("The VPN gateway is gw-west-2.")
    assert span_resolves("The VPN gateway is gw-west-2.", rec) is True


# --- the short-fragment guard (finding #7): a tiny fragment must not trivially match ---

def test_trivial_short_fragment_does_not_resolve():
    rec = _rec("The comprehensive quarterly budget report covers all departments and regions.")
    # "the" is a substring but carries no real support -> must not resolve.
    assert span_resolves("the", rec) is False
    assert span_resolves("all", rec) is False
