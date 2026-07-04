"""BT18: entailment gate (R-ENTAIL). A resolvable-but-unentailed claim must be dropped -> refuse;
an entailed claim must survive. Both directions pinned so the gate can't rot to a no-op."""

from docqa.core import answer_from_proposal
from docqa.entail import batch_entails
from docqa.types import ClaimRecord


def _rec(cid, text):
    return ClaimRecord(claim_id=cid, filename="region_registry.md", locator="#gw", text=text)


# The gw-north-4 red herring: the span mentions the gateway AND 'Reykjavik' but assigns no region.
_NORTH4 = _rec("c_n4", "Provisioning for gw-north-4 is pending; no region has been assigned yet. "
                       "Reykjavik remains a candidate site for a future gateway.")
_SOUTH1 = _rec("c_s1", "The gw-south-1 gateway is deployed in the Portland datacenter.")


def test_unentailed_claim_dropped_then_refuse():
    # Stubbed proposer forces the resolvable-but-unentailed edge: claims gw-north-4 is in Reykjavik,
    # citing the REAL gw-north-4 span (referential integrity passes). The gate must reject it.
    def judge(claim, span):
        return "reykjavik" not in claim.lower()  # the trap proposition is not entailed

    proposal = {"claims": [{"text": "gw-north-4 is deployed in Reykjavik.", "cite_ids": ["c_n4"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, [_NORTH4], entail_judge=judge)
    assert res.markers.refused                     # dropped -> nothing left -> refuse
    assert "Reykjavik" not in res.answer_text      # the trap answer never emitted


def test_entailed_claim_survives():
    # The gate is NOT a blanket refuse valve: a genuinely-entailed claim must be emitted.
    def judge(claim, span):
        return "portland" in span.lower()

    proposal = {"claims": [{"text": "gw-south-1 is in Portland.", "cite_ids": ["c_s1"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, [_SOUTH1], entail_judge=judge)
    assert not res.markers.refused
    assert "Portland" in res.answer_text


def test_mutation_entailment_disabled_lets_trap_through():
    # entailment-disabled mutant = always-accept judge. The trap claim now survives (BAD) — this is
    # exactly what T22-ENTAIL-REDHERRING must catch, proving the gate is load-bearing.
    always_accept = lambda claim, span: True  # noqa: E731
    proposal = {"claims": [{"text": "gw-north-4 is deployed in Reykjavik.", "cite_ids": ["c_n4"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, [_NORTH4], entail_judge=always_accept)
    # With the gate disabled the answer is emitted (the source span) — NOT a refusal. The eval
    # case reddening on this is what proves the gate matters.
    assert not res.markers.refused


def test_mutation_entailment_over_strict_reddens_positives():
    # entailment-over-strict mutant = always-reject. A genuinely-answerable claim now refuses,
    # which must redden positive cases (guards over-refusal in the sweep).
    always_reject = lambda claim, span: False  # noqa: E731
    proposal = {"claims": [{"text": "gw-south-1 is in Portland.", "cite_ids": ["c_s1"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, [_SOUTH1], entail_judge=always_reject)
    assert res.markers.refused  # over-strict -> drops a real answer -> refuse


def test_no_judge_preserves_bt13_behavior():
    # Without a judge, only referential integrity gates (the claim resolves and is emitted).
    proposal = {"claims": [{"text": "anything", "cite_ids": ["c_s1"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, [_SOUTH1], entail_judge=None)
    assert not res.markers.refused


def test_batch_entails_maps_over_pairs():
    judge = lambda c, s: c in s  # noqa: E731
    out = batch_entails(judge, [("Portland", "in Portland"), ("Reykjavik", "no region")])
    assert out == [True, False]
