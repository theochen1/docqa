"""BT07: deterministic claimizer. Atomic claims, value-separated proposition keys, byte-stable."""

from docqa.claimize import claimize, claimize_segment
from docqa.types import TextSegment, ValueType


def _seg(text, filename="handbook.md", locator="#PTO Policy"):
    return TextSegment(filename=filename, locator=locator, text=text)


def test_one_claim_per_sentence():
    seg = _seg("Full-time employees accrue 15 days of PTO. Remote work is allowed 3 days a week.")
    claims = claimize_segment(seg)
    assert len(claims) == 2
    assert all(c.filename == "handbook.md" and c.locator == "#PTO Policy" for c in claims)


def test_value_lifted_and_canonicalized():
    claims = claimize_segment(_seg("Full-time employees accrue 15 days of PTO per year."))
    c = claims[0]
    assert "15 days" in c.value_span
    assert c.value_type is ValueType.DURATION
    assert c.value_canon == "15 days"


def test_currency_and_id_values():
    inv = claimize_segment(_seg("Total amount due is $4,250.00.", filename="invoice.txt"))[0]
    assert inv.value_canon == "4250.00"
    net = claimize_segment(_seg("The gateway gw-west-2 serves the office.", filename="net.md"))[0]
    assert "gw-west-2" in net.value_span


def test_proposition_key_separates_subject_from_value():
    c = claimize_segment(_seg("Full-time employees accrue 15 days of PTO per year."))[0]
    # subject/predicate exclude the value so 15-vs-20 claims cluster on the same key.
    assert "15" not in c.subject_norm and "15" not in c.predicate_norm
    assert c.subject_norm  # non-empty


def test_claim_ids_stable_and_unique():
    seg = _seg("Alpha fact here. Beta fact there.")
    a = claimize_segment(seg)
    b = claimize_segment(seg)
    assert [c.claim_id for c in a] == [c.claim_id for c in b]      # stable
    assert len({c.claim_id for c in a}) == 2                        # unique per sentence


def test_determinism_over_segments():
    segs = [_seg("Fact one is 10."), _seg("Fact two is 20.", locator="#Two")]
    assert [c.model_dump() for c in claimize(segs)] == [c.model_dump() for c in claimize(segs)]
