"""Follow-up to the M3 review (conflict-backbone MEDIUM/LOW): value canonicalization must keep
genuinely-different values distinct. Each test reproduces a review finding."""

from docqa.canon import canonicalize
from docqa.claimize import _extract_value, claimize_segment
from docqa.types import TextSegment, ValueType

# --- finding: negative sign dropped -> -5 collided with 5 ---

def test_negative_sign_preserved_in_extraction():
    assert _extract_value("temperature is -5 degrees") == "-5"
    assert _extract_value("a delta of -15") == "-15"


def test_negative_and_positive_do_not_collide():
    assert canonicalize("-5")[1] != canonicalize("5")[1]
    seg_neg = claimize_segment(TextSegment(filename="f.md", locator="#h", text="Delta is -5."))[0]
    seg_pos = claimize_segment(TextSegment(filename="f.md", locator="#h", text="Delta is 5."))[0]
    assert seg_neg.value_canon != seg_pos.value_canon


# --- finding: spelled-out durations never extracted in claimize ---

def test_spelled_duration_extracted_and_matches_numeric():
    span = _extract_value("PTO is three weeks per year")
    assert span  # non-empty: "three weeks" is extracted
    assert canonicalize(span)[1] == canonicalize("21 days")[1] == "21 days"


# --- finding: huge integers canonicalized to 'inf' ---

def test_huge_integers_stay_distinct():
    big1 = "123456789012345678901234567890"
    big2 = "123456789012345678901234567891"
    v1 = canonicalize(big1)[1]
    v2 = canonicalize(big2)[1]
    assert v1 != "inf" and v2 != "inf"
    assert v1 != v2  # arbitrary-precision int, not float overflow


# --- finding: percent collided with the bare number ---

def test_percent_distinct_from_bare_number():
    assert canonicalize("92%")[1] != canonicalize("92")[1]
    assert canonicalize("92%")[0] is ValueType.NUMBER  # still numeric, but marked as percent
