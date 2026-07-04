"""BT07: value canonicalization — the deterministic backbone of conflict detection.

The load-bearing property: values that AGREE canonicalize equal (no false conflict) and values
that DISAGREE canonicalize distinct (real conflict fires) — without any model.
"""

from docqa.canon import canonicalize
from docqa.types import ValueType


def test_number_word_and_paren_forms_agree():
    # The over-conflict trap: '15', 'fifteen', 'fifteen (15)' must all canonicalize equal.
    _, a = canonicalize("15")
    _, b = canonicalize("fifteen")
    _, c = canonicalize("fifteen (15)")
    assert a == b == c == "15"


def test_distinct_numbers_stay_distinct():
    # The conflict case: 15 vs 20 must differ so a conflict can fire.
    assert canonicalize("15 days")[1] != canonicalize("20 days")[1]


def test_currency_normalizes_formatting():
    t, v = canonicalize("$4,250.00")
    assert t is ValueType.CURRENCY and v == "4250.00"
    assert canonicalize("$4,250")[1] == "4250.00"  # same regardless of trailing .00


def test_duration_unit_normalized_to_days():
    assert canonicalize("three weeks")[1] == canonicalize("21 days")[1] == "21 days"
    assert canonicalize("1 month")[1] == "30 days"


def test_iso_and_textual_dates_agree():
    t1, v1 = canonicalize("2025-03-01")
    t2, v2 = canonicalize("March 1, 2025")
    assert t1 is ValueType.DATE and v1 == "2025-03-01"
    assert v2 == "2025-03-01"


def test_entity_casefolded():
    t, v = canonicalize("gw-west-2")
    assert t is ValueType.ENTITY and v == "gw-west-2"
    assert canonicalize("Portland")[1] == "portland"


def test_percent_number():
    # Percent is kept distinct from the bare number (review fix): "92%" != "92".
    assert canonicalize("92%")[1] == "92%"
    assert canonicalize("92%")[1] != canonicalize("92")[1]


def test_empty_is_empty_string():
    assert canonicalize("")[1] == ""


def test_deterministic():
    for s in ["15 days", "$4,250.00", "March 1, 2025", "gw-west-2", "fifteen (15)"]:
        assert canonicalize(s) == canonicalize(s)
