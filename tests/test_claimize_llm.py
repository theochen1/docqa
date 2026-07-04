"""BT08: LLM claimizer path. Uses a stub decomposer (no real API). Falls back to the
deterministic path on no-decomposer / failure / empty output — indexing never breaks."""

from docqa.claimize import claimize_llm, claimize_segment_llm
from docqa.types import TextSegment, ValueType


def _seg(text):
    return TextSegment(filename="memo.md", locator="#Body", text=text)


def test_llm_path_uses_decomposer_output():
    # A stub decomposer that atomizes better than sentence-splitting would.
    def stub(text):
        return [
            {"text": "The PTO accrual is 15 days.", "value_span": "15 days"},
            {"text": "Remote work is allowed 3 days per week.", "value_span": "3 days"},
        ]

    claims = claimize_segment_llm(_seg("PTO is 15 days and remote is 3 days/week."), stub)
    assert len(claims) == 2
    # Values still canonicalized deterministically by canon.py.
    assert claims[0].value_canon == "15 days"
    assert claims[0].value_type is ValueType.DURATION
    assert all(c.filename == "memo.md" and c.locator == "#Body" for c in claims)


def test_llm_path_falls_back_when_no_decomposer():
    segs = [_seg("A fact worth 10 points.")]
    out = claimize_llm(segs, decomposer=None)  # no decomposer -> deterministic path
    assert len(out) == 1


def test_llm_path_falls_back_on_failure():
    def boom(text):
        raise RuntimeError("provider down")

    claims = claimize_segment_llm(_seg("A fact worth 10 points."), boom)
    assert len(claims) == 1  # deterministic fallback, not a crash


def test_llm_path_falls_back_on_empty():
    claims = claimize_segment_llm(_seg("A fact worth 10 points."), lambda t: [])
    assert len(claims) == 1


def test_llm_owns_value_extraction():
    # New contract (BT08b): the LLM is the value extractor. If it supplies the span, canon.py
    # gates it; if it omits it, we do NOT regex-recover on the primary path (the model judged
    # there is no salient value). This is the "agent does the NLP, script gates" division.
    with_value = claimize_segment_llm(
        _seg("x"), lambda t: [{"text": "Total is $4,250.00 due.", "value_span": "$4,250.00"}]
    )
    assert with_value[0].value_canon == "4250.00"  # canon gates the model-provided span

    omitted = claimize_segment_llm(_seg("x"), lambda t: [{"text": "Total is $4,250.00 due."}])
    assert omitted[0].value_span == ""  # no regex second-guessing on the LLM path
