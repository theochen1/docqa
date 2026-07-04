"""BT19: conflict detection + surfacing. Deterministic value-mismatch (not NLI); both sides + a
marker; agreeing/duplicate values don't fire (over-conflict guard)."""

from docqa.conflict import detect_conflicts
from docqa.core import answer_from_proposal
from docqa.types import ClaimRecord


def _c(cid, filename, value_canon, subject="pto accrual", predicate="is", text=None):
    return ClaimRecord(
        claim_id=cid, filename=filename, locator="#h",
        text=text or f"{subject} {value_canon}", subject_norm=subject, predicate_norm=predicate,
        value_span=value_canon, value_canon=value_canon,
    )


# --- detection ---

def test_distinct_values_two_files_is_conflict():
    claims = [_c("a", "handbook.md", "15 days"), _c("b", "hr_memo.md", "20 days")]
    cfs = detect_conflicts(claims)
    assert len(cfs) == 1
    assert {s.value_canon for s in cfs[0].sides} == {"15 days", "20 days"}


def test_agreeing_values_no_conflict():
    # Over-conflict guard: same canonical value from two files is NOT a conflict.
    claims = [_c("a", "handbook.md", "15 days"), _c("b", "summary.md", "15 days")]
    assert detect_conflicts(claims) == []


def test_same_file_distinct_values_not_conflict():
    # Needs >=2 distinct FILES; one doc restating itself isn't a cross-source contradiction.
    claims = [_c("a", "x.md", "15 days"), _c("b", "x.md", "20 days")]
    assert detect_conflicts(claims) == []


def test_different_propositions_not_conflict():
    claims = [_c("a", "x.md", "15 days", subject="pto"),
              _c("b", "y.md", "3 days", subject="remote work")]
    assert detect_conflicts(claims) == []


# --- surfacing through the answer path ---

def test_answer_surfaces_conflict_with_marker_and_both_sides():
    retrieved = [_c("a", "handbook.md", "15 days"), _c("b", "hr_memo.md", "20 days")]
    proposal = {"claims": [{"text": retrieved[0].text, "cite_ids": ["a"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, retrieved)
    assert res.markers.conflict is True
    assert "15 days" in res.answer_text and "20 days" in res.answer_text
    cited = {c.citation.filename for c in res.claims}
    assert {"handbook.md", "hr_memo.md"} <= cited


def test_mutation_silent_pick_would_lose_the_marker():
    # A "silent-pick" implementation emits only the proposer's one side, no marker. The T06 case's
    # conflict:true assertion is exactly what catches that regression. Here we prove the correct
    # path fires the marker (so its absence, under a mutant, reddens T06).
    retrieved = [_c("a", "handbook.md", "15 days"), _c("b", "hr_memo.md", "20 days")]
    proposal = {"claims": [{"text": retrieved[0].text, "cite_ids": ["a"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, retrieved)
    assert res.markers.conflict  # a silent-pick mutant would make this False -> reddens T06


def test_no_conflict_when_answer_untouched_proposition():
    # Conflict on PTO exists in the corpus, but a remote-work answer must NOT inherit it.
    retrieved = [
        _c("a", "handbook.md", "15 days", subject="pto"),
        _c("b", "hr_memo.md", "20 days", subject="pto"),
        _c("r", "handbook.md", "3 days", subject="remote work", text="Remote work up to 3 days."),
    ]
    proposal = {"claims": [{"text": "Remote work up to 3 days.", "cite_ids": ["r"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, retrieved)
    assert res.markers.conflict is False
    assert "3 days" in res.answer_text
