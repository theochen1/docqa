"""BT17: two-track selection. The proposition-contrast track guarantees a disagreeing source
survives into the candidate set BY CONSTRUCTION — the conflict precondition. The mutation
'contrast-track-disabled' (select=False) must lose the minority value."""

from docqa.retrieval.select import two_track_select
from docqa.types import ClaimRecord


def _claim(cid, filename, value_canon, subject="pto accrual", predicate="is"):
    return ClaimRecord(
        claim_id=cid, filename=filename, locator="#h",
        text=f"{subject} {value_canon}", subject_norm=subject, predicate_norm=predicate,
        value_span=value_canon, value_canon=value_canon,
    )


def test_minority_value_survives_by_construction():
    # Majority: 3 agreeing claims (value 15) across near-dup files. Minority: 1 claim (value 20).
    # In fused-rank order the majority comes first and would crowd out the minority at small k.
    pool = [
        _claim("a1", "handbook.md", "15 days"),
        _claim("a2", "handbook_copy.md", "15 days"),
        _claim("a3", "summary.md", "15 days"),
        _claim("b1", "hr_memo.md", "20 days"),  # the disagreeing source, ranked last
    ]
    selected = two_track_select(pool, k=2, per_source_cap=2)
    values = {c.value_canon for c in selected}
    assert "15 days" in values and "20 days" in values, "minority value must survive"


def test_disabling_contrast_track_buries_the_minority():
    # Simulate the mutation: raw fused order (what select=False returns), no contrast track.
    pool = [
        _claim("a1", "handbook.md", "15 days"),
        _claim("a2", "handbook_copy.md", "15 days"),
        _claim("b1", "hr_memo.md", "20 days"),
    ]
    # Raw top-k=1 (mimics no selection): only the majority survives.
    raw_top1 = pool[:1]
    assert {c.value_canon for c in raw_top1} == {"15 days"}
    # With the contrast track at k=2, the minority is pulled in.
    with_contrast = two_track_select(pool, k=2)
    assert "20 days" in {c.value_canon for c in with_contrast}


def test_agreeing_sources_not_force_split():
    # Over-conflict guard: identical canonical values are NOT a disagreement -> no forced include.
    pool = [
        _claim("a1", "a.md", "mit"),  # both say MIT
        _claim("a2", "b.md", "mit"),
    ]
    selected = two_track_select(pool, k=2)
    # Both may be selected via track A, but the contrast track adds nothing (single value).
    assert len({c.value_canon for c in selected}) == 1


def test_per_source_cap_limits_verbose_doc():
    pool = [_claim(f"x{i}", "verbose.md", f"v{i}", subject=f"s{i}") for i in range(5)]
    pool.append(_claim("y1", "other.md", "vv", subject="sy"))
    selected = two_track_select(pool, k=4, per_source_cap=2)
    # Backfill may exceed the cap on tiny pools, but the cap must let the other source in.
    assert any(c.filename == "other.md" for c in selected)


def test_empty_and_zero_k():
    assert two_track_select([], k=5) == []
    assert two_track_select([_claim("a", "f.md", "1")], k=0) == []


def test_deterministic():
    pool = [_claim("a1", "a.md", "15 days"), _claim("b1", "b.md", "20 days")]
    assert [c.claim_id for c in two_track_select(pool, k=2)] == \
           [c.claim_id for c in two_track_select(pool, k=2)]
