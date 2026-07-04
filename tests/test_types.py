"""BT03: the typed seam. Pure definitions — round-trip the result types and confirm the
Protocols are structurally checkable. No behavior is exercised."""

from docqa.interfaces import Generator, Parser, Retriever
from docqa.types import (
    EXIT_EMPTY_CORPUS,
    EXIT_EMPTY_QUERY,
    AnswerResult,
    Citation,
    Claim,
    ClaimRecord,
    Markers,
    SourceStatus,
    ValueType,
)


def test_claim_record_roundtrip():
    c = ClaimRecord(
        claim_id="abc123",
        filename="handbook.md",
        locator="#PTO Policy",
        text="Full-time employees accrue 15 days of paid time off per year.",
        subject_norm="full-time employees pto",
        predicate_norm="accrue",
        value_span="15 days",
        value_type=ValueType.DURATION,
        value_canon="15 days",
        source_status=SourceStatus.PARSED,
    )
    dumped = c.model_dump()
    restored = ClaimRecord.model_validate(dumped)
    assert restored == c
    assert restored.value_type is ValueType.DURATION


def test_answer_result_roundtrip():
    ar = AnswerResult(
        answer_text="Full-time employees get 15 days of PTO.",
        claims=[
            Claim(
                text="15 days of PTO",
                citation=Citation(filename="handbook.md", locator="#PTO Policy", span="15 days"),
                entailed=True,
                entail_score=0.98,
            )
        ],
        markers=Markers(refused=False, conflict=False),
        latency_ms=1234,
        meta={"gen_model": "test", "seed": 0},
    )
    restored = AnswerResult.model_validate(ar.model_dump())
    assert restored == ar
    assert restored.claims[0].entailed is True


def test_answer_result_defaults_are_safe():
    ar = AnswerResult()
    assert ar.answer_text == ""
    assert ar.claims == []
    assert ar.markers.refused is False
    assert ar.markers.refusal_token is None


def test_exit_codes_reserved():
    assert EXIT_EMPTY_CORPUS == 3
    assert EXIT_EMPTY_QUERY == 4


def test_protocols_are_runtime_checkable():
    # A minimal duck-typed object satisfies the Protocol structurally.
    class _R:
        def retrieve(self, query, k):
            return []

    assert isinstance(_R(), Retriever)

    class _P:
        def can_parse(self, path):
            return True

        def parse(self, path):
            return []

    assert isinstance(_P(), Parser)

    class _G:
        def propose(self, question, claims):
            return {}

    assert isinstance(_G(), Generator)
