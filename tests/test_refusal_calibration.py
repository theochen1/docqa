"""BT20a/b: refusal calibration (OUT_OF_SCOPE vs INSUFFICIENT_EVIDENCE via raw dense-cosine floor)
+ reserved degenerate-input exit codes."""

from docqa.core import answer_question
from docqa.types import EXIT_EMPTY_CORPUS, EXIT_EMPTY_QUERY, ClaimRecord


class _Retriever:
    """Stub retriever with a controllable top-similarity and a fixed retrieved set."""

    def __init__(self, sim, claims):
        self._sim = sim
        self._claims = claims

    def retrieve(self, query, k):
        return self._claims

    def top_similarity(self, query):
        return self._sim


class _RefuseGen:
    model_id = "refuse"

    def propose(self, question, claims):
        return {"claims": [], "refusal_token": None}  # nothing proposed -> refuse


_CLAIM = ClaimRecord(claim_id="c1", filename="f.md", locator="#h", text="an on-topic fact")


def test_below_floor_is_out_of_scope():
    # Low similarity + a refusal -> OUT_OF_SCOPE.
    r = _Retriever(sim=0.05, claims=[_CLAIM])
    res = answer_question("capital of France?", 5, r, _RefuseGen(), oos_floor=0.30)
    assert res.markers.refused
    assert res.markers.refusal_token == "OUT_OF_SCOPE"


def test_above_floor_is_insufficient_evidence():
    # On-topic (high similarity) but unanswerable -> INSUFFICIENT_EVIDENCE, NOT out-of-scope.
    r = _Retriever(sim=0.62, claims=[_CLAIM])
    res = answer_question("what is the retry limit?", 5, r, _RefuseGen(), oos_floor=0.30)
    assert res.markers.refused
    assert res.markers.refusal_token == "INSUFFICIENT_EVIDENCE"


def test_no_floor_keeps_insufficient_evidence():
    r = _Retriever(sim=0.01, claims=[_CLAIM])
    res = answer_question("q", 5, r, _RefuseGen(), oos_floor=None)
    assert res.markers.refusal_token == "INSUFFICIENT_EVIDENCE"


def test_empty_retrieval_below_floor_is_oos():
    r = _Retriever(sim=0.0, claims=[])
    res = answer_question("q", 5, r, _RefuseGen(), oos_floor=0.30)
    assert res.markers.refused and res.markers.refusal_token == "OUT_OF_SCOPE"


def test_answered_case_not_touched_by_oos():
    # An answer (not a refusal) must never be relabeled OUT_OF_SCOPE regardless of similarity.
    class _AnswerGen:
        model_id = "a"

        def propose(self, question, claims):
            return {"claims": [{"text": claims[0].text, "cite_ids": [claims[0].claim_id]}],
                    "refusal_token": None}

    r = _Retriever(sim=0.01, claims=[_CLAIM])
    res = answer_question("q", 5, r, _AnswerGen(), oos_floor=0.30)
    assert not res.markers.refused


# --- BT20b: reserved exit codes (unit-level: the CLI returns these) ---

def test_exit_codes_are_reserved_values():
    assert EXIT_EMPTY_CORPUS == 3
    assert EXIT_EMPTY_QUERY == 4


def test_empty_query_short_circuits_to_refusal():
    # answer_question itself refuses on empty query (the CLI maps it to EXIT_EMPTY_QUERY).
    r = _Retriever(sim=0.9, claims=[_CLAIM])
    res = answer_question("   ", 5, r, _RefuseGen())
    assert res.markers.refused
