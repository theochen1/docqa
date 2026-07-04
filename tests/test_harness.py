"""BT14: eval harness. check_case logic + a full stub-driven run over the sample corpus.

Uses a scripted stub generator (no API) that answers/refuses per question, so we test the harness
mechanics deterministically. The real API run is exercised by `docqa eval` with a key.
"""

from pathlib import Path

from docqa.embed import HashingEmbedder
from docqa.eval_harness import (
    DEFAULT_CASES,
    DEFAULT_CORPUS,
    check_case,
    format_scoreboard,
    run_eval,
)
from docqa.index_store import IndexStore
from docqa.ingest import build_index
from docqa.retrieval.dense import DenseRetriever
from docqa.types import AnswerResult, Citation, Claim, Markers

_REPO = Path(__file__).resolve().parent.parent


# --- check_case mechanics ---

def _answer(text, files):
    claims = [Claim(text=text, citation=Citation(filename=f, locator="#h", span=text),
                    entailed=True) for f in files]
    return AnswerResult(answer_text=text, claims=claims, markers=Markers())


def _refusal():
    return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))


def test_answerable_case_passes_with_gold_and_cite():
    case = {"id": "x", "question": "q",
            "expect": {"refused": False, "gold": ["15 days"], "cite_files": ["handbook.md"]}}
    r = check_case(case, _answer("Employees get 15 days of PTO.", ["handbook.md"]))
    assert r.passed


def test_answerable_missing_gold_fails():
    case = {"id": "x", "question": "q", "expect": {"refused": False, "gold": ["15 days"]}}
    r = check_case(case, _answer("Employees get some PTO.", ["handbook.md"]))
    assert not r.passed and any("gold" in reason for reason in r.reasons)


def test_answerable_missing_expected_citation_fails():
    case = {"id": "x", "question": "q",
            "expect": {"refused": False, "gold": ["15 days"], "cite_files": ["handbook.md"]}}
    r = check_case(case, _answer("Employees get 15 days.", ["wrong.md"]))
    assert not r.passed and any("citation" in reason for reason in r.reasons)


def test_over_refusal_fails_answerable_case():
    case = {"id": "x", "question": "q", "expect": {"refused": False, "gold": ["15 days"]}}
    r = check_case(case, _refusal())
    assert not r.passed and any("over-refusal" in reason for reason in r.reasons)


def test_refusal_case_passes_on_refusal():
    case = {"id": "x", "question": "q", "expect": {"refused": True, "forbidden": ["$"]}}
    assert check_case(case, _refusal()).passed


def test_refusal_case_fails_if_answered():
    case = {"id": "x", "question": "q", "expect": {"refused": True}}
    r = check_case(case, _answer("The budget is $5M.", ["x.md"]))
    assert not r.passed


def test_forbidden_substring_on_refusal_fails():
    case = {"id": "x", "question": "q", "expect": {"refused": True, "forbidden": ["Paris"]}}
    # A refusal that nonetheless leaked a forbidden token.
    res = AnswerResult(answer_text="It might be Paris.",
                       markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))
    assert not check_case(case, res).passed


# --- full stub-driven run over the real sample corpus ---

def test_full_run_with_scripted_stub(tmp_path):
    # Build a real index over the committed sample corpus with the hashing embedder (no download).
    idx = str(tmp_path / "eval.db")
    emb = HashingEmbedder(dim=256)
    build_index(DEFAULT_CORPUS, idx, emb)  # deterministic claimizer (no key needed)

    def build_retriever():
        return DenseRetriever(IndexStore(idx), emb)

    # Scripted generator: answers PTO + office-hours correctly, refuses the FY27 budget.
    class ScriptedGen:
        model_id = "scripted"

        def propose(self, question, claims):
            q = question.lower()
            if "pto" in q:
                for c in claims:
                    if "15 days" in c.text:
                        return {"claims": [{"text": c.text, "cite_ids": [c.claim_id]}],
                                "refusal_token": None}
            if "office hours" in q:
                for c in claims:
                    if "9am" in c.text:
                        return {"claims": [{"text": c.text, "cite_ids": [c.claim_id]}],
                                "refusal_token": None}
            return {"claims": [], "refusal_token": "INSUFFICIENT_EVIDENCE"}

    results, _ = run_eval(DEFAULT_CORPUS, DEFAULT_CASES, build_retriever, ScriptedGen(), k=8)
    by_id = {r.case_id: r for r in results}
    assert by_id["T00-BASE"].passed, by_id["T00-BASE"].reasons
    assert by_id["T-CARD1"].passed, by_id["T-CARD1"].reasons
    assert by_id["T03-ABSENT"].passed, by_id["T03-ABSENT"].reasons


def test_scoreboard_and_exit_semantics():
    from docqa.eval_harness import CaseResult

    good = [CaseResult("a", True), CaseResult("b", True)]
    board = format_scoreboard(good)
    assert "2/2" in board and "exit=0" in board
    mixed = [CaseResult("a", True), CaseResult("b", False, ["bad"])]
    assert "exit=1" in format_scoreboard(mixed)
