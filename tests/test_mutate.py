"""BT15: mutation sweep. Each mutant must redden the case(s) it breaks — proving the harness has
real regression value. Driven by a scripted stub generator over the committed sample corpus."""

import yaml

from docqa.embed import HashingEmbedder
from docqa.eval_harness import DEFAULT_CORPUS, run_eval
from docqa.index_store import IndexStore
from docqa.ingest import build_index
from docqa.mutate import BASE_MUTANTS, apply_mutant, sweep
from docqa.retrieval.dense import DenseRetriever

# A controlled mini-suite a keyword stub can drive cleanly: one answerable, one unanswerable.
# (The full guarantee suite runs against the real model via `docqa eval`.) The mutants must still
# redden these: eager-answer breaks the refusal case, drop-citations breaks the answerable one.
_MINI = {"cases": [
    {"id": "answer-remote", "question": "How many days per week may employees work remotely?",
     "expect": {"refused": False, "gold": ["3 days"], "cite_files": ["handbook.md"]}},
    {"id": "refuse-budget", "question": "What is the FY27 marketing budget?",
     "expect": {"refused": True, "forbidden": ["$"]}},
]}


class ScriptedGen:
    """Answers the remote-work question from retrieved claims; refuses everything else."""

    model_id = "scripted"

    def propose(self, question, claims):
        if "remotely" in question.lower():
            for c in claims:
                if "3 days" in c.text:
                    return {"claims": [{"text": c.text, "cite_ids": [c.claim_id]}],
                            "refusal_token": None}
        return {"claims": [], "refusal_token": "INSUFFICIENT_EVIDENCE"}


def _runner(tmp_path):
    idx = str(tmp_path / "eval.db")
    emb = HashingEmbedder(dim=256)
    build_index(DEFAULT_CORPUS, idx, emb)
    suite = tmp_path / "mini.yaml"
    suite.write_text(yaml.safe_dump(_MINI), encoding="utf-8")

    def run_case_fn(mutant):
        gen = apply_mutant(mutant, ScriptedGen())
        results, _ = run_eval(
            DEFAULT_CORPUS, str(suite),
            lambda: DenseRetriever(IndexStore(idx), emb), gen, k=12,
        )
        return results

    return run_case_fn


def test_baseline_all_pass(tmp_path):
    # Sanity: the honest generator passes every case (nothing reddened without a mutant).
    run_case_fn = _runner(tmp_path)

    class _Identity:
        name = "identity"
        wrap_generator = None

    reddened = run_case_fn(_Identity())
    assert all(r.passed for r in reddened), [r.reasons for r in reddened if not r.passed]


def test_every_mutant_reddens_something(tmp_path):
    run_case_fn = _runner(tmp_path)
    result = sweep(run_case_fn, BASE_MUTANTS)
    for name, reddened in result.items():
        assert reddened, f"mutant {name!r} reddened NO case (false-green: suite can't catch it)"


def test_eager_answer_reddens_the_absent_case(tmp_path):
    run_case_fn = _runner(tmp_path)
    result = sweep(run_case_fn, [m for m in BASE_MUTANTS if m.name == "eager-answer"])
    # Answering the unanswerable FY27-budget question must flip the refusal case red.
    assert "refuse-budget" in result["eager-answer"]


def test_drop_citations_reddens_answerable_cases(tmp_path):
    run_case_fn = _runner(tmp_path)
    result = sweep(run_case_fn, [m for m in BASE_MUTANTS if m.name == "drop-citations"])
    reddened = result["drop-citations"]
    assert "answer-remote" in reddened  # no citation resolves -> can't confirm the answer
