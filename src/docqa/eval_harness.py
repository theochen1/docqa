"""docqa eval harness — the mandatory deliverable + the regression gate.

Indexes the committed sample corpus, runs each case in cases.yaml through the SAME
core.answer_question the CLI uses, applies MECHANICAL pass/fail checks, prints a scoreboard, and
exits non-zero on any failure. No "it seemed to work" — every case is a deterministic assertion
over the AnswerResult.

Checks (all mechanical):
- refusal cases: markers.refused must be True; forbidden substrings must be ABSENT.
- answerable cases: markers.refused must be False; every gold substring present; each declared
  cite_file must be the source of at least one resolvable citation.

Determinism: generation is pinned temp=0. Assertions target the deterministic layer (gold
substrings, refusal marker, citation filenames), never exact prose — robust to wording drift.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from docqa.core import answer_question
from docqa.latency import LatencyReport, Timer
from docqa.types import AnswerResult

# src/docqa/eval_harness.py -> repo root is three parents up.
_REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS = str(_REPO / "sample_corpus")
DEFAULT_CASES = str(_REPO / "eval" / "cases.yaml")


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    guarantees: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def check_case(case: dict, result: AnswerResult) -> CaseResult:
    """Apply mechanical checks for one case against its AnswerResult.

    Checks are compositional, driven by which keys the case declares:
    - refused: true  -> must refuse; refused: false -> must NOT over-refuse; omitted -> either is
      acceptable (used by safe-behavior cases like the injection/entailment red herrings, where
      answering a DIFFERENT correct fact is fine as long as forbidden content never appears).
    - gold / cite_files -> only meaningful on an answer; skipped when the tool legitimately refuses.
    - forbidden -> ALWAYS checked against answer_text, refuse or answer.
    """
    expect = case.get("expect", {})
    reasons: list[str] = []
    answer_norm = _norm(result.answer_text)

    if "refused" in expect:
        if expect["refused"] and not result.markers.refused:
            reasons.append("expected refusal but tool answered")
        if not expect["refused"] and result.markers.refused:
            reasons.append("expected an answer but tool refused (over-refusal)")

    if "refusal_token" in expect and result.markers.refusal_token != expect["refusal_token"]:
        reasons.append(
            f"expected refusal_token {expect['refusal_token']!r}, "
            f"got {result.markers.refusal_token!r}"
        )

    if "conflict" in expect:
        if expect["conflict"] and not result.markers.conflict:
            reasons.append("expected a CONFLICT marker but none was surfaced")
        if not expect["conflict"] and result.markers.conflict:
            reasons.append("unexpected CONFLICT marker (over-conflict on agreeing sources)")

    # forbidden substrings must never appear, regardless of refuse/answer.
    for bad in expect.get("forbidden", []):
        if _norm(bad) and _norm(bad) in answer_norm:
            reasons.append(f"forbidden substring present: {bad!r}")

    # gold + citations only apply when the tool answered (not on a legitimate refusal).
    if not result.markers.refused:
        for gold in expect.get("gold", []):
            if _norm(gold) not in answer_norm:
                reasons.append(f"missing gold substring: {gold!r}")
        cited = {c.citation.filename for c in result.claims}
        for f in expect.get("cite_files", []):
            if f not in cited:
                reasons.append(f"expected a citation to {f!r}, got {sorted(cited)}")

    return CaseResult(
        case_id=case["id"],
        passed=not reasons,
        reasons=reasons,
        guarantees=case.get("guarantees", []),
    )


def load_cases(cases_path: str) -> list[dict]:
    data = yaml.safe_load(Path(cases_path).read_text(encoding="utf-8"))
    return data.get("cases", [])


def run_eval(
    corpus_dir: str,
    cases_path: str,
    build_retriever,
    generator,
    k: int = 8,
    verbose: bool = False,
    latency: LatencyReport | None = None,
    entail_judge=None,
    oos_floor: float | None = None,
) -> tuple[list[CaseResult], LatencyReport]:
    """Run all cases. build_retriever() -> a Retriever over the freshly-built index; generator is
    the proposer; entail_judge (optional) is the R-ENTAIL gate. All injected so tests can drive
    with stubs and the CLI with the real stack. Per-case latency is recorded into `latency`."""
    cases = load_cases(cases_path)
    retriever = build_retriever()
    report = latency if latency is not None else LatencyReport()
    results: list[CaseResult] = []
    for case in cases:
        with Timer() as t:
            result = answer_question(case["question"], k, retriever, generator,
                                     entail_judge=entail_judge, oos_floor=oos_floor)
        report.samples_ms.append(t.elapsed_ms)
        cr = check_case(case, result)
        results.append(cr)
        if verbose:
            print(f"  [{case['id']}] {'PASS' if cr.passed else 'FAIL'}: {case.get('notes','')}",
                  file=sys.stderr)
            for r in cr.reasons:
                print(f"      - {r}", file=sys.stderr)
    return results, report


def format_scoreboard(results: list[CaseResult]) -> str:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    lines = [f"docqa eval: {passed}/{total} cases passed"]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{mark}] {r.case_id}")
        for reason in r.reasons:
            lines.append(f"      - {reason}")
    lines.append(f"RESULT pass={passed} fail={total - passed} exit={0 if passed == total else 1}")
    return "\n".join(lines)
