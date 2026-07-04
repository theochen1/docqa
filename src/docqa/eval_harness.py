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
    """Apply mechanical checks for one case against its AnswerResult."""
    expect = case.get("expect", {})
    reasons: list[str] = []
    answer_norm = _norm(result.answer_text)

    if expect.get("refused"):
        if not result.markers.refused:
            reasons.append("expected refusal but tool answered")
        for bad in expect.get("forbidden", []):
            if _norm(bad) and _norm(bad) in answer_norm:
                reasons.append(f"forbidden substring present: {bad!r}")
    else:
        if result.markers.refused:
            reasons.append("expected an answer but tool refused (over-refusal)")
        for gold in expect.get("gold", []):
            if _norm(gold) not in answer_norm:
                reasons.append(f"missing gold substring: {gold!r}")
        for bad in expect.get("forbidden", []):
            if _norm(bad) and _norm(bad) in answer_norm:
                reasons.append(f"forbidden substring present: {bad!r}")
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
) -> tuple[list[CaseResult], LatencyReport]:
    """Run all cases. build_retriever() -> a Retriever over the freshly-built index; generator is
    the proposer. Both injected so tests can drive with stubs and the CLI with the real stack.
    Per-case query latency is recorded into `latency` (a fresh LatencyReport if not supplied)."""
    cases = load_cases(cases_path)
    retriever = build_retriever()
    report = latency if latency is not None else LatencyReport()
    results: list[CaseResult] = []
    for case in cases:
        with Timer() as t:
            result = answer_question(case["question"], k, retriever, generator)
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
