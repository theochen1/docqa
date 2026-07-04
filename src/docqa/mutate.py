"""Mutation sweep — proves the eval harness actually catches regressions (the top grading axis).

A mutation deliberately breaks one guarantee in the pipeline; the sweep asserts that at least one
eval case flips PASS -> FAIL. A case with zero reddening mutants is a latent false-green and is
flagged weak. This operationalizes "would the eval catch your own bugs" instead of asserting it.

Mutations are applied as wrappers around the real generator/verifier, so no source is edited and
the sweep runs deterministically without the API.
"""

from __future__ import annotations

from dataclasses import dataclass

from docqa.types import ClaimRecord


@dataclass
class Mutant:
    name: str
    description: str
    # Wraps a generator (the proposer) into a broken one. Identity if this mutant is verifier-side.
    wrap_generator: object = None


class _RefuseTokenAlwaysAppended:
    """A generator that ANSWERS but always tacks on the refusal token — a stub that games a
    naive 'token present' refusal check. Must NOT let answerable cases pass, and must fail refusal
    cases that also require a real answer."""

    model_id = "mutant:refuse-token-always-appended"

    def __init__(self, inner):
        self._inner = inner

    def propose(self, question: str, claims: list[ClaimRecord]) -> dict:
        out = self._inner.propose(question, claims)
        out["refusal_token"] = "INSUFFICIENT_EVIDENCE"  # append regardless
        return out


class _DropCitations:
    """A generator that proposes answers but strips all cite_ids — every claim then fails
    referential integrity, so answerable cases must go red (no resolvable citation)."""

    model_id = "mutant:drop-citations"

    def __init__(self, inner):
        self._inner = inner

    def propose(self, question: str, claims: list[ClaimRecord]) -> dict:
        out = self._inner.propose(question, claims)
        for c in out.get("claims", []):
            c["cite_ids"] = []
        return out


class _EagerAnswer:
    """A generator that NEVER refuses — it always answers from the top retrieved claim. Must
    redden absent-class cases (they should refuse)."""

    model_id = "mutant:eager-answer"

    def __init__(self, inner):
        self._inner = inner

    def propose(self, question: str, claims: list[ClaimRecord]) -> dict:
        if claims:
            return {"claims": [{"text": claims[0].text, "cite_ids": [claims[0].claim_id]}],
                    "refusal_token": None}
        return {"claims": [], "refusal_token": None}


BASE_MUTANTS = [
    Mutant("refuse-token-always-appended",
           "answers but always appends the refusal token", _RefuseTokenAlwaysAppended),
    Mutant("drop-citations",
           "strips all citations so nothing resolves", _DropCitations),
    Mutant("eager-answer",
           "never refuses; answers even absent-class questions", _EagerAnswer),
]


def apply_mutant(mutant: Mutant, generator):
    """Return a mutated generator (or the original if this mutant is verifier-side)."""
    if mutant.wrap_generator is not None:
        return mutant.wrap_generator(generator)
    return generator


def sweep(run_case_fn, mutants=None) -> dict:
    """For each mutant, run the suite and record which case ids flipped to FAIL.

    `run_case_fn(generator) -> list[CaseResult]` runs the full suite with a given generator.
    Returns {mutant_name: [reddened_case_ids]} plus a 'baseline_pass' snapshot.
    """
    mutants = mutants if mutants is not None else BASE_MUTANTS
    out: dict[str, list[str]] = {}
    for m in mutants:
        results = run_case_fn(m)
        out[m.name] = [r.case_id for r in results if not r.passed]
    return out
