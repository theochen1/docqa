"""The one shared entry point: answer_question(question, k) -> AnswerResult.

CLI, web, and eval all call this — no guarantee logic lives anywhere else. BT13 wires the thin
slice: retrieve -> propose -> resolve-check -> assemble-from-verified-claims / else refuse.

answer_text is ASSEMBLED by the verifier from surviving verified claims (never free LLM prose), so
no uncited sentence can appear. Conflict/entailment/calibration layer on in M5.
"""

from __future__ import annotations

from docqa.citations import verify_claim
from docqa.conflict import detect_conflicts
from docqa.generate import build_prompt
from docqa.types import AnswerResult, Citation, Claim, ClaimRecord, Markers


def _proposition_key(c: ClaimRecord) -> str:
    return f"{c.subject_norm}|{c.predicate_norm}".strip("|")


def _assemble(verified: list[Claim]) -> str:
    """answer_text is the verified claim texts joined — a function of claims[], not free prose."""
    return " ".join(c.text for c in verified)


def answer_from_proposal(
    proposal: dict,
    retrieved: list[ClaimRecord],
    entail_judge=None,
) -> AnswerResult:
    """Verify a proposer's output against the retrieved claims and assemble the result.

    Pure + provider-agnostic: takes an already-normalized proposal ({claims, refusal_token}) and
    the retrieved records. This is the DISPOSE step, unit-testable without any LLM.

    `entail_judge` (R-ENTAIL, BT18): optional callable (claim_text, span_text) -> bool. When
    supplied, a claim that resolves (referential integrity) but whose span does NOT entail the
    proposed claim is DROPPED. If dropping leaves nothing, we refuse. When None, only referential
    integrity gates (the BT13 behavior).
    """
    by_id = {c.claim_id: c for c in retrieved}

    # Explicit refusal from the proposer, or no claims proposed -> refuse.
    if proposal.get("refusal_token") or not proposal.get("claims"):
        return AnswerResult(
            markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"),
        )

    verified: list[Claim] = []
    seen_records: set[str] = set()
    for item in proposal["claims"]:
        text = item["text"]
        for cid in item.get("cite_ids", []):
            rec = by_id.get(cid)
            if rec is None:
                continue  # cite didn't resolve to a retrieved record (referential integrity)
            if rec.claim_id in seen_records:
                break  # this source already emitted; don't repeat the span
            vc = verify_claim(text, rec)
            if vc is None:
                continue
            # R-ENTAIL: the cited span must actually support the proposed claim, not just resolve.
            if entail_judge is not None and not entail_judge(text, rec.text):
                vc.entailed = False
                break  # unentailed -> drop this claim (don't emit an unsupported cell)
            verified.append(vc)
            seen_records.add(rec.claim_id)
            break  # one resolving+entailing citation is enough for this claim

    if not verified:
        # Proposer answered but nothing resolved to a real source -> refuse, don't emit uncited.
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

    # Conflict-surfacing: if the answer touches a proposition on which the RETRIEVED evidence
    # disagrees (>=2 distinct canonical values across >=2 files), surface BOTH sides + a marker,
    # rather than silently emitting the one side the proposer happened to pick.
    verified_props = _verified_props(verified, retrieved)
    conflicts = [cf for cf in detect_conflicts(retrieved) if cf.proposition in verified_props]
    if conflicts:
        conflict_claims: list[Claim] = []
        for cf in conflicts:
            for side in cf.sides:
                conflict_claims.append(
                    Claim(
                        text=side.text,
                        citation=Citation(filename=side.filename, locator=side.locator,
                                          span=side.text),
                        entailed=True,
                        entail_score=1.0,
                    )
                )
        answer = "Sources disagree: " + " | ".join(c.text for c in conflict_claims)
        return AnswerResult(
            answer_text=answer,
            claims=conflict_claims,
            markers=Markers(conflict=True, warning="conflicting sources surfaced"),
        )

    return AnswerResult(answer_text=_assemble(verified), claims=verified, markers=Markers())


def _verified_props(verified: list[Claim], retrieved: list[ClaimRecord]) -> set[str]:
    """Proposition keys of the retrieved records that back the verified (emitted) claims."""
    by_span = {(r.filename, r.locator, r.text): r for r in retrieved}
    props: set[str] = set()
    for c in verified:
        r = by_span.get((c.citation.filename, c.citation.locator, c.citation.span))
        if r is not None:
            props.add(_proposition_key(r))
    return props


def answer_question(question: str, k: int, retriever, generator, entail_judge=None) -> AnswerResult:
    """Full path. retriever + generator (+ optional entailment judge) are injected seams."""
    if not question or not question.strip():
        # Degenerate empty query — handled properly (reserved exit code) by the CLI at BT20b.
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

    retrieved = retriever.retrieve(question, k)
    if not retrieved:
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

    # The proposer sees id+text only; build_prompt's id_map is applied inside the generator.
    proposal = generator.propose(question, retrieved)
    result = answer_from_proposal(proposal, retrieved, entail_judge=entail_judge)
    # Stamp determinism/audit meta.
    result.meta = {
        "gen_model": getattr(generator, "model_id", "unknown"),
        "k": k,
    }
    return result


# Kept importable for callers that only need the prompt shape (e.g. debugging / --why later).
__all__ = ["answer_question", "answer_from_proposal", "build_prompt"]
