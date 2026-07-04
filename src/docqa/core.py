"""The one shared entry point: answer_question(question, k) -> AnswerResult.

CLI, web, and eval all call this — no guarantee logic lives anywhere else. BT13 wires the thin
slice: retrieve -> propose -> resolve-check -> assemble-from-verified-claims / else refuse.

answer_text is ASSEMBLED by the verifier from surviving verified claims (never free LLM prose), so
no uncited sentence can appear. Conflict/entailment/calibration layer on in M5.
"""

from __future__ import annotations

from docqa.citations import verify_claim
from docqa.generate import build_prompt
from docqa.types import AnswerResult, Claim, ClaimRecord, Markers


def _assemble(verified: list[Claim]) -> str:
    """answer_text is the verified claim texts joined — a function of claims[], not free prose."""
    return " ".join(c.text for c in verified)


def answer_from_proposal(
    proposal: dict,
    retrieved: list[ClaimRecord],
) -> AnswerResult:
    """Verify a proposer's output against the retrieved claims and assemble the result.

    Pure + provider-agnostic: takes an already-normalized proposal ({claims, refusal_token}) and
    the retrieved records. This is the DISPOSE step, unit-testable without any LLM.
    """
    by_id = {c.claim_id: c for c in retrieved}

    # Explicit refusal from the proposer, or no claims proposed -> refuse.
    if proposal.get("refusal_token") or not proposal.get("claims"):
        return AnswerResult(
            markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"),
        )

    verified: list[Claim] = []
    for item in proposal["claims"]:
        text = item["text"]
        for cid in item.get("cite_ids", []):
            rec = by_id.get(cid)
            if rec is None:
                continue
            vc = verify_claim(text, rec)
            if vc is not None:
                verified.append(vc)
                break  # one resolving citation is enough for this claim

    if not verified:
        # Proposer answered but nothing resolved to a real source -> refuse, don't emit uncited.
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

    return AnswerResult(answer_text=_assemble(verified), claims=verified, markers=Markers())


def answer_question(question: str, k: int, retriever, generator) -> AnswerResult:
    """Full thin-slice path. retriever + generator are injected (the pluggable seams)."""
    if not question or not question.strip():
        # Degenerate empty query — handled properly (reserved exit code) by the CLI at BT20b.
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

    retrieved = retriever.retrieve(question, k)
    if not retrieved:
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

    # The proposer sees id+text only; build_prompt's id_map is applied inside the generator.
    proposal = generator.propose(question, retrieved)
    result = answer_from_proposal(proposal, retrieved)
    # Stamp determinism/audit meta.
    result.meta = {
        "gen_model": getattr(generator, "model_id", "unknown"),
        "k": k,
    }
    return result


# Kept importable for callers that only need the prompt shape (e.g. debugging / --why later).
__all__ = ["answer_question", "answer_from_proposal", "build_prompt"]
