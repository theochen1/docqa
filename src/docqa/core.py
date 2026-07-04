"""The one shared entry point: answer_question(question, k) -> AnswerResult.

CLI, web, and eval all call this — no guarantee logic lives anywhere else. BT13 wires the thin
slice: retrieve -> propose -> resolve-check -> assemble-from-verified-claims / else refuse.

answer_text is ASSEMBLED by the verifier from surviving verified claims (never free LLM prose), so
no uncited sentence can appear. Conflict/entailment/calibration layer on in M5; the bounded
multi-hop retrieve-read loop layers on in BT23 (off by default).
"""

from __future__ import annotations

import re
import time

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
    require_multisource: bool = False,
) -> AnswerResult:
    """Verify a proposer's output against the retrieved claims and assemble the result.

    Pure + provider-agnostic: takes an already-normalized proposal ({claims, refusal_token}) and
    the retrieved records. This is the DISPOSE step, unit-testable without any LLM.

    `entail_judge` (R-ENTAIL, BT18): optional callable (claim_text, span_text) -> bool. When
    supplied, a claim that resolves (referential integrity) but whose span does NOT entail the
    proposed claim is DROPPED. If dropping leaves nothing, we refuse. When None, only referential
    integrity gates (the BT13 behavior).

    `require_multisource` (R-MULTIHOP, BT23): the join gate. When True, a NON-conflict answer must
    rest on >=2 verified claims from >=2 distinct source files, else we refuse. This is the
    load-bearing guarantee for a hop-produced answer: it forbids a "confident stitched guess" that
    cites only the target doc while the bridge fact was never verified (algorithm-design.md §6 —
    "only one source when >=2 needed -> refuse"). A CONFLICT outcome is exempt (surfacing a
    contradiction is a valid, non-stitched result).
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
        # Defensive: this is a public, separately-exported entry (unit-tested with hand-built
        # proposals), so don't assume normalize_proposal already ran — a text-less item is skipped,
        # never a KeyError (mirrors the get(...) guard on cite_ids below).
        text = (item.get("text") or "").strip()
        if not text:
            continue
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

    # R-MULTIHOP join gate: a non-conflict hop answer must join >=2 distinct source files, else it
    # is a single-source stitch with an unverified bridge — refuse rather than emit it.
    if require_multisource and len({c.citation.filename for c in verified}) < 2:
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

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


def _apply_oos(result: AnswerResult, retriever, question: str, oos_floor: float | None) -> None:
    """Refusal calibration: upgrade an INSUFFICIENT_EVIDENCE refusal to OUT_OF_SCOPE when the top
    raw dense cosine is below the off-domain floor (the query's topic isn't in the corpus at all).
    Uses raw cosine (absolute scale), never the RRF fused score. In-place on `result`."""
    if oos_floor is None or not result.markers.refused:
        return
    top = getattr(retriever, "top_similarity", None)
    if top is None:
        return
    if top(question) < oos_floor:
        result.markers.refusal_token = "OUT_OF_SCOPE"


def _dedup_by_claim_id(claims: list[ClaimRecord]) -> list[ClaimRecord]:
    """Order-preserving union keyed strictly on claim_id. Preserves retrieval rank (the invariant
    id_map / conflict / selection all depend on) and never drops a genuinely-new follow-up target
    as a near-duplicate — dedup is exact-id, nothing fuzzier."""
    seen: set[str] = set()
    out: list[ClaimRecord] = []
    for c in claims:
        if c.claim_id not in seen:
            seen.add(c.claim_id)
            out.append(c)
    return out


def _needs_lookup(proposal: dict) -> dict | None:
    """The proposer's optional structured hop request: {bridge_term, sub_question}. Absent/malformed
    -> None (the safe-fallback: no lookup requested, no hop). The proposer only emits this when it
    cites a claim referencing an entity it can't resolve (algorithm-design.md §6)."""
    nl = proposal.get("needs_lookup")
    if not isinstance(nl, dict):
        return None
    sub = (nl.get("sub_question") or "").strip()
    bridge = (nl.get("bridge_term") or "").strip()
    if not sub or not bridge:
        return None
    return {"bridge_term": bridge, "sub_question": sub}


def _bridge_grounded(bridge_term: str, pool: list[ClaimRecord]) -> bool:
    """Cheap pre-retrieval budget guard: don't spend a hop on a bridge term with ZERO support in
    the current pool. Matched on token boundaries (so 'gw-south-1' does not match 'gw-south-10').

    This is a budget check, NOT the fabrication stop — that is the >=2-file join gate in
    answer_from_proposal (require_multisource). A planner that copies a shown token still has to
    produce a genuine second cited source to be non-refused.
    """
    term = bridge_term.strip().lower()
    if not term:
        return False
    pat = re.compile(r"(?<![0-9a-z])" + re.escape(term) + r"(?![0-9a-z])")
    return any(pat.search(c.text.lower()) for c in pool)


def answer_question(question: str, k: int, retriever, generator, entail_judge=None,
                    oos_floor: float | None = None, max_hops: int = 0,
                    hop_deadline_ms: int | None = None) -> AnswerResult:
    """Full path. retriever + generator (+ optional entailment judge) are injected seams.

    `max_hops` (BT23, default 0 = OFF): the number of extra retrieve-read rounds. When 0, the loop
    body never executes and behavior is byte-identical to BT13-BT21 (backward-compatible). When >0
    and hop 1 refuses, the proposer's needs_lookup drives one follow-up retrieval; the union
    re-proposes once under the join gate. `hop_deadline_ms` is checked BEFORE spending a hop (R-LAT
    MUST outranks R-MULTIHOP SHOULD — refuse rather than blow the SLO).
    """
    if not question or not question.strip():
        # Degenerate empty query — handled properly (reserved exit code) by the CLI at BT20b.
        return AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))

    # Clock the WHOLE query from here (not from the loop) so the hop deadline reflects time already
    # burned by hop-1 retrieval + proposer: a slow hop-1 must be able to veto spending a hop-2 that
    # would blow the SLO (R-LAT MUST outranks R-MULTIHOP SHOULD).
    q_start = time.perf_counter()

    retrieved = retriever.retrieve(question, k)
    if not retrieved:
        res = AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))
        _apply_oos(res, retriever, question, oos_floor)
        res.meta = {"gen_model": getattr(generator, "model_id", "unknown"), "k": k, "hops": 0}
        return res

    # The proposer sees id+text only; build_prompt's id_map is applied inside the generator.
    proposal = generator.propose(question, retrieved)
    result = answer_from_proposal(proposal, retrieved, entail_judge=entail_judge)

    # --- Multi-hop bounded loop (BT23). Only engages when a hop was requested AND hop 1 refused;
    # a good single-hop answer is never overridden. Deadline-guarded, single extra round. ---
    hops = 0
    while (max_hops and hops < max_hops and result.markers.refused
           and not result.markers.conflict):
        lookup = _needs_lookup(proposal)
        if lookup is None:
            break  # safe fallback: proposer asked for nothing
        if not _bridge_grounded(lookup["bridge_term"], retrieved):
            break  # no pool support for the bridge -> don't spend the hop
        if (hop_deadline_ms is not None
                and (time.perf_counter() - q_start) * 1000.0 >= hop_deadline_ms):
            break  # budget already spent by hop-1 -> refuse rather than blow the SLO
        retrieved = _dedup_by_claim_id(retrieved + retriever.retrieve(lookup["sub_question"], k))
        proposal = generator.propose(question, retrieved)
        # The join gate is the load-bearing hop guarantee: a hop answer must join >=2 files.
        result = answer_from_proposal(proposal, retrieved, entail_judge=entail_judge,
                                      require_multisource=True)
        hops += 1

    _apply_oos(result, retriever, question, oos_floor)
    # Stamp determinism/audit meta.
    result.meta = {
        "gen_model": getattr(generator, "model_id", "unknown"),
        "k": k,
        "hops": hops,
    }
    return result


# Kept importable for callers that only need the prompt shape (e.g. debugging / --why later).
__all__ = ["answer_question", "answer_from_proposal", "build_prompt", "_dedup_by_claim_id"]
