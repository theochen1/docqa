"""BT23: multi-hop bounded loop (SHOULD, off by default).

Design-of-record: algorithm-design.md §6. A bounded single-extra-round retrieve-read loop that
reuses the proposer/verifier unchanged. The proposer emits an optional needs_lookup:{bridge_term,
sub_question}; one follow-up retrieve runs; the union re-proposes once; every hop is verified
identically. The LOAD-BEARING guarantee is NOT a substring bridge check — it is the join gate: a
hop-produced answer must carry >=2 verified claims from >=2 distinct files, else refuse. There is no
"confident stitched guess" branch.

Why the loop is proven here with a STUB retriever, not the end-to-end eval: the sample corpus is
tiny, so a real retriever can co-locate both docs in ONE retrieval and answer single-hop — the loop
would never run and the case would be a false-green (the design review's FATAL finding). A stub that
returns the bridge doc on the original query and the target doc ONLY on the follow-up makes the
single-hop join IMPOSSIBLE, so hops==1 is a real proof that the loop did the work.
"""

import time

from docqa.core import _dedup_by_claim_id, answer_from_proposal, answer_question
from docqa.latency import LatencyReport
from docqa.types import ClaimRecord

# --- fixtures: a genuine 2-hop join keyed on a unique codename that shares NO tokens with the
# question, so single-hop retrieval cannot reach the target doc. ---
_BRIDGE = ClaimRecord(
    claim_id="c_bridge", filename="network_ref.md", locator="#Gateways",
    text="The platform team's primary gateway is codenamed BLUEJAY.",
    subject_norm="platform team primary gateway", predicate_norm="codename",
    value_span="BLUEJAY", value_canon="bluejay",
)
_TARGET = ClaimRecord(
    claim_id="c_target", filename="onboarding_email.eml", locator="email:body",
    text="BLUEJAY is racked in the Portland datacenter.",
    subject_norm="bluejay", predicate_norm="location",
    value_span="Portland", value_canon="portland",
)

_QUESTION = "Where is the platform team's primary gateway located?"


class SpyRetriever:
    """Returns the bridge doc for the original question, the target doc ONLY for a query carrying
    the bridge codename. Records every query so a test can assert how many retrievals happened."""

    def __init__(self, bridge, target, marker="bluejay"):
        self.bridge = list(bridge)
        self.target = list(target)
        self.marker = marker
        self.calls: list[str] = []

    def retrieve(self, query: str, k: int):
        self.calls.append(query)
        if self.marker in query.lower():
            return list(self.target)
        return list(self.bridge)


class HopGen:
    """Pass 1 (bridge-only pool) refuses AND asks for a follow-up lookup on the codename. Pass 2
    (bridge + target present) joins both, citing each source once."""

    model_id = "hopgen"

    def __init__(self):
        self.calls = 0

    def propose(self, question, claims):
        self.calls += 1
        ids = {c.claim_id for c in claims}
        if "c_bridge" in ids and "c_target" in ids:
            return {"claims": [{"text": "bridge", "cite_ids": ["c_bridge"]},
                               {"text": "target", "cite_ids": ["c_target"]}],
                    "refusal_token": None}
        return {"claims": [], "refusal_token": "INSUFFICIENT_EVIDENCE",
                "needs_lookup": {"bridge_term": "BLUEJAY",
                                 "sub_question": "Where is BLUEJAY located?"}}


class SilentRefuseGen:
    """Refuses with NO needs_lookup — the safe-fallback path: no lookup requested, so no hop."""

    model_id = "silent"

    def propose(self, question, claims):
        return {"claims": [], "refusal_token": "INSUFFICIENT_EVIDENCE"}


# --- the loop engages and produces a genuine 2-source join ---

def test_hop_join_succeeds_and_reports_one_hop():
    ret = SpyRetriever([_BRIDGE], [_TARGET])
    gen = HopGen()
    res = answer_question(_QUESTION, k=6, retriever=ret, generator=gen, max_hops=1)
    assert not res.markers.refused
    assert "Portland" in res.answer_text
    assert {c.citation.filename for c in res.claims} == {"network_ref.md", "onboarding_email.eml"}
    assert res.meta["hops"] == 1
    assert len(ret.calls) == 2   # original question + one follow-up
    assert gen.calls == 2        # proposed once per hop, verified identically


# --- backward compatibility: the default (max_hops=0) never enters the loop ---

def test_max_hops_zero_never_hops():
    ret = SpyRetriever([_BRIDGE], [_TARGET])
    gen = HopGen()
    res = answer_question(_QUESTION, k=6, retriever=ret, generator=gen)  # max_hops defaults to 0
    assert res.markers.refused
    assert len(ret.calls) == 1
    assert gen.calls == 1
    assert res.meta["hops"] == 0


# --- latency MUST outranks multi-hop SHOULD: a spent budget skips the hop, never blows the SLO ---

def test_deadline_exhausted_skips_hop():
    ret = SpyRetriever([_BRIDGE], [_TARGET])
    gen = HopGen()
    res = answer_question(_QUESTION, k=6, retriever=ret, generator=gen,
                          max_hops=1, hop_deadline_ms=0)
    assert res.markers.refused
    assert len(ret.calls) == 1   # hop was never admitted
    assert res.meta["hops"] == 0


def test_slow_hop1_vetoes_hop_on_real_elapsed_time():
    # The deadline must reflect time ALREADY burned by hop-1, not just the 0>=0 short-circuit: a
    # slow hop-1 proposer that overruns a small nonzero budget must veto spending hop-2 (R-LAT
    # outranks R-MULTIHOP). Regression guard for the "clock started at the loop" dead-code bug.
    class SlowHopGen(HopGen):
        def propose(self, question, claims):
            time.sleep(0.05)  # hop-1 burns 50ms
            return super().propose(question, claims)

    ret = SpyRetriever([_BRIDGE], [_TARGET])
    gen = SlowHopGen()
    res = answer_question(_QUESTION, k=6, retriever=ret, generator=gen,
                          max_hops=1, hop_deadline_ms=10)  # budget < hop-1 cost
    assert res.markers.refused
    assert res.meta["hops"] == 0
    assert len(ret.calls) == 1   # hop-2 retrieval never issued -> SLO protected


# --- safe fallback: proposer that never asks for a lookup does not trigger a hop ---

def test_no_needs_lookup_means_no_hop():
    ret = SpyRetriever([_BRIDGE], [_TARGET])
    gen = SilentRefuseGen()
    res = answer_question(_QUESTION, k=6, retriever=ret, generator=gen, max_hops=1)
    assert res.markers.refused
    assert len(ret.calls) == 1


# --- the LOAD-BEARING gate: a hop answer must join >=2 files, else refuse (no single-source stitch)

def test_hop_answer_requires_two_distinct_sources():
    # Proposer cites ONLY the target doc: the platform-team -> BLUEJAY bridge was never verified, so
    # this is an ungrounded stitch. The join gate must refuse it.
    proposal = {"claims": [{"text": "t", "cite_ids": ["c_target"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, [_BRIDGE, _TARGET], require_multisource=True)
    assert res.markers.refused


def test_textless_claim_item_refuses_not_crashes():
    # answer_from_proposal is a public, separately-exported entry. A malformed proposal item with no
    # "text" key must be skipped safely (refuse), never raise KeyError.
    proposal = {"claims": [{"cite_ids": ["c_bridge"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, [_BRIDGE, _TARGET])
    assert res.markers.refused


def test_hop_answer_two_sources_passes_gate():
    proposal = {"claims": [{"text": "b", "cite_ids": ["c_bridge"]},
                           {"text": "t", "cite_ids": ["c_target"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, [_BRIDGE, _TARGET], require_multisource=True)
    assert not res.markers.refused
    assert len({c.citation.filename for c in res.claims}) == 2


def test_mutation_bypass_join_gate_stitches_single_source():
    # hop-fabricates-bridge mutant = the loop skips the join gate. The single-source (ungrounded)
    # stitch now ships instead of refusing — exactly the failure the gate exists to prevent. This
    # direct assertion is the mutation proof (per the BT18 entailment-gate precedent: gate mutants
    # are pinned as unit assertions, not CLI sweep entries).
    proposal = {"claims": [{"text": "t", "cite_ids": ["c_target"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, [_BRIDGE, _TARGET], require_multisource=False)
    assert not res.markers.refused  # gate bypassed -> ungrounded single-source answer emitted


# --- conflict x multi-hop (S01): a conflicted BRIDGE surfaces CONFLICT, doesn't silently pick ---

def test_conflicted_bridge_surfaces_conflict():
    # Two docs disagree on the bridge's value (codename BLUEJAY vs REDWING). Because the bridge is a
    # first-class emitted claim, its proposition is in verified_props, so the existing conflict path
    # (over the pool) fires — no separate per-slot conflict code needed.
    b1 = ClaimRecord(claim_id="c_b1", filename="network_ref.md", locator="#GW",
                     text="The platform team's primary gateway is codenamed BLUEJAY.",
                     subject_norm="platform team primary gateway", predicate_norm="codename",
                     value_canon="bluejay")
    b2 = ClaimRecord(claim_id="c_b2", filename="network_ref_alt.md", locator="#GW",
                     text="The platform team's primary gateway is codenamed REDWING.",
                     subject_norm="platform team primary gateway", predicate_norm="codename",
                     value_canon="redwing")
    proposal = {"claims": [{"text": "b", "cite_ids": ["c_b1"]}], "refusal_token": None}
    # The join gate must NOT suppress a conflict: conflict is a valid non-refusal outcome.
    res = answer_from_proposal(proposal, [b1, b2], require_multisource=True)
    assert res.markers.conflict
    assert "bluejay" in res.answer_text.lower() and "redwing" in res.answer_text.lower()


# --- determinism of the pool union ---

def test_dedup_preserves_order_and_dedupes_by_claim_id():
    out = _dedup_by_claim_id([_BRIDGE, _TARGET, _BRIDGE, _TARGET])
    assert [c.claim_id for c in out] == ["c_bridge", "c_target"]


def test_loop_is_deterministic_across_runs():
    def run():
        ret = SpyRetriever([_BRIDGE], [_TARGET])
        gen = HopGen()
        res = answer_question(_QUESTION, k=6, retriever=ret, generator=gen, max_hops=1)
        return res.answer_text, tuple(ret.calls)

    assert run() == run()


# --- latency instrument surfaces the hop cost so a grader sees multi-hop stayed off the fast path

def test_latency_reports_hop_fraction():
    r = LatencyReport()
    r.samples_ms = [100.0, 200.0, 300.0]
    r.hops_per_query = [0, 1, 0]
    assert r.hop_fraction == 1 / 3
    assert "hops" in r.line()


def test_latency_hop_fraction_zero_when_off():
    r = LatencyReport()
    r.samples_ms = [100.0, 200.0]
    r.hops_per_query = [0, 0]
    assert r.hop_fraction == 0.0
