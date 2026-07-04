"""BT13: core answer path. Resolve-or-refuse, assemble-from-verified-claims, R-PERSIST guard."""

import numpy as np

from docqa.core import answer_from_proposal, answer_question
from docqa.embed import HashingEmbedder
from docqa.index_store import IndexStore
from docqa.ingest import build_index
from docqa.query_session import CountingEmbedder, open_for_query
from docqa.retrieval.dense import DenseRetriever
from docqa.types import ClaimRecord


def _retrieved():
    return [
        ClaimRecord(claim_id="c_pto", filename="handbook.md", locator="#PTO",
                    text="Full-time employees accrue 15 days of PTO."),
        ClaimRecord(claim_id="c_vpn", filename="net.md", locator="#GW",
                    text="The VPN gateway is gw-west-2."),
    ]


def test_resolving_claim_is_emitted_with_citation():
    proposal = {"claims": [{"text": "Full-time employees accrue 15 days of PTO.",
                            "cite_ids": ["c_pto"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, _retrieved())
    assert not res.markers.refused
    assert "15 days" in res.answer_text
    assert res.claims[0].citation.filename == "handbook.md"
    assert res.claims[0].citation.locator == "#PTO"
    assert res.claims[0].entailed is True  # pre-gate default (BT18 narrows)


def test_answer_text_is_assembled_from_verified_claims_only():
    # Proposer text that does NOT resolve to any source must not appear in answer_text.
    proposal = {"claims": [{"text": "The CEO is Dana Reed.", "cite_ids": ["c_pto"]}],
                "refusal_token": None}
    res = answer_from_proposal(proposal, _retrieved())
    # "Dana Reed" doesn't round-trip to the PTO span -> claim dropped -> refuse (no uncited prose).
    assert res.markers.refused
    assert "Dana Reed" not in res.answer_text


def test_explicit_refusal_token_refuses():
    proposal = {"claims": [], "refusal_token": "INSUFFICIENT_EVIDENCE"}
    res = answer_from_proposal(proposal, _retrieved())
    assert res.markers.refused and res.markers.refusal_token == "INSUFFICIENT_EVIDENCE"


def test_wrong_cite_id_does_not_resolve():
    proposal = {"claims": [{"text": "Full-time employees accrue 15 days of PTO.",
                            "cite_ids": ["c_vpn"]}], "refusal_token": None}
    res = answer_from_proposal(proposal, _retrieved())
    # cited the VPN claim for a PTO statement -> span mismatch -> refuse
    assert res.markers.refused


def test_empty_query_refuses():
    class _R:
        def retrieve(self, q, k):
            return _retrieved()

    class _G:
        model_id = "stub"

        def propose(self, q, cs):
            return {"claims": [], "refusal_token": None}

    res = answer_question("   ", 5, _R(), _G())
    assert res.markers.refused


# --- R-PERSIST ---

def test_query_reuses_index_zero_corpus_embed(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "h.md").write_text("# PTO\nEmployees accrue 15 days of PTO.\n", encoding="utf-8")
    idx = str(tmp_path / "index.db")
    build_index(str(corpus), idx, HashingEmbedder(dim=64))

    # Open for query twice; the load-path says "loaded", corpus embed calls stay 0.
    store, counting, report = open_for_query(idx, HashingEmbedder(dim=64))
    assert report.load_path == "loaded persisted index"
    assert report.corpus_embed_calls == 0

    r = DenseRetriever(store, counting)
    r.retrieve("how much PTO", k=3)      # embeds the QUERY only
    r.retrieve("PTO again", k=3)
    assert isinstance(counting, CountingEmbedder)
    assert counting.query_calls == 2     # two queries, two query-embeds, no corpus re-embed


def test_open_for_query_missing_index_does_not_build(tmp_path):
    store, _, report = open_for_query(str(tmp_path / "nope.db"), HashingEmbedder())
    assert report.load_path == "no index"
    assert not store.exists()  # querying never creates an index


def test_persisted_vectors_not_recomputed(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "h.md").write_text("# X\nA fact about 42.\n", encoding="utf-8")
    idx = str(tmp_path / "index.db")
    build_index(str(corpus), idx, HashingEmbedder(dim=32))
    v1 = IndexStore(idx).load_vectors()
    # a query session loads the SAME persisted vectors, not freshly computed ones
    store, _, _ = open_for_query(idx, HashingEmbedder(dim=32))
    assert np.array_equal(store.load_vectors(), v1)
