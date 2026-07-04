"""BT11: dense retriever. Exact top-k, deterministic tie-break, satisfies the Retriever seam."""

import numpy as np

from docqa.embed import HashingEmbedder
from docqa.index_store import IndexStore
from docqa.ingest import build_index
from docqa.interfaces import Retriever
from docqa.retrieval.dense import DenseRetriever


def _corpus(root):
    (root / "pto.md").write_text(
        "# PTO\nFull-time employees accrue 15 days of paid time off per year.\n", encoding="utf-8"
    )
    (root / "net.md").write_text(
        "# Gateways\nThe VPN gateway gw-west-2 is in the Portland datacenter.\n", encoding="utf-8"
    )
    (root / "misc.md").write_text(
        "# Misc\nThe office plants are watered on Fridays.\n", encoding="utf-8"
    )


def _retriever(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    _corpus(corpus)
    idx = str(tmp_path / "index.db")
    emb = HashingEmbedder(dim=128)
    build_index(str(corpus), idx, emb)
    return DenseRetriever(IndexStore(idx), emb)


def test_retrieves_relevant_claim_first(tmp_path):
    r = _retriever(tmp_path)
    hits = r.retrieve("how many PTO days do employees get", k=3)
    assert hits
    assert "15 days" in hits[0].text  # the PTO claim ranks top


def test_satisfies_retriever_protocol(tmp_path):
    assert isinstance(_retriever(tmp_path), Retriever)


def test_deterministic_across_runs(tmp_path):
    r = _retriever(tmp_path)
    a = [c.claim_id for c in r.retrieve("gateway datacenter", k=3)]
    b = [c.claim_id for c in r.retrieve("gateway datacenter", k=3)]
    assert a == b


def test_k_bounds_results(tmp_path):
    r = _retriever(tmp_path)
    assert len(r.retrieve("anything", k=1)) == 1
    assert r.retrieve("anything", k=0) == []


def test_tie_break_is_stable_on_equal_scores():
    # Two identical vectors -> equal scores -> order must fall back to claim_id, deterministically.
    from docqa.types import ClaimRecord

    class _Store:
        def load_claims(self):
            return [
                ClaimRecord(claim_id="zzz", filename="b.md", locator="#x", text="same"),
                ClaimRecord(claim_id="aaa", filename="a.md", locator="#x", text="same"),
            ]

        def load_vectors(self):
            return np.ones((2, 4), dtype=np.float32)

    class _Emb:
        model_id = "t"

        def embed(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    r = DenseRetriever(_Store(), _Emb())
    ids = [c.claim_id for c in r.retrieve("q", k=2)]
    assert ids == ["aaa", "zzz"]  # tie broken by claim_id ascending
