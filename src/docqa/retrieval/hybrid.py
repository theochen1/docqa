"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion.

RRF is rank-based — no score normalization or weight tuning — so it degrades gracefully on an
unseen corpus (the whole reason to prefer it). Dense catches paraphrase; sparse catches exact
IDs/numbers/proper-nouns. Deterministic: both legs are deterministic and fusion + tie-break are
stable, so the fused ranking is byte-stable for a fixed index + query.
"""

from __future__ import annotations

from docqa.index_store import IndexStore
from docqa.retrieval.dense import DenseRetriever
from docqa.retrieval.select import two_track_select
from docqa.retrieval.sparse import BM25
from docqa.types import ClaimRecord


class HybridRetriever:
    def __init__(self, store: IndexStore, embedder, rrf_k: int = 60,
                 dense_n: int = 100, sparse_n: int = 100,
                 fuse_n: int = 60, per_source_cap: int = 2, select: bool = True):
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.dense_n = dense_n
        self.sparse_n = sparse_n
        self.fuse_n = fuse_n
        self.per_source_cap = per_source_cap
        self.select = select  # apply two-track selection (off = raw fused order, for A/B tests)
        self._dense = DenseRetriever(store, embedder)
        self._claims: list[ClaimRecord] | None = None
        self._bm25: BM25 | None = None

    def _ensure(self) -> None:
        if self._claims is None:
            self._claims = self.store.load_claims()
            self._bm25 = BM25(self._claims)

    def top_similarity(self, query: str) -> float:
        """Delegate to the dense leg's raw cosine — the absolute-scale off-domain signal."""
        return self._dense.top_similarity(query)

    def retrieve(self, query: str, k: int) -> list[ClaimRecord]:
        self._ensure()
        claims = self._claims or []
        if not claims or k <= 0:
            return []

        # Dense leg: rank by claim_id -> position.
        dense_hits = self._dense.retrieve_scored(query, self.dense_n)
        dense_rank = {sc.claim.claim_id: r for r, sc in enumerate(dense_hits)}

        # Sparse leg: BM25 over claim text.
        sparse_hits = self._bm25.rank(query, self.sparse_n)
        by_index = claims
        sparse_rank = {by_index[i].claim_id: r for r, (i, _s) in enumerate(sparse_hits)}

        # RRF fuse: score(c) = sum 1/(rrf_k + rank) over the lists c appears in.
        fused: dict[str, float] = {}
        for cid, r in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + r + 1)
        for cid, r in sparse_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + r + 1)

        by_id = {c.claim_id: c for c in claims}
        # Stable order: fused score desc, then claim_id asc.
        ranked_ids = sorted(fused.keys(), key=lambda cid: (-fused[cid], cid))
        pool = [by_id[cid] for cid in ranked_ids[: self.fuse_n]]
        if not self.select:
            return pool[:k]
        # Two-track selection guarantees a disagreeing source survives (conflict precondition).
        return two_track_select(pool, k, per_source_cap=self.per_source_cap)
