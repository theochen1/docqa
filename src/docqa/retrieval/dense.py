"""Dense retriever — exact brute-force cosine over claim vectors.

A complete Retriever implementation: the pipeline runs end-to-end on this alone. Hybrid (BT16)
and two-track selection (BT17) are measured upgrades that revert to this known-good baseline.

Exact search (a single numpy matmul over normalized vectors) is deterministic — no ANN recall
drift — so citation assertions never flake. Ties break on claim_id for a stable total order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from docqa.index_store import IndexStore
from docqa.types import ClaimRecord


@dataclass
class ScoredClaim:
    claim: ClaimRecord
    score: float


class DenseRetriever:
    def __init__(self, store: IndexStore, embedder):
        self.store = store
        self.embedder = embedder
        self._claims: list[ClaimRecord] | None = None
        self._vectors: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        if self._claims is None:
            self._claims = self.store.load_claims()
            self._vectors = self.store.load_vectors()

    def retrieve(self, query: str, k: int) -> list[ClaimRecord]:
        return [sc.claim for sc in self.retrieve_scored(query, k)]

    def retrieve_scored(self, query: str, k: int) -> list[ScoredClaim]:
        self._ensure_loaded()
        claims = self._claims or []
        if not claims or self._vectors is None or k <= 0:
            return []

        qv = self.embedder.embed([query])[0].astype(np.float32)
        qn = float(np.linalg.norm(qv))
        if qn == 0:
            return []
        qv = qv / qn

        # Vectors are stored normalized, so cosine == dot product.
        scores = self._vectors @ qv  # (n_claims,)

        # Rank by (-score, claim_id) for a stable, deterministic total order.
        order = sorted(
            range(len(claims)),
            key=lambda i: (-float(scores[i]), claims[i].claim_id),
        )
        top = order[:k]
        return [ScoredClaim(claim=claims[i], score=float(scores[i])) for i in top]
