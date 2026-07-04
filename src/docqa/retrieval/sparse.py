"""Sparse BM25 retrieval over claim text — catches exact IDs / numbers / proper nouns that dense
embedders blur (gw-west-2, $4,250.00, 8080). Deterministic Okapi BM25, dependency-free (~30 lines).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from docqa.types import ClaimRecord

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.$%][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75). Deterministic scoring for a fixed corpus + query."""

    def __init__(self, claims: list[ClaimRecord], k1: float = 1.5, b: float = 0.75):
        self.claims = claims
        self.k1 = k1
        self.b = b
        self.docs = [_tokenize(c.text) for c in claims]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        self.tf = [Counter(d) for d in self.docs]
        # document frequency per term
        df: Counter[str] = Counter()
        for d in self.docs:
            for term in set(d):
                df[term] += 1
        n = len(self.docs)
        self.idf = {
            t: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        out = [0.0] * len(self.claims)
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i]
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (freq * (self.k1 + 1)) / (denom or 1)
            out[i] = s
        return out

    def rank(self, query: str, n: int) -> list[tuple[int, float]]:
        """Return [(claim_index, score)] for the top-n, stable tie-break on index."""
        scored = list(enumerate(self.scores(query)))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [(i, s) for i, s in scored[:n] if s > 0.0]
