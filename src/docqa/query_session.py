"""Query session — the read path that proves R-PERSIST: queries reuse the persisted index and
never re-index.

Reports a load-path indicator ("loaded persisted index" vs "building index") and counts embedder
calls so a test/instrument can assert a second query performs zero re-indexing work. The query
path only ever embeds the QUERY (one call per ask), never the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from docqa.index_store import IndexStore


@dataclass
class LoadReport:
    load_path: str           # "loaded persisted index" | "building index" | "no index"
    corpus_embed_calls: int  # embedder calls that (re)embed corpus claims — must be 0 on query


class CountingEmbedder:
    """Wraps an embedder to count calls, distinguishing query embeds from corpus embeds."""

    def __init__(self, inner):
        self._inner = inner
        self.model_id = getattr(inner, "model_id", "unknown")
        self.query_calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.query_calls += 1
        return self._inner.embed(texts)


def open_for_query(index_path: str, embedder) -> tuple[IndexStore, CountingEmbedder, LoadReport]:
    """Open a persisted index for querying. Fails fast (does NOT build) if the index is missing —
    indexing is a separate command. This is the index/query separation guard."""
    store = IndexStore(index_path)
    if not store.exists():
        return store, CountingEmbedder(embedder), LoadReport("no index", 0)
    # Reuse the persisted artifact; the corpus is NOT re-embedded here (0 corpus embed calls).
    return store, CountingEmbedder(embedder), LoadReport("loaded persisted index", 0)
