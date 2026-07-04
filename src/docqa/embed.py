"""Embedders — text -> unit-normalized vectors. Local + key-free by default.

Two implementations behind one small interface:
- SentenceTransformerEmbedder: the real bge-small (lazy import; weights cached on first use).
- HashingEmbedder: deterministic, dependency-light, no weights/network — the test + offline
  fallback. Not semantically strong, but stable and real-valued, so the pipeline (and its
  determinism assertions) run anywhere without a model download.

Vectors are L2-normalized so cosine similarity == dot product (exact-search assumption).
"""

from __future__ import annotations

import hashlib

import numpy as np


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic hashing embedder — token hashing into a fixed-dim space, then normalize.

    Same text -> same vector, always, with no model. Good enough for wiring + tests; the real
    embedder replaces it at runtime. `model_id` is reported in the fingerprint so an index built
    with the fallback is never silently queried by a different embedder.
    """

    model_id = "hashing-v1"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _normalize(out)


class SentenceTransformerEmbedder:
    """Real local embedder (bge-small by default). Lazy import so tests never pay the load."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5"):
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(model_id: str, *, offline_fallback: bool = True):
    """Return the real embedder, falling back to the hashing embedder if it can't load."""
    emb = SentenceTransformerEmbedder(model_id)
    if not offline_fallback:
        return emb
    try:  # probe availability without downloading if the lib is simply absent
        import sentence_transformers  # noqa: F401

        return emb
    except ImportError:
        return HashingEmbedder()
