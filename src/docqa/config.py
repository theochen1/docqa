"""Configuration — a single pydantic Settings model, the one place tunables live.

Two principles from the design:
- **Fail fast.** A missing dependency (e.g. the answer-path API key) surfaces as one clear,
  actionable message naming the missing var — never a mid-query stack trace.
- **No magic numbers.** Every knob that affects a guarantee (thresholds, k, models) is here with
  a default, env-overridable, and documented as tuned-not-universal. Indexing needs no key; only
  the answer path does.

Env vars use the `DOCQA_` prefix. Values are read from the process environment (and a local `.env`
if present — loaded best-effort, no hard dependency on python-dotenv).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _load_dotenv(path: str = ".env") -> None:
    """Best-effort .env loader (no external dep). Does not overwrite already-set env vars."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings(BaseModel):
    """All docqa tunables. Construct via `Settings.load()` to pull from env + .env."""

    # --- models (pinned id+revision recorded in index_meta at index time) ---
    gen_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Answer proposer. Fast tier by default (Opus blows the p50<5s budget); "
        "override to a stronger model via DOCQA_GEN_MODEL.",
    )
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local embedder (key-free indexing). Changing it forces a reindex.",
    )
    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-small",
        description="Local entailment/NLI gate. Deterministic, no API call.",
    )

    # --- retrieval knobs ---
    k: int = Field(default=12, description="Final claims handed to the proposer.")
    dense_n: int = Field(default=100, description="Dense candidate depth before fusion.")
    sparse_n: int = Field(default=100, description="Sparse (BM25) candidate depth before fusion.")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant.")
    fuse_n: int = Field(default=60, description="Fused pool size into selection.")
    mmr_lambda: float = Field(default=0.5, description="MMR relevance-vs-diversity balance.")
    per_source_cap: int = Field(default=2, description="Max claims per file on the MMR track.")
    cluster_sim: float = Field(
        default=0.80,
        description="Proposition-key match threshold (subject+predicate cosine). "
        "The knob most likely to need per-corpus tuning.",
    )

    # --- verification thresholds ---
    entail_threshold: float = Field(
        default=0.5, description="Min P(entailment) for a claim's cited span to support it."
    )
    conflict_threshold: float = Field(
        default=0.5, description="Symmetric NLI-contradiction threshold (string/entity values)."
    )
    oos_floor: float = Field(
        default=0.45,
        description="Raw dense-cosine floor separating OUT_OF_SCOPE from INSUFFICIENT_EVIDENCE. "
        "Tuned on the sample corpus (off-domain ~0.38 vs on-domain ~0.59+); tuned-not-universal.",
    )

    # --- latency (the chosen operational constraint) ---
    latency_slo_ms: int = Field(
        default=5000,
        description="Blocking p50 SLO enforced on the FULL-scale corpus (BT24). The chosen "
        "constraint; sample-corpus latency is INFO-only.",
    )

    # --- multi-hop (SHOULD, off by default) ---
    multihop: bool = Field(default=False, description="Enable the bounded retrieve-read loop.")
    max_hops: int = Field(default=1, description="Extra retrieve-read rounds beyond the first.")
    hop_deadline_ms: int = Field(default=2500, description="Refuse hop-2 if the budget is gone.")

    # --- generation ---
    temperature: float = Field(default=0.0, description="Deterministic generation.")
    seed: int = Field(default=0, description="Fixed seed for reproducibility.")
    max_tokens: int = Field(default=512, description="Cap on proposer output.")

    # --- paths ---
    index_path: str = Field(default="index.db", description="Persisted index artifact.")

    @classmethod
    def load(cls, dotenv: bool = True) -> Settings:
        """Build Settings from environment (and .env if present)."""
        if dotenv:
            _load_dotenv()

        def num(name: str, default: float, cast):
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return cast(raw)
            except ValueError:
                return default

        return cls(
            gen_model=_env("DOCQA_GEN_MODEL", cls.model_fields["gen_model"].default),
            embed_model=_env("DOCQA_EMBED_MODEL", cls.model_fields["embed_model"].default),
            nli_model=_env("DOCQA_NLI_MODEL", cls.model_fields["nli_model"].default),
            k=int(num("DOCQA_K", cls.model_fields["k"].default, int)),
            dense_n=int(num("DOCQA_DENSE_N", cls.model_fields["dense_n"].default, int)),
            sparse_n=int(num("DOCQA_SPARSE_N", cls.model_fields["sparse_n"].default, int)),
            rrf_k=int(num("DOCQA_RRF_K", cls.model_fields["rrf_k"].default, int)),
            fuse_n=int(num("DOCQA_FUSE_N", cls.model_fields["fuse_n"].default, int)),
            mmr_lambda=num("DOCQA_MMR_LAMBDA", cls.model_fields["mmr_lambda"].default, float),
            per_source_cap=int(
                num("DOCQA_PER_SOURCE_CAP", cls.model_fields["per_source_cap"].default, int)
            ),
            cluster_sim=num("DOCQA_CLUSTER_SIM", cls.model_fields["cluster_sim"].default, float),
            entail_threshold=num(
                "DOCQA_ENTAIL_THRESHOLD", cls.model_fields["entail_threshold"].default, float
            ),
            conflict_threshold=num(
                "DOCQA_CONFLICT_THRESHOLD", cls.model_fields["conflict_threshold"].default, float
            ),
            oos_floor=num("DOCQA_OOS_FLOOR", cls.model_fields["oos_floor"].default, float),
            latency_slo_ms=int(
                num("DOCQA_LATENCY_SLO_MS", cls.model_fields["latency_slo_ms"].default, int)
            ),
            multihop=_env("DOCQA_MULTIHOP", "0") not in ("0", "", "false", "False"),
            max_hops=int(num("DOCQA_MAX_HOPS", cls.model_fields["max_hops"].default, int)),
            hop_deadline_ms=int(
                num("DOCQA_HOP_DEADLINE_MS", cls.model_fields["hop_deadline_ms"].default, int)
            ),
            temperature=num("DOCQA_TEMPERATURE", cls.model_fields["temperature"].default, float),
            seed=int(num("DOCQA_SEED", cls.model_fields["seed"].default, int)),
            max_tokens=int(num("DOCQA_MAX_TOKENS", cls.model_fields["max_tokens"].default, int)),
            index_path=_env("DOCQA_INDEX_PATH", cls.model_fields["index_path"].default),
        )


# The env var that carries the answer-path API key (required only for `ask`/`eval`, not `index`).
API_KEY_VAR = "ANTHROPIC_API_KEY"


class MissingDependencyError(RuntimeError):
    """Raised for a fail-fast preflight failure with an actionable message."""


def require_api_key() -> str:
    """Fail fast if the answer-path key is absent. Indexing never calls this."""
    key = os.environ.get(API_KEY_VAR, "").strip()
    if not key:
        raise MissingDependencyError(
            f"{API_KEY_VAR} is not set. The answer path (docqa ask / eval) needs it; indexing "
            f"does not. Set it in your environment or a .env file (see .env.example)."
        )
    return key
