"""Two-track selection — the conflict PRECONDITION.

From a fused candidate pool, select the final K via two merged tracks:
- (A) relevance/MMR track: fills most of K, diversity-aware, with a per-source cap so one verbose
  doc can't crowd out the minority source;
- (B) proposition-contrast track: cluster candidates by proposition-key (subject+predicate) and
  force-include one claim PER DISTINCT canonical value — so when two sources disagree, BOTH
  survive into the candidate set BY CONSTRUCTION (not by MMR luck). The per-source cap is bypassed
  on this track.

Conflict DETECTION happens later (BT19) over the surviving set; selection's only job is to
guarantee both sides are present to compare. This is why conflict-surfacing is reliable.
"""

from __future__ import annotations

from collections import defaultdict

from docqa.types import ClaimRecord


def _proposition_key(c: ClaimRecord) -> str:
    return f"{c.subject_norm}|{c.predicate_norm}".strip("|")


def two_track_select(
    candidates: list[ClaimRecord],
    k: int,
    per_source_cap: int = 2,
) -> list[ClaimRecord]:
    """Select final K from an already-fused (ranked) candidate list.

    `candidates` is in fused-rank order (best first). Deterministic: we iterate in that order and
    break ties by position, so the result is stable for a fixed input.
    """
    if k <= 0 or not candidates:
        return []

    selected: list[ClaimRecord] = []
    selected_ids: set[str] = set()

    # --- Track B first: proposition-contrast. For each proposition-cluster with >=2 distinct
    # canonical values, force-include the top-ranked claim of each distinct value. This is what
    # guarantees a disagreeing source survives.
    clusters: dict[str, dict[str, ClaimRecord]] = defaultdict(dict)
    for c in candidates:  # fused-rank order, so first-seen per value == top-ranked
        key = _proposition_key(c)
        if not key or not c.value_canon:
            continue
        clusters[key].setdefault(c.value_canon, c)
    for by_value in clusters.values():
        if len(by_value) >= 2:  # a genuine value disagreement on the same proposition
            for c in by_value.values():
                if c.claim_id not in selected_ids and len(selected) < k:
                    selected.append(c)
                    selected_ids.add(c.claim_id)

    # --- Track A: relevance/MMR-ish fill with a per-source cap. We approximate MMR by taking the
    # fused order (already relevance-ranked) and skipping near-duplicate sources beyond the cap.
    per_source: dict[str, int] = defaultdict(int)
    for c in candidates:
        if len(selected) >= k:
            break
        if c.claim_id in selected_ids:
            continue
        if per_source[c.filename] >= per_source_cap:
            continue
        selected.append(c)
        selected_ids.add(c.claim_id)
        per_source[c.filename] += 1

    # If the cap left us short of k (small corpora), backfill in fused order ignoring the cap.
    if len(selected) < k:
        for c in candidates:
            if len(selected) >= k:
                break
            if c.claim_id not in selected_ids:
                selected.append(c)
                selected_ids.add(c.claim_id)

    return selected[:k]
