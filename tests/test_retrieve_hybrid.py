"""BT16: hybrid retrieval (BM25 + RRF). Exact-ID/number recall the dense hashing leg misses,
plus determinism. Mutation: dropping the sparse leg must lose the ID retrieval."""

from docqa.embed import HashingEmbedder
from docqa.index_store import IndexStore
from docqa.ingest import build_index
from docqa.interfaces import Retriever
from docqa.retrieval.hybrid import HybridRetriever
from docqa.retrieval.sparse import BM25, _tokenize


def _corpus(root):
    (root / "net.md").write_text(
        "# Gateways\nThe VPN gateway gw-west-2 is in the Portland datacenter.\n"
        "# Other\nThe coffee machine is on the third floor.\n"
        "# More\nThe printer is near reception.\n",
        encoding="utf-8",
    )
    (root / "inv.txt").write_text("Total amount due is $4,250.00. Net 30.\n", encoding="utf-8")


def _hybrid(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    _corpus(corpus)
    idx = str(tmp_path / "index.db")
    emb = HashingEmbedder(dim=128)
    build_index(str(corpus), idx, emb)
    return HybridRetriever(IndexStore(idx), emb, dense_n=50, sparse_n=50), idx, emb


def test_satisfies_retriever_protocol(tmp_path):
    r, _, _ = _hybrid(tmp_path)
    assert isinstance(r, Retriever)


def test_exact_id_retrieved(tmp_path):
    r, _, _ = _hybrid(tmp_path)
    hits = r.retrieve("gw-west-2 datacenter", k=3)
    assert any("gw-west-2" in h.text for h in hits)


def test_exact_currency_retrieved(tmp_path):
    r, _, _ = _hybrid(tmp_path)
    hits = r.retrieve("$4,250.00 invoice total", k=3)
    assert any("4,250.00" in h.text for h in hits)


def test_deterministic(tmp_path):
    r, _, _ = _hybrid(tmp_path)
    a = [c.claim_id for c in r.retrieve("gw-west-2", k=3)]
    b = [c.claim_id for c in r.retrieve("gw-west-2", k=3)]
    assert a == b


def test_bm25_ranks_exact_term_first(tmp_path):
    # Direct BM25 sanity: the claim containing the rare term ranks first.
    r, idx, emb = _hybrid(tmp_path)
    claims = IndexStore(idx).load_claims()
    bm = BM25(claims)
    ranked = bm.rank("gw-west-2", n=3)
    top_idx = ranked[0][0]
    assert "gw-west-2" in claims[top_idx].text


def test_mutation_dropping_sparse_leg_loses_id_recall(tmp_path):
    # The mutation: dense-only (hashing embedder) should rank the exact-ID claim WORSE than hybrid.
    # This proves the sparse leg is load-bearing for ID recall.
    r, idx, emb = _hybrid(tmp_path)
    q = "gw-west-2"
    hybrid_top = r.retrieve(q, k=1)
    # Hybrid puts the gw-west-2 claim at rank 1.
    assert "gw-west-2" in hybrid_top[0].text
    # The sparse leg is load-bearing: BM25 alone ranks the exact-id claim first.
    claims = IndexStore(idx).load_claims()
    bm = BM25(claims)
    assert "gw-west-2" in claims[bm.rank(q, 1)[0][0]].text


# --- ID punctuation-variant recall (the gw-north-4 over-refusal fix) ---

def test_tokenize_expands_compound_id_into_parts():
    # A compound ID is kept WHOLE (exact-match precision) AND expanded into parts (recall on a
    # space-separated query). Symmetric: index + query run the same tokenizer.
    toks = _tokenize("gw-north-4 is pending")
    assert "gw-north-4" in toks              # whole ID preserved
    assert {"gw", "north", "4"} <= set(toks)  # ...and its parts, so 'gw north 4' can match


def test_tokenize_leaves_plain_words_alone():
    assert _tokenize("the printer is broken") == ["the", "printer", "is", "broken"]


def test_space_separated_id_query_recalls_hyphenated_claim(tmp_path):
    # The reported bug: "gw north 4" (spaces) must recall the claim about "gw-north-4" (hyphens),
    # which previously fell out of the candidate set entirely -> spurious refusal.
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "reg.md").write_text(
        "# gw-north-4\nProvisioning for gw-north-4 is pending; no region assigned yet.\n"
        "# noise\nThe cafeteria serves lunch at noon.\n"
        "# more\nParking is in the basement garage.\n",
        encoding="utf-8",
    )
    idx = str(tmp_path / "i.db")
    emb = HashingEmbedder(dim=128)
    build_index(str(corpus), idx, emb)
    r = HybridRetriever(IndexStore(idx), emb, dense_n=50, sparse_n=50)
    hits = r.retrieve("whats the status of gw north 4", k=5)
    assert any("gw-north-4" in h.text for h in hits), [h.text for h in hits]


def test_exact_hyphenated_id_still_precise(tmp_path):
    # Guard against over-expansion: the whole-ID query must still rank its exact claim first (the
    # parts are additive recall, they don't drown out exact-ID precision).
    r, idx, emb = _hybrid(tmp_path)
    claims = IndexStore(idx).load_claims()
    bm = BM25(claims)
    assert "gw-west-2" in claims[bm.rank("gw-west-2", 1)[0][0]].text
