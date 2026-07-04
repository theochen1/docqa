"""BT10: end-to-end index build + manifest reconciliation.

Uses the HashingEmbedder (no model download) so the wiring is tested offline + deterministically.
"""

import numpy as np

from docqa.embed import HashingEmbedder, get_embedder
from docqa.index_store import IndexStore
from docqa.ingest import build_index, parse_corpus


def _mixed_corpus(root):
    (root / "handbook.md").write_text(
        "# PTO Policy\nFull-time employees accrue 15 days of PTO per year.\n", encoding="utf-8"
    )
    (root / "notes.txt").write_text(
        "line one\nThe total is $4,250.00 due.\nline three\n", encoding="utf-8"
    )
    (root / "msg.eml").write_text(
        "From: a@x.example\nSubject: Budget\n\nThe travel budget is $9,900.\n", encoding="utf-8"
    )
    (root / "empty.md").write_text("   \n", encoding="utf-8")            # skip: empty
    (root / "blob.dat").write_bytes(b"\x00\x01\x02not-a-doc")           # skip: unsupported
    (root / "notes.bak").write_text("ignored ext", encoding="utf-8")    # skip: unsupported ext


def test_manifest_reconciles_and_accounts_every_file(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _mixed_corpus(corpus)

    claims, manifest = parse_corpus(str(corpus))
    # 6 files discovered; 3 parsed (md, txt, eml); 3 skipped (empty.md, blob.dat, notes.bak).
    assert manifest.discovered == 6
    assert manifest.parsed == 3
    assert manifest.skipped == 3
    assert manifest.reconciles()  # discovered == parsed + skipped (closed accounting)
    # Every skipped file carries a reason.
    for f in manifest.files:
        if f.status == "skipped":
            assert f.reason


def test_build_index_persists_claims_and_vectors(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _mixed_corpus(corpus)
    index_path = str(tmp_path / "index.db")

    manifest = build_index(str(corpus), index_path, HashingEmbedder(dim=64))
    assert manifest.total_claims > 0

    store = IndexStore(index_path)
    assert store.count() == manifest.total_claims
    vecs = store.load_vectors()
    assert vecs is not None and vecs.shape[0] == manifest.total_claims
    # provenance present on every stored claim
    assert all(c.filename and c.locator for c in store.load_claims())
    # fingerprint records the embedder so a mismatched query is caught later
    assert store.read_meta()["embed_model"] == "hashing-v1"


def test_reindex_is_byte_stable(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _mixed_corpus(corpus)
    a_claims, _ = parse_corpus(str(corpus))
    b_claims, _ = parse_corpus(str(corpus))
    assert [c.model_dump() for c in a_claims] == [c.model_dump() for c in b_claims]
    # hashing embeddings are deterministic too
    e = HashingEmbedder(dim=32)
    assert np.array_equal(e.embed(["x y z"]), e.embed(["x y z"]))


def test_empty_corpus_reconciles(tmp_path):
    corpus = tmp_path / "empty_corpus"
    corpus.mkdir()
    claims, manifest = parse_corpus(str(corpus))
    assert claims == []
    assert manifest.discovered == 0 and manifest.reconciles()


def test_get_embedder_returns_something_usable():
    # In this env sentence-transformers IS installed, so we get the real class (not loaded yet).
    emb = get_embedder("BAAI/bge-small-en-v1.5")
    assert hasattr(emb, "embed") and hasattr(emb, "model_id")
