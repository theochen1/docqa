"""BT09: index store. Round-trip, atomic write, fingerprint guard."""

import sqlite3

import numpy as np
import pytest

from docqa.index_store import IndexStore
from docqa.types import ClaimRecord


def _claims(n=3):
    return [
        ClaimRecord(
            claim_id=f"id{i}", filename=f"f{i}.md", locator=f"#H{i}", text=f"fact {i} is {i*10}",
            subject_norm=f"fact {i}", predicate_norm="is", value_span=str(i * 10),
            value_canon=str(i * 10),
        )
        for i in range(n)
    ]


def test_roundtrip_claims_and_vectors(tmp_path):
    store = IndexStore(str(tmp_path / "index.db"))
    claims = _claims(3)
    vecs = np.random.RandomState(0).rand(3, 8).astype(np.float32)
    store.build(claims, vecs, meta={"embed_model": "bge-small", "dim": 8})

    got = store.load_claims()
    assert [c.model_dump() for c in got] == [c.model_dump() for c in claims]
    gv = store.load_vectors()
    assert gv.shape == (3, 8)
    assert np.allclose(gv, vecs, atol=1e-6)
    assert store.count() == 3


def test_meta_fingerprint(tmp_path):
    store = IndexStore(str(tmp_path / "index.db"))
    store.build(_claims(1), None, meta={"embed_model": "bge-small", "dim": 384})
    assert store.fingerprint_matches({"embed_model": "bge-small"})
    assert not store.fingerprint_matches({"embed_model": "other-model"})
    assert store.read_meta()["schema_version"] == "1"


def test_atomic_write_no_half_store_on_crash(tmp_path):
    p = tmp_path / "index.db"
    store = IndexStore(str(p))
    store.build(_claims(2), None, meta={"m": 1})  # good index exists

    # Simulate a crash mid-build by making the tmp write fail after it starts.
    # A vectors/claims length mismatch raises before the atomic rename.
    with pytest.raises(ValueError):
        store.build(_claims(2), np.zeros((1, 4), dtype=np.float32), meta={"m": 2})

    # The original index must be intact (rename never happened).
    assert store.count() == 2
    assert store.read_meta()["m"] == 1


def test_vector_length_mismatch_rejected(tmp_path):
    store = IndexStore(str(tmp_path / "index.db"))
    with pytest.raises(ValueError):
        store.build(_claims(3), np.zeros((2, 4), dtype=np.float32), meta={})


def test_read_missing_index_raises(tmp_path):
    store = IndexStore(str(tmp_path / "nope.db"))
    assert not store.exists()
    with pytest.raises(FileNotFoundError):
        store.load_claims()


def test_single_artifact_holds_provenance_and_vectors(tmp_path):
    # Provenance + vectors in ONE file — they cannot desync.
    p = tmp_path / "index.db"
    IndexStore(str(p)).build(_claims(2), np.ones((2, 4), dtype=np.float32), meta={})
    con = sqlite3.connect(str(p))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"claims", "vectors", "meta"} <= tables
