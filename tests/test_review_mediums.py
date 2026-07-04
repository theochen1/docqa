"""Follow-up to the M3 review (remaining MEDIUMs). Each reproduces a finding, then pins the fix.

1. query-path fingerprint: a mismatched embedder must fail fast with a clear message, not crash in
   the matmul with a dimension error.
2. provenance: files with the same basename in different subdirs must not collide — the locator/
   filename must be unique across the corpus.
3. datamark: the [CLAIM] delimiter token must not be recomputable from public inputs alone (secret
   entropy), so document content can't forge it and break out of its block.
"""


from docqa.embed import HashingEmbedder
from docqa.generate import build_prompt
from docqa.ingest import build_index, parse_corpus
from docqa.query_session import open_for_query
from docqa.types import ClaimRecord

# --- 1. query-path fingerprint guard ---

def test_query_fingerprint_mismatch_fails_clearly(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "h.md").write_text("# H\nA fact about 42.\n", encoding="utf-8")
    idx = str(tmp_path / "index.db")
    build_index(str(corpus), idx, HashingEmbedder(dim=64))

    # Open for query with a DIFFERENT embedder id -> must be reported as a mismatch, not proceed.
    class _OtherEmbedder(HashingEmbedder):
        model_id = "different-embedder"

    _store, _counting, report = open_for_query(idx, _OtherEmbedder(dim=32))
    assert report.load_path == "fingerprint mismatch"


def test_query_matching_fingerprint_loads(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "h.md").write_text("# H\nA fact about 42.\n", encoding="utf-8")
    idx = str(tmp_path / "index.db")
    build_index(str(corpus), idx, HashingEmbedder(dim=64))
    _s, _c, report = open_for_query(idx, HashingEmbedder(dim=64))
    assert report.load_path == "loaded persisted index"


# --- 2. provenance uniqueness across subdirectories ---

def test_subdir_files_do_not_collide(tmp_path):
    corpus = tmp_path / "c"
    (corpus / "a").mkdir(parents=True)
    (corpus / "b").mkdir(parents=True)
    # Same basename, different content, different subdirs.
    (corpus / "a" / "notes.md").write_text("# X\nAlpha fact is 10.\n", encoding="utf-8")
    (corpus / "b" / "notes.md").write_text("# X\nBeta fact is 20.\n", encoding="utf-8")

    claims, _ = parse_corpus(str(corpus))
    filenames = {c.filename for c in claims}
    # The two notes.md must be distinguishable by their provenance.
    assert len(filenames) == 2, f"provenance collided: {filenames}"
    # And claim_ids must be unique (they mix filename + locator + text).
    assert len({c.claim_id for c in claims}) == len(claims)


# --- 3. datamark secret entropy ---

def test_datamark_not_recomputable_from_public_inputs():
    claims = [ClaimRecord(claim_id="c1", filename="f.md", locator="#h", text="a fact")]
    p1 = build_prompt("q", claims)
    p2 = build_prompt("q", claims)
    # Same inputs -> DIFFERENT mark across runs (per-run secret entropy), so a document author who
    # knows the question + claim ids still cannot predict the closing delimiter.
    assert p1["mark"] != p2["mark"]
    assert p1["mark"].startswith("MARK_")


def test_datamark_stable_within_a_single_prompt():
    claims = [ClaimRecord(claim_id="c1", filename="f.md", locator="#h", text="a fact")]
    p = build_prompt("q", claims)
    # The same mark must open and close every block within one prompt.
    assert p["user"].count(p["mark"]) >= 2
