"""BT08b: LLM claimization is the PRIMARY path — the model supplies subject/predicate/value, and
canon.py only gates value equality. Deterministic fallback still covers no-key. Stub decomposer,
no API call."""

from docqa.claimize import claimize_segment_llm
from docqa.claimizer_llm import _parse_decomposition
from docqa.embed import HashingEmbedder
from docqa.ingest import build_index, parse_corpus
from docqa.types import TextSegment, ValueType


def _seg(text):
    return TextSegment(filename="memo.md", locator="#Body", text=text)


def test_llm_subject_predicate_used_verbatim():
    def stub(text):
        return [
            {"text": "The PTO accrual is 15 days.", "subject": "pto accrual",
             "predicate": "is", "value_span": "15 days"},
        ]

    c = claimize_segment_llm(_seg("..."), stub)[0]
    # subject/predicate come from the MODEL, not the regex heuristic
    assert c.subject_norm == "pto accrual"
    assert c.predicate_norm == "is"
    # canon.py still gates the value for equality
    assert c.value_canon == "15 days"
    assert c.value_type is ValueType.DURATION


def test_llm_value_the_regex_would_have_botched():
    # The regex path dropped the negative sign; the LLM hands us the span directly.
    def stub(text):
        return [{"text": "The delta is -5 units.", "subject": "delta",
                 "predicate": "is", "value_span": "-5"}]

    c = claimize_segment_llm(_seg("..."), stub)[0]
    assert c.value_span == "-5"
    assert c.value_canon != HashingEmbedder  # sanity
    from docqa.canon import canonicalize
    assert c.value_canon == canonicalize("-5")[1] != canonicalize("5")[1]


def test_fallback_when_decomposer_missing_subject():
    # If the model omits subject/predicate, the heuristic fills in (defensive).
    def stub(text):
        return [{"text": "Revenue was 100 dollars.", "value_span": "100"}]

    c = claimize_segment_llm(_seg("Revenue was 100 dollars."), stub)[0]
    assert c.subject_norm or c.predicate_norm  # heuristic populated something


def test_ingest_uses_llm_path_when_decomposer_supplied(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "h.md").write_text("# H\nThe budget is $9,900.\n", encoding="utf-8")

    def stub(text):
        return [{"text": "The budget is $9,900.", "subject": "budget",
                 "predicate": "is", "value_span": "$9,900"}]

    claims, manifest = parse_corpus(str(corpus), decomposer=stub)
    assert manifest.claimizer == "llm"
    assert any(c.subject_norm == "budget" for c in claims)


def test_ingest_deterministic_without_decomposer(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "h.md").write_text("# H\nThe budget is $9,900.\n", encoding="utf-8")
    claims, manifest = parse_corpus(str(corpus), decomposer=None)
    assert manifest.claimizer == "deterministic"
    assert claims  # fallback still produces claims


def test_build_index_records_claimizer_in_meta(tmp_path):
    from docqa.index_store import IndexStore

    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "h.md").write_text("# H\nA fact about 42.\n", encoding="utf-8")
    idx = str(tmp_path / "index.db")
    build_index(str(corpus), idx, HashingEmbedder(dim=32),
                decomposer=lambda t: [{"text": "A fact about 42.", "subject": "fact",
                                       "predicate": "about", "value_span": "42"}])
    assert IndexStore(idx).read_meta()["claimizer"] == "llm"


# --- JSON parsing tolerance ---

def test_parse_clean_json():
    raw = '[{"text":"x","subject":"s","predicate":"p","value":"5"}]'
    out = _parse_decomposition(raw)
    assert out == [{"text": "x", "subject": "s", "predicate": "p", "value_span": "5"}]


def test_parse_salvages_trailing_prose():
    raw = 'Here are the claims: [{"text":"x","value":"5"}] hope that helps!'
    out = _parse_decomposition(raw)
    assert out and out[0]["text"] == "x" and out[0]["value_span"] == "5"


def test_parse_garbage_returns_empty():
    assert _parse_decomposition("not json at all") == []
    assert _parse_decomposition('{"not":"a list"}') == []
