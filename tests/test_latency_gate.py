"""BT24: full-scale corpus builder + the BLOCKING p50 gate.

The chosen operational constraint is LATENCY (p50<5s), and it MUST be enforced at REAL scale. This
pins the gate decision (pass/fail/skip) as a pure function so the blocking behavior is provable
offline, and the corpus builder as deterministic + offline so the scale corpus is reproducible.
The live end-to-end number needs a key + a built scale corpus (documented in the README).
"""

import sys
from pathlib import Path

from docqa.latency import GateResult, LatencyReport, evaluate_gate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_corpus  # noqa: E402

# --- the gate decision (pure, unit-testable, no I/O) ---

def _full_report(p50_ms, n=9):
    # samples whose nearest-rank p50 equals p50_ms (odd count -> middle element).
    return LatencyReport(samples_ms=[p50_ms] * n, n_docs=200, n_claims=5000, scale="full")


def test_gate_passes_under_slo():
    g = evaluate_gate(_full_report(1800), slo_ms=5000, corpus_present=True)
    assert g.status == "pass"
    assert not g.blocking
    assert "PASS" in g.line() and "SLO" in g.line()


def test_gate_fails_at_or_over_slo():
    g = evaluate_gate(_full_report(5200), slo_ms=5000, corpus_present=True)
    assert g.status == "fail"
    assert g.blocking  # THIS is what the CLI turns into a non-zero exit
    assert "FAIL" in g.line()


def test_gate_fail_boundary_is_inclusive():
    # p50 exactly at the SLO must FAIL (the SLO is a ceiling: p50 must be strictly under).
    g = evaluate_gate(_full_report(5000), slo_ms=5000, corpus_present=True)
    assert g.status == "fail" and g.blocking


def test_gate_skips_when_corpus_absent():
    # Offline / fresh-clone: no scale corpus -> SKIP with a reason, NOT a pass, NOT blocking.
    g = evaluate_gate(_full_report(1800), slo_ms=5000, corpus_present=False)
    assert g.status == "skip"
    assert not g.blocking
    assert "skipped" in g.line()


def test_gate_never_blocks_on_sample_scale():
    # A sample-scale report (scale != "full") can only ever SKIP here — the gate never reports a
    # blocking number without a full-scale corpus behind it.
    sample = LatencyReport(samples_ms=[9999], scale="sample")
    g = evaluate_gate(sample, slo_ms=5000, corpus_present=True)
    assert g.status == "skip" and not g.blocking


def test_gate_skips_on_empty_samples():
    g = evaluate_gate(LatencyReport(scale="full"), slo_ms=5000, corpus_present=True)
    assert g.status == "skip"


def test_gateresult_skip_line_is_honest():
    g = GateResult("skip", "scale corpus absent")
    assert "skipped" in g.line() and "scale corpus absent" in g.line()


# --- the corpus builder (deterministic + offline) ---

def test_build_corpus_is_deterministic(tmp_path):
    a = build_corpus.build(str(tmp_path / "a"), docs=9, words_per_doc=50, seed=7)
    b = build_corpus.build(str(tmp_path / "b"), docs=9, words_per_doc=50, seed=7)
    # Same seed -> byte-identical files (so the gate is reproducible run to run).
    files_a = sorted(p.name for p in Path(a["dir"]).iterdir())
    files_b = sorted(p.name for p in Path(b["dir"]).iterdir())
    assert files_a == files_b
    for name in files_a:
        assert (Path(a["dir"]) / name).read_text() == (Path(b["dir"]) / name).read_text()


def test_build_corpus_mixes_three_handled_formats(tmp_path):
    summary = build_corpus.build(str(tmp_path / "c"), docs=9, words_per_doc=40, seed=1)
    assert summary["formats"] == {"md": 3, "txt": 3, "eml": 3}  # even mix
    exts = {p.suffix for p in Path(summary["dir"]).iterdir()}
    assert exts == {".md", ".txt", ".eml"}  # only handled formats; no PDFs


def test_build_corpus_embeds_retrievable_facts(tmp_path):
    summary = build_corpus.build(str(tmp_path / "d"), docs=3, words_per_doc=60, seed=2)
    blob = "\n".join(p.read_text() for p in Path(summary["dir"]).iterdir())
    # Real answerable content, not just filler — numbers, entities, doc refs.
    assert "days of paid time off" in blob
    assert "datacenter" in blob
    assert "DOC-" in blob


def test_build_corpus_word_count_scales(tmp_path):
    small = build_corpus.build(str(tmp_path / "s"), docs=5, words_per_doc=50, seed=3)
    big = build_corpus.build(str(tmp_path / "b"), docs=5, words_per_doc=300, seed=3)
    assert big["words"] > small["words"]
