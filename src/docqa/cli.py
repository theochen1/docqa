"""docqa command-line entry point.

At BT01 this is version-only; `index` / `ask` / `eval` / `doctor` land in later tasks.
Kept deliberately thin — the CLI is an adapter over `docqa.core`, never a home for logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docqa import __version__


def _cmd_doctor(args: argparse.Namespace) -> int:
    # Imported lazily so `--version` and help never pay the import cost.
    from docqa.doctor import format_report, run_checks

    print(format_report(run_checks()))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from docqa.config import Settings
    from docqa.core import answer_question
    from docqa.tui import Spinner, quiet_ml_logs, render_answer, use_color

    verbose = getattr(args, "verbose", False)

    def diag(msg: str) -> None:
        """Grader/debug diagnostics — only under --verbose so a normal answer stays clean."""
        if verbose:
            print(f"[docqa] {msg}", file=sys.stderr)

    # Quiet the ML stack's advisory noise BEFORE the embedder imports (env flags read at import).
    quiet_ml_logs()

    from docqa.embed import get_embedder
    from docqa.query_session import open_for_query

    settings = Settings.load()
    embedder = get_embedder(settings.embed_model)
    store, counting, report = open_for_query(settings.index_path, embedder)
    diag(report.load_path)
    if report.load_path == "no index":
        print(
            f"ERROR: no index at {settings.index_path}. Run `docqa index <folder>` first.",
            file=sys.stderr,
        )
        return 1
    if report.load_path == "fingerprint mismatch":
        print(
            f"ERROR: index at {settings.index_path} was built with a different embedder than "
            f"{embedder.model_id}. Re-run `docqa index` (or set DOCQA_EMBED_MODEL to match).",
            file=sys.stderr,
        )
        return 1

    from docqa.entail import AnthropicEntailmentJudge
    from docqa.generate import AnthropicGenerator
    from docqa.retrieval.hybrid import HybridRetriever
    from docqa.types import EXIT_EMPTY_CORPUS, EXIT_EMPTY_QUERY

    # Degenerate empty/whitespace query -> reserved exit code (BT20b), not a spoofable message.
    if not args.question or not args.question.strip():
        diag("empty query")
        return EXIT_EMPTY_QUERY

    # Empty index (empty corpus or all files skipped) -> reserved exit code.
    if store.count() == 0:
        diag("empty corpus: index has zero claims")
        return EXIT_EMPTY_CORPUS

    retriever = HybridRetriever(store, counting, rrf_k=settings.rrf_k,
                                dense_n=settings.dense_n, sparse_n=settings.sparse_n)
    # Multi-hop (BT23) is off unless DOCQA_MULTIHOP is set. The engage decision is single-sourced
    # here: the lookup-enabled proposer and max_hops>0 are wired together or not at all.
    generator = AnthropicGenerator(settings.gen_model, settings.max_tokens,
                                   allow_lookup=settings.multihop)
    judge = AnthropicEntailmentJudge(settings.gen_model)
    max_hops = settings.max_hops if settings.multihop else 0
    hop_deadline_ms = settings.hop_deadline_ms if settings.multihop else None

    with Spinner("Thinking"):
        result = answer_question(args.question, settings.k, retriever, generator,
                                 entail_judge=judge, oos_floor=settings.oos_floor,
                                 max_hops=max_hops, hop_deadline_ms=hop_deadline_ms)

    # Query path must never re-embed the corpus (R-PERSIST): only the query is embedded.
    diag(f"corpus_embed_calls={report.corpus_embed_calls}")
    diag(f"hops={result.meta.get('hops', 0)} model={result.meta.get('gen_model', '?')}")

    print(render_answer(result, color=use_color()))
    return 0


# Probe queries for the blocking latency gate (BT24). They match the fact templates
# scripts/build_corpus.py embeds, so each exercises the FULL pipeline (retrieve + propose + entail)
# against real claims — the gate times pipeline cost, not answer correctness (the graded cases,
# which reference the sample docs, can't run against the scale corpus).
_LATENCY_PROBES = [
    "Where is the platform team's primary gateway deployed?",
    "How many days of paid time off do full-time staff accrue?",
    "What is the approved tooling budget for the data team?",
    "Which datacenter hosts the security team's gateway?",
    "Who approves requests above the standard cap?",
    "What is the document reference number for the travel policy?",
    "How many vacation days does the finance department get?",
    "What is the on-call rotation policy?",
]


def _run_latency_gate(settings, corpus: str, embedder, sample_corpus: str) -> int:
    """BT24: the BLOCKING p50 gate on the full-scale corpus.

    - scale corpus absent (offline / fresh clone) -> SKIP with an explicit reason, exit 0 (green).
      Never reports a blocking number without a real scale corpus behind it.
    - scale corpus present -> index it, time the probe queries (warm p50/p95), evaluate the gate,
      exit non-zero iff p50 >= SLO. LLM-call time is visible via the per-query latency line.
    """
    import tempfile

    from docqa.config import API_KEY_VAR
    from docqa.core import answer_question
    from docqa.entail import AnthropicEntailmentJudge
    from docqa.generate import AnthropicGenerator
    from docqa.index_store import IndexStore
    from docqa.ingest import build_index
    from docqa.latency import LatencyReport, Timer, evaluate_gate
    from docqa.retrieval.hybrid import HybridRetriever

    # A scale corpus is REQUIRED for a real number; the sample corpus can only ever SKIP.
    corpus_present = (corpus != sample_corpus and Path(corpus).is_dir()
                      and any(Path(corpus).iterdir()))
    if not corpus_present:
        report = LatencyReport(scale="full")
        gate = evaluate_gate(report, float(settings.latency_slo_ms), corpus_present=False,
                             reason=f"scale corpus '{corpus}' absent — build it with "
                                    f"`python scripts/build_corpus.py` (offline/fresh-clone safe)")
        print(f"[docqa] {gate.line()}", file=sys.stderr)
        return 0  # skip is GREEN by design

    import os
    if not os.environ.get(API_KEY_VAR, "").strip():
        # No key -> can't run the answer path at all; skip honestly rather than fake a number.
        from docqa.latency import GateResult
        gate = GateResult("skip", f"{API_KEY_VAR} not set — the answer path can't run",
                          slo_ms=float(settings.latency_slo_ms))
        print(f"[docqa] {gate.line()}", file=sys.stderr)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        idx = str(Path(tmp) / "scale_index.db")
        from docqa.claimizer_llm import AnthropicClaimizer
        decomposer = AnthropicClaimizer(settings.gen_model)
        print(f"[docqa] indexing scale corpus {corpus} (one-time, LLM claimizer)...",
              file=sys.stderr)
        build_index(corpus, idx, embedder, decomposer=decomposer)

        store = IndexStore(idx)
        retriever = HybridRetriever(store, embedder, rrf_k=settings.rrf_k,
                                    dense_n=settings.dense_n, sparse_n=settings.sparse_n)
        generator = AnthropicGenerator(settings.gen_model, settings.max_tokens)
        judge = AnthropicEntailmentJudge(settings.gen_model)

        def _one(q):
            return answer_question(q, settings.k, retriever, generator, entail_judge=judge,
                                   oos_floor=settings.oos_floor)

        # Mandatory warmup (excluded) so cold model/embedder load doesn't distort steady state.
        _one(_LATENCY_PROBES[0])
        report = LatencyReport(scale="full")
        for q in _LATENCY_PROBES:
            with Timer() as t:
                _one(q)
            report.samples_ms.append(t.elapsed_ms)
            report.hops_per_query.append(0)
        claims = store.load_claims()
        report.n_claims = len(claims)
        report.n_docs = len({c.filename for c in claims})
        report.n_words = sum(len(c.text.split()) for c in claims)

    print(f"[docqa] {report.line()}", file=sys.stderr)
    gate = evaluate_gate(report, float(settings.latency_slo_ms), corpus_present=True)
    print(f"[docqa] {gate.line()}", file=sys.stderr)
    return 1 if gate.blocking else 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import tempfile

    from docqa.config import Settings
    from docqa.embed import get_embedder
    from docqa.eval_harness import DEFAULT_CASES, DEFAULT_CORPUS, format_scoreboard, run_eval
    from docqa.generate import AnthropicGenerator
    from docqa.index_store import IndexStore
    from docqa.ingest import build_index
    from docqa.retrieval.hybrid import HybridRetriever

    settings = Settings.load()
    corpus = args.corpus or DEFAULT_CORPUS
    cases = args.suite or DEFAULT_CASES
    embedder = get_embedder(settings.embed_model)

    # --- BT24: blocking full-scale latency gate. Distinct from the case run — it times probe
    # queries against the scale corpus and blocks on p50 >= SLO. Offline-safe: no corpus -> skip.
    if args.latency:
        return _run_latency_gate(settings, corpus, embedder, DEFAULT_CORPUS)

    # Build a throwaway index over the sample corpus (deterministic; not the user's index.db).
    with tempfile.TemporaryDirectory() as tmp:
        idx = str(Path(tmp) / "eval_index.db")
        # LLM claimizer if a key is present; deterministic fallback otherwise (harness still runs).
        import os

        from docqa.config import API_KEY_VAR

        decomposer = None
        if os.environ.get(API_KEY_VAR, "").strip():
            from docqa.claimizer_llm import AnthropicClaimizer

            decomposer = AnthropicClaimizer(settings.gen_model)
        build_index(corpus, idx, embedder, decomposer=decomposer)

        def build_retriever():
            return HybridRetriever(IndexStore(idx), embedder, rrf_k=settings.rrf_k)

        generator = AnthropicGenerator(settings.gen_model, settings.max_tokens,
                                       allow_lookup=settings.multihop)
        judge = None
        if decomposer is not None:  # a key is present -> use the real entailment gate
            from docqa.entail import AnthropicEntailmentJudge
            judge = AnthropicEntailmentJudge(settings.gen_model)
        max_hops = settings.max_hops if settings.multihop else 0
        hop_deadline_ms = settings.hop_deadline_ms if settings.multihop else None
        results, latency = run_eval(corpus, cases, build_retriever, generator, k=settings.k,
                                    verbose=args.verbose, entail_judge=judge,
                                    oos_floor=settings.oos_floor, max_hops=max_hops,
                                    hop_deadline_ms=hop_deadline_ms)
        # Stamp corpus size onto the latency report (sample scale — INFO only, not the SLO).
        store = IndexStore(idx)
        claims = store.load_claims()
        latency.n_claims = len(claims)
        latency.n_docs = len({c.filename for c in claims})
        latency.n_words = sum(len(c.text.split()) for c in claims)

    print(format_scoreboard(results))
    print(f"[docqa] {latency.line()}", file=sys.stderr)

    if args.mutate:
        from docqa.mutate import BASE_MUTANTS, apply_mutant, sweep

        # Rebuild a persistent index for the sweep (temp dir above is gone).
        with tempfile.TemporaryDirectory() as tmp2:
            midx = str(Path(tmp2) / "mutate_index.db")
            build_index(corpus, midx, embedder, decomposer=decomposer)

            def run_case_fn(mutant):
                base_gen = AnthropicGenerator(settings.gen_model, settings.max_tokens)
                gen = apply_mutant(mutant, base_gen)
                res, _ = run_eval(
                    corpus, cases,
                    lambda: HybridRetriever(IndexStore(midx), embedder, rrf_k=settings.rrf_k),
                    gen, k=settings.k,
                )
                return res

            print("[docqa] mutation sweep (each mutant MUST redden >=1 case):", file=sys.stderr)
            weak = []
            for name, reddened in sweep(run_case_fn, BASE_MUTANTS).items():
                status = "OK" if reddened else "WEAK (false-green!)"
                print(f"  {name}: reddened {reddened or '[]'} [{status}]", file=sys.stderr)
                if not reddened:
                    weak.append(name)
            if weak:
                print(f"[docqa] WEAK mutants (no case caught them): {weak}", file=sys.stderr)

    return 0 if all(r.passed for r in results) else 1


def _cmd_index(args: argparse.Namespace) -> int:
    import os

    from docqa.config import API_KEY_VAR, Settings
    from docqa.embed import get_embedder
    from docqa.ingest import build_index

    settings = Settings.load()
    embedder = get_embedder(settings.embed_model)

    # LLM claimization is the PRIMARY path when a key is present; else deterministic fallback.
    # --no-llm forces the fallback (key-free / cost-free indexing, documented as degraded quality).
    decomposer = None
    meter = None
    use_llm = bool(os.environ.get(API_KEY_VAR, "").strip()) and not args.no_llm
    if use_llm:
        from docqa.claimizer_llm import AnthropicClaimizer, UsageMeter

        meter = UsageMeter()
        decomposer = AnthropicClaimizer(settings.gen_model, meter=meter)
    else:
        reason = "--no-llm" if args.no_llm else f"{API_KEY_VAR} not set"
        print(
            f"[docqa] LLM claimizer off ({reason}) — using deterministic fallback.",
            file=sys.stderr,
        )

    manifest = build_index(args.corpus, settings.index_path, embedder, decomposer=decomposer)
    print(manifest.summary())
    if meter is not None:
        print(f"[docqa] {meter.summary()}", file=sys.stderr)
    if not manifest.reconciles():
        print(
            "ERROR: manifest does not reconcile (discovered != parsed + skipped)",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docqa",
        description="Grounded document Q&A: cite, refuse, surface conflicts, resist injection.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"docqa {__version__}",
    )
    parser.set_defaults(func=None)

    sub = parser.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor", help="Fail-fast preflight: key, deps, index status.")
    doctor.set_defaults(func=_cmd_doctor)

    index = sub.add_parser("index", help="Index a folder of documents into index.db.")
    index.add_argument("corpus", help="Path to the folder of documents.")
    index.add_argument(
        "--no-llm",
        action="store_true",
        help="Force the deterministic claimizer (no API key / no cost; degraded quality).",
    )
    index.set_defaults(func=_cmd_index)

    ask = sub.add_parser("ask", help="Ask a question against the persisted index.")
    ask.add_argument("question", help="The natural-language question.")
    ask.add_argument("--verbose", "-v", action="store_true",
                     help="Print pipeline diagnostics (load path, embed calls, hops) to stderr.")
    ask.set_defaults(func=_cmd_ask)

    ev = sub.add_parser("eval", help="Run the eval harness against the sample corpus.")
    ev.add_argument("--corpus", help="Corpus dir (default: bundled sample_corpus).")
    ev.add_argument("--suite", help="Cases YAML (default: eval/cases.yaml).")
    ev.add_argument("--verbose", action="store_true", help="Print per-case detail to stderr.")
    ev.add_argument("--mutate", action="store_true",
                    help="Run the mutation sweep: each mutant must redden >=1 case.")
    ev.add_argument("--latency", action="store_true",
                    help="Run the BLOCKING full-scale p50 gate (needs --corpus scale_corpus/); "
                         "skips green if the scale corpus is absent.")
    ev.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        # No subcommand yet (BT01). Print help and exit cleanly.
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
