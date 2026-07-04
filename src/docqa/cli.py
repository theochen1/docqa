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
    from docqa.embed import get_embedder
    from docqa.generate import AnthropicGenerator
    from docqa.query_session import open_for_query
    from docqa.retrieval.dense import DenseRetriever

    settings = Settings.load()
    embedder = get_embedder(settings.embed_model)
    store, counting, report = open_for_query(settings.index_path, embedder)
    print(f"[docqa] {report.load_path}", file=sys.stderr)
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

    retriever = DenseRetriever(store, counting)
    generator = AnthropicGenerator(settings.gen_model, settings.max_tokens)
    result = answer_question(args.question, settings.k, retriever, generator)

    # Query path must never re-embed the corpus (R-PERSIST): only the query is embedded.
    print(f"[docqa] corpus_embed_calls={report.corpus_embed_calls}", file=sys.stderr)

    if result.markers.refused:
        print(result.markers.refusal_token or "INSUFFICIENT_EVIDENCE")
        return 0
    print(result.answer_text)
    for c in result.claims:
        print(f"  - {c.citation.filename} {c.citation.locator}", file=sys.stderr)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import tempfile

    from docqa.config import Settings
    from docqa.embed import get_embedder
    from docqa.eval_harness import DEFAULT_CASES, DEFAULT_CORPUS, format_scoreboard, run_eval
    from docqa.generate import AnthropicGenerator
    from docqa.index_store import IndexStore
    from docqa.ingest import build_index
    from docqa.retrieval.dense import DenseRetriever

    settings = Settings.load()
    corpus = args.corpus or DEFAULT_CORPUS
    cases = args.suite or DEFAULT_CASES
    embedder = get_embedder(settings.embed_model)

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
            return DenseRetriever(IndexStore(idx), embedder)

        generator = AnthropicGenerator(settings.gen_model, settings.max_tokens)
        results, latency = run_eval(corpus, cases, build_retriever, generator, k=settings.k,
                                    verbose=args.verbose)
        # Stamp corpus size onto the latency report (sample scale — INFO only, not the SLO).
        store = IndexStore(idx)
        claims = store.load_claims()
        latency.n_claims = len(claims)
        latency.n_docs = len({c.filename for c in claims})
        latency.n_words = sum(len(c.text.split()) for c in claims)

    print(format_scoreboard(results))
    print(f"[docqa] {latency.line()}", file=sys.stderr)
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
    ask.set_defaults(func=_cmd_ask)

    ev = sub.add_parser("eval", help="Run the eval harness against the sample corpus.")
    ev.add_argument("--corpus", help="Corpus dir (default: bundled sample_corpus).")
    ev.add_argument("--suite", help="Cases YAML (default: eval/cases.yaml).")
    ev.add_argument("--verbose", action="store_true", help="Print per-case detail to stderr.")
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
