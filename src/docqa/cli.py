"""docqa command-line entry point.

At BT01 this is version-only; `index` / `ask` / `eval` / `doctor` land in later tasks.
Kept deliberately thin — the CLI is an adapter over `docqa.core`, never a home for logic.
"""

from __future__ import annotations

import argparse
import sys

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


def _cmd_index(args: argparse.Namespace) -> int:
    from docqa.config import Settings
    from docqa.embed import get_embedder
    from docqa.ingest import build_index

    settings = Settings.load()
    embedder = get_embedder(settings.embed_model)
    manifest = build_index(args.corpus, settings.index_path, embedder)
    print(manifest.summary())
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
    index.set_defaults(func=_cmd_index)

    ask = sub.add_parser("ask", help="Ask a question against the persisted index.")
    ask.add_argument("question", help="The natural-language question.")
    ask.set_defaults(func=_cmd_ask)

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
