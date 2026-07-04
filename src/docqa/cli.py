"""docqa command-line entry point.

At BT01 this is version-only; `index` / `ask` / `eval` / `doctor` land in later tasks.
Kept deliberately thin — the CLI is an adapter over `docqa.core`, never a home for logic.
"""

from __future__ import annotations

import argparse
import sys

from docqa import __version__


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
