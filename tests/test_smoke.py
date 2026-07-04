"""BT01 smoke test: the package imports and the CLI version wiring works.

Proves the fresh-clone scaffold is runnable before any real logic exists.
"""

from docqa import __version__
from docqa.cli import build_parser, main


def test_version_is_set():
    assert __version__ == "0.1.0"


def test_cli_runs_with_no_args(capsys):
    # No subcommand yet: prints help, exits 0, never raises.
    code = main([])
    assert code == 0
    out = capsys.readouterr().out
    assert "docqa" in out


def test_parser_builds():
    parser = build_parser()
    assert parser.prog == "docqa"
