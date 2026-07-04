# docqa

Grounded question-answering over a folder of mixed personal documents. Point it at a directory,
ask a natural-language question, and get an answer where **every factual claim cites its source**
— or an honest refusal when the answer isn't in your documents.

Design goals (why this exists):

- **Citations, always.** Every claim resolves to `filename + locator` (heading / page / line). An
  untraceable claim is treated as a bug, not an answer.
- **Refuses when it can't answer.** A confident wrong answer is worse than "I don't know."
- **Surfaces conflicts.** When two documents disagree, it shows both sides — it doesn't silently pick.
- **Resists prompt injection.** Document text is treated as data, never as instructions.

> **Status:** early build. This README grows into full run/setup/triage docs as the tool lands
> (see the build plan). Right now it installs and runs a version check on a fresh clone.

## Quickstart (fresh clone)

```bash
uv sync --extra dev      # install the locked toolchain (Python 3.11)
uv run docqa --version   # smoke check
uv run pytest -q         # run the test gate
```

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

## License

Apache-2.0.
