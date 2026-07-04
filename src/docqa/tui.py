"""Terminal presentation for the CLI — the thin layer between a verified AnswerResult and a human.

Kept separate from cli.py (wiring) and core.py (logic): rendering is pure string formatting, so it
is unit-testable without an LLM or a TTY. The CLI decides *when* to show a spinner / color (I/O
concerns); this module decides *what the text looks like*.

Two jobs:
- quiet the ML stack's stderr chatter (HF token warning, weight-loading progress bars) that make a
  product CLI read like a debug log — without hiding our own errors;
- render an answer / refusal / conflict cleanly, with a Sources block, color only on a real TTY.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from itertools import cycle

from docqa.types import AnswerResult

# --- color (only on a real TTY, honoring the NO_COLOR convention) ---
_RESET = "\033[0m"
_COLORS = {"dim": "\033[2m", "bold": "\033[1m", "green": "\033[32m",
           "yellow": "\033[33m", "red": "\033[31m", "cyan": "\033[36m"}


def use_color(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, color: str, enabled: bool) -> str:
    if not enabled or color not in _COLORS:
        return text
    return f"{_COLORS[color]}{text}{_RESET}"


def quiet_ml_logs() -> None:
    """Silence the sentence-transformers / huggingface stack's advisory noise on the query path.

    Sets the env flags the libs read at import time (progress bars, tokenizer fork warning) and
    raises their loggers to ERROR so the 'Loading weights' bar and friends don't clutter a normal
    answer. Errors still surface (ERROR level). Called before the embedder is imported so the env
    flags take effect.

    The 'unauthenticated requests to the HF Hub' notice comes from the native hf_xet extension
    (written to stderr directly, bypassing Python logging), fired by the Hub freshness check. The
    query path only ever loads an ALREADY-cached model, so we default the Hub to offline — no
    request, no notice. It's a soft default (`setdefault`): a first run that still needs to download
    the embedder can override with `HF_HUB_OFFLINE=0`.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import logging

    for name in ("huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _locator_sep(locator: str) -> str:
    """Render a locator readably: '#Remote Work' -> 'Remote Work', 'L12-L18'/'email:body' as-is."""
    if locator.startswith("#"):
        return locator[1:]
    return locator


def render_citations(result: AnswerResult, color: bool) -> list[str]:
    """The de-duplicated Sources block lines (filename · locator), in first-seen order."""
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for c in result.claims:
        key = (c.citation.filename, c.citation.locator)
        if key in seen:
            continue
        seen.add(key)
        loc = _locator_sep(c.citation.locator)
        bullet = paint("•", "dim", color)
        name = paint(c.citation.filename, "cyan", color)
        lines.append(f"  {bullet} {name} {paint('·', 'dim', color)} {loc}")
    return lines


_REFUSAL_MESSAGES = {
    "OUT_OF_SCOPE": "No answer — this topic isn't covered by the indexed documents.",
    "INSUFFICIENT_EVIDENCE": "No answer — the documents don't state this.",
}


def render_answer(result: AnswerResult, color: bool = False) -> str:
    """Format a verified AnswerResult for a human. Pure: same result -> same string."""
    if result.markers.conflict:
        head = paint("⚠ Sources disagree:", "yellow", color)
        lines = [head]
        for c in result.claims:
            loc = _locator_sep(c.citation.locator)
            bullet = paint("•", "dim", color)
            name = paint(c.citation.filename, "cyan", color)
            lines.append(f"  {bullet} {c.text}  {paint('—', 'dim', color)} {name} "
                         f"{paint('·', 'dim', color)} {loc}")
        return "\n".join(lines)

    if result.markers.refused:
        token = result.markers.refusal_token or "INSUFFICIENT_EVIDENCE"
        msg = _REFUSAL_MESSAGES.get(token, "No answer.")
        tag = paint(f"({token})", "dim", color)
        return f"{paint(msg, 'yellow', color)} {tag}"

    parts = [result.answer_text.strip()]
    cites = render_citations(result, color)
    if cites:
        parts.append("")
        parts.append(paint("Sources:", "bold", color))
        parts.extend(cites)
    return "\n".join(parts)


class Spinner:
    """A minimal stderr spinner so the LLM wait isn't dead air. Animates only on a TTY (a daemon
    thread); a no-op otherwise (piped output / CI stays clean). Use as a context manager."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "Thinking", stream=None, enabled: bool | None = None):
        self.label = label
        self.stream = stream or sys.stderr
        self.enabled = (self.stream.isatty() if enabled is None
                        else enabled) and not os.environ.get("DOCQA_NO_SPINNER")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        for frame in cycle(self._FRAMES):
            if self._stop.is_set():
                break
            self.stream.write(f"\r{frame} {self.label}… ")
            self.stream.flush()
            time.sleep(0.08)

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            # Clear the spinner line so the answer starts on a clean row.
            self.stream.write("\r" + " " * (len(self.label) + 6) + "\r")
            self.stream.flush()
        return False
