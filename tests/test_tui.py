"""CLI presentation layer — pure render functions + spinner gating, no LLM/TTY needed."""

from docqa.tui import (
    Spinner,
    paint,
    quiet_ml_logs,
    render_answer,
    render_citations,
    use_color,
)
from docqa.types import AnswerResult, Citation, Claim, Markers


def _answer(text, cites):
    claims = [Claim(text=text, citation=Citation(filename=f, locator=loc, span=text), entailed=True)
              for f, loc in cites]
    return AnswerResult(answer_text=text, claims=claims, markers=Markers())


# --- answer rendering ---

def test_answer_shows_text_and_sources():
    r = _answer("Employees may work remotely up to 3 days per week.",
                [("handbook.md", "#Remote Work")])
    out = render_answer(r, color=False)
    assert "Employees may work remotely up to 3 days per week." in out
    assert "Sources:" in out
    assert "handbook.md" in out
    assert "Remote Work" in out          # '#' stripped for readability
    assert "#Remote Work" not in out


def test_citations_deduplicated_in_first_seen_order():
    r = _answer("x", [("a.md", "#H"), ("a.md", "#H"), ("b.md", "#K")])
    lines = render_citations(r, color=False)
    assert len(lines) == 2                # the duplicate (a.md #H) collapses
    assert "a.md" in lines[0] and "b.md" in lines[1]


def test_refusal_out_of_scope_is_human_readable():
    r = AnswerResult(markers=Markers(refused=True, refusal_token="OUT_OF_SCOPE"))
    out = render_answer(r, color=False)
    assert "isn't covered" in out
    assert "OUT_OF_SCOPE" in out          # token still shown (for the curious / graders)


def test_refusal_insufficient_evidence_distinct_message():
    r = AnswerResult(markers=Markers(refused=True, refusal_token="INSUFFICIENT_EVIDENCE"))
    out = render_answer(r, color=False)
    assert "don't state this" in out
    assert "INSUFFICIENT_EVIDENCE" in out


def test_conflict_shows_both_sides_with_marker():
    claims = [
        Claim(text="15 days of PTO.", citation=Citation(filename="handbook.md", locator="#PTO",
              span="15 days of PTO."), entailed=True),
        Claim(text="20 days of PTO.", citation=Citation(filename="hr_memo.md", locator="#PTO",
              span="20 days of PTO."), entailed=True),
    ]
    r = AnswerResult(answer_text="...", claims=claims,
                     markers=Markers(conflict=True, warning="conflicting sources surfaced"))
    out = render_answer(r, color=False)
    assert "disagree" in out.lower()
    assert "15 days" in out and "20 days" in out
    assert "handbook.md" in out and "hr_memo.md" in out


# --- color gating ---

def test_paint_noop_when_disabled():
    assert paint("hi", "green", enabled=False) == "hi"
    assert "\033[" in paint("hi", "green", enabled=True)


def test_use_color_false_when_not_a_tty():
    class _NotTTY:
        def isatty(self):
            return False
    assert use_color(_NotTTY()) is False


def test_use_color_respects_no_color(monkeypatch):
    class _TTY:
        def isatty(self):
            return True
    monkeypatch.setenv("NO_COLOR", "1")
    assert use_color(_TTY()) is False


# --- spinner gating (no animation when not a TTY) ---

def test_spinner_is_noop_off_tty():
    s = Spinner("Thinking", enabled=False)
    with s:
        pass
    assert s._thread is None              # never spawned a thread


def test_spinner_disabled_by_env(monkeypatch):
    class _TTY:
        def isatty(self):
            return True

        def write(self, *_):
            pass

        def flush(self):
            pass
    monkeypatch.setenv("DOCQA_NO_SPINNER", "1")
    s = Spinner("Thinking", stream=_TTY())
    assert s.enabled is False


# --- log quieting sets the env flags the ML libs read at import ---

def test_quiet_ml_logs_sets_env_flags(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    quiet_ml_logs()
    import os
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"          # default offline: cached model, no Hub call


def test_quiet_ml_logs_offline_is_soft_default(monkeypatch):
    # A first run needing a download can override; quiet_ml_logs must NOT clobber an explicit value.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    quiet_ml_logs()
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "0"
