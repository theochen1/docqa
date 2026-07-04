"""CLI presentation layer — pure render functions + spinner gating, no LLM/TTY needed."""

from docqa.tui import (
    Spinner,
    banner,
    paint,
    quiet_ml_logs,
    render_answer,
    render_citations,
    run_chat,
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


# --- chat REPL loop (scripted I/O, no TTY / no LLM) ---

def _scripted_reader(lines):
    """A read() that returns queued lines then raises EOFError (like Ctrl-D)."""
    it = iter(lines)

    def read(_prompt):
        try:
            return next(it)
        except StopIteration as e:
            raise EOFError from e
    return read


def _capture():
    out = []
    return out, (lambda s="": out.append(s))


def test_chat_answers_each_question_then_exits_on_eof():
    answered = []

    def answer_fn(q, verbose):
        answered.append(q)
        return f"ANSWER<{q}>"

    out, write = _capture()
    code = run_chat(answer_fn, banner_text="BANNER",
                    read=_scripted_reader(["what is PTO?", "office hours?"]), write=write)
    assert code == 0
    assert answered == ["what is PTO?", "office hours?"]      # both questions ran
    assert "BANNER" in out[0]
    assert "ANSWER<what is PTO?>" in out and "ANSWER<office hours?>" in out


def test_chat_exit_command_stops_before_more_input():
    answered = []
    out, write = _capture()
    code = run_chat(lambda q, v: answered.append(q) or "x", banner_text="",
                    read=_scripted_reader(["/exit", "should-not-run"]), write=write)
    assert code == 0
    assert answered == []                                     # /exit stopped the loop immediately


def test_chat_blank_lines_are_skipped():
    calls = []
    run_chat(lambda q, v: calls.append(q) or "x", banner_text="",
             read=_scripted_reader(["", "   ", "real question"]), write=lambda s="": None)
    assert calls == ["real question"]                         # blanks never hit the answer fn


def test_chat_help_command_prints_help_and_continues():
    out, write = _capture()
    run_chat(lambda q, v: "x", banner_text="",
             read=_scripted_reader(["/help"]), write=write)
    assert any("commands:" in line for line in out)


def test_chat_verbose_toggle_flips_state_passed_to_answer_fn():
    seen = []
    run_chat(lambda q, v: seen.append(v) or "x", banner_text="",
             read=_scripted_reader(["q1", "/verbose", "q2"]),
             write=lambda s="": None, verbose_state=False)
    assert seen == [False, True]                              # toggled on between the two questions


def test_chat_unknown_slash_command_warns_not_answers():
    answered = []
    out, write = _capture()
    run_chat(lambda q, v: answered.append(q) or "x", banner_text="",
             read=_scripted_reader(["/bogus"]), write=write)
    assert answered == []                                     # not treated as a question
    assert any("unknown command" in line for line in out)


def test_banner_reports_loaded_index():
    b = banner("index.db", n_claims=55, n_docs=11, color=False)
    assert "55 claims" in b and "11 docs" in b and "index.db" in b
