"""BT04: markdown + text parsers. Locator correctness + the no-crash / no-silent-garbage floor."""

from docqa.parsers.markdown import MarkdownParser
from docqa.parsers.text import TextParser

# --- Markdown ---

def test_md_heading_locators():
    p = MarkdownParser()
    content = (
        "# Intro\nWelcome to Aldermere.\n\n"
        "## PTO Policy\nFull-time employees accrue 15 days of PTO per year.\n\n"
        "## Security\nUse the VPN.\n"
    )
    out = p.parse_text("handbook.md", content)
    assert out.usable
    locs = {s.locator for s in out.segments}
    assert "#PTO Policy" in locs
    pto = next(s for s in out.segments if s.locator == "#PTO Policy")
    assert "15 days" in pto.text


def test_md_text_before_first_heading_is_kept():
    p = MarkdownParser()
    out = p.parse_text("x.md", "Some preamble text here.\n\n# Later\nBody.")
    assert out.usable
    assert any(s.locator == "#(top)" for s in out.segments)


def test_md_empty_is_skipped_not_crashed():
    p = MarkdownParser()
    out = p.parse_text("empty.md", "   \n\n  ")
    assert out.skipped
    assert out.segments == []
    assert out.skip_reason


def test_md_zero_byte_file_no_crash(tmp_path):
    f = tmp_path / "empty.md"
    f.write_bytes(b"")
    out = MarkdownParser().parse_file(str(f))
    assert out.skipped and out.segments == []


def test_md_non_utf8_no_crash(tmp_path):
    f = tmp_path / "latin.md"
    # 0xff is invalid UTF-8; parser must fall back to latin-1, never raise.
    f.write_bytes(b"# Caf\xe9\nText about the caf\xe9.\n")
    out = MarkdownParser().parse_file(str(f))
    assert out.segments  # recovered via latin-1


# --- Text ---

def test_txt_line_range_locator():
    p = TextParser()
    content = "line1\nline2\nline3\n\nblock two starts here\nmore\n"
    out = p.parse_text("notes.txt", content)
    assert out.usable
    first = out.segments[0]
    assert first.locator == "L1-L3"
    second = out.segments[1]
    assert second.locator.startswith("L5")


def test_txt_locator_resolves_to_the_right_lines():
    p = TextParser()
    content = "\n".join(f"line {n}" for n in range(1, 11))  # 10 lines, one block
    out = p.parse_text("x.txt", content)
    seg = out.segments[0]
    assert seg.locator == "L1-L10"
    assert "line 5" in seg.text


def test_txt_empty_skipped():
    out = TextParser().parse_text("e.txt", "")
    assert out.skipped and out.segments == []


def test_can_parse_is_extension_scoped():
    assert MarkdownParser().can_parse("a.md")
    assert not MarkdownParser().can_parse("a.txt")
    assert TextParser().can_parse("a.txt")
    assert not TextParser().can_parse("a.md")
