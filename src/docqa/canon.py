"""Value canonicalization — the deterministic backbone of conflict detection.

Pure functions: a raw value span -> (ValueType, canonical string). "same value" must be decidable
WITHOUT a model, so that '15' == 'fifteen (15)' (no conflict) but '15' != '20' (conflict). This is
why conflict detection does not rely on NLI for numbers/dates/currency (small NLI scores them
'neutral', not 'contradiction' — see algorithm-design.md §5.3).
"""

from __future__ import annotations

import re

from docqa.types import ValueType

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}

# duration units -> days, so "three weeks" and "21 days" canonicalize equal
_DURATION_DAYS = {
    "day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30,
    "year": 365, "years": 365,
}

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"]
    )
}
_MONTHS_ABBR = {m[:3]: i for m, i in _MONTHS.items()}

_CURRENCY_RE = re.compile(r"[$£€]\s?[\d,]+(?:\.\d+)?")
_DURATION_RE = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_TEXT_DATE_RE = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b")
_INT_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_PAREN_NUM_RE = re.compile(r"\((\d+)\)")


def _word_to_number(text: str) -> int | None:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens or any(t not in _NUMBER_WORDS for t in tokens):
        return None
    total = 0
    current = 0
    for t in tokens:
        v = _NUMBER_WORDS[t]
        if v in (100, 1000):
            current = (current or 1) * v
            total += current
            current = 0
        else:
            current += v
    return total + current


def _num_canon(digits: str, is_percent: bool) -> str | None:
    """Canonical number string. Arbitrary-precision int when whole (no float 'inf' on huge values);
    float only when a decimal point is present. Percent is suffixed so '92%' != '92'."""
    try:
        if "." in digits:
            f = float(digits)
            base = str(int(f)) if f.is_integer() else str(f)
        else:
            base = str(int(digits))  # arbitrary precision
    except ValueError:
        return None
    return base + "%" if is_percent else base


def canonicalize(value_span: str) -> tuple[ValueType, str]:
    """Map a raw value span to (type, canonical string). Deterministic + pure."""
    s = value_span.strip()
    if not s:
        return ValueType.STRING, ""

    # currency (before plain numbers)
    if _CURRENCY_RE.search(s):
        m = _CURRENCY_RE.search(s)
        digits = re.sub(r"[^\d.]", "", m.group(0))
        try:
            return ValueType.CURRENCY, f"{float(digits):.2f}"
        except ValueError:
            pass

    # duration (unit-normalized to days)
    dm = _DURATION_RE.search(s)
    if dm:
        qty_raw, unit = dm.group(1), dm.group(2).lower()
        qty = int(qty_raw) if qty_raw.isdigit() else _word_to_number(qty_raw)
        if qty is not None:
            return ValueType.DURATION, f"{qty * _DURATION_DAYS[unit]} days"

    # ISO date
    im = _ISO_DATE_RE.search(s)
    if im:
        y, mo, d = im.groups()
        return ValueType.DATE, f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    # textual date ("March 3, 2025" / "Mar 3 2025")
    tm = _TEXT_DATE_RE.search(s)
    if tm:
        mon_raw, d, y = tm.group(1).lower(), tm.group(2), tm.group(3)
        mon = _MONTHS.get(mon_raw) or _MONTHS_ABBR.get(mon_raw[:3])
        if mon:
            return ValueType.DATE, f"{int(y):04d}-{mon:02d}-{int(d):02d}"

    # entity/id token (e.g. "gw-west-2"): a single hyphen/alnum token that isn't a pure number
    # and isn't a spelled-out number word. Checked BEFORE bare-number extraction so the trailing
    # digit in "gw-west-2" isn't lifted, but "fifteen" still falls through to the number branch.
    if (
        re.fullmatch(r"[A-Za-z0-9][\w-]*", s)
        and len(s) <= 40
        and re.search(r"[A-Za-z]", s)
        and _word_to_number(s) is None
    ):
        return ValueType.ENTITY, s.casefold()

    # number, incl. "fifteen (15)" -> 15 and "fifteen" -> 15. Percent is kept distinct from the
    # bare number ("92%" != "92") and huge integers use arbitrary precision (no float 'inf').
    is_percent = "%" in s
    pm = _PAREN_NUM_RE.search(s)
    if pm:
        return ValueType.NUMBER, _num_canon(pm.group(1), is_percent)
    nm = _INT_RE.search(s)
    if nm:
        digits = nm.group(0).replace(",", "")
        canon = _num_canon(digits, is_percent)
        if canon is not None:
            return ValueType.NUMBER, canon
    wn = _word_to_number(s)
    if wn is not None:
        return ValueType.NUMBER, _num_canon(str(wn), is_percent)

    return ValueType.STRING, " ".join(s.casefold().split())
