"""Claimizer: TextSegment -> atomic ClaimRecords. Deterministic path (BT07).

One claim per sentence. Each claim carries a value-separated proposition key:
- value_span/value_canon: the salient value (number/currency/date/duration/entity) via canon.py
- subject_norm + predicate_norm: the sentence minus the value, normalized (the proposition key
  claims cluster on, so '15 days' and '20 days' about the same subject are comparable)

Deterministic by construction: same bytes + same claimizer version -> byte-identical claims
(R-CHUNK). The LLM decomposition path (BT08) improves quality behind this fallback.
"""

from __future__ import annotations

import hashlib
import re

from docqa.canon import canonicalize
from docqa.types import ClaimRecord, TextSegment, ValueType

CLAIMIZER_VERSION = "det-1"

# Sentence split: end punctuation followed by whitespace. Deterministic, dependency-free.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Salient value spans to lift out of a sentence, in priority order.
# Id tokens (gw-west-2) come BEFORE plain numbers so the trailing digit isn't lifted alone.
_VALUE_PATTERNS = [
    re.compile(r"[$£€]\s?[\d,]+(?:\.\d+)?"),                                    # currency
    re.compile(r"\b\d+\s+(?:days?|weeks?|months?|years?)\b", re.IGNORECASE),    # duration
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),                                   # iso date
    re.compile(r"\b[A-Za-z]+\s+\d{1,2},?\s+\d{4}\b"),                           # textual date
    re.compile(r"\b[a-z]+(?:-[a-z0-9]+)+\b", re.IGNORECASE),                    # id token gw-west-2
    re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b"),                                    # number / percent
]

_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "per", "and"}


def _claim_id(filename: str, locator: str, text: str) -> str:
    h = hashlib.sha1(f"{filename}\x00{locator}\x00{text}".encode()).hexdigest()
    return h[:12]


def _split_sentences(text: str) -> list[str]:
    # Collapse intra-segment newlines to spaces first so a wrapped sentence stays one claim.
    flat = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_RE.split(flat) if s.strip()]


def _extract_value(sentence: str) -> str:
    for pat in _VALUE_PATTERNS:
        m = pat.search(sentence)
        if m:
            return m.group(0)
    return ""


def _proposition_key(sentence: str, value_span: str) -> tuple[str, str]:
    """subject_norm, predicate_norm = the sentence minus the value, split heuristically."""
    without_value = sentence.replace(value_span, " ") if value_span else sentence
    raw = re.findall(r"[A-Za-z0-9%$-]+", without_value.lower())
    tokens = [t for t in raw if t not in _STOPWORDS]
    if not tokens:
        return "", ""
    # First ~⅔ of content tokens = subject; the rest = predicate. Crude but deterministic;
    # the LLM path (BT08) does this properly.
    cut = max(1, (len(tokens) * 2) // 3)
    return " ".join(tokens[:cut]), " ".join(tokens[cut:])


def claimize_segment(seg: TextSegment) -> list[ClaimRecord]:
    claims: list[ClaimRecord] = []
    for sentence in _split_sentences(seg.text):
        value_span = _extract_value(sentence)
        vtype, vcanon = canonicalize(value_span) if value_span else (ValueType.STRING, "")
        subj, pred = _proposition_key(sentence, value_span)
        claims.append(
            ClaimRecord(
                claim_id=_claim_id(seg.filename, seg.locator, sentence),
                filename=seg.filename,
                locator=seg.locator,
                text=sentence,
                subject_norm=subj,
                predicate_norm=pred,
                value_span=value_span,
                value_type=vtype,
                value_canon=vcanon,
                source_status=seg.source_status,
            )
        )
    return claims


def claimize(segments: list[TextSegment]) -> list[ClaimRecord]:
    """Deterministic claimization over parser output. LLM path (BT08) wraps this as fallback."""
    out: list[ClaimRecord] = []
    for seg in segments:
        out.extend(claimize_segment(seg))
    return out
