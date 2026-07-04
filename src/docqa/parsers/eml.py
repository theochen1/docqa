""".eml parser: extract readable body + key headers via stdlib email. Never index MIME/base64.

Locator is structured: "email:headers" for the header block and "email:body" for the decoded
plain-text body (not a raw-file line number, which is meaningless after MIME is stripped).
"""

from __future__ import annotations

from email import message_from_string, policy
from email.message import EmailMessage
from pathlib import Path

from docqa.parsers import MIN_USABLE_CHARS, ParseOutcome
from docqa.types import TextSegment

_HEADER_KEYS = ("From", "To", "Cc", "Subject", "Date")


class EmlParser:
    def can_parse(self, path: str) -> bool:
        return path.lower().endswith(".eml")

    def parse_text(self, filename: str, raw: str) -> ParseOutcome:
        try:
            msg: EmailMessage = message_from_string(raw, policy=policy.default)  # type: ignore[assignment]
        except Exception as e:  # noqa: BLE001 - never crash on a malformed message
            return ParseOutcome(skipped=True, skip_reason=f"unparseable .eml: {e}")

        segments: list[TextSegment] = []

        # Header block — who/when/subject, often the answerable fact.
        header_lines = [f"{k}: {msg[k]}" for k in _HEADER_KEYS if msg[k]]
        if header_lines:
            segments.append(
                TextSegment(
                    filename=filename,
                    locator="email:headers",
                    text="\n".join(header_lines),
                )
            )

        # Body — prefer plain-text; decode transfer-encoding; skip attachments/base64 blobs.
        body = self._extract_body(msg)
        if body and len(body.strip()) >= MIN_USABLE_CHARS:
            segments.append(
                TextSegment(filename=filename, locator="email:body", text=body.strip())
            )

        if not segments:
            return ParseOutcome(skipped=True, skip_reason="no usable headers or body in .eml")
        return ParseOutcome(segments=segments)

    def _extract_body(self, msg: EmailMessage) -> str:
        try:
            part = msg.get_body(preferencelist=("plain",))
            if part is not None:
                return part.get_content()
            # Fall back to any text/plain part; ignore attachments and non-text.
            for p in msg.walk():
                if p.get_content_type() == "text/plain" and not p.is_attachment():
                    return p.get_content()
        except Exception:  # noqa: BLE001 - malformed part; degrade to empty, never crash
            return ""
        return ""

    def parse(self, path: str) -> list[TextSegment]:
        return self.parse_file(path).segments

    def parse_file(self, path: str) -> ParseOutcome:
        p = Path(path)
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            return ParseOutcome(skipped=True, skip_reason=f"unreadable .eml: {e}")
        return self.parse_text(p.name, raw)
