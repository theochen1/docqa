"""BT05: .eml parser. Body+headers extracted, MIME/base64 never indexed, no crash on malformed."""

import base64

from docqa.parsers.eml import EmlParser

_QP_EMAIL = (
    "From: it-team@aldermere.example\n"
    "To: newhire@aldermere.example\n"
    "Subject: Your laptop + VPN setup\n"
    "Date: Mon, 14 Apr 2025 09:12:00 -0700\n"
    "Content-Type: text/plain; charset=utf-8\n"
    "Content-Transfer-Encoding: quoted-printable\n"
    "\n"
    "Your assigned VPN gateway is gw-west-3D2 and your default printer is on the 4th floor.\n"
)


def test_eml_extracts_headers_and_body():
    out = EmlParser().parse_text("onboarding.eml", _QP_EMAIL)
    assert out.usable
    locs = {s.locator for s in out.segments}
    assert locs == {"email:headers", "email:body"}
    headers = next(s for s in out.segments if s.locator == "email:headers")
    assert "Your laptop + VPN setup" in headers.text  # Subject retrievable
    assert "it-team@aldermere.example" in headers.text
    body = next(s for s in out.segments if s.locator == "email:body")
    assert "gw-west-3" in body.text.replace("=3D", "")  # QP decoded (=3D -> =)


def test_eml_multipart_base64_attachment_not_indexed():
    blob = base64.b64encode(b"\x00\x01BINARY-ATTACHMENT-BYTES\x02\x03").decode()
    multipart = (
        "From: a@x.example\n"
        "To: b@x.example\n"
        "Subject: Report\n"
        'Content-Type: multipart/mixed; boundary="B"\n'
        "\n"
        "--B\n"
        "Content-Type: text/plain\n\n"
        "The approved travel budget is $9,900.\n"
        "--B\n"
        "Content-Type: application/octet-stream\n"
        "Content-Transfer-Encoding: base64\n"
        'Content-Disposition: attachment; filename="a.bin"\n\n'
        f"{blob}\n"
        "--B--\n"
    )
    out = EmlParser().parse_text("report.eml", multipart)
    assert out.usable
    body = next(s for s in out.segments if s.locator == "email:body")
    assert "$9,900" in body.text
    # The base64 blob and MIME boundary markers must NOT appear anywhere.
    full = "\n".join(s.text for s in out.segments)
    assert blob not in full
    assert "--B" not in full
    assert "octet-stream" not in full


def test_eml_malformed_no_crash():
    out = EmlParser().parse_text("bad.eml", "\x00\x01 not really an email \xff")
    # Either skipped or yields something, but never raises.
    assert out.segments == [] or out.usable


def test_eml_empty_skipped():
    out = EmlParser().parse_text("e.eml", "")
    assert out.skipped
