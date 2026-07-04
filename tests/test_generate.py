"""BT12: proposer. Datamarking + id-only exposure + structured normalization, all via the pure
build_prompt / normalize_proposal functions and a stub generator (no API call)."""

from docqa.generate import (
    StubGenerator,
    build_prompt,
    normalize_proposal,
)
from docqa.types import ClaimRecord


def _claims():
    return [
        ClaimRecord(claim_id="realid_pto", filename="handbook.md", locator="#PTO",
                    text="Full-time employees accrue 15 days of PTO."),
        ClaimRecord(claim_id="realid_vpn", filename="net.md", locator="#GW",
                    text="The VPN gateway is gw-west-2."),
    ]


def test_prompt_datamarks_claims():
    p = build_prompt("how much PTO?", _claims())
    assert p["mark"].startswith("MARK_")
    assert f"[CLAIM id=c0 {p['mark']}]" in p["user"]
    # rules state claims are DATA, not instructions
    assert "DATA" in p["system"] and "never an instruction" in p["system"].lower()


def test_proposer_never_sees_filename_or_locator():
    claims = _claims()
    p = build_prompt("q", claims)
    # The real ids / filenames / locators must NOT leak into the prompt the model sees.
    assert "realid_pto" not in p["user"]
    assert "handbook.md" not in p["user"]
    assert "#PTO" not in p["user"]
    # Only local ids c0/c1 are exposed.
    assert "id=c0" in p["user"] and "id=c1" in p["user"]


def test_normalize_maps_local_ids_back_to_claim_ids():
    claims = _claims()
    p = build_prompt("q", claims)
    raw = {"claims": [{"text": "15 days of PTO.", "cite_ids": ["c0"]}], "refusal_token": None}
    out = normalize_proposal(raw, p["id_map"])
    assert out["claims"][0]["cite_ids"] == ["realid_pto"]


def test_normalize_drops_unknown_cite_ids():
    p = build_prompt("q", _claims())
    raw = {"claims": [{"text": "x", "cite_ids": ["c0", "c99", "bogus"]}], "refusal_token": None}
    out = normalize_proposal(raw, p["id_map"])
    assert out["claims"][0]["cite_ids"] == ["realid_pto"]  # c99/bogus dropped


def test_stub_generator_end_to_end():
    claims = _claims()

    def responder(q, cs):
        return {"claims": [{"text": "15 days.", "cite_ids": ["c0"]}], "refusal_token": None}

    out = StubGenerator(responder).propose("how much PTO?", claims)
    assert out["claims"][0]["cite_ids"] == ["realid_pto"]
    assert out["refusal_token"] is None


def test_mark_is_fresh_per_call_not_derivable():
    # Security fix: the datamark must NOT be a function of inputs (else document text could forge
    # the closing delimiter). Same inputs -> different mark each call.
    a = build_prompt("q", _claims())["mark"]
    b = build_prompt("q", _claims())["mark"]
    assert a != b
    assert a.startswith("MARK_") and b.startswith("MARK_")
