"""BT02: config loads with defaults, env overrides work, and the answer-path key fails fast
with an actionable message. Indexing needs no key."""

import pytest

from docqa.config import API_KEY_VAR, MissingDependencyError, Settings, require_api_key


def test_defaults_present():
    s = Settings.load(dotenv=False)
    # A few load-bearing defaults from algorithm-design.md §9.
    assert s.k == 8
    assert s.rrf_k == 60
    assert s.mmr_lambda == 0.5
    assert s.cluster_sim == 0.80
    assert s.entail_threshold == 0.5
    assert s.multihop is False
    assert "haiku" in s.gen_model.lower()  # fast tier by default, not Opus


def test_env_override(monkeypatch):
    monkeypatch.setenv("DOCQA_K", "12")
    monkeypatch.setenv("DOCQA_CLUSTER_SIM", "0.9")
    monkeypatch.setenv("DOCQA_MULTIHOP", "1")
    s = Settings.load(dotenv=False)
    assert s.k == 12
    assert s.cluster_sim == 0.9
    assert s.multihop is True


def test_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DOCQA_K", "not-a-number")
    s = Settings.load(dotenv=False)
    assert s.k == 8  # invalid value ignored, default kept


def test_require_api_key_fails_fast_with_named_var(monkeypatch):
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    with pytest.raises(MissingDependencyError) as exc:
        require_api_key()
    msg = str(exc.value)
    assert API_KEY_VAR in msg          # names the missing var
    assert "index" in msg.lower()      # explains indexing does not need it


def test_require_api_key_returns_when_set(monkeypatch):
    monkeypatch.setenv(API_KEY_VAR, "sk-test-123")
    assert require_api_key() == "sk-test-123"


def test_doctor_reports_missing_key(monkeypatch):
    from docqa.doctor import run_checks

    monkeypatch.delenv(API_KEY_VAR, raising=False)
    checks = run_checks(Settings.load(dotenv=False))
    key_check = next(c for c in checks if API_KEY_VAR in c.name)
    assert key_check.ok is False
