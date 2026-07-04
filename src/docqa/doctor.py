"""`docqa doctor` — fail-fast preflight.

Surfaces fresh-clone problems as one clear report instead of a mid-query crash. Each check is
non-fatal here (doctor reports status); the actual fail-fast on a missing key happens when the
answer path calls `config.require_api_key()`.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from docqa.config import API_KEY_VAR, Settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _has_module(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def run_checks(settings: Settings | None = None) -> list[Check]:
    settings = settings or Settings.load()
    checks: list[Check] = []

    # API key (required for the answer path only).
    key_set = bool(os.environ.get(API_KEY_VAR, "").strip())
    checks.append(
        Check(
            name=f"answer-path key ({API_KEY_VAR})",
            ok=key_set,
            detail="set" if key_set else "MISSING — needed for `ask`/`eval`, not for `index`",
        )
    )

    # Core dependency present.
    checks.append(
        Check(name="pydantic", ok=_has_module("pydantic"), detail="import check")
    )

    # Index artifact (informational — absent is fine before first index).
    idx = Path(settings.index_path)
    checks.append(
        Check(
            name=f"index ({settings.index_path})",
            ok=True,
            detail="present" if idx.exists() else "absent (run `docqa index` to build)",
        )
    )

    # Config echo — proves knobs are loaded, aids reproducibility.
    checks.append(
        Check(
            name="config",
            ok=True,
            detail=f"gen={settings.gen_model} embed={settings.embed_model} k={settings.k}",
        )
    )

    return checks


def format_report(checks: list[Check]) -> str:
    lines = ["docqa doctor:"]
    for c in checks:
        mark = "ok  " if c.ok else "MISS"
        lines.append(f"  [{mark}] {c.name}: {c.detail}")
    return "\n".join(lines)
