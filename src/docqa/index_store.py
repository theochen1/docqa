"""Index store — one SQLite file holding claim records + vectors + a fingerprint.

Design choice (BT09): the DEFAULT store is exact brute-force search over vectors kept as blobs in
the same `index.db` as the claim text + provenance. Rationale:
- provenance lives in the SAME artifact as the vectors, so a citation can never desync into an
  untraceable claim (R-PROV / the loudest pitfall);
- exact search is the design's choice anyway (deterministic, no ANN recall drift) and is cheap at
  <100k claims;
- it avoids sqlite-vec's pre-v1 wheel risk on a fresh clone.
sqlite-vec can slot in behind this same interface if ANN ever becomes necessary.

Writes are atomic: build into `index.db.tmp`, then rename — a crashed index never leaves a
half-written store a later `ask` would read.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from docqa.types import ClaimRecord

SCHEMA_VERSION = "1"


class IndexStore:
    def __init__(self, path: str):
        self.path = path

    # --- write path ---

    def build(
        self,
        claims: list[ClaimRecord],
        vectors: np.ndarray | None,
        meta: dict,
    ) -> None:
        """Write claims (+ optional aligned vectors) atomically. `meta` is the fingerprint."""
        if vectors is not None and len(vectors) != len(claims):
            raise ValueError("vectors and claims length mismatch")

        tmp = self.path + ".tmp"
        Path(tmp).unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        try:
            con.execute(
                "CREATE TABLE claims ("
                "row INTEGER PRIMARY KEY, claim_id TEXT, filename TEXT, locator TEXT, text TEXT, "
                "subject_norm TEXT, predicate_norm TEXT, value_span TEXT, value_type TEXT, "
                "value_canon TEXT, source_status TEXT)"
            )
            con.execute("CREATE TABLE vectors (row INTEGER PRIMARY KEY, vec BLOB, dim INTEGER)")
            con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

            for i, c in enumerate(claims):
                con.execute(
                    "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        i, c.claim_id, c.filename, c.locator, c.text, c.subject_norm,
                        c.predicate_norm, c.value_span, str(c.value_type), c.value_canon,
                        str(c.source_status),
                    ),
                )
                if vectors is not None:
                    v = np.asarray(vectors[i], dtype=np.float32)
                    con.execute(
                        "INSERT INTO vectors VALUES (?,?,?)", (i, v.tobytes(), int(v.shape[0]))
                    )

            stamped = dict(meta)
            stamped["schema_version"] = SCHEMA_VERSION
            for k, v in stamped.items():
                con.execute("INSERT INTO meta VALUES (?,?)", (k, json.dumps(v)))
            con.commit()
        finally:
            con.close()

        # Atomic swap: the tmp file becomes the live index only on full success.
        Path(tmp).replace(self.path)

    # --- read path ---

    def exists(self) -> bool:
        return Path(self.path).is_file()

    def _connect(self) -> sqlite3.Connection:
        if not self.exists():
            raise FileNotFoundError(f"no index at {self.path} (run `docqa index`)")
        return sqlite3.connect(self.path)

    def read_meta(self) -> dict:
        con = self._connect()
        try:
            rows = con.execute("SELECT key, value FROM meta").fetchall()
        finally:
            con.close()
        return {k: json.loads(v) for k, v in rows}

    def fingerprint_matches(self, expected: dict) -> bool:
        """A query-path guard: the stored fingerprint must match the runtime's config."""
        stored = self.read_meta()
        return all(stored.get(k) == v for k, v in expected.items())

    def load_claims(self) -> list[ClaimRecord]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT claim_id, filename, locator, text, subject_norm, predicate_norm, "
                "value_span, value_type, value_canon, source_status FROM claims ORDER BY row"
            ).fetchall()
        finally:
            con.close()
        return [
            ClaimRecord(
                claim_id=r[0], filename=r[1], locator=r[2], text=r[3], subject_norm=r[4],
                predicate_norm=r[5], value_span=r[6], value_type=r[7], value_canon=r[8],
                source_status=r[9],
            )
            for r in rows
        ]

    def load_vectors(self) -> np.ndarray | None:
        con = self._connect()
        try:
            rows = con.execute("SELECT vec, dim FROM vectors ORDER BY row").fetchall()
        finally:
            con.close()
        if not rows:
            return None
        return np.stack([np.frombuffer(v, dtype=np.float32) for v, _ in rows])

    def count(self) -> int:
        con = self._connect()
        try:
            return con.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        finally:
            con.close()
