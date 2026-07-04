"""Ingestion orchestration: a folder -> parse -> claimize -> embed -> persist, with a manifest.

The manifest's ground truth is an INDEPENDENT directory scan (os.walk), never the indexer's own
self-report — so `discovered == parsed + skipped` is a real reconciliation, not a tautology
(closes the circular-manifest false-green from the test plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from docqa.claimize import CLAIMIZER_VERSION, claimize, claimize_llm
from docqa.index_store import IndexStore
from docqa.parsers import ParseOutcome
from docqa.parsers.eml import EmlParser
from docqa.parsers.markdown import MarkdownParser
from docqa.parsers.pdf import PdfParser
from docqa.parsers.text import TextParser
from docqa.types import ClaimRecord

_PARSERS = [MarkdownParser(), TextParser(), EmlParser(), PdfParser()]
_SUPPORTED_EXT = {".md", ".markdown", ".txt", ".text", ".eml", ".pdf"}


@dataclass
class FileResult:
    filename: str
    status: str          # "parsed" | "skipped"
    reason: str = ""
    claim_count: int = 0


@dataclass
class Manifest:
    discovered: int = 0
    parsed: int = 0
    skipped: int = 0
    total_claims: int = 0
    claimizer: str = "deterministic"
    files: list[FileResult] = field(default_factory=list)

    def reconciles(self) -> bool:
        return self.discovered == self.parsed + self.skipped

    def summary(self) -> str:
        lines = [
            f"indexed {self.parsed}/{self.discovered} files, "
            f"{self.skipped} skipped, {self.total_claims} claims "
            f"(claimizer: {self.claimizer})",
        ]
        for f in self.files:
            tag = f"{f.status}" + (f" ({f.reason})" if f.reason else "")
            extra = f" claims={f.claim_count}" if f.claim_count else ""
            lines.append(f"  {f.filename}: {tag}{extra}")
        return "\n".join(lines)


def _parser_for(path: str):
    for p in _PARSERS:
        if p.can_parse(path):
            return p
    return None


def _discover(corpus_dir: str) -> list[str]:
    """Independent ground-truth file discovery — deterministic, sorted, recursive."""
    root = Path(corpus_dir)
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file())


def parse_corpus(corpus_dir: str, decomposer=None) -> tuple[list[ClaimRecord], Manifest]:
    """Parse + claimize every file. Returns claims + a reconciled manifest. No per-file crash.

    `decomposer` (LLM) is the PRIMARY claimization path when supplied; without it, the deterministic
    regex fallback runs. Either way canon.py gates value equality.
    """
    manifest = Manifest()
    manifest.claimizer = "llm" if decomposer is not None else "deterministic"
    all_claims: list[ClaimRecord] = []

    for path in _discover(corpus_dir):
        manifest.discovered += 1
        name = Path(path).name
        ext = Path(path).suffix.lower()
        parser = _parser_for(path)

        if parser is None or ext not in _SUPPORTED_EXT:
            manifest.skipped += 1
            manifest.files.append(
                FileResult(name, "skipped", reason=f"unsupported format: {ext or 'no ext'}")
            )
            continue

        outcome: ParseOutcome = parser.parse_file(path)
        if not outcome.usable:
            manifest.skipped += 1
            manifest.files.append(
                FileResult(name, "skipped", reason=outcome.skip_reason)
            )
            continue

        # Relabel provenance to the corpus-root-relative path so files with the same basename in
        # different subdirs don't collide (and claim_ids, which hash filename+locator+text, stay
        # unique across the corpus).
        rel = str(Path(path).relative_to(corpus_dir))
        for seg in outcome.segments:
            seg.filename = rel

        claims = claimize_llm(outcome.segments, decomposer) if decomposer else claimize(
            outcome.segments
        )
        all_claims.extend(claims)
        manifest.parsed += 1
        manifest.total_claims += len(claims)
        manifest.files.append(
            FileResult(name, "parsed", claim_count=len(claims))
        )

    return all_claims, manifest


def build_index(corpus_dir: str, index_path: str, embedder, decomposer=None) -> Manifest:
    """Full index build: parse -> claimize -> embed -> persist. Returns the manifest.

    `decomposer` (LLM) is the primary claimizer when supplied; else the deterministic fallback.
    """
    claims, manifest = parse_corpus(corpus_dir, decomposer=decomposer)
    vectors = embedder.embed([c.text for c in claims]) if claims else None
    meta = {
        "embed_model": embedder.model_id,
        "claimizer_version": CLAIMIZER_VERSION,
        "claimizer": manifest.claimizer,
        "claim_count": len(claims),
    }
    IndexStore(index_path).build(claims, vectors, meta)
    return manifest
