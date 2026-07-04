#!/usr/bin/env python3
"""Build a full-scale synthetic corpus for the blocking latency gate (BT24).

The chosen operational constraint is LATENCY (p50 < 5s), and the constraint MUST be enforced at
real scale — a p50 on the tiny sample corpus proves nothing. This script generates a large,
deterministic, offline corpus of mixed formats (.md / .txt / .eml — the handled formats; PDFs are a
documented skip) into `scale_corpus/` (git-ignored), so the blocking latency gate
(`docqa eval --latency --corpus scale_corpus/`) has something real to time against.

Deterministic (seeded) so re-runs are byte-identical and the gate is reproducible. Offline (pure
string assembly, no network / no LLM). Facts are embedded with recognizable entities so the latency
probe queries actually retrieve + answer, exercising the whole pipeline rather than empty refusals.

Usage:
    python scripts/build_corpus.py                       # 200 docs, ~default scale
    python scripts/build_corpus.py --docs 200 --words-per-doc 3000   # ~600k words (assess. scale)
    python scripts/build_corpus.py --out scale_corpus --docs 50      # smaller, faster
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# Domain vocabulary — HR / ops / finance, matching the sample corpus flavor so the corpus reads as a
# plausible "pile of company documents" rather than lorem ipsum.
_TEAMS = ["platform", "data", "security", "growth", "billing", "infra", "mobile", "research"]
_SITES = ["Portland", "Denver", "Austin", "Dublin", "Fairbanks", "Boise", "Reno", "Tucson"]
_DEPTS = ["Engineering", "Finance", "People Ops", "Legal", "Support", "Marketing"]
_TOPICS = [
    "onboarding", "expense policy", "incident review", "quarterly planning", "access control",
    "vendor management", "data retention", "on-call rotation", "travel policy", "benefits update",
]
_FILLER = [
    "This document is maintained by the {dept} team and reviewed each quarter.",
    "Employees should direct questions to their manager or the {dept} partner.",
    "The policy below supersedes any earlier guidance on the same subject.",
    "Exceptions require written approval and are logged for the annual audit.",
    "Related procedures are described in the internal knowledge base.",
    "All figures are stated in US dollars unless otherwise noted.",
    "Changes take effect on the first business day of the following month.",
    "Managers are responsible for communicating updates to their reports.",
    "This section is informational and does not create a contractual obligation.",
    "Records are retained according to the standard retention schedule.",
]


def _fact_sentences(rng: random.Random, seq: int) -> list[str]:
    """A handful of sentences carrying retrievable facts (numbers/dates/entities) unique-ish per doc
    so the corpus has real answerable content, not just filler."""
    team = rng.choice(_TEAMS)
    site = rng.choice(_SITES)
    dept = rng.choice(_DEPTS)
    pto = rng.choice([12, 15, 18, 20, 25])
    budget = rng.choice([5, 8, 10, 12, 15, 20]) * 1000
    year = rng.choice([2024, 2025, 2026])
    return [
        f"The {team} team's primary gateway is deployed in the {site} datacenter.",
        f"Full-time staff in {dept} accrue {pto} days of paid time off per year.",
        f"The approved {team} tooling budget for FY{year % 100} is ${budget:,}.",
        f"Document reference number is DOC-{seq:05d}.",
        f"The {dept} lead approves requests above the standard cap.",
    ]


def _paragraph(rng: random.Random, dept: str, n_sentences: int) -> str:
    return " ".join(rng.choice(_FILLER).format(dept=dept) for _ in range(n_sentences))


def _doc_body(rng: random.Random, seq: int, words_target: int) -> tuple[str, list[str]]:
    """Return (title, section_blocks). Sections mix embedded facts with filler to reach the word
    target."""
    topic = rng.choice(_TOPICS)
    dept = rng.choice(_DEPTS)
    title = f"{topic.title()} — Ref DOC-{seq:05d}"
    facts = _fact_sentences(rng, seq)
    blocks: list[str] = []
    words_so_far = 0
    section_idx = 0
    # Spread the facts across sections, padding with filler until we reach the word target.
    while words_so_far < words_target:
        heading = f"Section {section_idx + 1}"
        parts = []
        if section_idx < len(facts):
            parts.append(facts[section_idx])
        parts.append(_paragraph(rng, dept, rng.randint(3, 7)))
        block = f"## {heading}\n\n" + " ".join(parts)
        blocks.append(block)
        words_so_far += sum(len(p.split()) for p in parts)
        section_idx += 1
        if section_idx > 200:  # hard stop so a huge words-per-doc can't spin forever
            break
    return title, blocks


def _write_markdown(path: Path, title: str, blocks: list[str]) -> None:
    path.write_text(f"# {title}\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")


def _write_text(path: Path, title: str, blocks: list[str]) -> None:
    # Plain text: strip the markdown headings to a simple upper-case line.
    body = "\n\n".join(b.replace("## ", "").upper().split("\n")[0] + "\n" +
                       "\n".join(b.split("\n")[1:]) for b in blocks)
    path.write_text(f"{title}\n\n{body}\n", encoding="utf-8")


def _write_eml(path: Path, title: str, blocks: list[str], seq: int) -> None:
    body = "\n\n".join(b.replace("## ", "").replace("\n", " ") for b in blocks)
    path.write_text(
        f"From: ops@aldermere.example\n"
        f"To: all@aldermere.example\n"
        f"Subject: {title}\n"
        f"Date: Mon, 06 Jan 2025 09:{seq % 60:02d}:00 -0800\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


def build(out_dir: str, docs: int, words_per_doc: int, seed: int = 1234) -> dict:
    """Generate the corpus. Returns a summary dict. Deterministic for fixed (docs, words, seed)."""
    rng = random.Random(seed)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    counts = {"md": 0, "txt": 0, "eml": 0}
    total_words = 0
    for seq in range(docs):
        title, blocks = _doc_body(rng, seq, words_per_doc)
        total_words += sum(len(b.split()) for b in blocks)
        fmt = ["md", "txt", "eml"][seq % 3]  # even mix of the three handled formats
        if fmt == "md":
            _write_markdown(root / f"doc_{seq:05d}.md", title, blocks)
        elif fmt == "txt":
            _write_text(root / f"doc_{seq:05d}.txt", title, blocks)
        else:
            _write_eml(root / f"doc_{seq:05d}.eml", title, blocks, seq)
        counts[fmt] += 1

    return {"dir": str(root), "docs": docs, "words": total_words, "formats": counts, "seed": seed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a full-scale synthetic corpus for the "
                                             "blocking latency gate (BT24).")
    ap.add_argument("--out", default="scale_corpus", help="Output dir (default: scale_corpus).")
    ap.add_argument("--docs", type=int, default=200, help="Number of documents (default: 200).")
    ap.add_argument("--words-per-doc", type=int, default=3000,
                    help="Approx words per doc (default: 3000 -> ~600k total, assessment scale).")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed for reproducibility.")
    args = ap.parse_args(argv)

    summary = build(args.out, args.docs, args.words_per_doc, args.seed)
    print(f"built {summary['docs']} docs (~{summary['words']:,} words) into {summary['dir']}/ "
          f"[md={summary['formats']['md']} txt={summary['formats']['txt']} "
          f"eml={summary['formats']['eml']}, seed={summary['seed']}]")
    print("next: docqa eval --latency --corpus " + summary["dir"] + "/   (needs ANTHROPIC_API_KEY)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
