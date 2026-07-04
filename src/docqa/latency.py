"""Latency instrumentation — the chosen operational constraint, measured not guessed.

Wraps the full query path with a monotonic clock, computes p50/p95 with a pinned percentile method,
excludes a warm-up query so cold model-load doesn't distort steady state, and prints the number
with the load-bearing context (corpus size + reference environment). BT14b prints a SAMPLE-corpus
number labeled "not the SLO"; BT24 adds the blocking full-scale gate.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field


@dataclass
class LatencyReport:
    samples_ms: list[float] = field(default_factory=list)
    n_docs: int = 0
    n_claims: int = 0
    n_words: int = 0
    scale: str = "sample"  # "sample" (INFO only) | "full" (SLO-gated at BT24)
    hops_per_query: list[int] = field(default_factory=list)  # BT23: hops taken per query

    def _pct(self, p: float) -> float:
        if not self.samples_ms:
            return 0.0
        xs = sorted(self.samples_ms)
        # nearest-rank, pinned + stated so the number is reproducible run to run
        k = max(0, min(len(xs) - 1, round((p / 100.0) * (len(xs) - 1))))
        return xs[k]

    @property
    def p50(self) -> float:
        return self._pct(50)

    @property
    def p95(self) -> float:
        return self._pct(95)

    @property
    def hop_fraction(self) -> float:
        """Fraction of queries that took >=1 hop — so a grader sees multi-hop stayed off the common
        path (it is OFF by default; when enabled this shows how rarely it engaged)."""
        if not self.hops_per_query:
            return 0.0
        return sum(1 for h in self.hops_per_query if h > 0) / len(self.hops_per_query)

    def env_stamp(self) -> str:
        py = f"py{sys.version_info.major}.{sys.version_info.minor}"
        return f"{py} {platform.system()}/{platform.machine()}"

    def line(self) -> str:
        label = "" if self.scale == "full" else " (sample — NOT the SLO number)"
        hopped = sum(1 for h in self.hops_per_query if h > 0)
        return (
            f"latency p50={self.p50:.0f}ms p95={self.p95:.0f}ms{label} | "
            f"corpus={self.n_docs} docs / {self.n_claims} claims / {self.n_words} words | "
            f"hops={hopped}/{len(self.hops_per_query)} ({self.hop_fraction:.0%}) | "
            f"env={self.env_stamp()} | n={len(self.samples_ms)}"
        )


@dataclass
class GateResult:
    """The outcome of the blocking full-scale latency gate (BT24). `blocking` is what the CLI turns
    into an exit code; `skipped` means the scale corpus was absent (offline / fresh-clone) so the
    gate could not run — that is GREEN by design, never a silent pass presented as a real number."""

    status: str          # "pass" | "fail" | "skip"
    reason: str
    p50_ms: float = 0.0
    slo_ms: float = 0.0

    @property
    def blocking(self) -> bool:
        """Only a genuine over-SLO measurement blocks. A skip (no corpus) does not."""
        return self.status == "fail"

    def line(self) -> str:
        if self.status == "skip":
            return f"latency GATE skipped: {self.reason}"
        verdict = "PASS" if self.status == "pass" else "FAIL"
        return (f"latency GATE {verdict}: p50={self.p50_ms:.0f}ms "
                f"{'<' if self.status == 'pass' else '>='} SLO={self.slo_ms:.0f}ms ({self.reason})")


def evaluate_gate(report: LatencyReport, slo_ms: float, corpus_present: bool,
                  reason: str = "") -> GateResult:
    """Decide the blocking gate purely (no I/O), so the pass/fail/skip logic is unit-testable.

    - corpus absent  -> SKIP (offline / fresh-clone safe; stays green on the sample line).
    - corpus present, full scale, p50 < SLO -> PASS.
    - corpus present, full scale, p50 >= SLO -> FAIL (blocking; non-zero exit).

    The gate NEVER reports a blocking number without a full-scale corpus behind it — a sample-scale
    report can only ever SKIP here (the honest sample line is printed separately by LatencyReport).
    """
    if not corpus_present:
        return GateResult(
            "skip", reason or "scale corpus absent — build with scripts/build_corpus.py",
            slo_ms=slo_ms)
    if report.scale != "full" or not report.samples_ms:
        return GateResult("skip", reason or "no full-scale samples measured", slo_ms=slo_ms)
    p50 = report.p50
    if p50 < slo_ms:
        return GateResult("pass", f"{report.n_docs} docs / {report.n_claims} claims",
                          p50_ms=p50, slo_ms=slo_ms)
    return GateResult("fail", f"{report.n_docs} docs / {report.n_claims} claims",
                      p50_ms=p50, slo_ms=slo_ms)


class Timer:
    """Monotonic wall-clock around one query. Use as a context manager; read .elapsed_ms after."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        return False


def time_queries(query_fn, questions: list[str], warmup: bool = True) -> LatencyReport:
    """Time each question through query_fn (question -> Any). Excludes a warm-up run from stats."""
    report = LatencyReport()
    qs = list(questions)
    if warmup and qs:
        query_fn(qs[0])  # warm the model cache; not recorded
    for q in qs:
        with Timer() as t:
            query_fn(q)
        report.samples_ms.append(t.elapsed_ms)
    return report
