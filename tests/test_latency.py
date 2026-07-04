"""BT14b: latency instrument — percentile math, warmup exclusion, env stamp, sample label."""

from docqa.latency import LatencyReport, Timer, time_queries


def test_p50_p95_nearest_rank():
    r = LatencyReport(samples_ms=[10, 20, 30, 40, 100])
    assert r.p50 == 30
    assert r.p95 == 100  # nearest-rank top


def test_percentiles_empty_safe():
    r = LatencyReport()
    assert r.p50 == 0.0 and r.p95 == 0.0


def test_line_labels_sample_as_not_slo():
    r = LatencyReport(samples_ms=[100.0], n_docs=3, n_claims=10, n_words=200, scale="sample")
    line = r.line()
    assert "NOT the SLO" in line
    assert "3 docs" in line and "10 claims" in line
    assert "env=" in line and "n=1" in line


def test_full_scale_line_has_no_sample_disclaimer():
    r = LatencyReport(samples_ms=[100.0], scale="full")
    assert "NOT the SLO" not in r.line()


def test_timer_measures_positive_elapsed():
    with Timer() as t:
        sum(range(1000))
    assert t.elapsed_ms >= 0.0


def test_time_queries_excludes_warmup():
    calls = []
    r = time_queries(lambda q: calls.append(q), ["a", "b", "c"], warmup=True)
    # 3 questions + 1 warmup call = 4 invocations, but only 3 recorded samples.
    assert len(calls) == 4
    assert len(r.samples_ms) == 3
