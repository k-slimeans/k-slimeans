"""`Result.trace`: one plain-Python row per KMeans fit, purely additive to `summary_kmeans`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kslimeans.core import _objective as core_objective
from kslimeans.core import summary_kmeans
from kslimeans.results import RunDir
from kslimeans.schedules import FixedSchedule, State, Step, SummarySchedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_equivalence import ClosestToMeanSummarizer, LookupEmbedder, synthetic  # noqa: E402,F401
from synth import N_BLOBS  # noqa: E402

TRACE_KEYS = {
    "step",
    "inertia",
    "prev_inertia",
    "frac_changed",
    "summary_shift",
    "last_n_iter",
    "last_max_iter",
    "summarize",
    "summary_inertia",
}


def _run(synthetic, schedule, tol=-1.0):
    texts, X, embed = synthetic
    return summary_kmeans(
        texts,
        N_BLOBS,
        ClosestToMeanSummarizer(embed),
        LookupEmbedder(embed),
        schedule,
        text_features=X,
        tol=tol,
        random_state=0,
        verbose=False,
    )


def _objective(X, centers):
    d2 = ((X[:, None, :].astype(np.float64) - centers[None].astype(np.float64)) ** 2).sum(-1)
    return float(d2.min(axis=1).sum())


def test_fixed_schedule_trace_shape(synthetic):
    trace = _run(synthetic, FixedSchedule(5, 100, True, True)).trace

    assert len(trace) == 6
    assert [r["step"] for r in trace] == list(range(6))
    assert [r["summarize"] for r in trace] == [False] + [True] * 5
    assert all(set(r) == TRACE_KEYS for r in trace)
    for i in range(1, 6):
        assert trace[i]["prev_inertia"] == trace[i - 1]["inertia"]
        assert trace[i]["last_max_iter"] == 100 // 6
        assert isinstance(trace[i]["summary_inertia"], float)
        assert isinstance(trace[i]["summary_shift"], float)
    assert trace[0]["last_max_iter"] == 100 // 6
    for key in ("prev_inertia", "frac_changed", "summary_shift", "summary_inertia"):
        assert trace[0][key] is None
    json.dumps(trace)


def test_last_max_iter_follows_schedule_blocks(synthetic):
    trace = _run(synthetic, FixedSchedule(3, 30, False, False)).trace
    assert [r["last_max_iter"] for r in trace] == [1, 30 // 2, 30 // 2, 1]


def test_summary_inertia_matches_manual_objective(synthetic):
    _, X, _ = synthetic
    result = _run(synthetic, FixedSchedule(5, 100, True, True))
    summary_rows = result.trace[1:]
    assert len(summary_rows) == len(result.centroids_evolution)
    for row, start in zip(summary_rows, result.centroids_evolution):
        np.testing.assert_allclose(row["summary_inertia"], _objective(X, start), rtol=1e-9)


def test_core_objective_is_chunk_invariant(synthetic):
    _, X, _ = synthetic
    centers = X[:N_BLOBS]
    expected = _objective(X, centers)
    for chunk_rows in (7, len(X) - 1, len(X), len(X) + 1):
        np.testing.assert_allclose(core_objective(X, centers, chunk_rows=chunk_rows), expected, rtol=1e-12)


def test_tol_break_row_count(synthetic):
    result = _run(synthetic, FixedSchedule(10, 100, True, True), tol=1e-4)
    assert result.converged
    # The converging summary step stops before fitting, so it has no row.
    assert len(result.trace) == 1 + len(result.centroids_evolution)
    assert len(result.summaries_evolution) == len(result.centroids_evolution) + 1


class _MixedSchedule(SummarySchedule):
    _steps = [Step(False, 3), Step(True, 5), Step(False, 4)]

    def first_iters(self) -> int:
        return 10

    def next_step(self, state: State) -> Step | None:
        return self._steps[state.step] if state.step < len(self._steps) else None


def test_mixed_schedule_rows(synthetic):
    trace = _run(synthetic, _MixedSchedule()).trace

    assert [r["summarize"] for r in trace] == [False, False, True, False]
    assert [r["last_max_iter"] for r in trace] == [10, 3, 5, 4]
    for i in (1, 3):
        assert trace[i]["summary_shift"] is None
        assert trace[i]["summary_inertia"] is None
    assert trace[2]["summary_shift"] is not None
    assert trace[2]["summary_inertia"] is not None
    json.dumps(trace)


def test_log_trace_writes_keyed_lines(tmp_path):
    run = RunDir(tmp_path, "t", {})
    run.log_trace({"seed": 0, "method": "m"}, [{"step": 0, "x": np.float32(1.5)}, {"step": 1, "x": None}])
    lines = [json.loads(line) for line in (run.path / "trace.jsonl").read_text().splitlines()]
    assert lines == [
        {"seed": 0, "method": "m", "step": 0, "x": 1.5},
        {"seed": 0, "method": "m", "step": 1, "x": None},
    ]


def test_log_trace_empty_creates_no_file(tmp_path):
    run = RunDir(tmp_path, "t", {})
    run.log_trace({"seed": 0}, [])
    assert not (run.path / "trace.jsonl").exists()
