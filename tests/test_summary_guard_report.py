"""`scripts/summary_guard_report.py` (#47 stage 2) on hand-written records."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kslimeans.runlog import RunRecord

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summary_guard_report.py"
spec = importlib.util.spec_from_file_location("summary_guard_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def _trace(step, calls=0, changed=None, guarded=(0, 0, 0)):
    n, dup, men = guarded
    return {
        "step": step,
        "n_calls": calls,
        "frac_changed": changed,
        "n_guarded": n,
        "n_guarded_duplicate": dup,
        "n_guarded_mentions": men,
    }


def _record(method, assignments, trace, summaries=()):
    return RunRecord(
        method=method,
        seed=3,
        trace=trace,
        summaries_evolution=[list(s) for s in summaries],
        assignments_evolution=[np.asarray(a) for a in assignments],
        summary_source_steps=list(range(len(summaries))),
    )


def test_guard_counts_split_by_flag():
    # step 1: 3 guarded, 2 duplicates and 3 mentions (so 2 are both); step 2: a duplicate that does not mention.
    record = _record(
        "g",
        [[0, 1], [0, 1], [0, 1]],
        [_trace(0), _trace(1, 4, 0.5, (3, 2, 3)), _trace(2, 2, 0.1, (1, 1, 0))],
        summaries=[["a"] * 4, ["a"] * 4],
    )
    df = report.guard_counts(record)
    assert list(df.step) == [1, 2]
    first, second = df.iloc[0], df.iloc[1]
    assert (first.n_guarded, first.n_duplicate, first.n_mentions, first.n_both) == (3, 2, 3, 2)
    assert (first.n_duplicate_only, first.n_mentions_only) == (0, 1)
    assert first.share == pytest.approx(3 / 4)
    assert (second.n_both, second.n_duplicate_only, second.n_mentions_only) == (0, 1, 0)
    assert first.seed == 3


def test_recomputed_flags_use_the_surface_rules():
    summaries = ["How do I book?", "How do I book?", "A cluster of travel questions", None, "Pay a bill"]
    dup, mentions, union, ids = report.recomputed_flags(summaries)
    assert (dup, mentions, union, ids) == (2, 1, 3, [0, 1, 2])


def test_base_flag_counts_number_steps_like_guard_counts():
    record = _record("b", [[0], [0]], [], summaries=[["x", "x", "y"], ["a cluster", "b", "c"]])
    df = report.base_flag_counts(record)
    assert list(df.step) == [1, 2]
    assert list(df.n_flagged_base) == [2, 1]
    assert list(df.n_duplicate_base) == [2, 0]
    assert list(df.n_mentions_base) == [0, 1]


def test_expected_calls_counts_member_sets_never_summarized():
    a0 = np.array([0, 0, 1, 1, 2, 2])
    a1 = np.array([0, 0, 1, 1, 2, 1])  # cluster 0 repeats; clusters 1 and 2 are new member sets
    a2 = np.array([0, 0, 1, 1, 2, 2])  # all three come back from step 0
    assert report.expected_calls([a0, a1, a2], 3, k=4) == [3, 2, 0]


def test_expected_calls_skips_empty_clusters():
    assert report.expected_calls([np.array([0, 0, 2, 2])], 1, k=3) == [2]


def test_step_table_and_paired_intervals():
    gold = np.array([0, 0, 1, 1, 2, 2])
    perfect, wrong = [0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 2, 2]
    trace = [_trace(0), _trace(1, 3, 0.2)]
    guard = _record("g", [perfect, perfect], trace)
    base = _record("b", [perfect, wrong], [_trace(0), _trace(1, 3, 0.6)])
    small = _record("s", [perfect, perfect], trace)
    table = report.step_table(guard, base, small, gold)
    assert list(table.step) == [0, 1]
    assert table.loc[0, "ari_between"] == pytest.approx(1.0)
    assert table.loc[1, "n_differ"] == 2
    assert table.loc[1, "ari_between"] < 1
    assert table.loc[1, "ari_guard_small"] == pytest.approx(1.0)
    assert table.loc[1, "acc_guard"] == pytest.approx(1.0)
    assert table.loc[1, "acc_base"] == pytest.approx(4 / 6)
    assert table.loc[1, "frac_changed_guard"] == pytest.approx(0.2)

    # Two seeds with a difference of 1/3 and 1/6 in ACC at step 1.
    other = table.assign(seed=4)
    other.loc[other.step == 1, "acc_base"] = 5 / 6
    two = pd.concat([table, other], ignore_index=True)
    paired = report.steps_paired(two)
    row = paired[(paired.step == 1) & (paired.metric == "acc")].iloc[0]
    assert row.mean_diff == pytest.approx((1 / 3 + 1 / 6) / 2)
    assert row.n == 2 and row.wins == 2
    assert row.ci_low < row.mean_diff < row.ci_high
    zero = paired[(paired.step == 0) & (paired.metric == "acc")].iloc[0]
    assert zero.mean_diff == 0 and zero.wins == 0


def test_gap_closed_is_gain_over_gap_and_nan_when_reference_is_not_above():
    rows = []
    # guard - base = +0.01 on every metric; guard - kmeans = -0.03 (gap 0.04); guard - small = +0.02 (small below base).
    for metric in report.METRICS:
        for ref, diff in (("base", 0.01), ("kmeans", -0.03), ("small", 0.02)):
            rows.append({"reference": ref, "metric": metric, "mean_diff": diff})
    out = report.gap_closed(pd.DataFrame(rows), "base", ["kmeans", "small"])
    acc = out[out.metric == "acc"].set_index("reference")
    assert acc.loc["kmeans", "gap_to_base"] == pytest.approx(0.04)
    assert acc.loc["kmeans", "share_closed"] == pytest.approx(0.25)
    assert acc.loc["small", "gap_to_base"] == pytest.approx(-0.01)
    assert np.isnan(acc.loc["small", "share_closed"])
