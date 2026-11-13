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


# --- `build` on small synthetic run directories -------------------------------------------------------------------

import json  # noqa: E402

import yaml  # noqa: E402

from kslimeans.data import Dataset  # noqa: E402

GOLD = np.array([0, 0, 1, 1, 2, 2])
A0 = [0, 0, 1, 1, 2, 2]
P1 = [0, 0, 1, 1, 2, 1]
STEP1 = ["a cluster", "b", "b"]  # 2 duplicates, 1 mention, 3 flagged
STEP2 = ["x", "y", "z"]


def _run_dir(path, method, label, model, partitions, seeds, calls, guarded=None, scores=(0.5, 0.6, 0.4)):
    path.mkdir(parents=True)
    summarizer = {"type": "hf", "model_id": f"Qwen/{model}"} if model else {"type": "lsa", "top_k": 5}
    cfg = {"name": "r", "dataset": "synth", "embedder": "emb", "method_name": method, "summarizer": summarizer}
    (path / "config.yaml").write_text(yaml.safe_dump(cfg))
    metrics, trace, summaries = [], [], []
    for seed in seeds:
        acc, nmi, ari = (v + 0.01 * seed for v in scores)
        if method == "k-NLPmeans":
            metrics.append({"seed": seed, "method": "kmeans", "acc": 0.45 + 0.01 * seed, "nmi": 0.5, "ari": 0.3})
        metrics.append({"seed": seed, "method": method, "acc": acc, "nmi": nmi, "ari": ari, "runtime_s": 1.0})
        g = guarded or [(0, 0, 0)] * 2
        for t in range(3):
            row = {"seed": seed, "method": method, "step": t, "n_calls": 0 if t == 0 else calls[t - 1],
                   "frac_changed": None if t == 0 else 0.1}
            n, dup, men = (0, 0, 0) if t == 0 else g[t - 1]
            row.update(n_guarded=n, n_guarded_duplicate=dup, n_guarded_mentions=men)
            trace.append(row)
        summaries.append(
            {"seed": seed, "method": method, "summaries": STEP2, "assignments": partitions[-1],
             "summaries_evolution": [STEP1, STEP2], "assignments_evolution": partitions,
             "summary_source_steps": [0, 1]}
        )
    for name, rows in (("metrics", metrics), ("trace", trace), ("summaries", summaries)):
        (path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def runs(tmp_path, monkeypatch):
    ds = Dataset(name="synth", texts=["t"] * 6, labels=GOLD, n_clusters=3, prompt="", text_type="x")
    monkeypatch.setattr(report, "load_run_dataset", lambda log, cache_dir=None: (ds, None))

    def make(guard_seeds=(0, 1), step1=None):
        # guarded: fits A0, P1, A0 -> calls [3, 2]; unguarded: A0, B1, P1 -> calls [3, 2] too.
        base_fits = [A0, [0, 0, 1, 2, 2, 2], P1]
        guard = _run_dir(
            tmp_path / "g", "k-SLMmeans-guard", None, "Qwen3-4B", [A0, step1 or P1, A0], guard_seeds, [3, 2],
            guarded=[(3, 2, 1), (0, 0, 0)],
        )
        base = _run_dir(tmp_path / "b", "k-SLMmeans", None, "Qwen3-4B", base_fits, (0, 1), [3, 2], scores=(0.4, 0.5, 0.3))
        small = _run_dir(tmp_path / "s", "k-SLMmeans", None, "Qwen3-1.7B", base_fits, (0, 1), [3, 2], scores=(0.6, 0.7, 0.5))
        lsa = _run_dir(tmp_path / "l", "k-NLPmeans", None, None, base_fits, (0, 1), [3, 2], scores=(0.55, 0.65, 0.45))
        return guard, base, small, lsa

    return make


def _stage1(path, fit=P1):
    s = report.partition_scores(GOLD, np.asarray(fit))
    rows = [{"seed": seed, "variant": "b", "flag": "meta", "setting": "fixed",
             **{f"end_{m}": s[m] for m in report.METRICS}} for seed in (0, 1)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_pairs_by_seed_and_maps_labels(runs, tmp_path):
    frames, checks = report.build(*runs(), cache_dir="unused", stage1_csv=_stage1(tmp_path / "s1.csv"))
    paired = frames["paired"]
    assert set(paired.reference) == {"k-SLMmeans (Qwen3-4B)", "k-means", "k-NLPmeans (LSA)", "k-SLMmeans (Qwen3-1.7B)"}
    acc = paired[paired.metric == "acc"].set_index("reference")
    # guarded ACC (0.5, 0.51) against the unguarded 4B (0.4, 0.41), k-means (0.45, 0.46) and the 1.7B (0.6, 0.61).
    assert acc.loc["k-SLMmeans (Qwen3-4B)", "mean_diff"] == pytest.approx(0.1)
    assert acc.loc["k-means", "mean_diff"] == pytest.approx(0.05)
    assert acc.loc["k-SLMmeans (Qwen3-1.7B)", "mean_diff"] == pytest.approx(-0.1)
    assert (paired.n == 2).all()
    assert frames["counts"].n_guarded.tolist() == [3, 0, 3, 0]
    assert frames["counts"].n_flagged_base.tolist() == [3, 0, 3, 0]
    assert len(frames["steps"]) == 6
    assert any("max abs diff" in line for line in checks)


def test_build_raises_on_a_missing_seed(runs):
    with pytest.raises(ValueError, match="seed sets differ"):
        report.build(*runs(guard_seeds=(0,)), cache_dir="unused")


def test_build_raises_when_step_one_differs_from_stage_one(runs, tmp_path):
    with pytest.raises(ValueError, match="differs from stage 1 variant b"):
        report.build(*runs(), cache_dir="unused", stage1_csv=_stage1(tmp_path / "s1.csv", fit=A0))


def test_build_raises_when_logged_calls_differ_from_the_partitions(runs):
    # A guarded step-1 partition with other member sets implies other call counts than the logged [3, 2].
    with pytest.raises(ValueError, match="differ from those implied"):
        report.build(*runs(step1=[0, 1, 1, 2, 2, 2]), cache_dir="unused")


def test_step_table_raises_on_unequal_run_lengths():
    trace = [_trace(0), _trace(1, 3, 0.2)]
    short = _record("g", [A0], trace[:1])
    long = _record("b", [A0, A0], trace)
    with pytest.raises(ValueError, match="numbers of fits"):
        report.step_table(short, long, long, GOLD)
