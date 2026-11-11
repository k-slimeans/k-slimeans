"""`kslimeans.synthesis` on tiny hand-made frames with known answers (no models, no files, no clocks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from kslimeans import synthesis as syn


def _quality(run="cfg_a/20260101-000000", n_clusters=3, steps=(0, 1), seeds=(0,)):
    rows = [
        {"dataset": "d", "method": "m", "run": run, "seed": s, "step": step, "cluster": c, "n_members": 10, "purity": 0.5}
        for s in seeds for step in steps for c in range(n_clusters)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def _diag(run="cfg_a", n_clusters=3, steps=(1, 2), seeds=(0,)):
    rows = [
        {"run": run, "method": "raw", "seed": s, "step": step, "cluster": c, "size_pre": 10, "drift": 1.0}
        for s in seeds for step in steps for c in range(n_clusters)
    ]  # fmt: skip
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the join


def test_one_to_one_join_reports_counts():
    left = pd.DataFrame({"k": [1, 2, 3], "a": [1, 2, 3]})
    right = pd.DataFrame({"k": [2, 3, 4], "b": [20, 30, 40]})
    merged, rep = syn.one_to_one_join(left, right, ["k"], "t", left_only_ok=True, right_only_ok=True)
    assert (rep["n_left"], rep["n_right"], rep["matched"], rep["left_only"], rep["right_only"]) == (3, 3, 2, 1, 1)
    assert len(merged) == 3 and merged["b"].isna().sum() == 1


def test_one_to_one_join_refuses_unmatched_and_duplicates():
    left = pd.DataFrame({"k": [1, 2], "a": [1, 2]})
    with pytest.raises(syn.JoinError, match="left rows have no partner"):
        syn.one_to_one_join(left, pd.DataFrame({"k": [1], "b": [1]}), ["k"], "t")
    with pytest.raises(syn.JoinError, match="right rows have no partner"):
        syn.one_to_one_join(left, pd.DataFrame({"k": [1, 2, 3], "b": [1, 2, 3]}), ["k"], "t")
    with pytest.raises(syn.JoinError, match="share a key"):
        syn.one_to_one_join(left, pd.DataFrame({"k": [1, 1, 2], "b": [1, 2, 3]}), ["k"], "t")


def test_step_indices_align_and_match_counts():
    quality = syn.prepare_quality(_quality(steps=(0, 1)))
    assert list(quality["quality_step"].unique()) == [0, 1] and list(quality["step"].unique()) == [1, 2]
    joined, reports = syn.join_quality_diagnostic(quality, _diag(steps=(1, 2)))
    assert len(joined) == 6 and joined["has_diag"].all() and joined["drift"].notna().all()
    assert reports[0]["matched"] == 6 and reports[0]["left_only"] == 0
    assert "method" in joined and "dup" not in joined  # the #34 label is kept, #33's spelling dropped


def test_join_fails_when_indexing_is_off_by_one():
    quality = syn.prepare_quality(_quality(steps=(0, 1)))
    with pytest.raises(syn.JoinError):
        syn.join_quality_diagnostic(quality, _diag(steps=(0, 1)))  # #33 steps start at 1


def test_runs_without_diagnostic_are_kept_and_flagged_and_unscored_diag_runs_raise():
    quality = syn.prepare_quality(pd.concat([_quality(), _quality(run="old/2026")], ignore_index=True))
    joined, reports = syn.join_quality_diagnostic(quality, _diag())
    assert joined.groupby("run_config")["has_diag"].all().to_dict() == {"cfg_a": True, "old": False}
    assert joined.loc[~joined["has_diag"], "drift"].isna().all()
    assert [r["left_only"] for r in reports] == [0, 6]
    with pytest.raises(syn.JoinError, match="no #34 scores"):
        syn.join_quality_diagnostic(quality, pd.concat([_diag(), _diag(run="other")]))


def test_attach_run_level_requires_every_row_to_match():
    frame = pd.DataFrame({"run": ["a", "a", "b"], "seed": [0, 0, 0]})
    run_level = pd.DataFrame({"run": ["a"], "seed": [0], "final_acc": [0.5]})
    with pytest.raises(syn.JoinError):
        syn.attach_run_level(frame, run_level, ["run", "seed"], "t")
    out, rep = syn.attach_run_level(frame, pd.concat([run_level, run_level.assign(run="b")]), ["run", "seed"], "t")
    assert rep["matched"] == 3 and list(out["final_acc"]) == [0.5] * 3


# ------------------------------------------------------------------ cost and outcome columns


def test_unsent_clusters_cost_zero_and_are_flagged():
    cost = syn.cost_rows(4, sent=[0, 2], input_tokens={0: 100, 2: 50}, output_tokens={0: 7, 2: 9})
    assert list(cost["sent"]) == [True, False, True, False]
    assert list(cost["input_tokens"]) == [100, 0, 50, 0] and list(cost["output_tokens"]) == [7, 0, 9, 0]
    # a stray count for an unsent cluster (a memo hit) is not charged
    cost = syn.cost_rows(2, sent=[0], input_tokens={0: 5, 1: 99}, output_tokens={1: 3})
    assert list(cost["input_tokens"]) == [5, 0] and list(cost["output_tokens"]) == [0, 0]


def test_after_overlap_follows_members_and_the_majority_intent():
    gold = np.array([0, 0, 0, 1, 1, 1])
    source = np.array([0, 0, 0, 1, 1, 1])
    after = np.array([0, 0, 1, 1, 1, 1])  # one gold-0 document moved into cluster 1
    out = syn.after_overlap(source, after, gold, 2).set_index("cluster")
    assert out.loc[0, "majority_before"] == 0 and out.loc[0, "majority_after"] == 0
    assert out.loc[1, "agree_majority"] == 1.0
    assert out.loc[0, "member_agree_after"] == pytest.approx(2 / 3)  # the moved doc now sits in a gold-1 cluster
    assert out.loc[1, "member_agree_after"] == 1.0
    assert out.loc[0, "jaccard_major_after"] == pytest.approx(2 / 3)  # {0,1} vs intent {0,1,2}


def test_after_overlap_flags_a_cluster_that_lost_everyone():
    gold = np.array([0, 0, 1, 1, 1])
    out = syn.after_overlap(np.array([0, 0, 1, 1, 1]), np.array([1, 1, 1, 1, 1]), gold, 2).set_index("cluster")
    assert pd.isna(out.loc[0, "majority_after"]) and np.isnan(out.loc[0, "agree_majority"])
    assert out.loc[0, "member_agree_after"] == 0.0  # its members now sit in a cluster of the other intent


def test_add_derived_outcomes():
    df = _quality(steps=(1,), n_clusters=2).assign(
        run_config="cfg",
        has_diag=True, purity_after=[0.7, 0.3], n_lost=[2, 10], n_gained=[1, 0], n_after=[9, 0],
        member_agree_after=[0.65, 0.2], input_tokens=[10, 0], output_tokens=[2, 0], jaccard_majority=0.5,
        jaccard_major_after=[0.6, 0.0], flu_copy=[True, False], flu_echo=[True, True], label_top1=[True, False],
    )  # fmt: skip
    out = syn.add_derived(df)
    assert list(out["d_purity"].round(2)) == [0.2, -0.2]
    assert list(out["lost_share"]) == [0.2, 1.0] and list(out["emptied"]) == [0.0, 1.0]
    assert list(out["d_member_purity"].round(2)) == [0.15, -0.3]
    assert list(out["helped"]) == [1.0, 0.0] and list(out["harmed"]) == [0.0, 1.0]
    assert list(out["n_flu_flags"]) == [2, 1] and list(out["cost_tokens"]) == [12, 0]
    assert out["step1"].all() and list(out["method_id"]) == ["m", "m"]


# ------------------------------------------------------------------ seed-unit statistics


def test_seed_interval_is_a_t_interval_over_seeds():
    iv = syn.seed_interval([1.0, 2.0, 3.0])
    half = stats.t.ppf(0.975, 2) / np.sqrt(3)
    assert iv["mean"] == 2.0 and iv["lo"] == pytest.approx(2 - half) and iv["hi"] == pytest.approx(2 + half)
    assert iv["n_seeds"] == 3
    assert syn.seed_interval([1.0, np.nan, 3.0])["n_seeds"] == 2  # NaNs are not seeds
    one = syn.seed_interval([4.0])
    assert one["mean"] == 4.0 and np.isnan(one["lo"])
    assert syn.seed_interval([2.0, 2.0, 2.0])["hi"] == 2.0  # no variation, no width
    assert syn.seed_interval([])["n_seeds"] == 0


def test_summarise_seeds_needs_min_seeds():
    assert np.isnan(syn.summarise_seeds([1.0, 2.0])["mean"]) and syn.summarise_seeds([1.0, 2.0])["n_seeds"] == 2
    assert syn.summarise_seeds([1.0, 2.0, 3.0])["mean"] == 2.0


def test_interval_width_depends_on_seeds_not_clusters():
    """Many clusters per seed must not narrow the interval: only the per-seed values count."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"seed": np.repeat([0, 1, 2], 4), "y": np.repeat([0.1, 0.2, 0.3], 4) + rng.normal(0, 0.01, 12)})
    many = pd.concat([base] * 50, ignore_index=True)
    key = lambda d: syn.stratified_table(d.assign(s="a", method_id="m"), "s", ["y"]).iloc[0]  # noqa: E731
    assert key(base)["y_hi"] - key(base)["y_lo"] == pytest.approx(key(many)["y_hi"] - key(many)["y_lo"])


def test_spearman_and_partial_spearman():
    x = np.arange(10.0)
    assert syn.spearman(x, x**3) == pytest.approx(1.0)
    assert syn.spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(syn.spearman(x, np.ones(10))) and np.isnan(syn.spearman([1, 2], [1, 2]))
    rng = np.random.default_rng(1)
    z = rng.normal(size=300)
    a, b = z + rng.normal(scale=0.3, size=300), z + rng.normal(scale=0.3, size=300)
    assert syn.spearman(a, b) > 0.8
    assert abs(syn.spearman(a, b, pd.DataFrame({"z": z}))) < 0.2  # the shared cause is partialled out
    assert syn.spearman([1, 2, np.nan, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)  # NaN rows dropped


def test_correlation_table_is_per_seed():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"seed": np.repeat(range(4), 30), "method_id": "m", "x": rng.normal(size=120)})
    df["y"] = df["x"] + rng.normal(scale=0.2, size=120)
    row = syn.correlation_table(df, ["x"], ["y"]).iloc[0]
    assert row["n_seeds"] == 4 and row["n_clusters"] == 120 and row["rho"] > 0.8 and row["lo"] < row["rho"] < row["hi"]


# ------------------------------------------------------------------ strata


def _strata_frame():
    rows = []
    for seed, (low, high) in enumerate([(0.8, 0.1), (0.6, 0.1), (0.7, 0.3)]):
        rows += [{"seed": seed, "method_id": "m", "cov": 0.1, "y": low}] * 4
        rows += [{"seed": seed, "method_id": "m", "cov": 0.9, "y": high}] * 4
    return pd.DataFrame(rows)


def test_stratified_table_means_are_means_of_seed_means():
    df = add = _strata_frame().assign(coverage_nearest=lambda d: d["cov"])
    df = syn.add_bins(add)
    table = syn.stratified_table(df, "bin_coverage_nearest", ["y"]).set_index("stratum")
    assert table.loc["(0,.25]", "y"] == pytest.approx(0.7) and table.loc["(.75,1]", "y"] == pytest.approx(0.5 / 3)
    assert table.loc["(0,.25]", "n_clusters"] == 12 and table.loc["(0,.25]", "n_seeds"] == 3
    assert table.loc["(0,.25]", "y_lo"] < 0.7 < table.loc["(0,.25]", "y_hi"]


def test_stratified_table_drops_seeds_with_too_few_clusters():
    df = _strata_frame().assign(coverage_nearest=lambda d: d["cov"])
    df = df[~((df["seed"] == 2) & (df["cov"] == 0.1)) | (df.groupby(["seed", "cov"]).cumcount() < 2)]  # seed 2: 2 rows
    table = syn.stratified_table(syn.add_bins(df), "bin_coverage_nearest", ["y"]).set_index("stratum")
    assert table.loc["(0,.25]", "n_seeds"] == 2 and np.isnan(table.loc["(0,.25]", "y"])  # below MIN_SEEDS


def test_bins_use_the_documented_edges():
    df = pd.DataFrame({"coverage_nearest": [0.0, 0.25, 0.26, 1.0], "specificity_margin": [-0.1, 0.0, 0.1, 0.5]})
    out = syn.add_bins(df)
    assert list(out["bin_coverage_nearest"]) == ["0", "(0,.25]", "(.25,.5]", "(.75,1]"]
    assert list(out["bin_specificity_margin"]) == ["<=0", "<=0", "(0,.1]", ">.2"]


def test_stratified_difference_adjusts_within_strata():
    """Confounded by construction: fluent clusters sit in the high stratum. Raw difference is large, adjusted is 0."""
    rows = []
    for seed in range(3):
        rows += [{"seed": seed, "stratum": "hi", "flag": 0.0, "y": 1.0}] * 8 + [{"seed": seed, "stratum": "hi", "flag": 1.0, "y": 1.0}] * 4
        rows += [{"seed": seed, "stratum": "lo", "flag": 1.0, "y": 0.0}] * 8 + [{"seed": seed, "stratum": "lo", "flag": 0.0, "y": 0.0}] * 4
    df = pd.DataFrame(rows)
    raw = syn.stratified_difference(df, "flag", "y")
    adj = syn.stratified_difference(df, "flag", "y", ["stratum"])
    assert raw["mean"] == pytest.approx(-0.333, abs=1e-3) and adj["mean"] == pytest.approx(0.0)
    assert adj["n_strata_used"] == 2 and adj["n_seeds"] == 3


def test_stratified_difference_skips_cells_without_both_groups():
    df = pd.DataFrame({"seed": [0] * 6, "flag": [1.0] * 6, "y": range(6)})
    out = syn.stratified_difference(df, "flag", "y")
    assert np.isnan(out["mean"]) and out["n_strata_used"] == 0.0
