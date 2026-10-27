"""Fast checks for `kslimeans.aggregate` on hand-written synthetic run directories."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from kslimeans import aggregate
from kslimeans.data import Dataset


def _write_run(
    path,
    summarizer,
    method_name,
    method_rows,
    baseline_rows=(),
    summaries=None,
    dataset="synth",
    embedder="emb",
):
    """method_rows / baseline_rows: [(seed, acc, nmi, ari, runtime_s)]; a None runtime_s is omitted."""
    path.mkdir(parents=True)
    cfg = {"name": "r", "dataset": dataset, "embedder": embedder, "method_name": method_name, "summarizer": summarizer}
    (path / "config.yaml").write_text(yaml.safe_dump(cfg))
    lines = []
    for method, rows in (("kmeans", baseline_rows), (method_name, method_rows)):
        for seed, acc, nmi, ari, rt in rows:
            line = {"seed": seed, "method": method, "acc": acc, "nmi": nmi, "ari": ari}
            if rt is not None:
                line["runtime_s"] = rt
            lines.append(line)
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    if summaries is not None:
        (path / "summaries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in summaries))
    return path


HF = {"type": "hf", "model_id": "Qwen/Qwen3-0.6B"}
LSA = {"type": "lsa", "top_k": 5}
KM = [(0, 0.5, 0.6, 0.3, 1.0), (1, 0.7, 0.8, 0.5, 3.0)]


def _own(*scores):
    """Own-method rows over seeds 0 and 1 (same seeds as KM), one (acc, nmi, ari) per seed."""
    return [(seed, *score, 2.0) for seed, score in enumerate(scores)]


OWN = _own((0.1, 0.2, 0.3), (0.2, 0.3, 0.4))


def test_labels_and_baseline_dedupe_and_order(tmp_path):
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", OWN, KM)
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", _own((0.4, 0.5, 0.6), (0.5, 0.6, 0.7)), KM)
    df = aggregate.aggregate_runs([a, b])
    assert list(df["method"]) == ["k-means", "k-NLPmeans (LSA)", "k-SLMmeans (Qwen3-0.6B)"]
    assert list(df["n_seeds"]) == [2, 2, 2]


@pytest.mark.filterwarnings("ignore:.*seeds")
def test_mean_std_and_formatting(tmp_path):
    a = _write_run(tmp_path / "a", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], KM)
    df = aggregate.aggregate_runs([a])
    km = df.iloc[0]
    assert km["acc_mean"] == pytest.approx(0.6)
    assert km["acc_std"] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert km["runtime_s"] == pytest.approx(2.0)
    assert np.isnan(df.iloc[1]["nmi_std"])  # a single seed has no sample std

    table = aggregate.format_table(df)
    assert list(table.columns) == ["Method", "ACC", "NMI", "ARI", "s/seed", "seeds"]
    assert table.loc[0, "ACC"] == "0.600 ± 0.141"
    assert table.loc[0, "s/seed"] == "2.0"
    assert table.loc[1, "NMI"] == "0.200 ± n/a"
    assert table.loc[0, "seeds"] == "2"


@pytest.mark.filterwarnings("ignore:.*seeds")
def test_to_markdown_shape(tmp_path):
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", [(0, 0.1, 0.2, 0.3, 2.0)], KM)
    lines = aggregate.to_markdown(aggregate.format_table(aggregate.aggregate_runs([a]))).splitlines()
    assert lines[0] == "| Method | ACC | NMI | ARI | s/seed | seeds |"
    assert lines[1] == "| --- | ---: | ---: | ---: | ---: | ---: |"
    assert len(lines) == 4
    assert lines[3].startswith("| k-NLPmeans (LSA) | 0.100 ± n/a |")


def test_baseline_mismatch_warns_and_keeps_first(tmp_path):
    other = [(0, 0.5, 0.6, 0.3, 1.0), (1, 0.7, 0.8, 0.51, 3.0)]
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", OWN, KM)
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", _own((0.4, 0.5, 0.6), (0.5, 0.6, 0.7)), other)
    with pytest.warns(UserWarning, match="k-means"):
        df = aggregate.aggregate_runs([a, b])
    assert df.iloc[0]["ari_mean"] == pytest.approx(0.4)  # first occurrence kept


def test_identical_baselines_do_not_warn(tmp_path, recwarn):
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", OWN, KM)
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", _own((0.4, 0.5, 0.6), (0.5, 0.6, 0.7)), KM)
    aggregate.aggregate_runs([a, b])
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_same_own_label_in_two_runs_raises(tmp_path):
    a = _write_run(tmp_path / "a", HF, "k-SLMmeans", OWN, KM)
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", _own((0.4, 0.5, 0.6), (0.5, 0.6, 0.7)), KM)
    with pytest.raises(ValueError, match=r"(?s)/a.*/b.*k-SLMmeans \(Qwen3-0.6B\)"):
        aggregate.aggregate_runs([a, b])


def test_baseline_seed_sets_differing_or_disjoint_warn(tmp_path):
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", OWN, KM)
    subset = _write_run(tmp_path / "b", HF, "k-SLMmeans", OWN, KM[:1])
    with pytest.warns(UserWarning, match="k-means.*seeds"):
        aggregate.aggregate_runs([a, subset])
    disjoint = _write_run(tmp_path / "c", HF, "k-SLMmeans", OWN, [(5, 0.5, 0.6, 0.3, 1.0), (6, 0.7, 0.8, 0.5, 3.0)])
    with pytest.warns(UserWarning, match="k-means.*seeds"):
        aggregate.aggregate_runs([a, disjoint])


@pytest.mark.parametrize("key, kwargs", [("dataset", {"dataset": "other"}), ("embedder", {"embedder": "e2"})])
def test_differing_dataset_or_embedder_warns(tmp_path, key, kwargs):
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", OWN, KM)
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", OWN, KM, **kwargs)
    with pytest.warns(UserWarning, match=rf"{key}.*{tmp_path / 'b'}|{tmp_path / 'b'}.*{key}"):
        aggregate.aggregate_runs([a, b])


def test_partial_own_seed_set_warns(tmp_path):
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", OWN, KM)
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", OWN[:1], KM)
    with pytest.warns(UserWarning, match="k-SLMmeans.*partial or crashed"):
        aggregate.aggregate_runs([a, b])


@pytest.mark.parametrize("runtime", [None, float("nan")])
def test_missing_runtime_renders_na(tmp_path, runtime):
    rows = [(0, 0.1, 0.2, 0.3, runtime), (1, 0.2, 0.3, 0.4, runtime)]
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", rows)
    df = aggregate.aggregate_runs([a])
    assert np.isnan(df.loc[0, "runtime_s"])
    assert df.loc[0, "n_seeds"] == 2
    table = aggregate.format_table(df)
    assert table.loc[0, "s/seed"] == "n/a"
    assert table.loc[0, "seeds"] == "2"


def test_resolve_gold():
    names = ["alpha", "7", "beta"]
    assert aggregate._resolve_gold(2, names, 3) == 2
    assert aggregate._resolve_gold("beta", names, 3) == 2
    assert aggregate._resolve_gold("7", names, 3) == 1  # a name wins over a numeric parse
    assert aggregate._resolve_gold("2", names, 3) == 2
    assert aggregate._resolve_gold("-0", None, 3) == 0
    for bad in ("--5", "gamma"):
        with pytest.raises(ValueError, match="Unknown gold class name"):
            aggregate._resolve_gold(bad, names, 3)
    with pytest.raises(ValueError, match="no label names"):
        aggregate._resolve_gold("--5", None, 3)
    for bad in (3, -1, "3", "-1"):
        with pytest.raises(ValueError, match="out of range"):
            aggregate._resolve_gold(bad, names, 3)


# ----------------------------------------------------------------------------
# find_examples
# ----------------------------------------------------------------------------

LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def _fake_dataset(names):
    return Dataset("synth", [f"d{i}" for i in range(8)], LABELS, 2, "", "", names)


def _example_run(tmp_path, with_assignments=True):
    # Predicted cluster 1 holds gold class 0 docs 0-2 (+ doc 4); cluster 0 holds the rest.
    row = {"seed": 0, "method": "k-SLMmeans", "summaries": ["zero\nsummary", "one summary"]}
    if with_assignments:
        row["assignments"] = [1, 1, 1, 0, 1, 0, 0, 0]
    return _write_run(tmp_path / "e", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=[row])


def test_examples_overlap_mapping_by_id_and_name(tmp_path, monkeypatch):
    run = _example_run(tmp_path)
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(["alpha", "beta"]))
    by_name = aggregate.find_examples([run], ["alpha"], seed=0)
    assert by_name == aggregate.find_examples([run], [0], seed=0)
    assert "### alpha (class 0)" in by_name
    assert "- **k-SLMmeans (Qwen3-0.6B)**: one summary (3/4 gold docs, cluster size 4)" in by_name
    beta = aggregate.find_examples([run], ["1"], seed=0)  # numeric strings are class ids
    assert "zero summary (3/4 gold docs, cluster size 4)" in beta  # newline collapsed
    assert aggregate.find_examples([run], ["alpha", 0], seed=0) == by_name  # same class twice: one entry


def test_examples_errors(tmp_path, monkeypatch):
    run = _example_run(tmp_path)
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(None))
    with pytest.raises(ValueError, match="no label names"):
        aggregate.find_examples([run], ["alpha"])
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(["alpha", "beta"]))
    with pytest.raises(ValueError, match="Unknown gold class name"):
        aggregate.find_examples([run], ["gamma"])
    with pytest.raises(ValueError, match="no summaries"):
        aggregate.find_examples([run], [0], seed=3)


def test_examples_missing_assignments_errors(tmp_path, monkeypatch):
    run = _example_run(tmp_path / "old", with_assignments=False)
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(["alpha", "beta"]))
    with pytest.raises(ValueError, match="assignments"):
        aggregate.find_examples([run], [0])


def test_examples_overlap_tie_picks_lowest_cluster(tmp_path, monkeypatch):
    # Gold class 0 (docs 0-3) splits 2/2 over clusters 0 and 1.
    row = {
        "seed": 0,
        "method": "k-SLMmeans",
        "summaries": ["low", "high"],
        "assignments": [0, 0, 1, 1, 0, 0, 1, 1],
    }
    run = _write_run(tmp_path / "t", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=[row])
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(None))
    assert "low (2/4 gold docs, cluster size 4)" in aggregate.find_examples([run], [0])


def test_examples_two_runs_share_gold_header(tmp_path, monkeypatch):
    def row(method, summary):
        return {"seed": 0, "method": method, "summaries": [summary, "other"], "assignments": [0] * 4 + [1] * 4}

    a = _write_run(tmp_path / "a", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=[row("k-SLMmeans", "from slm")])
    b = _write_run(tmp_path / "b", LSA, "k-NLPmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=[row("k-NLPmeans", "from lsa")])
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(["alpha", "beta"]))
    out = aggregate.find_examples([a, b], ["alpha"])
    assert out.count("### alpha (class 0)") == 1
    assert out.index("**k-SLMmeans (Qwen3-0.6B)**: from slm") < out.index("**k-NLPmeans (LSA)**: from lsa")


def test_examples_select_requested_seed_among_several(tmp_path, monkeypatch):
    rows = [
        {"seed": seed, "method": method, "summaries": [f"{method}-{seed}", "x"], "assignments": [0] * 4 + [1] * 4}
        for method in ("kmeans", "k-SLMmeans")
        for seed in (0, 1, 2)
    ]
    run = _write_run(tmp_path / "s", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=rows)
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(None))
    assert "k-SLMmeans-2 (4/4" in aggregate.find_examples([run], [0], seed=2)


# ----------------------------------------------------------------------------
# per_seed_table / rank_examples
# ----------------------------------------------------------------------------


def test_per_seed_table_rows_labels_and_dedupe(tmp_path):
    a = _write_run(tmp_path / "a", LSA, "k-NLPmeans", OWN, KM)
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", _own((0.4, 0.5, 0.6), (0.5, 0.6, 0.7)), KM)
    df = aggregate.per_seed_table([a, b])
    assert list(df.columns) == ["method", "seed", "acc", "nmi", "ari", "runtime_s"]
    assert list(df["method"]) == ["k-means"] * 2 + ["k-NLPmeans (LSA)"] * 2 + ["k-SLMmeans (Qwen3-0.6B)"] * 2
    assert list(df["seed"]) == [0, 1] * 3
    slm = df[df["method"] == "k-SLMmeans (Qwen3-0.6B)"].set_index("seed")
    assert slm.loc[1, "nmi"] == 0.6
    assert list(df[df["method"] == "k-means"]["runtime_s"]) == [1.0, 3.0]


def test_per_seed_table_missing_runtime_is_nan(tmp_path):
    run = _write_run(tmp_path / "a", LSA, "k-NLPmeans", [(0, 0.1, 0.2, 0.3, None)])
    assert aggregate.per_seed_table([run])["runtime_s"].isna().all()


def test_rank_examples_scores_min_over_runs_and_breaks_ties_by_id(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "_resolve_dataset", lambda spec: _fake_dataset(["alpha", "beta"]))
    perfect = {"seed": 0, "method": "k-SLMmeans", "summaries": ["a", "b"], "assignments": [0, 0, 0, 0, 1, 1, 1, 1]}
    # Class 0 -> 3/4 in cluster 1; class 1 -> 3/4 in cluster 0.
    noisy = {**perfect, "assignments": [1, 1, 1, 0, 1, 0, 0, 0]}
    a = _write_run(tmp_path / "a", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=[perfect])
    b = _write_run(tmp_path / "b", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=[noisy])
    assert aggregate.rank_examples([a], seed=0) == [("alpha", 1.0), ("beta", 1.0)]
    assert aggregate.rank_examples([a, b], seed=0) == [("alpha", 0.75), ("beta", 0.75)]
    only_beta_noisy = {**perfect, "assignments": [0, 0, 0, 0, 1, 0, 0, 0]}
    c = _write_run(tmp_path / "c", HF, "k-SLMmeans", [(0, 0.1, 0.2, 0.3, 2.0)], summaries=[only_beta_noisy])
    assert aggregate.rank_examples([a, c], seed=0) == [("alpha", 1.0), ("beta", 0.75)]


# ----------------------------------------------------------------------------
# paired_differences / cost_table
# ----------------------------------------------------------------------------


def _frame(rows):
    """rows: [(method, seed, acc, nmi, ari)] -> per-seed frame."""
    return pd.DataFrame(rows, columns=["method", "seed", "acc", "nmi", "ari"])


def test_paired_differences_stats_wins_ties_and_order():
    ref = [0.50, 0.60, 0.70, 0.80]
    other = [0.55, 0.58, 0.70, 0.83]
    rows = [("ref", s, a, a, a) for s, a in enumerate(ref)] + [("m", s, a, a, a) for s, a in enumerate(other)]
    rows += [("z", s, a, a, a) for s, a in enumerate(ref)]  # identical to the reference
    paired = aggregate.paired_differences(_frame(rows), ["ref"])
    assert list(paired["method"]) == ["m"] * 3 + ["z"] * 3
    assert list(paired["metric"]) == ["acc", "nmi", "ari"] * 2
    row = paired.iloc[0]
    diff = np.array(other) - np.array(ref)
    sd = diff.std(ddof=1)
    half = stats.t.ppf(0.975, 3) * sd / 2
    assert (row["n"], row["wins"], row["ties"]) == (4, 2, 1)
    assert row["mean_diff"] == pytest.approx(diff.mean())
    assert row["sd"] == pytest.approx(sd)
    assert row["ci_low"] == pytest.approx(diff.mean() - half)
    assert row["ci_high"] == pytest.approx(diff.mean() + half)
    zero = paired.iloc[3]
    assert (zero["mean_diff"], zero["sd"], zero["wins"], zero["ties"]) == (0, 0, 0, 4)


def test_paired_differences_uses_common_seeds_only_and_nan_ci_for_one_seed():
    rows = [("ref", 0, 0.5, 0.5, 0.5), ("ref", 1, 0.6, 0.6, 0.6), ("m", 1, 0.7, 0.7, 0.7), ("m", 2, 0.9, 0.9, 0.9)]
    paired = aggregate.paired_differences(_frame(rows), ["ref"], metrics=("acc",))
    row = paired.iloc[0]
    assert (row["n"], row["wins"]) == (1, 1)
    assert row["mean_diff"] == pytest.approx(0.1)
    assert np.isnan(row["sd"]) and np.isnan(row["ci_low"]) and np.isnan(row["ci_high"])


def test_paired_differences_multiple_references_and_unknown_reference():
    rows = [(m, s, v, v, v) for m, v in (("a", 0.1), ("b", 0.2), ("c", 0.3)) for s in (0, 1)]
    paired = aggregate.paired_differences(_frame(rows), ["b", "a"], metrics=("acc",))
    assert list(zip(paired["reference"], paired["method"])) == [("b", "a"), ("b", "c"), ("a", "b"), ("a", "c")]
    with pytest.raises(ValueError, match="Unknown reference"):
        aggregate.paired_differences(_frame(rows), ["nope"])


def test_format_paired_markdown():
    rows = [("ref", 0, 0.50, 0.5, 0.5), ("ref", 1, 0.60, 0.5, 0.5), ("m", 0, 0.505, 0.5, 0.5), ("m", 1, 0.591, 0.5, 0.5)]
    one = aggregate.paired_differences(_frame(rows[:1] + rows[2:3]), ["ref"], metrics=("acc",))
    assert "| m | ACC | +0.005 | n/a | 1/1 |" in aggregate.format_paired_markdown(one)
    two = aggregate.paired_differences(_frame(rows), ["ref"], metrics=("acc",))
    lines = aggregate.format_paired_markdown(two).splitlines()
    assert lines[0] == "Paired difference vs ref (method minus reference):"
    assert lines[2] == "| Method | Metric | mean diff | 95% CI | wins/n |"
    assert lines[4].startswith("| m | ACC | -0.002 | [-0.0") and lines[4].endswith("| 1/2 |")
    both = aggregate.paired_differences(_frame(rows), ["ref", "m"], metrics=("acc",))
    assert aggregate.format_paired_markdown(both).count("Paired difference vs") == 2


COST = {
    "n_calls": 10,
    "input_tokens": 1000,
    "output_tokens": 100,
    "summarizer_seconds": 4.0,
    "n_empty_summaries": 0,
    "n_truncated_clusters": 1,
    "n_dropped_texts": 2,
    "peak_gpu_mem_gb": 2.5,
    "peak_gpu_reserved_gb": None,
}


def _cost_run(path, step0=("a", "a"), env=True, **kwargs):
    """HF run with two seeds whose summary rows carry cost fields (calls 10 / 20) and step0 hashes."""
    run = _write_run(path, HF, "k-SLMmeans", OWN, KM, **kwargs)
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    for line in lines:
        line["step0_sha1"] = "a" if line["method"] == "kmeans" else step0[line["seed"]]
        if line["method"] == "k-SLMmeans":
            line.update(COST, n_calls=COST["n_calls"] * (1 + line["seed"]))
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    if env:
        info = {"gpu_name": "RTX", "summarizer": {"model_id": "m", "dtype": "torch.bfloat16", "batch_size": 8}}
        (run / "environment.json").write_text(json.dumps(info))
    return run


def test_cost_table_with_environment_and_nan_baseline(tmp_path):
    costs = aggregate.cost_table([_cost_run(tmp_path / "a")]).set_index("method")
    assert list(costs.index) == ["k-means", "k-SLMmeans (Qwen3-0.6B)"]
    slm = costs.loc["k-SLMmeans (Qwen3-0.6B)"]
    assert slm["n_calls_mean"] == pytest.approx(15)
    assert slm["n_calls_std"] == pytest.approx(np.std([10, 20], ddof=1))
    assert slm["input_tokens_mean"] == 1000 and slm["input_tokens_std"] == 0
    assert slm["runtime_s_mean"] == pytest.approx(2.0)
    assert np.isnan(slm["peak_gpu_reserved_gb_mean"])  # logged as null (no CUDA)
    assert (slm["batch_size"], slm["dtype"], slm["gpu"]) == (8, "torch.bfloat16", "RTX")
    km = costs.loc["k-means"]
    assert np.isnan(km["n_calls_mean"]) and np.isnan(km["input_tokens_std"])
    assert km["runtime_s_mean"] == pytest.approx(2.0)
    assert pd.isna(km["batch_size"]) and pd.isna(km["gpu"])

    table = aggregate.format_cost_table(costs.reset_index()).set_index("Method")
    assert table.loc["k-SLMmeans (Qwen3-0.6B)", "calls"] == "15 ± 7"
    assert table.loc["k-SLMmeans (Qwen3-0.6B)", "reserved GPU GB"] == "n/a"
    assert table.loc["k-SLMmeans (Qwen3-0.6B)", "batch size"] == "8"
    assert table.loc["k-means", "calls"] == "n/a" and table.loc["k-means", "GPU"] == "n/a"


def test_cost_table_without_environment_or_cost_fields(tmp_path):
    old = _write_run(tmp_path / "old", LSA, "k-NLPmeans", OWN, KM)
    costs = aggregate.cost_table([old])
    assert costs["n_calls_mean"].isna().all()
    assert costs["batch_size"].isna().all() and costs["gpu"].isna().all()
    assert (costs["n_seeds"] == 2).all()
    table = aggregate.format_cost_table(costs)
    assert (table.drop(columns=["Method", "total s"]) == "n/a").all().all()
    no_env = aggregate.cost_table([_cost_run(tmp_path / "b", env=False)]).set_index("method")
    assert no_env.loc["k-SLMmeans (Qwen3-0.6B)", "n_calls_mean"] == pytest.approx(15)
    assert pd.isna(no_env.loc["k-SLMmeans (Qwen3-0.6B)", "gpu"])


def test_step0_sha1_summary_shared_and_differing(tmp_path):
    shared = _cost_run(tmp_path / "a")
    assert aggregate.step0_sha1_summary([shared]) == "step0_sha1 shared across all methods for seeds 0-1: yes"
    differing = _cost_run(tmp_path / "b", step0=("a", "b"))
    assert (
        aggregate.step0_sha1_summary([differing])
        == "step0_sha1 shared across all methods for seeds 0-1: no (differs for seeds 1)"
    )
    old = _write_run(tmp_path / "c", LSA, "k-NLPmeans", OWN, KM)
    assert aggregate.step0_sha1_summary([old]) == "step0_sha1 not logged in these runs."
