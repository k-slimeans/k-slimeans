"""Fast checks for `kslimeans.aggregate` on hand-written synthetic run directories."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

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
