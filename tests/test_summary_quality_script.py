"""The table builder of `scripts/summary_quality.py` on a handful of hand-written rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summary_quality.py"


def load_script():
    spec = importlib.util.spec_from_file_location("summary_quality_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FLAGS = ["flag_not_fluent", "flu_no_question", "flu_echo", "flu_copy", "flu_markup", "flu_script", "flu_empty", "flag_low_coverage",
         "flag_generic", "flag_duplicate", "flag_near_duplicate", "flag_mentions", "flag_low_margin",
         "flag_unsupported"]  # fmt: skip


def frames():
    # One method, two seeds, steps 0 and 1; plus an old-format method with a single approximate step.
    rows = []
    for method, steps, approx in (("m1", (0, 1), False), ("old", (4,), True)):
        for step in steps:
            for cluster, cov in enumerate((0.2, 0.8)):
                rows.append({
                    "dataset": "toy", "method": method, "run": "r", "seed": 0, "step": step, "source_step": step,
                    "approx_partition": approx, "cluster": cluster, "n_members": 10, "coverage_nearest": cov,
                    "coverage_intents": 1.0, "specificity_margin": 0.1, "max_summary_cos": 0.3,
                    "unsupported_lemma_share": 0.0, "n_kept": 8, "n_lost": 2, "n_gained": 1,
                    "n_members_uncovered": 4, "n_lost_uncovered": 1, **{f: cluster == 0 for f in FLAGS},
                })  # fmt: skip
    clusters = pd.DataFrame(rows)
    steps = clusters.drop_duplicates(["method", "step"])[["dataset", "method", "run", "seed", "step", "source_step",
                                                          "approx_partition"]].assign(
        genericity=0.2, genericity_centroids=0.3, label_align_sim=0.4, label_align_acc=0.5)  # fmt: skip
    return clusters, steps


def test_build_table_scopes_and_link_columns():
    script = load_script()
    table = script.build_table(*frames()).set_index(["method", "scope"])
    assert set(table.index) == {("m1", "all steps"), ("m1", "last step"), ("old", "final (approx.)")}
    assert table.loc[("m1", "all steps"), "n"] == 4 and table.loc[("m1", "last step"), "n"] == 2
    assert table.loc[("m1", "all steps"), "coverage_nearest_mean"] == pytest.approx(0.5)
    assert table.loc[("m1", "all steps"), "flag_generic_share"] == 0.5
    assert table.loc[("m1", "last step"), "kept"] == pytest.approx(0.8)
    # Per cluster: 10 members, 4 uncovered of which 1 lost, so 1 of 6 covered members lost: 2 lost - 1.
    assert table.loc[("m1", "last step"), "lost_if_uncovered"] == pytest.approx(1 / 4)
    assert table.loc[("m1", "last step"), "lost_if_covered"] == pytest.approx(1 / 6)
    assert table.loc[("m1", "last step"), "member_uncovered"] == pytest.approx(0.4)
    # Cluster 0 of every group is a flagged duplicate; the "distinct" columns keep cluster 1 only.
    assert table.loc[("m1", "all steps"), "coverage_nearest_mean_distinct"] == pytest.approx(0.8)
    assert table.loc[("m1", "all steps"), "flag_low_coverage_share_distinct"] == 0.0
    assert table.loc[("m1", "all steps"), "genericity"] == pytest.approx(0.2)


def test_final_step_without_a_partition_after_is_left_out_of_the_member_columns():
    script = load_script()
    clusters, steps = frames()
    last = (clusters["method"] == "m1") & (clusters["step"] == 1)
    clusters.loc[last, ["n_kept", "n_lost", "n_gained", "n_members_uncovered", "n_lost_uncovered"]] = float("nan")
    table = script.build_table(clusters, steps).set_index(["method", "scope"])
    assert table.loc[("m1", "all steps"), "kept"] == pytest.approx(0.8)  # step 0 only
    assert pd.isna(table.loc[("m1", "last step"), "kept"])


def test_markdown_names_the_approximation_and_every_family():
    script = load_script()
    md = script.format_markdown(script.build_table(*frames()))
    assert "### toy" in md and "FINAL assignments" in md and "old (final (approx.))" in md
    for heading in ("**Scores**", "**Flags**", "**Fluency checks**", "**Members at the step**"):
        assert heading in md


def test_method_label_names_the_model():
    from kslimeans.summary_quality import method_label

    assert method_label({"summarizer": {"type": "lsa"}}, "k-NLPmeans") == "k-NLPmeans"
    assert method_label({"summarizer": {"type": "hf", "model_id": "Qwen/Qwen3-4B"}}, "k-SLMmeans") == (
        "k-SLMmeans (Qwen3-4B)"
    )
