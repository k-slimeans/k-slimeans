"""The headline re-run config and plan (`configs/rq2_rerun.yaml`, `rq2.plan_phase_r`) and the truncation check's helpers.

No models and no data: the config is compared with the #31 config it must follow, the plan is built on synthetic
adaptive traces, and the truncation helpers run on a fake prompt budget.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_rq2 import _row, _trace, _write_run  # noqa: E402

from kslimeans import rq2  # noqa: E402
from kslimeans.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rerun_cfg() -> dict:
    return load_config(ROOT / "configs" / "rq2_rerun.yaml")


# ---------------------------------------------------------------------------------------------- the config


def test_rerun_config_follows_the_31_protocol(rerun_cfg):
    protocol = load_config(ROOT / "configs" / "clinc150_summarizers.yaml")
    # Everything that defines the experiment is #31's, except the seeds (0-9 instead of 0-4) and the method label.
    for key in rq2.IDENTITY_KEYS:
        if key in ("seeds", "method_name"):
            continue
        assert rerun_cfg[key] == protocol[key], key
    assert rerun_cfg["seeds"] == list(range(10)) and protocol["seeds"] == list(range(5))
    assert rerun_cfg["summarizer"]["max_input_tokens"] == 4096  # spelled out, as in #31
    assert rerun_cfg["cache_dir"] == protocol["cache_dir"] and rerun_cfg["results_dir"] == protocol["results_dir"]


def test_rerun_config_differs_from_15_only_in_seeds_and_the_explicit_token_cap(rerun_cfg):
    sweep = load_config(ROOT / "configs" / "rq2_clinc150.yaml")
    assert rerun_cfg["seeds"] != sweep["seeds"]
    assert {**rerun_cfg["summarizer"]} == {**sweep["summarizer"], "max_input_tokens": 4096}  # 4096 is the default
    for key in ("dataset", "embedder", "baselines", "force_context_length", "tol", "schedule", "method_name"):
        assert rerun_cfg[key] == sweep[key], key
    assert rerun_cfg["name"] == "rq2r_fixed_n5" != sweep["name"]  # never the #15 directories


# ----------------------------------------------------------------------------------------------- the plan


def test_random_name_keeps_the_run_prefix():
    assert rq2.random_name("rq2_adaptive_d0.3_r0.4", 2) == "rq2_random_d0.3_r0.4_seed2"
    assert rq2.random_name("rq2_adaptive_d0.6_r0.7_ext", 5) == "rq2_random_d0.6_r0.7_seed5"
    assert rq2.random_name("rq2r_adaptive_d0.6_r0.7", 9) == "rq2r_random_d0.6_r0.7_seed9"


def test_phase_r_plan_before_the_adaptive_run_is_done(rerun_cfg, tmp_path):
    specs, rule = rq2.plan_phase("R", rerun_cfg, tmp_path)
    assert rule is None
    names = [s.name for s in specs]
    assert names[:4] == ["rq2r_fixed_n5", "rq2r_fixed_n1", "rq2r_fixed_n2", "rq2r_adaptive_d0.6_r0.7"]
    assert names[4:] == [f"rq2r_random_d0.6_r0.7_seed{s}" for s in range(10)]
    assert all(s.cfg is not None for s in specs[:4]) and all(s.cfg is None and s.blocked for s in specs[4:])
    assert all(s.phase == "R" for s in specs)
    fixed5, fixed1, fixed2, adaptive = (s.cfg for s in specs[:4])
    assert fixed5["baselines"] == {"kmeans": {}} and fixed1["baselines"] == fixed2["baselines"] == adaptive["baselines"] == {}
    assert [c["schedule"]["n_summaries"] for c in (fixed1, fixed2, fixed5)] == [1, 2, 5]
    assert [c["method_name"] for c in (fixed5, fixed1, fixed2, adaptive)] == ["fixed-n5", "fixed-n1", "fixed-n2", "adaptive-d0.6-r0.7"]
    for cfg in (fixed5, fixed1, fixed2, adaptive):
        assert cfg["seeds"] == list(range(10)) and cfg["summarizer"] == rerun_cfg["summarizer"]
    assert adaptive["schedule"] == {"type": "adaptive", "n_summaries": 5, "max_iter": 120, "initial_iter": True,
                                    "final_iter": True, "tau_drift": 0.6, "tau_churn": 0.7, "s_max": None}
    assert rerun_cfg["name"] == "rq2r_fixed_n5" and rerun_cfg["schedule"]["n_summaries"] == 5  # base not mutated


def test_phase_r_controls_follow_the_adaptive_runs_calls_per_seed(rerun_cfg, tmp_path):
    adaptive = rq2.plan_phase("R", rerun_cfg, tmp_path)[0][3]
    m = adaptive.cfg["method_name"]
    trace = [r for s in range(10) for r in _trace(m, s, [150, None, 3 + s, 2, 0])]
    rows = [_row(m, s, 0.5, 155 + s) for s in range(10)]
    _write_run(tmp_path, adaptive.cfg, trace=trace, rows=rows)
    specs, _ = rq2.plan_phase("R", rerun_cfg, tmp_path)
    controls = specs[4:]
    assert len(controls) == 10 and all(c.cfg is not None for c in controls)
    for s, c in enumerate(controls):
        assert c.name == f"rq2r_random_d0.6_r0.7_seed{s}" and c.cfg["seeds"] == [s]
        assert c.cfg["method_name"] == "random-d0.6-r0.7" and c.cfg["baselines"] == {}
        assert c.cfg["schedule"]["type"] == "random_subset" and c.cfg["schedule"]["seed"] == s
        assert c.cfg["schedule"]["counts"] == [150, 0, 3 + s, 2, 0]
        assert c.cfg["summarizer"] == rerun_cfg["summarizer"]


def test_sweep_phases_and_default_driver_phases_are_unchanged():
    assert rq2.PHASES == ("A", "B", "C", "D") and rq2.RERUN_PHASE == "R"
    with pytest.raises(ValueError, match="Unknown phase"):
        rq2.plan_phase("Z", {}, ".")


def test_runner_is_executable_and_scripts_are_not():
    assert (ROOT / "scripts" / "run_rq2_rerun.sh").stat().st_mode & 0o111
    script = ROOT / "scripts" / "rq2_truncation_check.py"
    assert not script.stat().st_mode & 0o111 and not script.read_text().startswith("#!")


# ------------------------------------------------------------------------------- truncation check helpers


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("rq2_truncation_check", ROOT / "scripts" / "rq2_truncation_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_iters_is_the_first_block(check):
    sched = {"max_iter": 120, "initial_iter": True}
    assert check.first_iters({"schedule": {**sched, "n_summaries": 5}}) == 20
    assert check.first_iters({"schedule": {**sched, "n_summaries": 1}}) == 60
    assert check.first_iters({"schedule": {**sched, "n_summaries": 2}}) == 40
    assert check.first_iters({"schedule": {**sched, "n_summaries": 5, "initial_iter": False}}) == 1


def test_changed_clusters_diffs_consecutive_steps(check):
    evolution = [["a", "b", "c"], ["a", "B", "c"], ["a", "B", "C"]]
    assert check.changed_clusters(evolution, 0) == [0, 1, 2]
    assert check.changed_clusters(evolution, 1) == [1]
    assert check.changed_clusters(evolution, 2) == [2]
    assert check.changed_clusters([[None, "b"], ["x", "b"]], 0) == [1]  # a masked schedule's unsummarized cluster
    assert check.changed_clusters([[None, "b"], ["x", "b"]], 1) == [0]


class FakeBudget:
    """Every member costs one token; a prompt fits when it has at most `limit` members."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.texts = [str(i) for i in range(100)]
        self.max_tokens = limit

    def may_overflow(self, doc_ids) -> bool:
        return len(doc_ids) > self.limit

    def kept_prefix(self, doc_ids) -> int:
        return min(len(doc_ids), self.limit)

    def kept_random(self, doc_ids) -> int:
        return min(len(doc_ids), self.limit)


def test_cluster_events_reports_only_overflowing_clusters_and_lost_intents(check, monkeypatch):
    monkeypatch.setattr(check, "N_CLUSTERS", 2)
    gold = np.array([0] * 4 + [1] * 4 + [2] * 2 + [3] * 2)  # dataset order is grouped by intent
    partition = np.array([0] * 8 + [1] * 4)  # cluster 0 holds intents 0 and 1, cluster 1 intents 2 and 3
    events = check.cluster_events(FakeBudget(5), gold, partition, [0, 1], "prefix")
    assert [(e["cluster"], e["size"], e["kept"], e["dropped"]) for e in events] == [(0, 8, 5, 3)]
    assert events[0]["n_intents"] == 2 and events[0]["lost_intents"] == 0  # the prefix still reaches intent 1
    events = check.cluster_events(FakeBudget(4), gold, partition, [0, 1], "prefix")
    assert events[0]["lost_intents"] == 1  # the prefix of 4 is intent 0 alone: intent 1 is gone
    assert check.cluster_events(FakeBudget(8), gold, partition, [0, 1], "prefix") == []


def test_lost_intents_prefix_versus_random(check):
    gold = np.repeat(np.arange(10), 10)  # 10 intents of 10, grouped
    doc_ids = np.arange(100)
    n, lost_prefix = check._lost_intents(gold, doc_ids, 50, random=False)
    assert (n, lost_prefix) == (10, 5)  # the first 50 are intents 0-4
    _, lost_random = check._lost_intents(gold, doc_ids, 50, random=True)
    assert lost_random == 0  # a seeded random half of every intent's members survives
