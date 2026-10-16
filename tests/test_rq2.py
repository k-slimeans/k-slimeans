"""Fast checks for `kslimeans.rq2` on synthetic run directories (no models, no data)."""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from kslimeans import rq2
from kslimeans.config import load_config

BASE_YAML = """
name: rq2_fixed_n5
dataset: clinc150_intent
embedder: distilbert
seeds: [0, 1, 2, 3, 4]
baselines:
  kmeans: {}
method_name: fixed-n5
force_context_length: 0
tol: 1.0e-4
schedule: {type: fixed, n_summaries: 5, max_iter: 120, initial_iter: true, final_iter: true}
summarizer: {type: hf, model_id: Qwen/Qwen3-0.6B, max_new_tokens: 64, batch_size: 8}
cache_dir: cache
results_dir: results
"""


@pytest.fixture
def base_path(tmp_path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(BASE_YAML)
    return path


@pytest.fixture
def base(base_path) -> dict:
    return load_config(base_path)


def _write_run(root: Path, cfg: dict, stamp="20260101-000000", rows=(), trace=(), done=True) -> Path:
    path = root / cfg["name"] / stamp
    path.mkdir(parents=True)
    (path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if rows:
        (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if trace:
        (path / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    if done:
        (path / "summary.csv").write_text("method\n")
    return path


def _row(method, seed, acc, calls, nmi=None, runtime_s=1.0):
    row = {"seed": seed, "method": method, "acc": acc, "nmi": acc if nmi is None else nmi, "ari": acc, "runtime_s": runtime_s}
    if calls is not None:
        row["n_calls"] = calls
    return row


def _trace(method, seed, per_step):
    """per_step: n_summarized at steps 1..; None means a plain continuation. Step 0 is the initial fit."""
    rows = [{"seed": seed, "method": method, "step": 0, "summarize": False, "n_calls": 0, "n_summarized": 0}]
    for i, n in enumerate(per_step, start=1):
        rows.append({"seed": seed, "method": method, "step": i, "summarize": n is not None,
                     "n_calls": n or 0, "n_summarized": n or 0})
    return rows


# ------------------------------------------------------------------------------------------------ planning


def test_phase_a_names_configs_and_baselines(base):
    specs = rq2.plan_phase_a(base)
    assert [s.name for s in specs] == ["rq2_fixed_n5", "rq2_fixed_n1", "rq2_fixed_n2", "rq2_fixed_n3", "rq2_fixed_n4"]
    for s in specs:
        n = s.cfg["schedule"]["n_summaries"]
        assert s.cfg["method_name"] == f"fixed-n{n}" and s.cfg["seeds"] == [0, 1, 2, 3, 4]
        assert s.cfg["baselines"] == ({"kmeans": {}} if n == 5 else {})
    assert base["name"] == "rq2_fixed_n5" and base["baselines"] == {"kmeans": {}}  # base not mutated


def test_phase_b_grid(base):
    specs = rq2.plan_phase_b(base)
    names = [s.name for s in specs]
    assert len(specs) == 11 and len(set(names)) == 11
    assert "rq2_adaptive_d0.1_r0.15" in names and "rq2_adaptive_d0.6_r0.7_s2" in names
    s = specs[names.index("rq2_adaptive_d0.3_r0.4_s2")]
    assert s.cfg["method_name"] == "adaptive-d0.3-r0.4-s2"
    assert s.cfg["seeds"] == [0, 1, 2] and s.cfg["baselines"] == {}
    assert s.cfg["schedule"] == {"type": "adaptive", "n_summaries": 5, "max_iter": 120, "initial_iter": True,
                                 "final_iter": True, "tau_drift": 0.3, "tau_churn": 0.4, "s_max": 2}
    assert specs[0].cfg["schedule"]["s_max"] is None


def test_overrides_apply_to_every_run(base):
    specs = rq2.plan_phase_a(base, ["dataset=bank77", "seeds=[0]"]) + rq2.plan_phase_b(base, ["seeds=[0]"])
    assert all(s.cfg["seeds"] == [0] for s in specs)
    assert specs[0].cfg["dataset"] == "bank77"


def test_counts_come_from_n_calls_not_n_summarized():
    rows = _trace("m", 0, [150, 10])
    rows[2]["n_calls"] = 7  # 10 clusters selected, 3 of them memo-cache hits
    assert rq2.counts_from_trace(rows, 0, 2) == [150, 7]


def test_counts_from_trace_with_plain_continuation():
    rows = _trace("m", 0, [150, None, 7, None, None]) + _trace("m", 1, [150, 5, 5, 5, 5])
    assert rq2.counts_from_trace(rows, 0, 5) == [150, 0, 7, 0, 0]
    assert rq2.counts_from_trace(rows, 1, 5) == [150, 5, 5, 5, 5]


def test_counts_from_trace_recovers_a_tol_stopped_step():
    rows = _trace("m", 0, [150, 20])[:3]  # steps 0..2 logged; step 3 stopped on tol and logged nothing
    assert rq2.counts_from_trace(rows, 0, 4, total_calls=190) == [150, 20, 20, 0]  # 190 - 170 on step 3
    assert rq2.counts_from_trace(rows, 0, 4, total_calls=170) == [150, 20, 0, 0]
    with pytest.raises(ValueError, match="spends"):
        rq2.counts_from_trace(rows, 0, 4, total_calls=100)


def test_counts_from_trace_requires_rows_for_the_seed():
    with pytest.raises(ValueError, match="seed 3"):
        rq2.counts_from_trace(_trace("m", 0, [1, 1]), 3, 2)


def test_phase_c_budget_matched_per_seed_in_calls(base, tmp_path):
    adaptive = rq2.plan_phase_b(base)[0]
    m = adaptive.cfg["method_name"]
    adaptive.cfg["seeds"] = [0, 1]
    trace = _trace(m, 0, [150, None, 4, 2, None]) + _trace(m, 1, [150, 9, 9, 9, 9])
    trace[2]["n_summarized"] = 12  # selected 12 clusters at seed 0 step 1, 4 of which cost nothing...
    trace[3]["n_summarized"] = 40  # ...the control follows n_calls, not n_summarized
    _write_run(tmp_path, adaptive.cfg, trace=trace, rows=[_row(m, 0, 0.5, 156), _row(m, 1, 0.5, 186)])
    specs = rq2.plan_random_controls(adaptive.cfg, rq2.find_done_run(tmp_path, adaptive.cfg), "C")
    assert [s.name for s in specs] == [f"rq2_random_d0.1_r0.15_seed{i}" for i in range(2)]
    s0, s1 = specs
    assert s0.cfg["schedule"] == {"type": "random_subset", "n_summaries": 5, "max_iter": 120, "initial_iter": True,
                                  "final_iter": True, "counts": [150, 0, 4, 2, 0], "seed": 0}
    assert s0.cfg["seeds"] == [0] and s0.cfg["baselines"] == {} and s0.cfg["method_name"] == "random-d0.1-r0.15"
    assert s1.cfg["schedule"]["counts"] == [150, 9, 9, 9, 9] and s1.cfg["schedule"]["seed"] == 1
    adaptive.cfg["seeds"] = [0, 1, 2]
    _write_run(tmp_path, adaptive.cfg, stamp="20260102-000000", trace=trace, rows=[_row(m, s, 0.5, 156) for s in range(3)])
    with pytest.raises(ValueError, match="seed 2"):  # seed 2 has no trace rows
        rq2.plan_random_controls(adaptive.cfg, rq2.find_done_run(tmp_path, adaptive.cfg), "C")


def test_phase_c_blocked_until_adaptive_run_done(base, tmp_path):
    specs, _ = rq2.plan_phase("C", base, tmp_path)
    assert len(specs) == 33 and all(s.cfg is None and s.blocked for s in specs)
    adaptive = rq2.plan_phase_b(base)[0]
    _write_run(tmp_path, adaptive.cfg, trace=_trace("adaptive-d0.1-r0.15", 0, [150, 1, 1, 1, 1]), done=False)
    assert all(s.cfg is None for s in rq2.plan_phase("C", base, tmp_path)[0])  # crashed run does not unblock


def test_select_headline_prefers_best_acc_within_budget():
    stats = {"a": (400.0, 0.60), "b": (900.0, 0.70), "c": (450.0, 0.65), "d": (600.0, 0.80)}
    name, rule = rq2.select_headline(stats, fixed5_calls=1000.0)
    assert name == "c" and "50%" in rule


def test_select_headline_fallback_cheapest_third():
    stats = {k: (c, a) for k, c, a in [("a", 900, 0.5), ("b", 600, 0.6), ("c", 700, 0.9), ("d", 800, 0.7),
                                       ("e", 650, 0.55), ("f", 950, 0.99)]}
    name, rule = rq2.select_headline(stats, fixed5_calls=1000.0)
    assert name == "b" and "cheapest third" in rule  # cheapest two: b (0.6) and e (0.55)
    assert rq2.select_headline({"x": (5, 1.0)}, 1.0)[0] == "x"


# ------------------------------------------------------------------------------------------ done-detection


def test_done_detection_completed_crashed_and_mismatch(base, tmp_path):
    spec = rq2.plan_phase_b(base)[0]
    assert rq2.find_done_run(tmp_path, spec.cfg) is None
    _write_run(tmp_path, spec.cfg, stamp="20260101-000000", done=False)
    assert rq2.find_done_run(tmp_path, spec.cfg) is None  # crashed
    done = _write_run(tmp_path, spec.cfg, stamp="20260102-000000")
    assert rq2.find_done_run(tmp_path, spec.cfg) == done
    other = {**spec.cfg, "seeds": [0]}
    assert rq2.find_done_run(tmp_path, other) is None  # config mismatch
    renamed = {**spec.cfg, "cache_dir": "elsewhere", "results_dir": "x"}
    assert rq2.find_done_run(tmp_path, renamed) == done  # location keys do not matter


def test_baselines_empty_and_missing_are_equivalent(base):
    a = {**base, "baselines": {}}
    b = {k: v for k, v in base.items() if k != "baselines"}
    c = {**base, "baselines": None}
    assert rq2.identity(a) == rq2.identity(b) == rq2.identity(c)


def test_planned_fixed_n3_matches_cli_launched_run(base_path, base, tmp_path):
    """A run launched as `run_offline.py <base> --set name=... --set method_name=... --set schedule.n_summaries=3
    --set 'baselines={}'` is recognized as done by the planner."""
    from kslimeans.results import RunDir

    cli_cfg = load_config(base_path, ["name=rq2_fixed_n3", "method_name=fixed-n3", "schedule.n_summaries=3",
                                      "baselines={}"])
    run = RunDir(tmp_path, cli_cfg["name"], cli_cfg)
    run.log_metrics(_row("fixed-n3", 0, 0.5, 1))
    run.finish()
    planned = next(s for s in rq2.plan_phase_a(base) if s.name == "rq2_fixed_n3")
    assert rq2.find_done_run(tmp_path, planned.cfg) == run.path
    n5 = load_config(base_path)  # the unmodified base is n5 as launched
    assert rq2.identity(n5) == rq2.identity(rq2.plan_phase_a(base)[0].cfg)


def test_in_progress_detection(base, tmp_path):
    spec = rq2.plan_phase_a(base)[0]
    assert rq2.run_status(tmp_path, spec.cfg) == ("todo", None)
    run = _write_run(tmp_path, spec.cfg, rows=[_row("fixed-n5", 0, 0.5, 1)], done=False)
    assert rq2.run_status(tmp_path, spec.cfg) == ("running", run)  # freshly written, no summary.csv
    old = time.time() - 3 * 3600
    for f in [run, *run.iterdir()]:
        os.utime(f, (old, old))
    assert rq2.run_status(tmp_path, spec.cfg) == ("todo", None)  # untouched for 3 h: crashed, redo
    assert rq2.run_status(tmp_path, spec.cfg, stale_minutes=24 * 60)[0] == "running"
    # a fresh dir with a different config is not "running" for this spec
    other = rq2.plan_phase_a(base)[1]
    _write_run(tmp_path, {**other.cfg, "name": spec.cfg["name"]}, stamp="20260102-000000", done=False)
    assert rq2.run_status(tmp_path, spec.cfg) == ("todo", None)
    (run / "summary.csv").write_text("x\n")
    assert rq2.run_status(tmp_path, spec.cfg) == ("done", run)


# ----------------------------------------------------------------------------------------------- fake data

# calls per candidate step of each adaptive config (index into ADAPTIVE_GRID); fixed-n has 200n calls
_STEPS = {4: [150, 100, 50, 50, 50], 5: [300, 200, 100, 100, 100]}  # d0.3-r0.4 cheap + good, d0.3-r0.7 over budget
_ACC = {4: 0.69, 5: 0.75}


def _fake_rows(spec) -> tuple[list[dict], list[dict]]:
    """Synthetic (metrics, trace) rows for one planned run; ACC jitters by 0.001 * seed."""
    cfg, m = spec.cfg, spec.cfg["method_name"]
    seeds = cfg["seeds"]
    if m.startswith("fixed-n"):
        n = int(m.removeprefix("fixed-n"))
        rows = [_row(m, s, 0.5 + 0.04 * n + 0.001 * s, 200 * n) for s in seeds]
        rows += [_row("kmeans", s, 0.5, None) for s in seeds if cfg["baselines"]]
        return rows, []
    if m.startswith("adaptive-"):
        idx = [rq2.adaptive_method(c) for c in rq2.ADAPTIVE_GRID].index(m)
        steps = _STEPS.get(idx, [150, 150, 60, 60, 60])
        acc = _ACC.get(idx, 0.60)
        return ([_row(m, s, acc + 0.001 * s, sum(steps)) for s in seeds],
                [r for s in seeds for r in _trace(m, s, steps)])
    counts = cfg["schedule"]["counts"]  # random control: same calls, 0.05 worse
    s = seeds[0]
    return [_row(m, s, 0.60 + 0.001 * s, sum(counts))], _trace(m, s, counts)


def _populate(root: Path, base: dict, overrides=None, skip=()) -> None:
    """Write every planned run (phases A, B, C, D, D again for the controls) as a finished fake run."""
    for phase in ("A", "B", "C", "D", "D"):
        for spec in rq2.plan_phase(phase, base, root, overrides)[0]:
            if spec.cfg is None or spec.name in skip or rq2.find_done_run(root, spec.cfg):
                continue
            rows, trace = _fake_rows(spec)
            _write_run(root, spec.cfg, rows=rows, trace=trace)


@pytest.fixture
def populated(base, tmp_path):
    _populate(tmp_path, base)
    return tmp_path


def test_phase_d_plan_after_a_and_b(base, tmp_path):
    f5, adaptive = rq2.plan_phase_a(base)[0], rq2.plan_phase_b(base)
    specs, rule = rq2.plan_phase_d(base, tmp_path)
    assert rule is None and specs[0].blocked and specs[1].name == "rq2_fixed_n5_ext"  # headline unknown yet
    assert specs[1].cfg["seeds"] == [5, 6, 7, 8, 9] and specs[1].cfg["baselines"] == {}
    assert specs[1].cfg["method_name"] == "fixed-n5"
    _populate(tmp_path, base, skip={"rq2_fixed_n5_ext", "rq2_fixed_n2_ext", "rq2_adaptive_d0.3_r0.4_ext"})
    # config 4 (d0.3-r0.4) is cheap and best; 5 is more accurate but over the calls budget
    specs, rule = rq2.plan_phase_d(base, tmp_path)
    # headline calls (400) equal fixed-n2's, so n2 brackets it on both sides; n5 is always extended
    assert rule and [s.name for s in specs[:3]] == [f"{adaptive[4].name}_ext", "rq2_fixed_n5_ext", "rq2_fixed_n2_ext"]
    assert specs[0].cfg["seeds"] == list(range(3, 10)) and specs[0].cfg["schedule"] == adaptive[4].cfg["schedule"]
    assert specs[2].cfg["seeds"] == [5, 6, 7, 8, 9] and specs[2].cfg["baselines"] == {}
    assert specs[2].cfg["method_name"] == "fixed-n2" and specs[2].cfg["schedule"]["n_summaries"] == 2
    assert [s.name for s in specs[3:]] == [f"rq2_random_d0.3_r0.4_seed{i}" for i in range(3, 10)]
    assert all(s.blocked for s in specs[3:])  # need the extension run's trace
    assert f5.cfg["seeds"] == [0, 1, 2, 3, 4]


def test_phase_d_controls_unblock_once_extension_is_done(populated, base):
    specs, _ = rq2.plan_phase_d(base, populated)
    assert len(specs) == 10 and all(s.cfg is not None for s in specs)
    assert specs[3].cfg["schedule"]["counts"] == [150, 100, 50, 50, 50]
    assert rq2.find_done_run(populated, specs[3].cfg) is not None


@pytest.mark.parametrize(
    "calls, expected",
    [(50, [5, 1]),  # below fixed-n1: just n1 (plus n5)
     (300, [5, 1, 2]),  # between n1 and n2
     (400, [5, 2]),  # exactly fixed-n2
     (900, [5, 4]),  # between n4 (800) and n5 (1000)
     (1000, [5]), (1200, [5])],
)
def test_bracketing_ns(calls, expected):
    fixed_calls = {n: 200.0 * n for n in range(1, 6)}
    assert rq2.bracketing_ns(fixed_calls, calls) == expected


def test_phase_d_waits_for_all_of_phase_a_to_pick_brackets(base, tmp_path):
    _populate(tmp_path, base, skip={"rq2_fixed_n3", "rq2_fixed_n3_ext"})
    names = [s.name for s in rq2.plan_phase_d(base, tmp_path)[0]]
    assert "rq2_fixed_n<bracket>_ext" in names and "rq2_fixed_n5_ext" in names


# ----------------------------------------------------------------------------------------------- report


def test_planned_run_dirs_ignore_stale_config_and_unplanned_dirs(populated, base):
    spec = rq2.plan_phase_b(base)[0]
    # a newer run of a planned name but with another config (stale), plus an unplanned dir
    stale = {**spec.cfg, "schedule": {**spec.cfg["schedule"], "tau_drift": 0.99}}
    _write_run(populated, stale, stamp="29990101-000000", rows=[_row(spec.cfg["method_name"], 0, 0.99, 1)])
    _write_run(populated, {**spec.cfg, "name": "rq2_other"}, rows=[_row("x", 0, 0.1, 1)])
    runs, notes = rq2.planned_run_dirs(base, populated)
    assert not any("29990101" in str(r) for r in runs)
    assert any("rq2_other" in n and "no planned run" in n for n in notes)
    assert not any(n.startswith(spec.name + ":") for n in notes)  # a matching completed run exists too
    per_seed = rq2.collect_per_seed(runs)
    assert per_seed[per_seed["variant"] == spec.cfg["method_name"]]["acc"].max() < 0.9
    only_stale = populated / "x"
    _write_run(only_stale, stale, rows=[_row("y", 0, 0.5, 1)])
    _, notes = rq2.planned_run_dirs(base, only_stale)
    assert any("stale config" in n for n in notes)


def test_collect_and_summary_table(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    variants = list(per_seed["variant"].unique())
    assert variants[:5] == [f"fixed-n{n}" for n in range(1, 6)] and variants[-1] == "k-means"
    table = rq2.summary_table(per_seed).set_index("variant")
    assert table.loc["fixed-n5", "seeds"] == "0-9" and table.loc["fixed-n5", "calls_mean"] == 1000
    assert table.loc["fixed-n2", "calls_pct"] == pytest.approx(40)
    ad = table.loc["adaptive-d0.3-r0.4"]
    assert ad["seeds"] == "0-9" and ad["calls_mean"] == 400 and ad["calls_pct"] == pytest.approx(40)
    assert ad["n_summarized_mean"] == 400
    assert table.loc["adaptive-d0.6-r0.7", "seeds"] == "0-2"
    assert table.loc["k-means", "calls_mean"] == 0 and table.loc["k-means", "seeds"] == "0-4"
    text = rq2.format_summary_table(rq2.summary_table(per_seed))
    assert text.loc[text["Variant"] == "adaptive-d0.3-r0.4", "calls"].item() == "400 ± 0"


def test_report_headline_equals_drivers_choice(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    method, rule = rq2.headline_variant(per_seed)
    specs, driver_rule = rq2.plan_phase_d(base, populated)
    assert specs[0].cfg["method_name"] == method == "adaptive-d0.3-r0.4" and rule == driver_rule


def test_matched_comparison_values_on_selection_seeds(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    m = rq2.matched_comparison(per_seed, rq2.SELECTION_SEEDS).set_index("variant")
    row = m.loc["adaptive-d0.3-r0.4"]
    # fixed-n5 ACC 0.70 + 0.001 s; adaptive 0.69 + 0.001 s -> constant delta -0.01
    assert row["d_acc5"] == pytest.approx(-0.01) and row["d_acc5_std"] == pytest.approx(0)
    assert row["d_acc5_min"] == pytest.approx(-0.01) and row["ratio5"] == pytest.approx(0.4)
    assert row["d_acc_rand"] == pytest.approx(0.09) and row["calls_dev_max"] == 0
    assert row["seeds"] == "0-2" and bool(row["h2"]) is True
    # fixed ACC = 0.5 + 0.04n at calls 200n (+0.002 mean jitter): at 400 calls that is exactly fixed-n2
    assert row["d_acc_interp"] == pytest.approx(0.69 + 0.001 - (0.5 + 0.08 + 0.001))
    assert row["lower_n"] == 2 and row["upper_n"] == 2 and row["d_acc_lower"] == pytest.approx(row["d_acc_upper"])
    assert not np.isnan(m.loc["adaptive-d0.1-r0.15", "d_acc_lower"])  # 480 calls: between n=2 and n=3
    assert m.loc["adaptive-d0.1-r0.15", "lower_n"] == 2 and m.loc["adaptive-d0.1-r0.15", "upper_n"] == 3


def test_matched_below_fixed_n1_is_na(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    per_seed.loc[per_seed["variant"] == "adaptive-d0.3-r0.4", "calls"] = 50  # cheaper than fixed-n1 (200)
    row = rq2.matched_comparison(per_seed, rq2.SELECTION_SEEDS).set_index("variant").loc["adaptive-d0.3-r0.4"]
    assert np.isnan(row["d_acc_interp"]) and np.isnan(row["d_acc_lower"]) and row["upper_n"] == 1


def test_fixed_point_missing_a_seed_is_na_with_warning(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    per_seed = per_seed[~((per_seed["variant"] == "fixed-n3") & (per_seed["seed"] == 1))]
    with pytest.warns(UserWarning, match="fixed-n3 lacks seeds"):
        m = rq2.matched_comparison(per_seed, rq2.SELECTION_SEEDS).set_index("variant")
    two_three = m.loc["adaptive-d0.1-r0.15"]  # 480 calls: brackets are fixed-n2 and fixed-n3
    assert np.isnan(two_three["d_acc_interp"]) and np.isnan(two_three["d_acc_upper"])
    assert two_three["lower_n"] == 2 and not np.isnan(two_three["d_acc_lower"])  # the complete side still works
    assert "fixed-n3 lacks seeds" in two_three["fixed_notes"]
    assert not np.isnan(m.loc["adaptive-d0.3-r0.4", "d_acc_interp"])  # 400 calls: only fixed-n2 is used
    assert "d_acc5" in m  # fixed-n5 comparisons are unaffected


@pytest.mark.parametrize(
    "acc_fixed, acc_adaptive, calls_adaptive, expected",
    [(0.70, 0.69, 500, True), (0.70, 0.60, 400, False), (0.70, 0.69, 501, False), (0.70, 0.75, 400, True)],
)
def test_h2_verdict(acc_fixed, acc_adaptive, calls_adaptive, expected):
    assert rq2.h2_verdict(acc_fixed, acc_adaptive, 1000, calls_adaptive) is expected


def test_interpolate_fixed():
    pts = [(400.0, 0.6), (200.0, 0.5), (600.0, 0.8)]
    assert rq2.interpolate_fixed(pts, 300) == pytest.approx(0.55)
    assert np.isnan(rq2.interpolate_fixed(pts, 100)) and np.isnan(rq2.interpolate_fixed(pts[:1], 400))


def test_heldout_verdict_uses_only_heldout_seeds(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    text = rq2.matched_markdown(per_seed)
    assert "Overall H2 (held-out seeds 3-9, paired seeds 3-9): SUPPORTED" in text
    assert "in-sample" in text and "winner" in text and "chord" in text
    assert "matched on summarizer calls, not on summary resets" in text
    # Ruin the selection seeds only: the held-out verdict must not change.
    ruined = per_seed.copy()
    ruined.loc[(ruined["variant"] == "adaptive-d0.3-r0.4") & ruined["seed"].isin(rq2.SELECTION_SEEDS), "acc"] = 0.1
    assert "NOT SUPPORTED" not in rq2.matched_markdown(ruined).split("# Part 2")[1]
    # Ruin the held-out seeds: now it flips, and the in-sample numbers are unaffected.
    held = per_seed.copy()
    held.loc[(held["variant"] == "adaptive-d0.3-r0.4") & held["seed"].isin(rq2.HELDOUT_SEEDS), "acc"] = 0.1
    part2 = rq2.matched_markdown(held).split("# Part 2")[1]
    assert "NOT SUPPORTED" in part2
    held_row = rq2.matched_comparison(held, rq2.HELDOUT_SEEDS, ["adaptive-d0.3-r0.4"], fixed_family=False).iloc[0]
    assert held_row["seeds"] == "3-9" and held_row["n_paired"] == 7
    # Held-out run missing -> n/a, not a verdict on seeds 0-2.
    missing = per_seed[~(per_seed["variant"].str.startswith("adaptive-") & per_seed["seed"].isin(rq2.HELDOUT_SEEDS))]
    assert "Overall H2:** n/a" in rq2.matched_markdown(missing)


def test_spread_columns_in_markdown(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    text = rq2.matched_markdown(per_seed)
    assert "-0.010 (sd 0.000; -0.010..-0.010)" in text


def test_budget_mismatch_flagged_on_max_per_seed_deviation(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    assert "MISMATCH" not in rq2.matched_markdown(per_seed)
    mask = (per_seed["variant"] == "random-d0.3-r0.4") & (per_seed["seed"] == 1)
    per_seed.loc[mask, "calls"] = 300  # one seed 25% under: the mean ratio is only ~0.92 but the max is flagged
    m = rq2.matched_comparison(per_seed, rq2.SELECTION_SEEDS).set_index("variant").loc["adaptive-d0.3-r0.4"]
    assert m["calls_dev_max"] == pytest.approx(0.25)
    assert "MISMATCH" in rq2.matched_markdown(per_seed)


def test_merged_labels_for_coincident_points():
    rows = pd.DataFrame({
        "variant": ["adaptive-d0.3-r0.15", "adaptive-d0.6-r0.15", "adaptive-d0.1-r0.4"],
        "calls_mean": [246.0, 246.0, 229.0], "acc_mean": [0.456, 0.456, 0.436],
    })
    labels = {l for _, _, l in rq2.merged_labels(rows, "acc_mean")}
    assert labels == {"d0.3-r0.15 = d0.6-r0.15", "d0.1-r0.4"}


def test_write_report_files(populated, base, tmp_path):
    out = tmp_path / "out"
    paths = rq2.write_report(populated, out, base, plot=False)
    assert {p.name for p in paths} == {"rq2_clinc150_table.md", "rq2_clinc150_table.csv",
                                       "rq2_clinc150_per_seed.csv", "rq2_clinc150_matched.md"}
    table = (out / "rq2_clinc150_table.md").read_text()
    assert "fixed-n5" in table and "different seed sets" not in table and "own seeds" in table
    with pytest.raises(ValueError, match="no completed"):
        rq2.write_report(tmp_path / "empty", out, base, plot=False)


def test_pareto_plot_writes_png_on_selection_seeds(populated, base, tmp_path):
    pytest.importorskip("matplotlib")
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    png = tmp_path / "pareto.png"
    numeric = rq2.summary_table(per_seed[per_seed["seed"].isin(rq2.SELECTION_SEEDS)])
    rq2.plot_pareto(numeric, png, "seeds 0-2")
    assert png.stat().st_size > 1000


def test_t_interval_on_a_small_example():
    delta = pd.Series([0.01, 0.03, 0.02, 0.04])  # mean 0.025, sd 0.012910, n=4, t(0.975, 3) = 3.182446
    rec: dict = {}
    rq2._paired(rec, "d", delta)
    assert rec["d"] == pytest.approx(0.025) and rec["d_std"] == pytest.approx(0.0129099, abs=1e-6)
    assert rec["d_se"] == pytest.approx(0.00645497, abs=1e-6)
    assert rec["d_ci_lo"] == pytest.approx(0.025 - 3.182446 * 0.00645497, abs=1e-6)
    assert rec["d_ci_hi"] == pytest.approx(0.025 + 3.182446 * 0.00645497, abs=1e-6)
    one: dict = {}
    rq2._paired(one, "d", pd.Series([0.1]))
    assert np.isnan(one["d_se"]) and np.isnan(one["d_ci_lo"])


def test_heldout_section_has_interval_fixed_family_and_ci_check(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    part2 = rq2.matched_markdown(per_seed).split("# Part 2")[1]
    assert "95% CI [" in part2 and "se " in part2
    assert "Additional check" in part2 and "lower end of the 95% t-interval" in part2 and "supported at 95%" in part2
    assert "vs interpolated fixed family" in part2 and "vs fixed-n2 (lower neighbour)" in part2
    assert "chord" in part2


def test_heldout_incomplete_when_fixed_n5_only_covers_some_seeds(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    only_34 = per_seed[~((per_seed["variant"] == "fixed-n5") & per_seed["seed"].isin(range(5, 10)))]
    part2 = rq2.matched_markdown(only_34).split("# Part 2")[1]
    assert "Overall H2: incomplete" in part2 and "2 of 7 held-out seeds paired: 3-4" in part2
    assert "SUPPORTED" not in part2.split("###")[0]  # no verdict, not even NOT SUPPORTED
    row = rq2.matched_comparison(only_34, rq2.HELDOUT_SEEDS, ["adaptive-d0.3-r0.4"]).iloc[0]
    assert row["paired5"] == (3, 4) and row["n_paired"] == 2


def test_heldout_random_line_incomplete_and_part1_shows_paired_seeds(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    cut = per_seed[~((per_seed["variant"] == "random-d0.3-r0.4") & (per_seed["seed"] > 5))]
    part2 = rq2.matched_markdown(cut).split("# Part 2")[1]
    assert "random control: **incomplete** (3 of 7 held-out seeds paired: 3-5" in part2
    ragged = per_seed[~((per_seed["variant"] == "fixed-n5") & (per_seed["seed"] == 1))]
    part1 = rq2.matched_markdown(ragged).split("# Part 2")[0]
    assert "| adaptive-d0.3-r0.4 | 0,2 |" in part1  # fixed-n5 pairing shows 0,2, not the adaptive run's 0-2


def test_heldout_fixed_family_incomplete_when_bracket_lacks_seeds(populated, base):
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    cut = per_seed[~((per_seed["variant"] == "fixed-n2") & (per_seed["seed"] > 6))]
    with pytest.warns(UserWarning, match="fixed-n2 lacks seeds"):
        part2 = rq2.matched_markdown(cut).split("# Part 2")[1]
    assert "Incomplete / n/a" in part2 and "fixed-n2 lacks seeds [7, 8, 9]" in part2


def test_heldout_uses_planned_brackets_even_if_heldout_calls_cross_a_boundary(populated, base):
    """The planner extended fixed-n2 (headline calls 400 on seeds 0-2 equal fixed-n2's). If the held-out calls
    were 650 the nearest-by-calls points would be n3/n4, which have no seeds 5-9; the report must still compare
    against the extended n2 and clamp the chord, not go n/a."""
    per_seed = rq2.collect_per_seed(rq2.planned_run_dirs(base, populated)[0])
    held = per_seed["variant"].eq("adaptive-d0.3-r0.4") & per_seed["seed"].isin(rq2.HELDOUT_SEEDS)
    per_seed.loc[held, "calls"] = 650
    assert rq2._planned_bracket(per_seed, "adaptive-d0.3-r0.4") == (2, 2)
    row = rq2.matched_comparison(per_seed, rq2.HELDOUT_SEEDS, ["adaptive-d0.3-r0.4"], bracket=(2, 2)).iloc[0]
    assert row["lower_n"] == row["upper_n"] == 2 and not np.isnan(row["d_acc_interp"])
    naive = rq2.matched_comparison(per_seed, rq2.HELDOUT_SEEDS, ["adaptive-d0.3-r0.4"]).iloc[0]
    assert "lower_n" not in naive and "upper_n" not in naive  # by held-out calls: n3/n4, which were not extended
    part2 = rq2.matched_markdown(per_seed).split("# Part 2")[1]
    assert "vs fixed-n2 (lower neighbour)" in part2 and "Incomplete / n/a" not in part2


def test_empty_seed_tuple_renders_none():
    assert rq2._seed_cells([()]) == ["none"] and rq2._fmt_seeds(()) == "none"
    assert rq2._seed_cells([(0, 1, 2), "3-9", None]) == ["0-2", "3-9", "n/a"]
