"""RQ2 experiment: planning, resumability and reporting for the adaptive-schedule sweep on CLINC150.

Question (H2): can an adaptive summary schedule (`AdaptiveSchedule`: re-summarize only clusters whose
centroid drifted or whose membership churned) match the default fixed schedule's accuracy with a fraction
of the summarizer calls? Cost is `n_calls` from `metrics.jsonl`. The sweep has four phases, all derived from
the base config `configs/rq2_clinc150.yaml` (see its header):

- A: `FixedSchedule` with n_summaries 1..5 (n=5 is the base config and the only run with the k-means baseline);
- B: the adaptive grid `ADAPTIVE_GRID` on seeds 0-2;
- C: for every (adaptive config, seed), a `RandomSubsetSchedule` control that spends the same number of
  summarizer calls at every candidate step as that adaptive run (its trace's `n_calls`), drawn at random
  among the clusters whose membership changed since their last summary (an unchanged cluster would be a
  free memo-cache hit, so it cannot spend budget);
- D: the headline adaptive config (`select_headline`, chosen on seeds 0-2) and its random controls on the
  held-out seeds 3-9, plus a fixed-n5 extension (`rq2_fixed_n5_ext`, seeds 5-9; seeds 3-4 come from A).

Selection vs evaluation: the headline is the best of 11 configs on seeds 0-2, so its accuracy there is
optimistically biased (winner's curse). The overall H2 verdict is therefore judged on the held-out seeds
3-9 only; seeds 0-2 numbers are reported separately as in-sample.

`scripts/run_rq2.py` is the resumable driver and `scripts/rq2_report.py` the report CLI; both are thin
wrappers around this module. Planning, done-detection and the selection rule are pure functions of config
dicts and run directories, so they are unit-tested without running anything.
"""

from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as t_dist

from .aggregate import METRICS, to_markdown
from .config import apply_overrides, load_config

PHASES = ("A", "B", "C", "D")
FIXED_NS = (1, 2, 3, 4, 5)
FIXED_ORDER = (5, 1, 2, 3, 4)  # run order: the default schedule first, so the key run finishes first
# (tau_drift, tau_churn, s_max). Units: drift in RMS radii, churn = Jaccard distance. Chosen from a pilot
# of per-cluster drift / churn between consecutive fixed-schedule summary steps.
ADAPTIVE_GRID: tuple[tuple[float, float, int | None], ...] = (
    *((td, tr, None) for td in (0.1, 0.3, 0.6) for tr in (0.15, 0.4, 0.7)),
    (0.3, 0.4, 2),
    (0.6, 0.7, 2),
)
ADAPTIVE_SEEDS = [0, 1, 2]
SELECTION_SEEDS = (0, 1, 2)  # seeds on which `select_headline` compares configs (in-sample)
HELDOUT_SEEDS = tuple(range(3, 10))  # the headline config and its random control are evaluated here
FIXED_EXT_SEEDS = [5, 6, 7, 8, 9]  # fixed-n5 on 3-4 comes from phase A (seeds 0-4)
STALE_MINUTES = 90.0  # an unfinished run dir untouched for longer than this counts as crashed
CALLS_BUDGET = 0.5  # H2: at most this fraction of fixed-n5's calls ...
BUDGET_MISMATCH = 0.05  # flag a random control whose calls differ from its adaptive run's by more than this
ACC_TOLERANCE = 0.01  # ... with ACC at most this far below fixed-n5's
# Keys that define an experiment; name / cache_dir / results_dir only say where things live.
IDENTITY_KEYS = (
    "dataset",
    "embedder",
    "seeds",
    "baselines",
    "method_name",
    "force_context_length",
    "tol",
    "schedule",
    "summarizer",
)
_IDENTITY_DEFAULTS = {"seeds": [0], "baselines": {}, "force_context_length": 0, "tol": 1e-4}


# --------------------------------------------------------------------------------------------- planning


@dataclass
class RunSpec:
    """One planned run. `cfg` is None while `blocked` (it needs the traces of runs that are not done yet)."""

    name: str
    phase: str
    cfg: dict | None
    blocked: str | None = None


def _num(x: float) -> str:
    return f"{x:g}"


def config_tag(tau_drift: float, tau_churn: float, s_max: int | None) -> str:
    """`d0.3_r0.4` / `d0.3_r0.4_s2`, the suffix of run names."""
    return f"d{_num(tau_drift)}_r{_num(tau_churn)}" + (f"_s{s_max}" if s_max is not None else "")


def adaptive_name(config: tuple) -> str:
    return f"rq2_adaptive_{config_tag(*config)}"


def adaptive_method(config: tuple) -> str:
    return "adaptive-" + config_tag(*config).replace("_", "-")


def random_method(adaptive_method_name: str) -> str:
    return "random-" + adaptive_method_name.removeprefix("adaptive-")


def random_name(adaptive_run_name: str, seed: int) -> str:
    """`rq2_random_d0.3_r0.4_seed0` from `rq2_adaptive_d0.3_r0.4` (or its `_ext` run)."""
    tag = adaptive_run_name.removeprefix("rq2_adaptive_").removesuffix("_ext")
    return f"rq2_random_{tag}_seed{seed}"


def _block_params(schedule: dict) -> dict:
    return {k: schedule[k] for k in ("n_summaries", "max_iter", "initial_iter", "final_iter")}


def plan_phase_a(base: dict, overrides: list[str] | None = None) -> list[RunSpec]:
    """Fixed family n_summaries 1..5. Run-relevant config is the base plus exactly the overrides
    `--set name=rq2_fixed_n{n} --set method_name=fixed-n{n} --set schedule.n_summaries={n}`
    (and `--set 'baselines={}'` for n != 5), so earlier hand-launched runs are recognized as done."""
    specs = []
    for n in FIXED_ORDER:
        cfg = copy.deepcopy(base)
        cfg["name"] = f"rq2_fixed_n{n}"
        cfg["method_name"] = f"fixed-n{n}"
        cfg["schedule"]["n_summaries"] = n
        if n != 5:
            cfg["baselines"] = {}
        specs.append(RunSpec(cfg["name"], "A", apply_overrides(cfg, overrides)))
    return specs


def plan_phase_b(base: dict, overrides: list[str] | None = None) -> list[RunSpec]:
    specs = []
    for config in ADAPTIVE_GRID:
        td, tr, s_max = config
        cfg = copy.deepcopy(base)
        cfg["name"] = adaptive_name(config)
        cfg["method_name"] = adaptive_method(config)
        cfg["seeds"] = list(ADAPTIVE_SEEDS)
        cfg["baselines"] = {}
        cfg["schedule"] = {
            "type": "adaptive",
            **_block_params(base["schedule"]),
            "tau_drift": td,
            "tau_churn": tr,
            "s_max": s_max,
        }
        specs.append(RunSpec(cfg["name"], "B", apply_overrides(cfg, overrides)))
    return specs


def counts_from_trace(rows: list[dict], seed: int, n_summaries: int, total_calls: int | None = None) -> list[int]:
    """Summarizer calls spent at candidate steps 1..n_summaries of one seed's trace (`n_calls`).

    A plain-continuation step (`summarize` false) counts 0. A summary step that stopped on `tol` fits no
    KMeans and so logs no trace row, nothing runs after it, and its calls are missing from the trace: when
    `total_calls` (the run's metrics `n_calls`) is given, the first missing step gets
    `total_calls - sum(trace n_calls)` and later missing steps 0. Raises `ValueError` if the seed has no
    trace rows at all, or if the trace spends more calls than `total_calls`.
    """
    seed_rows = [r for r in rows if r["seed"] == seed]
    if not seed_rows:
        raise ValueError(f"trace has no rows for seed {seed}")
    by_step = {r["step"]: r for r in seed_rows}
    counts = [int(by_step[s]["n_calls"]) if s in by_step and by_step[s]["summarize"] else 0 for s in range(1, n_summaries + 1)]
    missing = [s for s in range(1, n_summaries + 1) if s not in by_step]
    if missing and total_calls is not None:
        remainder = int(total_calls) - sum(int(r["n_calls"]) for r in seed_rows)
        if remainder < 0:
            raise ValueError(f"seed {seed}: trace spends {sum(counts)} calls but the run reports {total_calls}")
        counts[missing[0] - 1] = remainder
    return counts


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plan_random_controls(adaptive_cfg: dict, adaptive_run: Path | None, phase: str) -> list[RunSpec]:
    """One `RandomSubsetSchedule` run per seed of an adaptive run, with that seed's per-step calls.

    `adaptive_run` is the completed run dir to read traces from; None (not done yet) yields blocked specs.
    """
    seeds = adaptive_cfg.get("seeds") or [0]
    rows = _read_jsonl(adaptive_run / "trace.jsonl") if adaptive_run is not None else None
    totals = {}
    if adaptive_run is not None:
        totals = {
            r["seed"]: r["n_calls"]
            for r in _read_jsonl(adaptive_run / "metrics.jsonl")
            if r["method"] == adaptive_cfg["method_name"]
        }
    specs = []
    for seed in seeds:
        name = random_name(adaptive_cfg["name"], seed)
        if rows is None:
            specs.append(RunSpec(name, phase, None, blocked=f"needs the trace of {adaptive_cfg['name']}"))
            continue
        params = _block_params(adaptive_cfg["schedule"])
        cfg = copy.deepcopy(adaptive_cfg)
        cfg["name"] = name
        cfg["method_name"] = random_method(adaptive_cfg["method_name"])
        cfg["seeds"] = [seed]
        cfg["baselines"] = {}
        cfg["schedule"] = {
            "type": "random_subset",
            **params,
            "counts": counts_from_trace(rows, seed, params["n_summaries"], totals.get(seed)),
            "seed": seed,
        }
        specs.append(RunSpec(name, phase, cfg))
    return specs


def plan_phase_c(adaptive_specs: list[RunSpec], results_dir: str | Path) -> list[RunSpec]:
    specs = []
    for spec in adaptive_specs:
        run = find_done_run(results_dir, spec.cfg)
        specs.extend(plan_random_controls(spec.cfg, run, "C"))
    return specs


def select_headline(stats: dict[str, tuple[float, float]], fixed5_calls: float) -> tuple[str, str]:
    """Pick the adaptive config to extend to seeds 3-4. `stats` maps config -> (mean calls, mean ACC)
    over the selection seeds. Returns `(config, rule)`.

    Rule 1: among configs with mean calls <= `CALLS_BUDGET` x fixed-n5's, the highest mean ACC.
    Rule 2 (none qualifies): the highest mean ACC among the cheapest third of configs (at least one) --
    the best ACC/calls trade-off available. Ties go to the cheaper, then the earlier, config.
    """
    if not stats:
        raise ValueError("no adaptive configs to choose from")
    order = {name: i for i, name in enumerate(stats)}

    def best(names: list[str]) -> str:
        return max(names, key=lambda n: (stats[n][1], -stats[n][0], -order[n]))

    qualifying = [n for n, (calls, _) in stats.items() if calls <= CALLS_BUDGET * fixed5_calls]
    if qualifying:
        return best(qualifying), f"highest ACC among configs with calls <= {CALLS_BUDGET:.0%} of fixed-n5"
    cheapest = sorted(stats, key=lambda n: (stats[n][0], order[n]))[: max(1, len(stats) // 3)]
    return best(cheapest), "none within the calls budget; highest ACC among the cheapest third"


def _mean_over_seeds(run: Path, method: str, seeds: tuple[int, ...]) -> tuple[float, float] | None:
    rows = [r for r in _read_jsonl(run / "metrics.jsonl") if r["method"] == method and r["seed"] in seeds]
    if not rows:
        return None
    return float(np.mean([r["n_calls"] for r in rows])), float(np.mean([r["acc"] for r in rows]))


def fixed_ext_spec(base: dict, n: int, overrides: list[str] | None = None) -> RunSpec:
    """`rq2_fixed_n{n}_ext`: fixed-n{n} on seeds 5-9 without baselines (seeds 3-4 come from phase A)."""
    cfg = copy.deepcopy(next(s for s in plan_phase_a(base) if s.cfg["schedule"]["n_summaries"] == n).cfg)
    cfg["name"] = f"rq2_fixed_n{n}_ext"
    cfg["seeds"] = list(FIXED_EXT_SEEDS)
    cfg["baselines"] = {}
    return RunSpec(cfg["name"], "D", apply_overrides(cfg, overrides))


def bracketing_ns(fixed_calls: dict[int, float], calls: float) -> list[int]:
    """The fixed-n values to extend to held-out seeds for a config with mean `calls`: n=5 (always) plus the
    lower and upper neighbour of `calls` among fixed-n by mean calls. Below fixed-n1 that is n1 alone;
    between n4 and n5 it is n4 and n5."""
    chosen = {5, *(n for n in bracket_pair(fixed_calls, calls) if n is not None)}
    return [5, *sorted(chosen - {5})]


def bracket_pair(fixed_calls: dict[int, float], calls: float) -> tuple[int | None, int | None]:
    """`(lower, upper)`: the fixed-n with the largest mean calls <= `calls` and the one with the smallest
    >= it (None when `calls` is outside the fixed family's range). Shared by the planner, which extends these
    points to held-out seeds, and the report, which must compare against exactly them."""
    ns = sorted(fixed_calls, key=fixed_calls.get)
    lower = [n for n in ns if fixed_calls[n] <= calls]
    upper = [n for n in ns if fixed_calls[n] >= calls]
    return (lower[-1] if lower else None), (upper[0] if upper else None)


def plan_phase_d(
    base: dict, results_dir: str | Path, overrides: list[str] | None = None
) -> tuple[list[RunSpec], str | None]:
    """Held-out evaluation. The headline adaptive config (chosen on seeds 0-2) runs on seeds 3-9 as
    `<name>_ext`, with a call-matched random control per seed. Fixed-n5 and the fixed-n points bracketing the
    headline's mean calls get `rq2_fixed_n{n}_ext` on seeds 5-9 (no baselines), so they exist on all of 3-9
    for the matched-calls comparison. Returns `(specs, rule)`; while the phases it depends on are incomplete
    the unknown parts are single blocked placeholders."""
    fixed5 = plan_phase_a(base, overrides)[0]
    adaptive = plan_phase_b(base, overrides)
    chosen = choose_headline(results_dir, fixed5.cfg, adaptive)
    if isinstance(chosen, str):
        return [
            RunSpec("rq2_adaptive_<headline>_ext", "D", None, blocked=chosen),
            fixed_ext_spec(base, 5, overrides),
        ], None
    parent, rule, calls = chosen
    ext = copy.deepcopy(parent.cfg)
    ext["name"] = f"{parent.name}_ext"
    ext["seeds"] = list(HELDOUT_SEEDS)
    ext = apply_overrides(ext, overrides)
    specs = [RunSpec(ext["name"], "D", ext)]

    fixed_calls = {}
    for spec in plan_phase_a(base, overrides):
        run = find_done_run(results_dir, spec.cfg)
        mean = _mean_over_seeds(run, spec.cfg["method_name"], SELECTION_SEEDS) if run else None
        if mean:
            fixed_calls[spec.cfg["schedule"]["n_summaries"]] = mean[0]
    if len(fixed_calls) == len(FIXED_NS):
        specs += [fixed_ext_spec(base, n, overrides) for n in bracketing_ns(fixed_calls, calls)]
    else:
        specs += [fixed_ext_spec(base, 5, overrides),
                  RunSpec("rq2_fixed_n<bracket>_ext", "D", None, blocked="needs all of phase A")]
    specs += plan_random_controls(ext, find_done_run(results_dir, ext), "D")
    return specs, rule


def choose_headline(
    results_dir: str | Path, fixed5_cfg: dict, adaptive: list[RunSpec]
) -> tuple[RunSpec, str, float] | str:
    """The Phase-D adaptive spec, the rule that chose it and its mean calls on `SELECTION_SEEDS`, from the
    completed A/B runs; a string explaining what is missing when it cannot be chosen yet."""
    fixed5 = find_done_run(results_dir, fixed5_cfg)
    runs = {s.name: find_done_run(results_dir, s.cfg) for s in adaptive}
    if fixed5 is None or any(r is None for r in runs.values()):
        return "needs phases A and B"
    fixed = _mean_over_seeds(fixed5, fixed5_cfg["method_name"], SELECTION_SEEDS)
    stats = {s.cfg["method_name"]: _mean_over_seeds(runs[s.name], s.cfg["method_name"], SELECTION_SEEDS) for s in adaptive}
    if fixed is None or None in stats.values():
        return "phase A/B runs lack seeds 0-2"
    method, rule = select_headline(stats, fixed[0])
    return next(s for s in adaptive if s.cfg["method_name"] == method), rule, stats[method][0]


def plan_phase(
    phase: str, base: dict, results_dir: str | Path, overrides: list[str] | None = None
) -> tuple[list[RunSpec], str | None]:
    """Specs of one phase given what is on disk now. The second value is the Phase-D selection rule used."""
    if phase == "A":
        return plan_phase_a(base, overrides), None
    if phase == "B":
        return plan_phase_b(base, overrides), None
    if phase == "C":
        return plan_phase_c(plan_phase_b(base, overrides), results_dir), None
    if phase == "D":
        return plan_phase_d(base, results_dir, overrides)
    raise ValueError(f"Unknown phase '{phase}', expected one of {PHASES}")


def load_base(path: str | Path, results_dir: str | Path | None = None) -> dict:
    base = load_config(path)
    if results_dir is not None:
        base["results_dir"] = str(results_dir)
    return base


# ------------------------------------------------------------------------------------------ done-detection


def identity(cfg: dict) -> dict:
    """The experiment-defining part of a config, normalized so equivalent spellings compare equal
    (missing / None `baselines` is `{}`, missing `seeds` is `[0]`, ...)."""
    out = {}
    for key in IDENTITY_KEYS:
        value = cfg.get(key)
        out[key] = _IDENTITY_DEFAULTS.get(key) if value is None else value
    return out


def _matches(run_dir: Path, cfg: dict) -> bool:
    """`run_dir`'s recorded config defines the same experiment as `cfg`."""
    if not (run_dir / "config.yaml").exists():
        return False
    with open(run_dir / "config.yaml") as f:
        return identity(yaml.safe_load(f) or {}) == identity(cfg)


def is_done(run_dir: Path, cfg: dict) -> bool:
    """`run_dir` finished (has `summary.csv`) and its recorded config defines the same experiment as `cfg`."""
    return (run_dir / "summary.csv").exists() and _matches(run_dir, cfg)


def _stamp_dirs(results_dir: str | Path, name: str) -> list[Path]:
    root = Path(results_dir) / name
    return sorted((p for p in root.iterdir() if p.is_dir()), reverse=True) if root.is_dir() else []


def find_done_run(results_dir: str | Path, cfg: dict) -> Path | None:
    """Newest completed run of `cfg["name"]` under `results_dir` that matches `cfg`, or None."""
    return next((p for p in _stamp_dirs(results_dir, cfg["name"]) if is_done(p, cfg)), None)


def _last_touched(run_dir: Path) -> float:
    return max([run_dir.stat().st_mtime, *(f.stat().st_mtime for f in run_dir.iterdir())])


def find_running_run(
    results_dir: str | Path, cfg: dict, stale_minutes: float = STALE_MINUTES, now: float | None = None
) -> Path | None:
    """A matching run dir without `summary.csv` that was written to within `stale_minutes` (another
    process is probably still running it). Older unfinished dirs count as crashed, so they are redone."""
    now = time.time() if now is None else now
    for path in _stamp_dirs(results_dir, cfg["name"]):
        if not (path / "summary.csv").exists() and _matches(path, cfg) and now - _last_touched(path) <= stale_minutes * 60:
            return path
    return None


def run_status(
    results_dir: str | Path, cfg: dict, stale_minutes: float = STALE_MINUTES, now: float | None = None
) -> tuple[str, Path | None]:
    """`("done" | "running" | "todo", run dir or None)` for a planned config."""
    done = find_done_run(results_dir, cfg)
    if done is not None:
        return "done", done
    running = find_running_run(results_dir, cfg, stale_minutes, now)
    return ("running", running) if running is not None else ("todo", None)


def planned_runs(
    base: dict, results_dir: str | Path, overrides: list[str] | None = None
) -> list[tuple[RunSpec, Path | None]]:
    """Every planned run of phases A-D (as the driver would plan them from what is on disk) with its
    completed run dir, or None."""
    out = []
    for phase in PHASES:
        specs, _ = plan_phase(phase, base, results_dir, overrides)
        out.extend((s, None if s.cfg is None else find_done_run(results_dir, s.cfg)) for s in specs)
    return out


# -------------------------------------------------------------------------------------------------- report


def _variant(method: str) -> str:
    return "k-means" if method == "kmeans" else method


def _family(variant: str) -> str:
    for prefix, family in (("fixed-n", "fixed"), ("adaptive-", "adaptive"), ("random-", "random")):
        if variant.startswith(prefix):
            return family
    return "k-means"


def _sort_key(variant: str) -> tuple:
    family = _family(variant)
    adaptive_order = [adaptive_method(c) for c in ADAPTIVE_GRID]
    if family == "fixed":
        return (0, int(variant.removeprefix("fixed-n")), "")
    if family == "adaptive":
        return (1, adaptive_order.index(variant) if variant in adaptive_order else 99, variant)
    if family == "random":
        twin = "adaptive-" + variant.removeprefix("random-")
        return (2, adaptive_order.index(twin) if twin in adaptive_order else 99, variant)
    return (3, 0, variant)


def _total_summarized(run: Path, rows: pd.DataFrame) -> list[float]:
    """Per metrics row: clusters selected for summarization over the whole run (sum of the trace's
    `n_summarized`), 0 for k-means, NaN without a trace. Memo-cache hits count here but cost no call."""
    if not (run / "trace.jsonl").exists():
        return [0.0 if v == "k-means" else np.nan for v in rows["variant"]]
    totals: dict[tuple, int] = {}
    for r in _read_jsonl(run / "trace.jsonl"):
        key = (r["method"], r["seed"])
        totals[key] = totals.get(key, 0) + int(r.get("n_summarized") or 0)
    return [
        0.0 if v == "k-means" else totals.get((m, sd), np.nan)
        for v, m, sd in zip(rows["variant"], rows["method"], rows["seed"])
    ]


_COLUMNS = ["variant", "seed", "calls", "n_summarized", *METRICS, "runtime_s"]


def collect_per_seed(run_dirs: list[Path]) -> pd.DataFrame:
    """Per-seed rows `variant, seed, calls, n_summarized, acc, nmi, ari, runtime_s` of the given runs.

    Runs of one variant (an adaptive run and its `_ext`, fixed-n5 and `rq2_fixed_n5_ext`, a random control
    per seed) are concatenated; if a (variant, seed) appears twice the first run wins. Rows without
    `n_calls` (k-means) get 0 calls.
    """
    frames = []
    for run in run_dirs:
        if not (run / "metrics.jsonl").exists():
            continue
        rows = pd.DataFrame(_read_jsonl(run / "metrics.jsonl"))
        rows["variant"] = [_variant(m) for m in rows["method"]]
        if "n_calls" not in rows:
            rows["n_calls"] = np.nan
        rows.loc[rows["variant"] == "k-means", "n_calls"] = 0
        if "runtime_s" not in rows:
            rows["runtime_s"] = np.nan
        rows["n_summarized"] = _total_summarized(run, rows)
        frames.append(rows.rename(columns={"n_calls": "calls"})[_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["variant", "seed"], keep="first")
    df["_key"] = df["variant"].map(_sort_key)
    return df.sort_values(["_key", "seed"], kind="stable").drop(columns="_key").reset_index(drop=True)


def planned_run_dirs(
    base: dict, results_dir: str | Path, overrides: list[str] | None = None
) -> tuple[list[Path], list[str]]:
    """Completed run dirs of the planned runs (identity-checked, like the driver), and warnings about
    `rq2_*` dirs under `results_dir` that match no planned run or have no completed run of that config."""
    planned = planned_runs(base, results_dir, overrides)
    notes = []
    names = {spec.name for spec, _ in planned}
    done_names = {spec.name for spec, run in planned if run is not None}
    for path in sorted(Path(results_dir).glob("rq2_*")):
        if path.name not in names:
            notes.append(f"{path.name}: matches no planned run, ignored")
        elif path.name not in done_names:
            notes.append(f"{path.name}: no completed run with the planned config (stale config or unfinished), ignored")
    return [run for _, run in planned if run is not None], notes


def _fmt_seeds(seeds) -> str:
    seeds = sorted(int(s) for s in seeds)
    if not seeds:
        return "none"
    if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ",".join(map(str, seeds))


def _seed_means(per_seed: pd.DataFrame, variant: str, seeds: set | None = None) -> pd.DataFrame:
    rows = per_seed[per_seed["variant"] == variant]
    if seeds is not None:
        rows = rows[rows["seed"].isin(seeds)]
    return rows.set_index("seed")


def _restrict(per_seed: pd.DataFrame, seeds) -> pd.DataFrame:
    return per_seed[per_seed["seed"].isin(set(seeds))]


def summary_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Numeric table, one row per variant: seeds, calls mean/std, calls as % of fixed-n5 (same seeds),
    metric mean/std (sample std, NaN for one seed) and mean runtime."""
    fixed5 = _seed_means(per_seed, "fixed-n5")
    records = []
    for variant, rows in per_seed.groupby("variant", sort=False):
        rec = {"variant": variant, "family": _family(variant), "seeds": _fmt_seeds(rows["seed"]), "n_seeds": len(rows)}
        rec["calls_mean"], rec["calls_std"] = rows["calls"].mean(), rows["calls"].std()
        rec["n_summarized_mean"] = rows["n_summarized"].mean()
        common = fixed5.index.intersection(rows["seed"])
        ref = fixed5.loc[common, "calls"].mean() if len(common) else np.nan
        mine = rows.set_index("seed").loc[common, "calls"].mean() if len(common) else np.nan
        rec["calls_pct"] = 100 * mine / ref if ref and not np.isnan(ref) else np.nan
        for metric in METRICS:
            rec[f"{metric}_mean"], rec[f"{metric}_std"] = rows[metric].mean(), rows[metric].std()
        rec["runtime_s"] = rows["runtime_s"].mean()
        records.append(rec)
    return pd.DataFrame(records)


def format_summary_table(numeric: pd.DataFrame) -> pd.DataFrame:
    def pm(mean: float, std: float, digits: int) -> str:
        if np.isnan(mean):
            return "n/a"
        return f"{mean:.{digits}f} ± " + ("n/a" if np.isnan(std) else f"{std:.{digits}f}")

    out = pd.DataFrame({"Variant": numeric["variant"], "seeds": numeric["seeds"]})
    out["calls"] = [pm(m, s, 0) for m, s in zip(numeric["calls_mean"], numeric["calls_std"])]
    out["clusters summarized"] = ["n/a" if np.isnan(x) else f"{x:.0f}" for x in numeric["n_summarized_mean"]]
    out["% of fixed-n5"] = ["n/a" if np.isnan(p) else f"{p:.1f}%" for p in numeric["calls_pct"]]
    for metric in METRICS:
        out[metric.upper()] = [pm(m, s, 3) for m, s in zip(numeric[f"{metric}_mean"], numeric[f"{metric}_std"])]
    out["s/seed"] = ["n/a" if np.isnan(t) else f"{t:.1f}" for t in numeric["runtime_s"]]
    return out


def h2_verdict(acc_fixed: float, acc_adaptive: float, calls_fixed: float, calls_adaptive: float) -> bool:
    """H2 holds: ACC within `ACC_TOLERANCE` of fixed-n5 (or better) with at most `CALLS_BUDGET` of its calls."""
    return (acc_fixed - acc_adaptive) <= ACC_TOLERANCE + 1e-12 and calls_adaptive <= CALLS_BUDGET * calls_fixed


def interpolate_fixed(points: list[tuple[float, float]], calls: float) -> float:
    """Linear interpolation of (calls, value) points at `calls`; NaN outside their range."""
    if len(points) < 2:
        return np.nan
    xs, ys = zip(*sorted(points))
    if not xs[0] <= calls <= xs[-1]:
        return np.nan
    return float(np.interp(calls, xs, ys))


def _paired(rec: dict, key: str, delta: pd.Series) -> None:
    """Store the paired per-seed deltas' mean under `key`, and under `key_std` / `_min` / `_max` / `_se` /
    `_ci_lo` / `_ci_hi` their sample std, range, standard error and 95% t-interval of the mean (NaN for
    fewer than two seeds)."""
    n = len(delta)
    rec[key] = delta.mean()
    sd = delta.std() if n > 1 else np.nan
    rec[f"{key}_std"] = sd
    rec[f"{key}_min"], rec[f"{key}_max"] = delta.min(), delta.max()
    rec[f"{key}_se"], rec[f"{key}_ci_lo"], rec[f"{key}_ci_hi"] = t_interval(rec[key], sd, n)


def t_interval(mean: float, sd: float, n: int, level: float = 0.95) -> tuple[float, float, float]:
    """`(standard error, lower, upper)` of the `level` t-interval of a mean of `n` paired deltas with sample
    std `sd`; NaNs for n < 2."""
    if n < 2 or np.isnan(sd):
        return np.nan, np.nan, np.nan
    se = sd / np.sqrt(n)
    half = float(t_dist.ppf(0.5 + level / 2, n - 1)) * se
    return float(se), float(mean - half), float(mean + half)


def _fixed_neighbours(
    per_seed: pd.DataFrame, seeds: set, calls: float, pair: tuple[int | None, int | None] | None = None
) -> tuple[dict | None, dict | None, list[str]]:
    """The fixed-n points bracketing `calls`, with their per-seed rows over exactly `seeds`. By default they
    are the points with the largest mean calls <= it and the smallest >= it (each variant's mean calls over
    all its seeds); `pair=(lower_n, upper_n)` names them explicitly (the planner's brackets). A neighbour
    lacking any of `seeds` is None with a warning and a note: no partial-seed averaging."""
    seeds_sorted = sorted(seeds)
    by_n = {}
    for variant in [v for v in per_seed["variant"].unique() if _family(v) == "fixed"]:
        rows = _seed_means(per_seed, variant)
        by_n[int(variant.removeprefix("fixed-n"))] = (rows["calls"].mean(), rows)
    if pair is None:
        pair = bracket_pair({n: c for n, (c, _) in by_n.items()}, calls)
    notes: list[str] = []

    def use(n: int | None) -> dict | None:
        if n is None or n not in by_n:
            return None
        _, rows = by_n[n]
        missing = seeds - set(rows.index)
        if missing:
            note = f"fixed-n{n} lacks seeds {sorted(missing)}"
            warnings.warn(f"{note}; fixed-family comparisons over seeds {_fmt_seeds(seeds)} are n/a for it", stacklevel=4)
            notes.append(note)
            return None
        r = rows.loc[seeds_sorted]
        return {"n": n, "calls": r["calls"].mean(), "acc": r["acc"], "nmi": r["nmi"]}

    return use(pair[0]), use(pair[1]), notes


def matched_comparison(
    per_seed: pd.DataFrame,
    seeds,
    variants: list[str] | None = None,
    fixed_family: bool = True,
    bracket: tuple[int | None, int | None] | None = None,
) -> pd.DataFrame:
    """One row per adaptive config, paired by seed, over `seeds` only. Deltas are adaptive minus reference;
    each `d_*` is the mean over the paired seeds, with `_std` / `_min` / `_max` / `_se` / `_ci_lo` / `_ci_hi`
    giving their spread and 95% t-interval (see `_paired`).

    (a) vs fixed-n5: `d_acc5`, `d_nmi5`, `ratio5` (calls ratio), the H2 verdict `h2` and `paired5` (the seeds
        actually paired, which can be fewer than `seeds`);
    (b) vs its random control: `d_acc_rand`, `d_nmi_rand`, `random_seeds`, `summarized_*`, `calls_*`,
        `calls_ratio_random` (mean per-seed ratio) and `calls_dev_max` (largest per-seed |ratio - 1|);
    (c) vs the fixed family at matched calls (`fixed_family`): `d_acc_interp` / `d_nmi_interp` against the
        linear interpolation between the two fixed-n points bracketing the config's mean calls (applied per
        seed with weights from the mean calls), and `d_acc_lower` / `d_acc_upper` (+ `d_nmi_*`) with
        `lower_n` / `upper_n` against those points alone. A bracketing point lacking any compared seed is
        n/a with a warning (`fixed_notes`). Outside the fixed family's calls range the missing side and the
        interpolation are NaN. `bracket=(lower_n, upper_n)` fixes the points instead of choosing them from
        the config's calls on `seeds`; the chord weight is then clamped to [0, 1].
    """
    seeds = set(seeds)
    fixed5 = _seed_means(per_seed, "fixed-n5", seeds)
    records = []
    for variant in [v for v in per_seed["variant"].unique() if _family(v) == "adaptive"]:
        if variants is not None and variant not in variants:
            continue
        mine = _seed_means(per_seed, variant, seeds)
        if mine.empty:
            continue
        common = fixed5.index.intersection(mine.index)
        rec: dict = {"variant": variant, "seeds": _fmt_seeds(mine.index), "n_paired": len(common)}
        rec["calls"] = mine["calls"].mean()
        rec["paired5"] = tuple(sorted(int(x) for x in common))
        if len(common):
            a, f = mine.loc[common], fixed5.loc[common]
            _paired(rec, "d_acc5", a["acc"] - f["acc"])
            _paired(rec, "d_nmi5", a["nmi"] - f["nmi"])
            rec["ratio5"] = a["calls"].mean() / f["calls"].mean()
            rec["h2"] = h2_verdict(f["acc"].mean(), a["acc"].mean(), f["calls"].mean(), a["calls"].mean())
        rand = _seed_means(per_seed, random_method(variant), seeds)
        both = rand.index.intersection(mine.index)
        if len(both):
            a, r = mine.loc[both], rand.loc[both]
            _paired(rec, "d_acc_rand", a["acc"] - r["acc"])
            _paired(rec, "d_nmi_rand", a["nmi"] - r["nmi"])
            ratios = np.where(a["calls"] > 0, r["calls"] / a["calls"].where(a["calls"] > 0), 1.0)
            rec |= {
                "random_seeds": tuple(sorted(int(x) for x in both)),
                "summarized_adaptive": a["n_summarized"].mean(),
                "summarized_random": r["n_summarized"].mean(),
                "calls_adaptive": a["calls"].mean(),
                "calls_random": r["calls"].mean(),
                "calls_ratio_random": float(np.mean(ratios)),
                "calls_dev_max": float(np.max(np.abs(ratios - 1))),
            }
        if fixed_family:
            x = mine["calls"].mean()
            lower, upper, notes = _fixed_neighbours(per_seed, set(mine.index), x, bracket)
            rec["fixed_notes"] = "; ".join(notes)
            for side, point in (("lower", lower), ("upper", upper)):
                if point is not None:
                    rec[f"{side}_n"] = point["n"]
                    for metric in ("acc", "nmi"):
                        _paired(rec, f"d_{metric}_{side}", mine[metric] - point[metric])
            if lower is not None and upper is not None:
                span = upper["calls"] - lower["calls"]
                w = min(1.0, max(0.0, (x - lower["calls"]) / span)) if span > 0 else 0.0
                for metric in ("acc", "nmi"):
                    ref = (1 - w) * lower[metric] + w * upper[metric]
                    _paired(rec, f"d_{metric}_interp", mine[metric] - ref)
        records.append(rec)
    return pd.DataFrame(records)


def _signed(x: float, digits: int = 3) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:+.{digits}f}"


def _delta_cell(row: pd.Series, key: str, full: bool = False) -> str:
    """`+0.012 (sd 0.004; +0.007..+0.018)` for a paired-delta column group, `n/a` if absent. With `full` it
    adds the standard error and the 95% t-interval: `...; se 0.002; 95% CI [+0.008, +0.016])`."""
    if key not in row or pd.isna(row[key]):
        return "n/a"
    sd = row.get(f"{key}_std")
    parts = ["sd n/a" if sd is None or pd.isna(sd) else f"sd {sd:.3f}", f"{row[f'{key}_min']:+.3f}..{row[f'{key}_max']:+.3f}"]
    if full:
        se = row.get(f"{key}_se")
        parts.append("se n/a; 95% CI n/a" if se is None or pd.isna(se)
                     else f"se {se:.3f}; 95% CI [{row[f'{key}_ci_lo']:+.3f}, {row[f'{key}_ci_hi']:+.3f}]")
    return f"{row[key]:+.3f} ({'; '.join(parts)})"


def headline_variant(per_seed: pd.DataFrame) -> tuple[str, str] | None:
    """The Phase-D config (method name) and rule, recomputed from the per-seed rows on `SELECTION_SEEDS`
    exactly as the driver's `choose_headline` does."""
    sel = _restrict(per_seed, SELECTION_SEEDS)
    fixed5 = _seed_means(sel, "fixed-n5")
    stats = {}
    for variant in [v for v in per_seed["variant"].unique() if _family(v) == "adaptive"]:
        rows = _seed_means(sel, variant)
        if len(rows):
            stats[variant] = (rows["calls"].mean(), rows["acc"].mean())
    if fixed5.empty or not stats:
        return None
    return select_headline(stats, fixed5["calls"].mean())


_SEED_NOTE = (
    "Means are over each variant's own seeds (the `seeds` column): fixed-n and k-means start on seeds 0-4 "
    "(fixed-n5 and the fixed-n points bracketing the Phase-D config are extended to 5-9), adaptive configs and "
    "random controls on 0-2 (the Phase-D config and its controls also on 3-9). Compare variants on common seeds "
    "in `rq2_clinc150_matched.md`, not by eye across rows."
)


def _seed_cells(values) -> list[str]:
    """Seed cells: tuples of seeds are compacted, strings (already formatted) pass through, missing is n/a."""
    return [
        v if isinstance(v, str) else "n/a" if v is None or (not isinstance(v, tuple) and pd.isna(v)) else _fmt_seeds(v)
        for v in values
    ]


def matched_markdown(per_seed: pd.DataFrame) -> str:
    """The `rq2_clinc150_matched.md` text: in-sample paired comparisons on the selection seeds, then the
    held-out H2 evaluation of the Phase-D config."""
    sel_label = f"selection seeds {_fmt_seeds(SELECTION_SEEDS)} (in-sample)"
    held_label = f"held-out seeds {_fmt_seeds(HELDOUT_SEEDS)}"
    m = matched_comparison(per_seed, SELECTION_SEEDS)
    if m.empty:
        return "# RQ2 matched comparisons\n\nNo adaptive runs found on the selection seeds.\n"

    def col(name: str) -> pd.Series:
        return m[name] if name in m else pd.Series([np.nan] * len(m), index=m.index)

    def cells(key: str) -> list[str]:
        return [_delta_cell(r, key) for _, r in m.iterrows()]

    def table(seeds: str, columns: dict[str, list]) -> str:
        """`seeds` names the column holding the seeds actually paired for this comparison."""
        return to_markdown(pd.DataFrame({"Config": m["variant"], "paired seeds": _seed_cells(col(seeds)), **columns}))

    ratio = lambda series: ["n/a" if pd.isna(x) else f"{x:.2f}" for x in series]  # noqa: E731
    neighbour = lambda side: [  # noqa: E731
        "n/a" if pd.isna(d) else f"{_signed(d)} (n={int(n)})"
        for n, d in zip(col(f"{side}_n"), col(f"d_acc_{side}"))
    ]
    parts = [
        "# RQ2 matched comparisons",
        "All deltas are adaptive minus reference over the seeds actually paired (listed per table), shown as "
        "`mean (sd across seeds; min..max)`. The adaptive configs were compared and one was picked on seeds "
        f"{_fmt_seeds(SELECTION_SEEDS)}, so everything in the first part is in-sample and optimistically biased for "
        f"the winner; the H2 verdict is judged on {held_label} in the last section.",
        f"# Part 1: {sel_label}",
        "## vs fixed-n5 (default schedule)",
        table("paired5", {"ΔACC": cells("d_acc5"), "ΔNMI": cells("d_nmi5"), "calls / fixed-n5": ratio(col("ratio5"))}),
        "## vs random control matched on calls",
        "The control is matched on summarizer calls, not on summary resets: it is given the adaptive run's "
        "per-step calls and spends them on random clusters among those whose membership changed. The adaptive "
        "schedule may also re-apply cached summaries of unchanged clusters for free (drift- or `s_max`-triggered "
        "resets), which the control cannot, so `summarized` can differ while calls match. Calls ratios are computed "
        f"per seed; the flag fires when any seed deviates from 1 by more than {BUDGET_MISMATCH:.0%}.",
        table(
            "random_seeds",
            {
                "ΔACC": cells("d_acc_rand"),
                "ΔNMI": cells("d_nmi_rand"),
                "summarized (adaptive / random)": [
                    "n/a" if pd.isna(a) else f"{a:.0f} / {r:.0f}"
                    for a, r in zip(col("summarized_adaptive"), col("summarized_random"))
                ],
                "calls (adaptive / random)": [
                    "n/a" if pd.isna(a) else f"{a:.0f} / {r:.0f}" for a, r in zip(col("calls_adaptive"), col("calls_random"))
                ],
                "random / adaptive calls": ["n/a" if pd.isna(x) else f"{x:.3f}" for x in col("calls_ratio_random")],
                "max per-seed deviation": ["n/a" if pd.isna(x) else f"{x:.1%}" for x in col("calls_dev_max")],
                "budget": [
                    "n/a" if pd.isna(x) else ("MISMATCH" if x > BUDGET_MISMATCH else "ok") for x in col("calls_dev_max")
                ],
            },
        ),
        "## vs the fixed family at matched calls",
        _FIXED_FAMILY_NOTE,
        table(
            "seeds",
            {
                "mean calls": [f"{x:.0f}" for x in m["calls"]],
                "ΔACC interp.": [_signed(x) for x in col("d_acc_interp")],
                "ΔNMI interp.": [_signed(x) for x in col("d_nmi_interp")],
                "ΔACC vs lower neighbour": neighbour("lower"),
                "ΔACC vs upper neighbour": neighbour("upper"),
            },
        ),
        f"## H2 per config, {sel_label}",
        f"H2 holds for a config if fixed-n5's ACC minus the adaptive ACC is at most {ACC_TOLERANCE:.2f} "
        f"(within one ACC point, or better) and its calls are at most {CALLS_BUDGET:.0%} of fixed-n5's.",
        table(
            "paired5",
            {
                "ΔACC": cells("d_acc5"),
                "calls / fixed-n5": ratio(col("ratio5")),
                "H2": ["n/a" if pd.isna(x) else ("yes" if x else "no") for x in col("h2")],
            },
        ),
        f"# Part 2: {held_label}",
        _heldout_markdown(per_seed),
    ]
    return "\n\n".join(parts) + "\n"


_FIXED_FAMILY_NOTE = (
    "Interpolation: the two fixed-n points bracketing the adaptive config's mean calls are joined by a chord and "
    "evaluated at those calls (per seed). Caveat: accuracy vs calls is typically concave, so a chord lies below "
    "the true curve and the interpolated delta can favour the adaptive config; the deltas against the bracketing "
    "fixed-n points alone show the same comparison without interpolating. A bracketing point lacking any compared "
    "seed is n/a (no partial-seed averaging), and below fixed-n1's or above fixed-n5's calls the missing side is n/a. "
    "In the held-out section the bracketing points are the ones the planner extended to seeds 5-9, chosen from "
    "the config's calls on the selection seeds; if its held-out calls fall outside their range the chord is "
    "clamped to the nearer point."
)


def _missing_note(wanted, paired) -> str:
    missing = sorted(set(wanted) - set(paired))
    return f"{len(set(wanted) & set(paired))} of {len(wanted)} held-out seeds paired: {_fmt_seeds(paired) if paired else 'none'}; missing {_fmt_seeds(missing)}"


def _heldout_markdown(per_seed: pd.DataFrame) -> str:
    """Part 2: the verdict on the held-out seeds. A verdict is issued only if every held-out seed is paired
    with fixed-n5; otherwise the section says "incomplete" and lists which seeds were paired."""
    held_label = f"held-out seeds {_fmt_seeds(HELDOUT_SEEDS)}"
    headline = headline_variant(per_seed)
    if headline is None:
        return "**Overall H2:** n/a (needs fixed-n5 and adaptive runs on the selection seeds to choose the config)."
    variant, rule = headline
    chosen = f"Config `{variant}` was chosen on seeds {_fmt_seeds(SELECTION_SEEDS)}: {rule}."
    held = matched_comparison(per_seed, HELDOUT_SEEDS, [variant], bracket=_planned_bracket(per_seed, variant))
    if held.empty or "h2" not in held:
        return (f"{chosen}\n\n**Overall H2:** n/a (the held-out runs of `{variant}` and fixed-n5 on seeds "
                f"{_fmt_seeds(HELDOUT_SEEDS)} are not complete).")
    row = held.iloc[0]
    wanted = set(HELDOUT_SEEDS)
    paired5 = set(row["paired5"])
    criterion = f"criterion: fixed-n5 ACC − adaptive ACC ≤ {ACC_TOLERANCE:.2f} and calls ≤ {CALLS_BUDGET:.0%}"
    stats_line = (f"ΔACC vs fixed-n5 {_delta_cell(row, 'd_acc5', full=True)} at {row['ratio5']:.2f} of its calls "
                  f"({criterion})")
    lines = [chosen]
    if wanted <= paired5:
        verdict = "SUPPORTED" if row["h2"] else "NOT SUPPORTED"
        lines.append(f"**Overall H2 ({held_label}, paired seeds {_fmt_seeds(paired5)}): {verdict}.** {stats_line}.")
        lo = row.get("d_acc5_ci_lo")
        if lo is None or pd.isna(lo):
            lines.append("95% interval check: n/a (fewer than two paired seeds).")
        else:
            above = lo > -ACC_TOLERANCE
            lines.append(
                f"Additional check (separate from the verdict above): the lower end of the 95% t-interval of ΔACC "
                f"(adaptive − fixed-n5) is {lo:+.3f}, which is {'above' if above else 'not above'} {-ACC_TOLERANCE:+.2f}, "
                f"so \"within one ACC point\" is {'also' if above else 'not'} supported at 95%."
            )
    else:
        lines.append(
            f"**Overall H2: incomplete** ({_missing_note(wanted, paired5)}); no verdict. "
            f"Provisional, over the paired seeds only: {stats_line}."
        )
    rand_seeds = set(row["random_seeds"]) if "random_seeds" in row and isinstance(row["random_seeds"], tuple) else set()
    if rand_seeds:
        text = (f"ΔACC {_delta_cell(row, 'd_acc_rand', full=True)}, ΔNMI {_delta_cell(row, 'd_nmi_rand', full=True)}, "
                f"max per-seed calls deviation {row['calls_dev_max']:.1%}")
        if wanted <= rand_seeds:
            lines.append(f"vs the call-matched random control (paired seeds {_fmt_seeds(rand_seeds)}): {text}.")
        else:
            lines.append(f"vs the call-matched random control: **incomplete** ({_missing_note(wanted, rand_seeds)}); "
                         f"provisional {text}.")
    else:
        lines.append("vs the call-matched random control: no held-out random-control runs to pair with.")

    lines.append(f"### vs the fixed family at matched calls, {held_label}")
    lines.append(_FIXED_FAMILY_NOTE)
    lines.append(_heldout_fixed_family(row))
    lines.append(
        f"For reference, on the selection seeds {_fmt_seeds(SELECTION_SEEDS)} (in-sample, optimistically biased): "
        + _in_sample_line(per_seed, variant)
    )
    sub = _restrict(per_seed, HELDOUT_SEEDS)
    sub = sub[sub["variant"].isin(["fixed-n5", variant, random_method(variant)])]
    return "\n\n".join(lines) + "\n\n" + to_markdown(format_summary_table(summary_table(sub)))


def _planned_bracket(per_seed: pd.DataFrame, variant: str) -> tuple[int | None, int | None] | None:
    """The fixed-n points the planner extended for `variant`: `bracket_pair` of its mean calls vs fixed-n mean
    calls, both on the selection seeds (the planner's inputs), not on the held-out seeds."""
    sel = _restrict(per_seed, SELECTION_SEEDS)
    mine = _seed_means(sel, variant)
    fixed = {
        int(v.removeprefix("fixed-n")): _seed_means(sel, v)["calls"].mean()
        for v in sel["variant"].unique()
        if _family(v) == "fixed"
    }
    return bracket_pair(fixed, mine["calls"].mean()) if len(mine) and fixed else None


def _heldout_fixed_family(row: pd.Series) -> str:
    """The headline's held-out ΔACC / ΔNMI vs the interpolated fixed family and each bracketing fixed-n."""
    cells = []
    for side in ("lower", "upper"):
        n = row.get(f"{side}_n")
        if n is not None and not pd.isna(n):
            cells.append((f"vs fixed-n{int(n)} ({side} neighbour)", f"d_acc_{side}", f"d_nmi_{side}"))
    if "d_acc_interp" in row and not pd.isna(row["d_acc_interp"]):
        cells.insert(0, ("vs interpolated fixed family", "d_acc_interp", "d_nmi_interp"))
    if not cells:
        notes = row.get("fixed_notes") or "the bracketing fixed-n points are outside the fixed family's calls range"
        return f"**Incomplete / n/a:** {notes}. The fixed-n points bracketing the held-out calls need runs on all of seeds {_fmt_seeds(HELDOUT_SEEDS)}."
    lines = [f"- {label}: ΔACC {_delta_cell(row, a, True)}; ΔNMI {_delta_cell(row, n, True)}" for label, a, n in cells]
    if row.get("fixed_notes"):
        lines.append(f"- n/a for the other side or the interpolation: {row['fixed_notes']}.")
    elif len(cells) < 3:
        lines.append("- the interpolation and the other neighbour are n/a (the held-out calls are outside the fixed family's range).")
    return "\n".join(lines)


def _in_sample_line(per_seed: pd.DataFrame, variant: str) -> str:
    sel = matched_comparison(per_seed, SELECTION_SEEDS, [variant], fixed_family=False)
    if sel.empty or "d_acc5" not in sel:
        return "n/a."
    r = sel.iloc[0]
    return f"ΔACC vs fixed-n5 {_delta_cell(r, 'd_acc5')} at {r['ratio5']:.2f} of its calls (paired seeds {_fmt_seeds(r['paired5'])})."


# Okabe-Ito colorblind-safe palette
_COLORS = {"fixed": "#0072B2", "adaptive": "#D55E00", "random": "#009E73", "k-means": "#666666"}


def merged_labels(rows: pd.DataFrame, ymean: str, digits: int = 6) -> list[tuple[float, float, str]]:
    """`(x, y, label)` per distinct point of adaptive rows; configs landing on the same point are merged
    into one label like `d0.3-r0.15 = d0.6-r0.15`."""
    groups: dict[tuple, list[str]] = {}
    for _, r in rows.iterrows():
        key = (round(r["calls_mean"], digits), round(r[ymean], digits))
        groups.setdefault(key, []).append(r["variant"].removeprefix("adaptive-"))
    return [(x, y, " = ".join(names)) for (x, y), names in groups.items()]


def plot_pareto(numeric: pd.DataFrame, path: str | Path, seeds_label: str = "") -> None:
    """ACC vs calls and NMI vs calls (mean ± std over seeds): fixed family as a labelled line, adaptive
    configs as labelled points (coincident ones share a label), random controls as hollow points, k-means as
    a dashed horizontal line. Pass the table of the common seeds; `seeds_label` goes in the title.
    matplotlib is imported here so the package works without the `plot` extra."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, metric in zip(axes, ("acc", "nmi")):
        ymean, ystd = f"{metric}_mean", f"{metric}_std"

        def draw(rows: pd.DataFrame, family: str, **kw) -> None:
            ax.errorbar(
                rows["calls_mean"],
                rows[ymean],
                xerr=rows["calls_std"].fillna(0),
                yerr=rows[ystd].fillna(0),
                color=_COLORS[family],
                ecolor=_COLORS[family],
                elinewidth=1,
                capsize=2,
                **kw,
            )

        fixed = numeric[numeric["family"] == "fixed"]
        if len(fixed):
            draw(fixed, "fixed", marker="o", linestyle="-", label="fixed (n = 1..5)")
            for _, r in fixed.iterrows():
                ax.annotate(
                    "n=" + r["variant"].removeprefix("fixed-n"),
                    (r["calls_mean"], r[ymean]),
                    textcoords="offset points",
                    xytext=(5, -12),
                    fontsize=8,
                    color=_COLORS["fixed"],
                )
        adaptive = numeric[numeric["family"] == "adaptive"]
        if len(adaptive):
            draw(adaptive, "adaptive", marker="D", linestyle="none", label="adaptive")
            for x, y, label in merged_labels(adaptive, ymean):
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    color=_COLORS["adaptive"],
                )
        rand = numeric[numeric["family"] == "random"]
        if len(rand):
            draw(rand, "random", marker="s", linestyle="none", markerfacecolor="none", label="random control")
        km = numeric[numeric["family"] == "k-means"]
        if len(km):
            ax.axhline(km[ymean].iloc[0], color=_COLORS["k-means"], linestyle="--", linewidth=1, label="k-means")
        ax.set_xlabel(f"summarizer calls (mean ± std over {seeds_label or 'seeds'})")
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"RQ2 on CLINC150: every variant on the common {seeds_label or 'seeds'} (selection seeds, in-sample)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    results_dir: str | Path,
    out_dir: str | Path,
    base: dict,
    overrides: list[str] | None = None,
    plot: bool = True,
) -> list[Path]:
    """Write the table (md + csv), per-seed csv, matched comparison and (if `plot`) Pareto plot to `out_dir`.

    The runs used are the planned runs of phases A-D built from `base` (+ `overrides`) exactly as the
    driver builds them, each identity-checked; other `rq2_*` dirs are warned about and ignored. The Pareto
    plot shows the selection seeds only, so every variant is averaged over the same seeds. Returns the
    written paths."""
    runs, notes = planned_run_dirs(base, results_dir, overrides)
    for note in notes:
        warnings.warn(note, stacklevel=2)
    per_seed = collect_per_seed(runs)
    if per_seed.empty:
        raise ValueError(f"no completed rq2_* runs under {results_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numeric = summary_table(per_seed)
    paths = {
        "table": out / "rq2_clinc150_table.md",
        "csv": out / "rq2_clinc150_table.csv",
        "per_seed": out / "rq2_clinc150_per_seed.csv",
        "matched": out / "rq2_clinc150_matched.md",
    }
    paths["table"].write_text(to_markdown(format_summary_table(numeric)) + "\n\n" + _SEED_NOTE + "\n")
    numeric.to_csv(paths["csv"], index=False)
    per_seed.to_csv(paths["per_seed"], index=False)
    paths["matched"].write_text(matched_markdown(per_seed))
    written = list(paths.values())
    if plot:
        png = out / "rq2_clinc150_pareto.png"
        plot_pareto(summary_table(_restrict(per_seed, SELECTION_SEEDS)), png, f"seeds {_fmt_seeds(SELECTION_SEEDS)}")
        written.append(png)
    return written
