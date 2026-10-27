"""Combine several run directories into one comparison table (and qualitative examples).

Each run directory (see `results.RunDir`) holds one summary-centroids method plus the baselines
configured for that run. `aggregate_runs` reads `config.yaml` + `metrics.jsonl` from each, labels
the rows, and reports mean +/- sample std over seeds. `scripts/aggregate_results.py` is the CLI
around it. `paired_differences` / `format_paired_markdown` give per-seed paired differences against
reference methods, and `cost_table` / `format_cost_table` / `step0_sha1_summary` report summarizer
cost (calls, tokens, time, memory) and run settings. `find_examples` reads `summaries.jsonl` to show, per gold class, which cluster best
matches it and what that cluster's final summary says.

Row labels: baselines keep their method name (`kmeans` is shown as `k-means`); the run's own
method is `<method_name> (<model basename>)` for the `hf` summarizer and `<method_name> (<TYPE>)`
for extractive ones, e.g. `k-SLMmeans (Qwen3-0.6B)` / `k-NLPmeans (LSA)`.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .config import default_method_name
from .experiment import _resolve_dataset

METRICS = ("acc", "nmi", "ari")
_DISPLAY_NAMES = {"kmeans": "k-means"}
_SAME_TOL = 1e-9
DEFAULT_REFERENCES = ("k-NLPmeans (LSA)", "k-means")
# Per-seed cost fields of metrics.jsonl: (column, display header, format spec).
_COST_FIELDS = (
    ("n_calls", "calls", ",.0f"),
    ("input_tokens", "input tokens", ",.0f"),
    ("output_tokens", "output tokens", ",.0f"),
    ("summarizer_seconds", "summarizer s", ".1f"),
    ("runtime_s", "total s", ".1f"),
    ("n_empty_summaries", "empty summaries", ".1f"),
    ("n_truncated_clusters", "truncated clusters", ".1f"),
    ("n_dropped_texts", "dropped texts", ".1f"),
    ("peak_gpu_mem_gb", "peak GPU GB", ".2f"),
    ("peak_gpu_reserved_gb", "reserved GPU GB", ".2f"),
)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_config(run_dir: Path) -> dict:
    with open(run_dir / "config.yaml") as f:
        return yaml.safe_load(f)


def method_label(cfg: dict) -> str:
    """Display label of the summary-centroids method of a run, from its config."""
    summarizer = cfg["summarizer"]
    name = cfg.get("method_name") or default_method_name(summarizer)
    if summarizer.get("type") == "hf":
        detail = str(summarizer["model_id"]).rstrip("/").split("/")[-1]
    else:
        detail = str(summarizer.get("type", "")).upper()
    return f"{name} ({detail})"


def _own_name(cfg: dict) -> str:
    return cfg.get("method_name") or default_method_name(cfg["summarizer"])


def _labelled_rows(run_dir: Path, cfg: dict) -> pd.DataFrame:
    """The run's metrics rows with `method` replaced by its display label."""
    own = _own_name(cfg)
    rows = pd.DataFrame(_read_jsonl(run_dir / "metrics.jsonl"))
    rows["method"] = [
        method_label(cfg) if m == own else _DISPLAY_NAMES.get(m, m) for m in rows["method"]
    ]
    return rows


def _warn_if_incomparable(run_dir: Path, cfg: dict, first_dir: Path, first_cfg: dict) -> None:
    for key in ("dataset", "embedder"):
        if cfg.get(key) != first_cfg.get(key):
            warnings.warn(
                f"{run_dir} has {key} {cfg.get(key)!r}, but {first_dir} has {first_cfg.get(key)!r}; "
                "the rows are not comparable",
                stacklevel=3,
            )


def _collect_runs(run_dirs: list[str | Path]) -> dict[str, pd.DataFrame]:
    """Per-seed metrics of every method, keyed by display label (order of first appearance).

    Baselines repeat in every run dir, so the first occurrence of a label wins; a later copy whose
    per-seed metrics or seed set differ is warned about, since identical seeds and embeddings
    should agree. A run's own method is never dropped: two run dirs with the same own label raise
    `ValueError`. Differing `dataset` / `embedder` configs, and row groups whose seeds differ from
    the first run's own-method seeds (partial or crashed runs), are warned about.
    """
    kept: dict[str, pd.DataFrame] = {}
    own_dirs: dict[str, Path] = {}
    first_dir: Path | None = None
    first_cfg: dict = {}
    reference_seeds: set | None = None
    for run_dir in map(Path, run_dirs):
        cfg = _load_config(run_dir)
        if first_dir is None:
            first_dir, first_cfg = run_dir, cfg
        else:
            _warn_if_incomparable(run_dir, cfg, first_dir, first_cfg)
        own_label = method_label(cfg)
        rows = _labelled_rows(run_dir, cfg)
        if own_label in set(rows["method"]) and reference_seeds is None:
            reference_seeds = set(rows.loc[rows["method"] == own_label, "seed"])
        for label, group in rows.groupby("method", sort=False):
            other = group.set_index("seed")
            if reference_seeds is not None and set(other.index) != reference_seeds:
                warnings.warn(
                    f"'{label}' in {run_dir} has seeds {sorted(other.index)}, expected "
                    f"{sorted(reference_seeds)} (partial or crashed run?)",
                    stacklevel=2,
                )
            if label == own_label:
                if label in own_dirs:
                    raise ValueError(
                        f"{own_dirs[label]} and {run_dir} both produce method '{label}'; "
                        "aggregate runs with distinct methods"
                    )
                own_dirs[label] = run_dir
                kept[label] = other
                continue
            if label not in kept:
                kept[label] = other
                continue
            first = kept[label]
            if set(first.index) != set(other.index):
                warnings.warn(
                    f"'{label}' in {run_dir} has seeds {sorted(other.index)}, but the first "
                    f"occurrence has {sorted(first.index)}; keeping the first occurrence",
                    stacklevel=2,
                )
            common = first.index.intersection(other.index)
            diff = (first.loc[common, list(METRICS)] - other.loc[common, list(METRICS)]).abs()
            if (diff > _SAME_TOL).any().any():
                warnings.warn(
                    f"'{label}' differs between runs (max |diff| {diff.to_numpy().max():.3g}) in "
                    f"{run_dir}; keeping the first occurrence",
                    stacklevel=2,
                )
    return kept


def aggregate_runs(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Numeric table, one row per method in order of first appearance.

    Columns: `method`, `<metric>_mean` / `<metric>_std` for acc/nmi/ari, `runtime_s` (mean seconds
    per seed; NaN when the runs have no `runtime_s`) and `n_seeds` (seeds with metrics). std is the
    sample std (ddof=1), as in `summary.csv`; NaN for one seed. Baseline de-duplication and the
    comparability warnings are described in `_collect_runs`.
    """
    kept = _collect_runs(run_dirs)
    records = []
    for label, group in kept.items():
        record = {"method": label}
        for metric in METRICS:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std()
        record["runtime_s"] = group["runtime_s"].mean() if "runtime_s" in group else np.nan
        record["n_seeds"] = int(group[list(METRICS)].notna().all(axis=1).sum())
        records.append(record)
    return pd.DataFrame(records)


def per_seed_table(run_dirs: list[str | Path]) -> pd.DataFrame:
    """One row per (method, seed): `method, seed, acc, nmi, ari, runtime_s`, labelled as in `aggregate_runs`.

    Use it for paired comparisons: every method of a run shares the seed. `runtime_s` is NaN
    when a run has none.
    """
    frames = []
    for label, group in _collect_runs(run_dirs).items():
        frame = group.drop(columns="method").reset_index()
        if "runtime_s" not in frame:
            frame["runtime_s"] = np.nan
        frame.insert(0, "method", label)
        frames.append(frame[["method", "seed", *METRICS, "runtime_s"]])
    return pd.concat(frames, ignore_index=True)


def format_table(numeric: pd.DataFrame) -> pd.DataFrame:
    """Display table: `Method | ACC | NMI | ARI | s/seed | seeds` with `mean ± std` strings."""

    def cell(mean: float, std: float) -> str:
        return f"{mean:.3f} ± " + ("n/a" if np.isnan(std) else f"{std:.3f}")

    out = pd.DataFrame({"Method": numeric["method"]})
    for metric in METRICS:
        out[metric.upper()] = [cell(m, s) for m, s in zip(numeric[f"{metric}_mean"], numeric[f"{metric}_std"])]
    out["s/seed"] = ["n/a" if np.isnan(t) else f"{t:.1f}" for t in numeric["runtime_s"]]
    out["seeds"] = [str(n) for n in numeric["n_seeds"]]
    return out


def to_markdown(table: pd.DataFrame) -> str:
    """Render a string-valued DataFrame as a GitHub markdown table (no `tabulate` needed)."""
    header = "| " + " | ".join(table.columns) + " |"
    divider = "| " + " | ".join(["---"] + ["---:"] * (len(table.columns) - 1)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in table.astype(str).itertuples(index=False)]
    return "\n".join([header, divider, *body])


def paired_differences(
    per_seed: pd.DataFrame, references: list[str], metrics: tuple[str, ...] = METRICS
) -> pd.DataFrame:
    """Paired per-seed differences of every method against each reference method.

    `per_seed` is `per_seed_table`'s output (`method`, `seed`, metric columns). For each reference,
    every other method (table order) and metric, over the seeds present for both: `n`,
    `mean_diff` (method minus reference), `sd` (sample std, ddof=1), the 95% Student-t interval
    `ci_low` / `ci_high` of the mean (NaN when n < 2), `wins` (seeds where the method is strictly
    higher) and `ties`. Rows are ordered by reference, method, metric. Raises `ValueError` for a
    reference that is not in `per_seed`.
    """
    methods = list(dict.fromkeys(per_seed["method"]))
    rows = []
    for reference in references:
        if reference not in methods:
            raise ValueError(f"Unknown reference method '{reference}', expected one of {methods}")
        ref = per_seed[per_seed["method"] == reference].set_index("seed")
        for method in methods:
            if method == reference:
                continue
            other = per_seed[per_seed["method"] == method].set_index("seed")
            for metric in metrics:
                diff = (other[metric] - ref[metric]).dropna()  # index alignment keeps common seeds only
                n = len(diff)
                mean = diff.mean() if n else np.nan
                sd = diff.std(ddof=1) if n > 1 else np.nan
                half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
                rows.append(
                    {
                        "reference": reference,
                        "method": method,
                        "metric": metric,
                        "n": n,
                        "mean_diff": mean,
                        "sd": sd,
                        "ci_low": mean - half,
                        "ci_high": mean + half,
                        "wins": int((diff > 0).sum()),
                        "ties": int((diff == 0).sum()),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["reference", "method", "metric", "n", "mean_diff", "sd", "ci_low", "ci_high", "wins", "ties"],
    )


def format_paired_markdown(paired: pd.DataFrame) -> str:
    """One markdown table per reference: `Method | Metric | mean diff | 95% CI | wins/n`.

    Differences are signed (method minus reference) with 3 decimals; the CI is `n/a` for n < 2.
    """

    def signed(x: float) -> str:
        return "n/a" if np.isnan(x) else f"{x:+.3f}"

    sections = []
    for reference, group in paired.groupby("reference", sort=False):
        table = pd.DataFrame(
            {
                "Method": group["method"],
                "Metric": group["metric"].str.upper(),
                "mean diff": [signed(m) for m in group["mean_diff"]],
                "95% CI": [
                    "n/a" if np.isnan(lo) else f"[{signed(lo)}, {signed(hi)}]"
                    for lo, hi in zip(group["ci_low"], group["ci_high"])
                ],
                "wins/n": [f"{w}/{n}" for w, n in zip(group["wins"], group["n"])],
            }
        )
        sections.append(f"Paired difference vs {reference} (method minus reference):\n\n{to_markdown(table)}")
    return "\n\n".join(sections)


def _read_environment(run_dir: Path) -> dict:
    path = run_dir / "environment.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def cost_table(run_dirs: list[str | Path]) -> pd.DataFrame:
    """One row per method (order of first appearance) with mean/std over seeds of the cost fields.

    Columns: `method`, `n_seeds`, `<field>_mean` / `<field>_std` for each of n_calls, input_tokens,
    output_tokens, summarizer_seconds, runtime_s, n_empty_summaries, n_truncated_clusters,
    n_dropped_texts, peak_gpu_mem_gb and peak_gpu_reserved_gb (std is the sample std, ddof=1), and
    `batch_size`, `dtype`, `gpu` from the run's `environment.json`. A field a method never logged
    (baselines, runs that predate cost logging) is NaN; the environment columns are missing (NaN) for
    baselines and for runs without `environment.json`.
    """
    own_env = {}
    for run_dir in map(Path, run_dirs):
        env = _read_environment(run_dir)
        own_env[method_label(_load_config(run_dir))] = env
    records = []
    for label, group in _collect_runs(run_dirs).items():
        env = own_env.get(label, {})
        summarizer = env.get("summarizer") or {}
        record = {"method": label, "n_seeds": len(group)}
        for field, _, _ in _COST_FIELDS:
            values = pd.to_numeric(group[field], errors="coerce") if field in group else pd.Series(dtype=float)
            record[f"{field}_mean"] = values.mean() if values.notna().any() else np.nan
            record[f"{field}_std"] = values.std() if values.notna().sum() > 1 else np.nan
        record["batch_size"] = summarizer.get("batch_size")
        record["dtype"] = summarizer.get("dtype")
        record["gpu"] = env.get("gpu_name")
        records.append(record)
    return pd.DataFrame(records)


def format_cost_table(costs: pd.DataFrame) -> pd.DataFrame:
    """Display table of `cost_table`: `mean ± std` strings, `n/a` for missing values."""

    def cell(mean: float, std: float, spec: str) -> str:
        if np.isnan(mean):
            return "n/a"
        return f"{mean:{spec}} ± " + ("n/a" if np.isnan(std) else f"{std:{spec}}")

    def text(value) -> str:
        if pd.isna(value):
            return "n/a"
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)  # NaNs make ints float

    out = pd.DataFrame({"Method": costs["method"]})
    for field, header, spec in _COST_FIELDS:
        out[header] = [cell(m, s, spec) for m, s in zip(costs[f"{field}_mean"], costs[f"{field}_std"])]
    out["batch size"] = [text(v) for v in costs["batch_size"]]
    out["dtype"] = [text(v) for v in costs["dtype"]]
    out["GPU"] = [text(v) for v in costs["gpu"]]
    return out


def step0_sha1_summary(run_dirs: list[str | Path]) -> str:
    """One line saying whether `step0_sha1` (hash of the step-0 labels) agrees across all methods per seed.

    Only methods that logged the hash count; runs that predate it give a "not logged" line. For a
    shared k-means++ initialization every seed must have one distinct hash.
    """
    by_seed: dict[int, dict[str, str]] = {}
    for label, group in _collect_runs(run_dirs).items():
        if "step0_sha1" not in group:
            continue
        for seed, sha in group["step0_sha1"].dropna().items():
            by_seed.setdefault(int(seed), {})[label] = sha
    if not by_seed:
        return "step0_sha1 not logged in these runs."
    seeds = sorted(by_seed)
    span = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {seeds[0]}-{seeds[-1]}"
    differ = [s for s in seeds if len(set(by_seed[s].values())) > 1]
    if not differ:
        return f"step0_sha1 shared across all methods for {span}: yes"
    return f"step0_sha1 shared across all methods for {span}: no (differs for seeds {', '.join(map(str, differ))})"


def _resolve_gold(gold: int | str, label_names: list[str] | None, n_classes: int) -> int:
    """Class id for `gold`: an int id, a class name (matched first), or a numeric string id."""
    if isinstance(gold, str):
        if label_names is not None and gold in label_names:
            return label_names.index(gold)
        if re.fullmatch(r"-?\d+", gold):
            gold = int(gold)
        elif label_names is None:
            raise ValueError(f"Gold class '{gold}' is a name, but this dataset has no label names; use an int id")
        else:
            raise ValueError(f"Unknown gold class name '{gold}', expected one of {label_names}")
    if not 0 <= gold < n_classes:
        raise ValueError(f"Gold class id {gold} is out of range for {n_classes} classes")
    return gold


def _load_seed_summaries(run_dir: Path, seed: int):
    """(config, dataset, summaries, assignments) of the run's summary method at `seed`."""
    cfg = _load_config(run_dir)
    own = _own_name(cfg)
    row = next(
        (r for r in _read_jsonl(run_dir / "summaries.jsonl") if r["method"] == own and r["seed"] == seed),
        None,
    )
    if row is None:
        raise ValueError(f"{run_dir}: no summaries for method '{own}' at seed {seed}")
    if "assignments" not in row:
        raise ValueError(
            f"{run_dir}: summaries.jsonl has no 'assignments' (run predates assignment logging); "
            "rerun the experiment to use --examples"
        )
    return cfg, _resolve_dataset(cfg["dataset"]), row["summaries"], np.asarray(row["assignments"])


def _gold_overlaps(labels: np.ndarray, assignments: np.ndarray, gold: int, n_clusters: int) -> np.ndarray:
    """Members of gold class `gold` per predicted cluster; `argmax` is the best-matching cluster."""
    return np.bincount(assignments[labels == gold], minlength=n_clusters)


def rank_examples(run_dirs: list[str | Path], seed: int = 0) -> list[tuple[str, float]]:
    """Gold classes ranked by how well *every* run recovers them, best first.

    A run's score for a class is overlap / gold class size of its best-matching cluster (the
    `find_examples` rule); a class scores the minimum over runs, so a high score means all methods
    recover it. Ties are broken by class id. Returns `(class name or id, score)`.
    """
    scores: dict[int, float] = {}
    names: list[str] | None = None
    for run_dir in map(Path, run_dirs):
        _, ds, summaries, assignments = _load_seed_summaries(Path(run_dir), seed)
        names = ds.label_names
        for g in range(ds.n_clusters):
            overlap = _gold_overlaps(ds.labels, assignments, g, len(summaries)).max()
            score = float(overlap / (ds.labels == g).sum())
            scores[g] = min(score, scores.get(g, score))
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(names[g] if names else str(g), score) for g, score in ranked]


def find_examples(run_dirs: list[str | Path], golds: list[int | str], seed: int = 0) -> str:
    """Markdown section per gold class: each run's best-matching cluster and its final summary.

    The best cluster is the predicted cluster holding the most members of the gold class, shown
    as `(overlap/gold docs, cluster size)`. Uses the run's summary method at `seed`.
    """
    sections: dict[int, list[str]] = {}
    header: dict[int, str] = {}
    for run_dir in map(Path, run_dirs):
        cfg, ds, summaries, assignments = _load_seed_summaries(run_dir, seed)
        label = method_label(cfg)
        # dict.fromkeys: one entry per class even if it is named twice (e.g. by id and by name)
        for g in dict.fromkeys(_resolve_gold(gold, ds.label_names, ds.n_clusters) for gold in golds):
            overlaps = _gold_overlaps(ds.labels, assignments, g, len(summaries))
            best = int(overlaps.argmax())
            size = int((assignments == best).sum())
            name = ds.label_names[g] if ds.label_names else str(g)
            header[g] = f"### {name} (class {g})"
            summary = " ".join(summaries[best].split())
            sections.setdefault(g, []).append(
                f"- **{label}**: {summary} ({overlaps[best]}/{int((ds.labels == g).sum())} gold docs, cluster size {size})"
            )
    return "\n\n".join(f"{header[g]}\n\n" + "\n".join(lines) for g, lines in sections.items())
