"""Writing experiment outputs.

Every run gets its own directory, `<results_dir>/<name>/<timestamp>/`, containing

- `config.yaml`      the fully resolved config (after `--set` overrides),
- `environment.json` once per run: timestamp, git commit and dirty flag, python / OS / WSL, torch, CUDA,
                     GPU and driver, package versions, the summarizer's description (model, dtype, batch
                     size, padding side, token limits, ...), embedder and dataset,
- `metrics.jsonl`    one JSON object per (seed, method[, year]) with the metrics from `metrics.cluster_metrics`
                     plus `runtime_s` and `step0_sha1` (sha1 of the step-0 labels as int64: equal across methods
                     for a seed iff they share the initialization). The summary-centroids method's row also has
                     the cost fields: `n_calls`, `input_tokens`, `output_tokens`, `summarizer_seconds` (time inside
                     `summarize_batch`), `n_empty_summaries`, `n_truncated_clusters`, `n_dropped_texts`
                     (per-seed totals, including a final summary step that stopped on `tol`) and
                     `peak_gpu_mem_gb` / `peak_gpu_reserved_gb` (None without CUDA),
- `summaries.jsonl`  final cluster summaries (and their evolution) and the final cluster assignments
                     (one id per document) per (seed, method[, year]). Offline runs also carry
                     `assignments_evolution` (labels after every KMeans fit; entry `t` belongs to trace row
                     `t`) and `summary_source_steps` (per entry of `summaries_evolution`, the trace step whose
                     assignments it was written from): summary step `j` was written from
                     `assignments_evolution[summary_source_steps[j]]` and its Lloyd block ended in
                     `assignments_evolution[summary_source_steps[j] + 1]` (absent for a final step that
                     stopped on `tol`, which fits no KMeans),
- `trace.jsonl`      one row per KMeans fit per (seed, method) from `Result.trace` (inertia, label churn,
                     summary shift, ...) and that step's cost deltas (`input_tokens`, `output_tokens`,
                     `summarizer_seconds`, `n_empty_summaries`, `n_truncated_clusters`, `n_dropped_texts`; 0 on
                     non-summary rows), written only for summary-centroids methods,

and `summary.csv` with mean / std over seeds is written when the run finishes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class RunDir:
    """Append-only writer for one experiment run."""

    def __init__(self, results_dir: str | Path, name: str, config: dict):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(results_dir) / name / stamp
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        self.rows: list[dict] = []

    def log_environment(self, environment: dict) -> None:
        """Write `environment.json` (see `environment.collect_environment`)."""
        with open(self.path / "environment.json", "w") as f:
            json.dump(_to_jsonable(environment), f, indent=2)
            f.write("\n")

    def log_metrics(self, row: dict) -> None:
        """Record one result row, e.g. `{"seed": 0, "method": "kmeans", "nmi": ..., ...}`."""
        row = _to_jsonable(row)
        self.rows.append(row)
        with open(self.path / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def log_summaries(self, row: dict) -> None:
        """Record summaries, e.g. `{"seed": 0, "method": "k-NLPmeans", "summaries": [...]}`."""
        with open(self.path / "summaries.jsonl", "a") as f:
            f.write(json.dumps(_to_jsonable(row)) + "\n")

    def log_trace(self, key: dict, trace: list[dict]) -> None:
        """Append one `trace.jsonl` line per trace row, each `{**key, **row}`; no file if `trace` is empty."""
        if not trace:
            return
        with open(self.path / "trace.jsonl", "a") as f:
            for row in trace:
                f.write(json.dumps(_to_jsonable({**key, **row})) + "\n")

    def finish(self, group_by: list[str] | None = None, metrics: list[str] | None = None) -> pd.DataFrame:
        """Write `summary.csv` (mean/std over seeds) and return it for printing."""
        group_by = group_by or ["method"]
        metrics = metrics or ["acc", "nmi", "ari"]
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        # Columns only some methods report (summarizer cost on baseline rows) are NaN/None elsewhere.
        df[metrics] = df[metrics].apply(pd.to_numeric, errors="coerce")
        summary = df.groupby(group_by, sort=False)[metrics].agg(["mean", "std"])
        summary.to_csv(self.path / "summary.csv")
        return summary
