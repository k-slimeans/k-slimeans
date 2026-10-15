"""Writing experiment outputs.

Every run gets its own directory, `<results_dir>/<name>/<timestamp>/`, containing

- `config.yaml`      the fully resolved config (after `--set` overrides),
- `metrics.jsonl`    one JSON object per (seed, method[, year]) with the metrics from `metrics.cluster_metrics`,
- `summaries.jsonl`  final cluster summaries (and their evolution) and the final cluster assignments
                     (one id per document) per (seed, method[, year]),
- `trace.jsonl`      one row per KMeans fit per (seed, method) from `Result.trace` (inertia, label churn,
                     summary shift, ...), written only for summary-centroids methods,

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
        summary = df.groupby(group_by, sort=False)[metrics].agg(["mean", "std"])
        summary.to_csv(self.path / "summary.csv")
        return summary
