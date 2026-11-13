"""Tables for the BBC News write-up (docs/results/bbc.md) that the run summaries do not give directly.

For run directories of one variant (TF-IDF or dense):

- `--oov-md` / `--oov-csv`: per method, the summary tokens outside the TF-IDF vocabulary and the summaries whose TF-IDF
  vector is all zeros, summed over seeds and summary steps from the `embed_*` counters in `metrics.jsonl` (runs of a
  dense variant have no such counters and are skipped).
- `--partitions-md` / `--partitions-csv`: for every seed, the shared step-0 partition (taken from the first run that
  logged partitions) and each run's final partition, as topic counts per cluster against the topics in the ids file
  (`data/bbc500_ids.csv`, in sample order; no article text is read), with the Hungarian-matched accuracy.

Usage:

    python scripts/bbc_report.py results/bbc_tfidf_lsa/<stamp> results/bbc_tfidf_qwen3-0.6b/<stamp> ... \\
        --ids data/bbc500_ids.csv --oov-md docs/results/bbc_tfidf_oov.md --oov-csv docs/results/bbc_tfidf_oov.csv \\
        --partitions-md docs/results/bbc_tfidf_partitions.md --partitions-csv docs/results/bbc_tfidf_partitions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from kslimeans.runlog import load_run

MIN_SHOWN = 10  # a topic is named in a cluster's composition from this many members


def load_topics(ids_csv: str | Path) -> tuple[np.ndarray, list[str]]:
    """Gold labels in sample order and the sorted topic names. The ids file starts with a `#` hash line."""
    topics = pd.read_csv(ids_csv, comment="#")["topic"]
    names = sorted(topics.unique())
    return topics.map(names.index).to_numpy(), names


def oov_rows(run_dirs) -> list[dict]:
    """One row per run and method that logged the `embed_*` counters, summed over seeds."""
    rows = []
    for run_dir in run_dirs:
        runlog = load_run(run_dir)
        for method in runlog.methods:
            records = runlog.method_records(method)
            if not any("embed_tokens" in r.metrics for r in records):
                continue
            tokens = sum(r.metrics["embed_tokens"] for r in records)
            oov = sum(r.metrics["embed_oov_tokens"] for r in records)
            rows.append(
                {
                    "run": runlog.config["name"],
                    "method": method,
                    "summaries": sum(s is not None for r in records for step in r.summaries_evolution for s in step),
                    "tokens": tokens,
                    "oov_tokens": oov,
                    "oov_share": oov / tokens if tokens else float("nan"),
                    "zero_vectors": sum(r.metrics["embed_zero_vectors"] for r in records),
                }
            )
    return rows


def crosstab(labels: np.ndarray, gold: np.ndarray, k: int) -> np.ndarray:
    """`k x k` counts, rows = clusters, columns = gold topics."""
    if int(labels.max()) >= k:
        raise ValueError(f"labels use {int(labels.max()) + 1} clusters but k is {k}: expected one cluster per topic")
    counts = np.zeros((k, k), dtype=np.int64)
    np.add.at(counts, (labels, gold), 1)
    return counts


def matched_accuracy(counts: np.ndarray) -> float:
    rows, cols = linear_sum_assignment(-counts)
    return float(counts[rows, cols].sum() / counts.sum())


def composition(row: np.ndarray, names: list[str]) -> str:
    """`business 93 + tech 95` for the topics with at least MIN_SHOWN members, largest first."""
    shown = [(n, c) for n, c in sorted(zip(names, row), key=lambda x: -x[1]) if c >= MIN_SHOWN]
    return " + ".join(f"{n} {c}" for n, c in shown) if shown else f"mixed (no topic has {MIN_SHOWN})"


def partition_rows(run_dirs, gold: np.ndarray, names: list[str]) -> list[dict]:
    """Step-0 partition per seed (once) and the final partition of every run, per cluster."""
    k = len(names)
    if int(gold.max()) + 1 != k:
        raise ValueError(f"gold labels have {int(gold.max()) + 1} topics but {k} names were given")
    rows = []
    step0_seen: set[int] = set()
    for run_dir in run_dirs:
        runlog = load_run(run_dir)
        for (method, seed), record in sorted(runlog.records.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if record.assignments_evolution is None:
                continue
            parts = [("final", record.assignments)]
            if seed not in step0_seen:
                step0_seen.add(seed)
                parts.insert(0, ("step0", record.assignments_evolution[0]))
            for which, labels in parts:
                counts = crosstab(labels, gold, k)
                acc = matched_accuracy(counts)
                label = "all methods" if which == "step0" else f"{runlog.config['name']} {method}"
                for cluster in range(k):
                    rows.append(
                        {
                            "partition": which,
                            "run": label,
                            "seed": seed,
                            "acc": acc,
                            "cluster": cluster,
                            "size": int(counts[cluster].sum()),
                            **{n: int(c) for n, c in zip(names, counts[cluster])},
                            "composition": composition(counts[cluster], names),
                        }
                    )
    return rows


def oov_markdown(rows: list[dict]) -> str:
    lines = [
        "| Run | Method | summaries embedded | tokens | outside the vocabulary | share | all-zero vectors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['method']} | {r['summaries']:,} | {r['tokens']:,} | {r['oov_tokens']:,} "
            f"| {100 * r['oov_share']:.1f}% | {r['zero_vectors']} |"
        )
    return "\n".join(lines)


def partitions_markdown(rows: list[dict]) -> str:
    """One line per seed and partition: the clusters' compositions, largest first."""
    frame = pd.DataFrame(rows)
    lines = ["| Partition | Run | Seed | ACC | Clusters (topics with at least 10 members) |", "| --- | --- | ---: | ---: | --- |"]
    for (which, run, seed), group in frame.groupby(["partition", "run", "seed"], sort=False):
        comps = " ; ".join(group.sort_values("size", ascending=False)["composition"])
        lines.append(f"| {which} | {run} | {seed} | {group['acc'].iloc[0]:.3f} | {comps} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--ids", default="data/bbc500_ids.csv", help="the ids file (default: %(default)s)")
    parser.add_argument("--oov-md")
    parser.add_argument("--oov-csv")
    parser.add_argument("--partitions-md")
    parser.add_argument("--partitions-csv")
    args = parser.parse_args()

    if args.oov_md or args.oov_csv:
        rows = oov_rows(args.run_dirs)
        if args.oov_md:
            Path(args.oov_md).write_text(oov_markdown(rows) + "\n")
        if args.oov_csv:
            pd.DataFrame(rows).to_csv(args.oov_csv, index=False)
    if args.partitions_md or args.partitions_csv:
        gold, names = load_topics(args.ids)
        rows = partition_rows(args.run_dirs, gold, names)
        if args.partitions_md:
            Path(args.partitions_md).write_text(partitions_markdown(rows) + "\n")
        if args.partitions_csv:
            pd.DataFrame(rows).to_csv(args.partitions_csv, index=False)


if __name__ == "__main__":
    main()
