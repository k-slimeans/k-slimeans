"""Worked examples for the summary-quality section: fluent summaries that do not represent their cluster.

Rule (no hand-picking). Among the summaries of one run at one seed and summary step that pass every fluency proxy,
have a nearest-summary share below `--max-share` and at least `--min-members` members, take the one that lost the
most members at the step. For it, print the summary, its scores, the gold intents of its members, the members'
fates (where the lost members went) and a few member texts.

    python scripts/summary_quality_examples.py RUN_DIR --scores results/summary_quality/summary_quality_per_cluster.csv \\
        --cache-dir ../k-slimeans/cache --seed 0 --step 4
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from kslimeans.runlog import load_run, load_run_dataset


def pick(scores: pd.DataFrame, run: str, seed: int, step: int, max_share: float, min_members: int) -> pd.Series | None:
    """The row the rule above selects, or `None` if no summary qualifies."""
    rows = scores[(scores["run"] == run) & (scores["seed"] == seed) & (scores["step"] == step)]
    rows = rows[~rows["flag_not_fluent"].astype(bool) & (rows["coverage_nearest"] < max_share)
                & (rows["n_members"] >= min_members)]  # fmt: skip
    return None if rows.empty else rows.sort_values(["n_lost", "cluster"], ascending=[False, True]).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--max-share", type=float, default=0.3)
    parser.add_argument("--min-members", type=int, default=50)
    args = parser.parse_args()

    runlog = load_run(args.run_dir)
    run = f"{runlog.config['name']}/{runlog.path.name}"
    row = pick(pd.read_csv(args.scores, low_memory=False), run, args.seed, args.step, args.max_share, args.min_members)
    if row is None:
        print(f"no qualifying summary in {run}")
        return
    ds, _ = load_run_dataset(runlog, cache_dir=args.cache_dir)
    method = next(m for m in runlog.methods if m != "kmeans")
    record = runlog.record(method, args.seed)
    src, after = record.source_partition(args.step), record.partition_after(args.step)
    c = int(row["cluster"])
    members = np.flatnonzero(src == c)
    names = lambda ids: Counter(ds.label_names[g] for g in ds.labels[ids]).most_common(4)  # noqa: E731
    print(f"{run} seed {args.seed} step {args.step} cluster {c}: {row['summary']!r}")
    print(f"members {len(members)}; nearest-summary share {row['coverage_nearest']:.2f}; intent coverage "
          f"{row['coverage_intents']:.2f}; margin {row['specificity_margin']:.3f}; flags: generic={row['flag_generic']} "
          f"unsupported={row['flag_unsupported']}")
    print(f"after the full Lloyd block (partition_after): kept {int(row['n_kept'])}, lost {int(row['n_lost'])}, "
          f"gained {int(row['n_gained'])}; lost in uncovered intents {int(row['n_lost_uncovered'])}")
    print("gold intents of the members:", names(members))
    lost = members[after[members] != c]
    print("lost members go to clusters:", Counter(after[lost].tolist()).most_common(4))
    print("gold intents of the lost:", names(lost))
    for i in np.random.default_rng(0).permutation(members)[:6]:
        print("  -", ds.texts[i], f"[{ds.label_names[ds.labels[i]]}; {'stays' if after[i] == c else 'moves'}]")


if __name__ == "__main__":
    main()
