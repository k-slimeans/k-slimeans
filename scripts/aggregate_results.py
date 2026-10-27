"""Combine run directories into one comparison table (mean ± std over seeds).

Reads `config.yaml` and `metrics.jsonl` from each run directory and prints (or writes) a markdown
table via `kslimeans.aggregate`. Baselines repeat in every run directory; the first occurrence
of each is kept. With `--examples`, also shows, for each gold class, the cluster that best
matches it in every run and that cluster's final summary (needs runs made after assignments
were logged to `summaries.jsonl`).

Usage:

    python scripts/aggregate_results.py results/prelim_bank77_lsa/<stamp> results/prelim_bank77_qwen3-0.6b/<stamp>

    # Write the table, and a CSV with the numeric means/stds.
    python scripts/aggregate_results.py RUN_DIR [RUN_DIR ...] --out-md table.md --out-csv table.csv

    # One row per (method, seed) for paired comparisons.
    python scripts/aggregate_results.py RUN_DIR [RUN_DIR ...] --per-seed-csv per_seed.csv

    # Paired per-seed differences against reference methods (default: LSA and k-means), and costs.
    python scripts/aggregate_results.py RUN_DIR [RUN_DIR ...] --paired-md paired.md --paired-csv paired.csv \\
        --cost-md cost.md

    # Print the 5 gold classes every run recovers best (min over runs of overlap / gold size).
    python scripts/aggregate_results.py RUN_DIR [RUN_DIR ...] --top-recovered 5 --seed 0

    # Qualitative examples for gold classes (id or name), at seed 0.
    python scripts/aggregate_results.py RUN_DIR [RUN_DIR ...] --examples card_arrival 12 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kslimeans.aggregate import (
    DEFAULT_REFERENCES,
    aggregate_runs,
    cost_table,
    find_examples,
    format_cost_table,
    format_paired_markdown,
    format_table,
    paired_differences,
    per_seed_table,
    rank_examples,
    step0_sha1_summary,
    to_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="run directories (results/<name>/<timestamp>)")
    parser.add_argument("--out-md", help="write the markdown here instead of printing it")
    parser.add_argument("--out-csv", help="write numeric means/stds to this CSV")
    parser.add_argument("--per-seed-csv", help="write per-seed metrics (method, seed, acc, nmi, ari, runtime_s) here")
    parser.add_argument("--paired-md", help="write paired differences vs the references (one table each) here")
    parser.add_argument("--paired-csv", help="write the numeric paired differences (mean, sd, 95%% CI, wins) here")
    parser.add_argument(
        "--references",
        nargs="+",
        default=list(DEFAULT_REFERENCES),
        metavar="METHOD",
        help="reference method labels for the paired tables (default: %(default)s)",
    )
    parser.add_argument("--cost-md", help="write the cost table (calls, tokens, time, memory, settings) here")
    parser.add_argument(
        "--top-recovered",
        type=int,
        metavar="N",
        help="print the N gold classes best recovered by every run (score = min over runs of best-cluster overlap / gold size)",
    )
    parser.add_argument(
        "--examples", nargs="+", metavar="GOLD", help="gold class ids or names to show example summaries for"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed to take examples from (default: 0)")
    args = parser.parse_args()

    numeric = aggregate_runs(args.run_dirs)
    markdown = to_markdown(format_table(numeric))
    if args.top_recovered:
        ranked = rank_examples(args.run_dirs, args.seed)[: args.top_recovered]
        markdown += f"\n\nTop recovered (seed {args.seed}):\n\n" + "\n".join(
            f"- {name}: {score:.3f}" for name, score in ranked
        )
    if args.examples:
        markdown += "\n\n" + find_examples(args.run_dirs, args.examples, args.seed)
    markdown += "\n"

    if args.per_seed_csv:
        per_seed_table(args.run_dirs).to_csv(args.per_seed_csv, index=False)
    if args.paired_md or args.paired_csv:
        paired = paired_differences(per_seed_table(args.run_dirs), args.references)
        if args.paired_md:
            Path(args.paired_md).write_text(format_paired_markdown(paired) + "\n")
        if args.paired_csv:
            paired.to_csv(args.paired_csv, index=False)
    if args.cost_md:
        costs = to_markdown(format_cost_table(cost_table(args.run_dirs)))
        Path(args.cost_md).write_text(f"{costs}\n\n{step0_sha1_summary(args.run_dirs)}\n")
    if args.out_csv:
        numeric.to_csv(args.out_csv, index=False)
    if args.out_md:
        Path(args.out_md).write_text(markdown)
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
