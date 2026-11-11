"""Tabulate fixed-workload output directories (`scripts/fixed_workload.py`) side by side.

One row per directory: the setup (model, prompt source, batch size), the surface counts of the first repeat and
the cost over repeats. The prompt column says "dataset" for the logged run's own prompt and "override" for a
`summarizer.prompt`; `--show-prompt` prints each override in full below the table.

Usage:

    python scripts/compare_workloads.py results/fixed_workload/old_4b/<stamp> results/fixed_workload/v2_4b/<stamp>
"""

from __future__ import annotations

import argparse
from statistics import mean

from kslimeans.workload import load_workload_dir

HEADER = (
    "| Run | Model | Prompt | Batch | Clusters | mention cluster/summar… | Duplicate groups (clusters) "
    "| Largest group | Mean words | Ends with \"?\" | Repeats | s per repeat | Input tokens | Output tokens "
    "| Truncated | Same across repeats | Equal to logged |"
)
RULE = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"


def _share(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.0f}%)" if total else str(count)


def format_row(w: dict) -> str:
    cfg, s, cost = w["config"], w["surface"], w["cost"]
    summarizer = cfg["summarizer"]
    model = str(summarizer.get("model_id") or summarizer.get("type")).rsplit("/", 1)[-1]
    prompt = "override" if summarizer.get("prompt") else "dataset"
    seconds = mean(row["seconds"] for row in cost)
    logged = s.get("logged_match")
    return (
        f"| {cfg['name']} | {model} | {prompt} | {summarizer.get('batch_size', '')} | {s['n']} "
        f"| {_share(s['mentions'], s['n'])} | {s['duplicate_groups']} ({s['clusters_in_duplicate_groups']}) "
        f"| {s['largest_duplicate_group']} | {s['mean_words']:.1f} "
        f"| {_share(s['ends_with_question'], s['n'])} | {s['repeats']} | {seconds:.1f} "
        f"| {cost[0].get('input_tokens', 0):,} | {cost[0].get('output_tokens', 0):,} "
        f"| {cost[0].get('n_truncated_clusters', 0)} | {'yes' if s['identical_across_repeats'] else 'NO'} "
        f"| {'n/a' if logged is None else f'{100 * logged:.0f}%'} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="WORKLOAD_DIR")
    parser.add_argument("--show-prompt", action="store_true", help="print each run's effective prompt after the table")
    args = parser.parse_args()

    workloads = [load_workload_dir(path) for path in args.run_dirs]
    print(HEADER)
    print(RULE)
    for w in workloads:
        print(format_row(w))
    if args.show_prompt:
        print()
        for w in workloads:
            print(f"- {w['config']['name']}: {w['config']['prompt']}")


if __name__ == "__main__":
    main()
