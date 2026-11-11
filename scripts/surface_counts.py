"""Print the summary surface checks (docs/results/clinc150_summarizers.md section 2.4) for run directories.

One row per method that logged summaries, pooled over its seeds; see `kslimeans.surface` for the rules.

Usage:

    python scripts/surface_counts.py results/clinc150_qwen3-4b/<stamp> results/prelim_bank77_qwen3-4b/<stamp>
"""

from __future__ import annotations

import argparse

from kslimeans.runlog import load_run
from kslimeans.surface import SurfaceCounts, surface_counts

HEADER = (
    "| Run | Model | summaries (all steps) | mean words | ends with \"?\" | mentions cluster/summar… "
    "| final: mentions cluster/summar… | final: clusters sharing an identical summary (per seed) "
    "| final: largest identical group | starts with quote | `<think>` |"
)


def _pct(count: int, total: int) -> str:
    return f"{count:,} ({100 * count / total:.0f}%)" if total else f"{count:,}"


def format_row(run: str, model: str, c: SurfaceCounts) -> str:
    return (
        f"| {run} | {model} | {c.n:,} | {c.mean_words:.1f} | {_pct(c.ends_with_question, c.n)} "
        f"| {_pct(c.mentions, c.n)} | {c.final_mentions}/{c.final_n} "
        f"| {c.final_shared_min}–{c.final_shared_max} | {c.final_largest_group} "
        f"| {c.starts_with_quote} | {c.think} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    args = parser.parse_args()

    print(HEADER)
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run_dir in args.run_dirs:
        runlog = load_run(run_dir)
        model = str(runlog.config["summarizer"].get("model_id") or runlog.config["summarizer"].get("type"))
        model = model.rsplit("/", 1)[-1]
        for method in runlog.methods:
            records = [r for r in runlog.method_records(method) if r.summaries_evolution]
            if records:
                counts = surface_counts(r.summaries_evolution for r in records)
                print(format_row(runlog.config["name"], f"{model} ({method})", counts))


if __name__ == "__main__":
    main()
