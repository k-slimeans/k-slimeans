"""Write the blinded hand-check sheet for the summary-quality audit (issue #34), plus a separate key.

Per dataset it takes the runs to compare (LSA and the Qwen3 sizes), each at its last summary step and one seed, draws
the audited gold intents by the rule in `kslimeans.summary_audit`, and writes

- `summary_audit_sheet_<dataset>.csv`: fillable, one row per summary with its cluster's size and members, method
  hidden (letters shuffled per item); `.md` is the same sheet for reading;
- `summary_audit_key_<dataset>.csv`: row id -> method, run, seed, step, cluster, plus the automatic scores and flags
  of `summary_quality.py`, which `audit_agreement.py` compares the filled sheet with. Do not open it while rating.

Runs without per-step assignments (BANKING77 before #30) are shown against their FINAL assignments, an
approximation (`approx_partition` in the key). The rubric is `docs/summary_audit_rubric.md`.

    python scripts/make_audit_sheet.py RUN_DIR... --scores results/summary_quality/summary_quality_per_cluster.csv \\
        --cache-dir ../k-slimeans/cache --out-dir docs/results
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from kslimeans.runlog import load_run, load_run_dataset
from kslimeans.summary_audit import MethodPartition, build_audit
from kslimeans.summary_quality import method_label, step_partitions

AUTO_COLUMNS = [
    "coverage_nearest", "specificity_margin", "unsupported_lemma_share",
    "flag_low_coverage", "flag_generic", "flag_unsupported", "flag_not_fluent",
]  # fmt: skip


def sheet_markdown(dataset: str, sheet: list[dict], approx: bool) -> str:
    lines = [f"# Summary audit sheet: {dataset}", "", "Rate each row by `docs/summary_audit_rubric.md`; fill the CSV."]
    if approx:
        lines += ["", "The summaries are shown against the FINAL assignments (approximation, see the key)."]
    item = None
    for row in sheet:
        if row["item"] != item:
            item = row["item"]
            lines += ["", f"## {item}"]
        members = "\n".join(f"  - {m.split('. ', 1)[1]}" for m in row["members"].split("\n"))
        lines += [
            "", f"### {row['row_id']}  (cluster of {row['cluster_size']}, {row['members_shown']} shown)", "",
            f"Summary: {row['summary']}", "", "Members:", members,
        ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--scores", required=True, help="summary_quality_per_cluster.csv from summary_quality.py")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    by_dataset: dict[str, list] = defaultdict(list)
    data: dict[str, tuple] = {}
    for run_dir in args.run_dirs:
        runlog = load_run(run_dir)
        ds, _ = load_run_dataset(runlog, cache_dir=args.cache_dir)
        data[ds.name] = (ds.texts, ds.labels, ds.label_names)
        for method in runlog.methods:
            record = runlog.records.get((method, args.seed))
            if record is None or not record.summaries_evolution:
                continue
            (part,) = step_partitions(record, final_only=True)
            by_dataset[ds.name].append(
                MethodPartition(
                    method=method_label(runlog.config, method),
                    run=f"{runlog.config['name']}/{runlog.path.name}",
                    seed=record.seed,
                    step=part.step,
                    summaries=record.summaries_evolution[part.step],
                    labels=part.labels,
                    approx=part.approx,
                )
            )

    scores = pd.read_csv(args.scores)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for dataset, partitions in by_dataset.items():
        texts, gold, label_names = data[dataset]
        sheet, key = build_audit(dataset, texts, gold, label_names, partitions, seed=args.seed)
        key_df = pd.DataFrame(key).merge(
            scores[["dataset", "method", "seed", "step", "cluster", *AUTO_COLUMNS]],
            on=["dataset", "method", "seed", "step", "cluster"], how="left", validate="one_to_one",
        )  # fmt: skip
        pd.DataFrame(sheet).to_csv(out / f"summary_audit_sheet_{dataset}.csv", index=False)
        key_df.to_csv(out / f"summary_audit_key_{dataset}.csv", index=False, float_format="%.6g")
        (out / f"summary_audit_sheet_{dataset}.md").write_text(
            sheet_markdown(dataset, sheet, any(p.approx for p in partitions))
        )
        print(f"{dataset}: {len(sheet)} rows ({len(sheet) // len(partitions)} items x {len(partitions)} methods)")


if __name__ == "__main__":
    main()
