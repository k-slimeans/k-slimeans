#!/usr/bin/env python
"""Score the human ratings of the summary hand check: one command, every number asked for (issue #34).

Reads, per dataset, the filled sheets of two or more raters (copies of `summary_audit_sheet_<dataset>.csv`) and the
key (`summary_audit_key_<dataset>.csv`, made by `make_audit_sheet.py --write-key`, not in the repo), and writes
`human_eval_report.md` and CSVs to `--out-dir`:

1. agreement between the raters per property (2 x 2, raw agreement, Cohen's kappa; for `members_fit` also the
   Spearman correlation of the counts and the mean absolute difference), and the disagreeing rows, listed not resolved;
2. each rater against the automatic flags, and the consensus (rows on which all raters agree) against them;
3. per-method rates for each rater, unblinded through the key;
4. the paraphrase check: `paraphrase_flagged_<dataset>.csv` (every summary with `flag_unsupported` = 1, the words,
   numbers and entities that set it off, each rater's value, an empty `category` column to fill by hand) and
   `paraphrase_reverse_<dataset>.csv` (a rater says unsupported, the flag does not fire). The triggers are recomputed
   from the cluster's FULL member set (the runs are read from `--runs-dir`, the datasets from `--cache-dir`) and must
   reproduce the key's flag on every row, or the script stops.

The report names the raters, the n per dataset and the fixed sample of 40 summaries per dataset. After filling
`category` (paraphrase, generalisation, inflection_tokenisation, unshown_members, cluster_talk, unsupported; `unshown_members`
only makes sense in the reverse CSV), COPY the CSVs out of `--out-dir` (for example to `results/human_eval_filled/`),
fill the copies, and run again with `--categories` and the copies to add the counts per category, separately for
each file. The script rewrites its own outputs on every run, so it refuses a `--categories` path that is one of them.

    python scripts/human_eval_report.py --raters is-sami naween321 --sheets-dir docs/results \\
        --key-dir results/audit_keys --runs-dir results --cache-dir cache \\
        --out-dir results/human_eval
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from kslimeans.human_eval import (
    CATEGORIES,
    analyse_rows,
    category_counts,
    check_flags,
    check_sheets,
    flagged_table,
    members_fit_agreement,
    paraphrase_summary,
    pairwise_agreement,
    rater_disagreements,
    reverse_table,
    validate_ratings,
    vs_flags,
)
from kslimeans.summary_audit import HAND_LABELS, agreement_markdown, method_rates

DATASETS = ("clinc150_intent", "bank77")
SAMPLE_SIZE = 40
SHEET_PATTERN = "summary_audit_human_{rater}_{dataset}.csv"


def pair_header(a: str, b: str) -> list[str]:
    return ["property", "n", f"{a} yes", f"{b} yes", "both", f"{a} only", f"{b} only", "neither", "raw agreement", "kappa"]


def rates_markdown(rates: pd.DataFrame, suffix: str = "") -> str:
    """Counts per method and property: the hand labels as `marked/rated`, or the automatic flags (`suffix` "_auto")."""
    names = [p.replace("_", " ") for p in HAND_LABELS]
    lines = ["| method | " + " | ".join(names) + " |", "| --- | " + " | ".join(["---:"] * len(names)) + " |"]
    for r in rates.to_dict("records"):
        cells = [f"{r[p + suffix]}" if suffix else f"{r[p]}/{r[p + '_rated']}" for p in HAND_LABELS]
        lines.append(f"| {r['method']} ({r['n']}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_markdown(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def load_members(key: pd.DataFrame, runs_dir: Path, cache_dir: Path) -> dict[str, list[str]]:
    """Each audited row's cluster, with ALL its members' texts, from the logged run named in the key."""
    from kslimeans.runlog import load_run, load_run_dataset
    from kslimeans.summary_quality import method_label, step_partitions

    out: dict[str, list[str]] = {}
    loaded: dict[str, tuple] = {}
    for r in key.to_dict("records"):
        if r["run"] not in loaded:
            runlog = load_run(runs_dir / r["run"])
            texts = load_run_dataset(runlog, cache_dir=cache_dir)[0].texts
            loaded[r["run"]] = (runlog, texts)
        runlog, texts = loaded[r["run"]]
        (method,) = [m for m in runlog.methods if method_label(runlog.config, m) == r["method"]]
        record = runlog.records[(method, int(r["seed"]))]
        (part,) = [p for p in step_partitions(record) if p.step == int(r["step"])]
        out[r["row_id"]] = [texts[i] for i in np.flatnonzero(np.asarray(part.labels) == int(r["cluster"]))]
    return out


def load_categories(paths: list[Path], datasets, out_dir: Path) -> dict[str, pd.DataFrame]:
    """Read and validate the filled paraphrase CSVs before anything is written.

    Each needs a `dataset` column holding only processed datasets and valid `category` values, and must not be a
    file this script writes into `out_dir` (it would be overwritten with an empty `category`).
    """
    outputs = {
        (out_dir / f"paraphrase_{kind}_{d}.csv").resolve() for kind in ("flagged", "reverse") for d in datasets
    }
    loaded: dict[str, pd.DataFrame] = {}
    for path in paths:
        if Path(path).resolve() in outputs:
            raise SystemExit(
                f"{path} is a file this run writes; copy the filled CSV elsewhere (for example "
                "results/human_eval_filled/) and pass the copy to --categories"
            )
        table = pd.read_csv(path)
        if "dataset" not in table or "category" not in table:
            raise SystemExit(f"{path}: needs the columns `dataset` and `category`")
        unknown = sorted(set(table["dataset"].dropna()) - set(datasets))
        if unknown:
            raise SystemExit(f"{path}: rows of datasets {unknown} that are not being processed ({list(datasets)})")
        try:
            category_counts(table)
        except ValueError as e:
            raise SystemExit(f"{path}: {e}")
        if Path(path).name in loaded:
            raise SystemExit(f"two --categories files are named {Path(path).name}; rename one")
        loaded[Path(path).name] = table
    return loaded


def dataset_section(
    dataset: str,
    sheets: Mapping[str, pd.DataFrame],
    key: pd.DataFrame,
    analysis: pd.DataFrame,
    categories: Mapping[str, pd.DataFrame],
    out_dir: Path,
) -> list[str]:
    raters = list(sheets)
    lines = ["", f"## {dataset} (n = {len(key)} summaries, sample fixed at {SAMPLE_SIZE})"]
    # 1. between raters
    lines += ["", "### 1. Agreement between the raters"]
    for (a, b), table in pairwise_agreement(sheets).items():
        table.to_csv(out_dir / f"rater_agreement_{a}_{b}_{dataset}.csv", index=False)
        fit = members_fit_agreement(sheets[a], sheets[b])
        lines += ["", f"#### {a} against {b}", "", agreement_markdown(table, pair_header(a, b)), "",
                  f"`members_fit` counts: n = {fit['n']}, identical counts on {fit['identical']}, Spearman "
                  f"{fit['spearman']:.2f}, mean absolute difference {fit['mean_abs_diff']:.2f}. "
                  "The low coverage row above is the label (fewer than half of the shown members fit)."]  # fmt: skip
    dis = rater_disagreements(sheets)
    dis.to_csv(out_dir / f"rater_disagreements_{dataset}.csv", index=False)
    labels = dis[dis["property"] != "members_fit"]
    lines += ["", f"#### Disagreeing labels (not resolved; {len(labels)} of {len(key) * len(HAND_LABELS)} labels per rater pair; "
              f"{int((dis['property'] == 'members_fit').sum())} rows with different `members_fit` counts are in "
              f"`rater_disagreements_{dataset}.csv`)", ""]  # fmt: skip
    lines += [table_markdown(labels.to_dict("records"), ["row_id", "property", *raters]) if len(labels) else "None."]
    # 2. human vs automatic
    lines += ["", "### 2. Human against the automatic flags", ""]
    lines += ["`hand` columns are the rater (or the consensus), `auto` the automatic flag."]
    for name, table in vs_flags(sheets, key).items():
        table.to_csv(out_dir / f"vs_flags_{name}_{dataset}.csv", index=False)
        title = "Consensus (rows on which all raters agree on that property)" if name == "consensus" else name
        lines += ["", f"#### {title} against the flags", "", agreement_markdown(table)]
    # 3. per method
    lines += ["", "### 3. Per-method rates (marked/rated; method (summaries))"]
    for r, sheet in sheets.items():
        rates = method_rates(sheet, key)
        rates.to_csv(out_dir / f"method_rates_{r}_{dataset}.csv", index=False)
        lines += ["", f"#### {r}", "", rates_markdown(rates)]
    lines += ["", "#### Automatic flags on the same summaries (flagged)", "",
              rates_markdown(method_rates(sheets[raters[0]], key), "_auto")]  # fmt: skip
    # 4. paraphrase check
    flagged = flagged_table(analysis, sheets)
    reverse = reverse_table(analysis, sheets)
    flagged.to_csv(out_dir / f"paraphrase_flagged_{dataset}.csv", index=False)
    reverse.to_csv(out_dir / f"paraphrase_reverse_{dataset}.csv", index=False)
    s = paraphrase_summary(flagged, raters)
    lines += ["", "### 4. Does the unsupported flag wrongly flag paraphrases?", "",
              f"- Summaries with `flag_unsupported` = 1: {s['flagged']} of {len(key)}.",
              f"- Of those rated by every rater ({s['rated_by_all']}), judged supported (unsupported = 0) by all "
              f"raters: {s['all_supported']} ({s['share_all_supported']:.0%})." if s["rated_by_all"] else
              "- No flagged summary is rated by every rater.",
              f"- Reverse: {len(reverse)} summaries a rater marks unsupported although the flag does not fire "
              f"(`paraphrase_reverse_{dataset}.csv`).",
              f"- Triggers per flagged summary and the `category` column to fill: `paraphrase_flagged_{dataset}.csv`."]  # fmt: skip
    for kind, table in categories.items():
        if "dataset" in table and (table["dataset"] == dataset).any():
            counts = category_counts(table[table["dataset"] == dataset])
            lines += ["", f"Categories filled in `{kind}` for {dataset}:", "",
                      table_markdown([counts], [*CATEGORIES, "unlabelled"])]  # fmt: skip
    return lines


def report(
    raters: list[str],
    sheets_dir: Path,
    key_dir: Path,
    runs_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    datasets=DATASETS,
    sheet_pattern: str = SHEET_PATTERN,
    kind: str = "Human ratings",
    category_files: list[Path] = (),
    blank_dir: Path | None = None,
) -> str:
    from kslimeans.summary_quality import load_nlp

    categories = load_categories(list(category_files), datasets, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nlp = load_nlp()
    keys, sizes = {}, {}
    body: list[str] = []
    for dataset in datasets:
        key = pd.read_csv(key_dir / f"summary_audit_key_{dataset}.csv")
        sheets = {r: pd.read_csv(sheets_dir / sheet_pattern.format(rater=r, dataset=dataset)) for r in raters}
        blank = pd.read_csv(blank_dir / f"summary_audit_sheet_{dataset}.csv") if blank_dir else None
        check_sheets(sheets, key, blank)
        validate_ratings(sheets)
        analysis = analyse_rows(next(iter(sheets.values())), key, load_members(key, runs_dir, cache_dir), nlp)
        check_flags(analysis)
        keys[dataset], sizes[dataset] = key, len(key)
        body += dataset_section(dataset, sheets, key, analysis, categories, out_dir)
    ns = ", ".join(f"{d} {n}" for d, n in sizes.items())
    head = [f"# {kind} of the summary hand check (issue #34)", "",
            f"**{kind} of the summary hand check by {', '.join(f'`{r}`' for r in raters)}.** n per dataset: {ns}; the sample is fixed at "
            f"{SAMPLE_SIZE} summaries per dataset (10 gold intents x 4 methods), rated blind to the method. "
            "Generated by `scripts/human_eval_report.py`. Properties are the hand labels of "
            "`docs/summary_audit_rubric.md` (low coverage: fewer than half of the shown members fit; generic; "
            "unsupported; not fluent); \"auto\" is the automatic flag of the same name. Kappa is 0 when only one of "
            "the two labellings ever marks a summary and `nan` when neither does."]  # fmt: skip
    text = "\n".join(head + body) + "\n"
    (out_dir / "human_eval_report.md").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raters", nargs="+", required=True, metavar="LOGIN", help="two or more rater names")
    parser.add_argument("--sheets-dir", required=True)
    parser.add_argument("--key-dir", required=True)
    parser.add_argument("--runs-dir", required=True, help="the directory holding the logged runs named in the key")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--sheet-pattern", default=SHEET_PATTERN, help="file name of a rater's sheet, with {rater} and {dataset}")
    parser.add_argument("--kind", default="Human ratings", help="what the ratings are, for the report header (\"AI ratings\")")
    parser.add_argument("--blank-dir", default=None, help="directory of the blank summary_audit_sheet_<dataset>.csv "
                        "the raters' fixed columns must equal (default: --sheets-dir)")
    parser.add_argument("--categories", nargs="*", default=[], metavar="CSV", help="filled paraphrase_*.csv files")
    args = parser.parse_args()
    if len(args.raters) < 2:
        parser.error("need at least two raters")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    report(
        args.raters, Path(args.sheets_dir), Path(args.key_dir), Path(args.runs_dir), Path(args.cache_dir),
        Path(args.out_dir), args.datasets, args.sheet_pattern, args.kind, [Path(p) for p in args.categories],
        Path(args.blank_dir or args.sheets_dir),
    )  # fmt: skip
    print(f"wrote {Path(args.out_dir) / 'human_eval_report.md'}")


if __name__ == "__main__":
    main()
