"""Score every logged summary for coverage, genericity, unsupported content and fluency (issue #34).

For each run directory: re-embed the logged summaries with the run's encoder (on CPU), take the cluster members from
the partition each summary was written from, and write

- `summary_quality_per_cluster.csv`: one row per (dataset, method, seed, step, cluster) with the four score families,
  their flags and, where the run logged per-step assignments, the members kept, lost and gained at the step;
- `summary_quality_per_step.csv`: one row per (dataset, method, seed, step) with the method-level genericity (#7),
  label alignment (#8) and a replay check of the logged `summary_inertia`;
- `summary_quality_table.csv` / `.md`: means, medians and flag shares per dataset, method and scope.

Runs without per-step assignments (the BANKING77 runs before #30) are scored at their last summary step against the
FINAL assignments. That is an approximation: the final summaries were written from the partition before the last
Lloyd block, which is not logged. Rows carry `approx_partition` and the tables say so.

CPU only: the script hides the GPU. Usage:

    python scripts/summary_quality.py results/clinc150_*/<stamp> results/prelim_bank77_*/<stamp> \\
        --cache-dir ../k-slimeans/cache --out-dir results/summary_quality
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kslimeans.summary_quality import aggregate_scores, load_nlp, method_label

KEYS = ["dataset", "method", "run", "seed", "step", "source_step", "approx_partition"]


def score_run(run_dir: str, cache_dir: str, corpora: dict, nlp, final_only: bool) -> tuple[list[dict], list[dict]]:
    """Cluster rows and step rows of one run directory (`corpora` caches the per-dataset spaCy features)."""
    from kslimeans.runlog import embed_summaries, load_run, load_run_dataset, load_run_embedder
    from kslimeans.summary_quality import analyze_corpus, score_step, step_partitions

    runlog = load_run(run_dir)
    config = runlog.config
    ds, X = load_run_dataset(runlog, cache_dir=cache_dir)
    if ds.name not in corpora:
        corpora[ds.name] = analyze_corpus(ds.texts, nlp)
    embedder = load_run_embedder(runlog, ds)
    label_emb = (
        embedder.encode([n.replace("_", " ") for n in ds.label_names]) if ds.label_names is not None else None
    )
    # Extractive summarizers are not asked for a question, so the "?" check applies to model runs only.
    asks_question = "question" in ds.prompt.lower() and config["summarizer"].get("type") == "hf"

    cluster_rows: list[dict] = []
    step_rows: list[dict] = []
    for method in runlog.methods:
        for record in runlog.method_records(method):
            if not record.summaries_evolution:
                continue
            for sp in step_partitions(record, final_only=final_only):
                summaries = record.summaries_evolution[sp.step]
                S = embed_summaries(embedder, summaries)
                rows, step = score_step(
                    summaries, S, X, sp.labels, ds.labels, corpora[ds.name], nlp,
                    prompt=ds.prompt, asks_question=asks_question, after=sp.after, label_emb=label_emb,
                )
                key = {
                    "dataset": ds.name,
                    "method": method_label(config, method),
                    "run": f"{config['name']}/{runlog.path.name}",
                    "seed": record.seed,
                    "step": sp.step,
                    "source_step": sp.source_step,
                    "approx_partition": sp.approx,
                }
                cluster_rows += [{**key, **row} for row in rows]
                logged = [r for r in record.trace if sp.source_step is not None and r["step"] == sp.source_step + 1]
                logged_inertia = logged[0].get("summary_inertia") if logged else None
                step["summary_inertia_logged"] = logged_inertia
                step["summary_inertia_rel_diff"] = (
                    abs(step["summary_inertia_replayed"] - logged_inertia) / logged_inertia
                    if logged_inertia else float("nan")
                )
                step_rows.append({**key, "n_scored": len(rows), **step})
            print(f"  {key['method']} seed {record.seed}: done", file=sys.stderr)
    return cluster_rows, step_rows


# ------------------------------------------------------------------ tables


def _scoped(df: pd.DataFrame) -> pd.DataFrame:
    """Stack the frame under its scopes: all steps, and the last step of each (dataset, method)."""
    last = df[df["step"] == df.groupby(["dataset", "method"])["step"].transform("max")]
    all_steps = df.assign(scope="all steps")
    last = last.assign(scope="last step")
    stacked = pd.concat([all_steps, last], ignore_index=True)
    # Old-format runs have a single (approximate) step: keep it once, labelled as such.
    approx = stacked["approx_partition"].astype(bool)
    stacked = stacked[~(approx & (stacked["scope"] == "all steps"))]
    stacked.loc[approx, "scope"] = "final (approx.)"
    return stacked


def build_table(clusters: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    by = ["dataset", "method", "scope"]
    table = aggregate_scores(_scoped(clusters), by)
    step_agg = _scoped(steps).groupby(by, sort=False)[["genericity", "genericity_centroids", "label_align_sim",
                                                       "label_align_acc"]].mean()
    table = table.join(step_agg)
    link = clusters[clusters["n_kept"].notna()] if "n_kept" in clusters else clusters.iloc[:0]
    if len(link):
        link = _scoped(link)
        link = link.assign(kept=link["n_kept"] / link["n_members"], lost=link["n_lost"] / link["n_members"],
                           gained=link["n_gained"] / link["n_members"],
                           member_uncovered=link["n_members_uncovered"] / link["n_members"])
        lost_total = link.groupby(by, sort=False)["n_lost"].sum()
        table = table.join(link.groupby(by, sort=False)[["kept", "lost", "gained", "member_uncovered"]].mean())
        table["lost_uncovered"] = link.groupby(by, sort=False)["n_lost_uncovered"].sum() / lost_total.where(lost_total > 0)
    return table.reset_index()


def _f(x, digits=3) -> str:
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def _pct(x) -> str:
    return "" if pd.isna(x) else f"{100 * x:.0f}%"


def _md(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def format_markdown(table: pd.DataFrame) -> str:
    out = []
    for dataset, sub in table.groupby("dataset", sort=False):
        out.append(f"### {dataset}\n")
        label = [f"{m} ({s})" for m, s in zip(sub["method"], sub["scope"])]
        if sub["scope"].eq("final (approx.)").any():
            out.append("Last summary step scored against the FINAL assignments (approximation: the summaries were "
                       "written from the partition before the last Lloyd block, which is not logged).\n")
        out.append("**Scores** (mean / median over clusters; `n` = summaries)\n")
        out.append(_md(
            ["method (scope)", "n", "nearest-summary share", "intent coverage", "specificity margin",
             "max cos to another summary", "unsupported lemma share", "genericity (#7)", "label align sim / acc (#8)"],
            [[lb, f"{r.n:,}", f"{_f(r.coverage_nearest_mean)} / {_f(r.coverage_nearest_median)}",
              _f(r.coverage_intents_mean),
              f"{_f(r.specificity_margin_mean)} / {_f(r.specificity_margin_median)}",
              _f(r.max_summary_cos_mean),
              f"{_f(r.unsupported_lemma_share_mean)} / {_f(r.unsupported_lemma_share_median)}",
              _f(r.genericity), f"{_f(r.label_align_sim)} / {_f(r.label_align_acc)}"]
             for lb, r in zip(label, sub.itertuples())]))
        out.append("\n**Flags** (share of summaries)\n")
        out.append(_md(
            ["method (scope)", "not fluent", "low coverage", "generic", "duplicate", "near-duplicate", "mentions",
             "margin <= 0", "unsupported"],
            [[lb, _pct(r.flag_not_fluent_share), _pct(r.flag_low_coverage_share), _pct(r.flag_generic_share),
              _pct(r.flag_duplicate_share), _pct(r.flag_near_duplicate_share), _pct(r.flag_mentions_share),
              _pct(r.flag_low_margin_share), _pct(r.flag_unsupported_share)]
             for lb, r in zip(label, sub.itertuples())]))
        out.append("\n**Fluency checks** (share of summaries failing each)\n")
        out.append(_md(
            ["method (scope)", "no \"?\"", "echoes prompt", "copies a member", "markup/quotes", "empty"],
            [[lb, _pct(r.flu_no_question_share), _pct(r.flu_echo_share), _pct(r.flu_copy_share),
              _pct(r.flu_markup_share), _pct(r.flu_empty_share)]
             for lb, r in zip(label, sub.itertuples())]))
        if "kept" in sub and sub["kept"].notna().any():
            out.append("\n**Members at the step** (mean per cluster, as a share of the cluster's members written "
                       "from; `lost in uncovered intents` is the share of lost members whose gold intent the summary "
                       "does not cover, to compare with `members in uncovered intents`)\n")
            out.append(_md(
                ["method (scope)", "kept", "lost", "gained", "lost in uncovered intents",
                 "members in uncovered intents"],
                [[lb, _pct(r.kept), _pct(r.lost), _pct(r.gained), _pct(r.lost_uncovered), _pct(r.member_uncovered)]
                 for lb, r in zip(label, sub.itertuples())]))
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--cache-dir", required=True, help="the embeddings cache the runs were made with")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--final-only", action="store_true", help="score only the last summary step of each run")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # analysis only: CPU, never the GPU
    start = time.perf_counter()
    nlp = load_nlp()
    corpora: dict = {}
    cluster_rows: list[dict] = []
    step_rows: list[dict] = []
    for run_dir in args.run_dirs:
        if not (Path(run_dir) / "summary.csv").exists():
            print(f"warning: skipping {run_dir}: no summary.csv (aborted run?)", file=sys.stderr)
            continue
        print(run_dir, file=sys.stderr)
        c, s = score_run(run_dir, args.cache_dir, corpora, nlp, args.final_only)
        cluster_rows += c
        step_rows += s

    clusters = pd.DataFrame(cluster_rows)
    steps = pd.DataFrame(step_rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(out / "summary_quality_per_cluster.csv", index=False, float_format="%.6g")
    steps.to_csv(out / "summary_quality_per_step.csv", index=False, float_format="%.6g")
    table = build_table(clusters, steps)
    table.to_csv(out / "summary_quality_table.csv", index=False, float_format="%.6g")
    (out / "summary_quality_table.md").write_text(format_markdown(table))
    worst = steps["summary_inertia_rel_diff"].max()
    print(f"replayed vs logged summary_inertia: max relative difference {worst:.2e}", file=sys.stderr)
    print(f"{len(clusters):,} summaries scored in {time.perf_counter() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
