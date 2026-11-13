"""Score every logged summary for coverage, genericity, unsupported content and fluency (issue #34).

For each run directory: re-embed the logged summaries with the run's encoder (on CPU), take the cluster members from
the partition each summary was written from, and write

- `summary_quality_per_cluster.csv`: one row per (dataset, method, run, seed, step, cluster) with the four score families,
  their flags and, where the run logged per-step assignments, the members kept, lost and gained at the step and the
  cluster's purity after it. `method` is the label (`k-NLPmeans`, `k-SLMmeans (Qwen3-4B)`): the Qwen runs share one
  method name, so join on `run` as well (or on the label);
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


def run_prompt(config: dict, dataset_prompt: str) -> str:
    """The prompt the run's summarizer was given: its own `summarizer.prompt`, else the dataset default."""
    return config["summarizer"].get("prompt") or dataset_prompt


def asks_for_question(config: dict, prompt: str) -> bool:
    """True for model runs whose prompt asks for a question (the "?" check does not apply to extractive runs)."""
    return "question" in prompt.lower() and config["summarizer"].get("type") == "hf"


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
    prompt = run_prompt(config, ds.prompt)
    asks_question = asks_for_question(config, prompt)

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
                    prompt=prompt, asks_question=asks_question, after=sp.after, label_emb=label_emb,
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
    scoped = _scoped(clusters)
    table = aggregate_scores(scoped, by)
    # The tie rule gives a group of exact duplicates one representative; show the scores without those rows too.
    distinct = scoped[~scoped["flag_duplicate"].astype(bool)].groupby(by, sort=False)
    table["coverage_nearest_mean_distinct"] = distinct["coverage_nearest"].mean()
    table["flag_low_coverage_share_distinct"] = distinct["flag_low_coverage"].apply(lambda s: s.astype(float).mean())
    step_agg = _scoped(steps).groupby(by, sort=False)[["genericity", "genericity_centroids", "label_align_sim",
                                                       "label_align_acc"]].mean()
    table = table.join(step_agg)
    link = scoped[scoped["n_kept"].notna()] if "n_kept" in scoped else scoped.iloc[:0]
    if len(link):
        # Ratios of sums (members pooled), so that tiny clusters do not dominate the shares.
        sums = link.groupby(by, sort=False)[
            ["n_members", "n_kept", "n_lost", "n_gained", "n_members_uncovered", "n_lost_uncovered"]
        ].sum()
        table = table.join(
            pd.DataFrame(
                {
                    "kept": sums["n_kept"] / sums["n_members"],
                    "lost": sums["n_lost"] / sums["n_members"],
                    "gained": sums["n_gained"] / sums["n_members"],
                    "member_uncovered": sums["n_members_uncovered"] / sums["n_members"],
                    "lost_if_uncovered": sums["n_lost_uncovered"] / sums["n_members_uncovered"].where(
                        sums["n_members_uncovered"] > 0
                    ),
                    "lost_if_covered": (sums["n_lost"] - sums["n_lost_uncovered"])
                    / (sums["n_members"] - sums["n_members_uncovered"]).where(
                        sums["n_members"] > sums["n_members_uncovered"]
                    ),
                }
            )
        )
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
             "max cos to another summary", "unsupported lemma share", "genericity (#7)", "label align sim / acc (#8)",
             "nearest-summary share without exact duplicates"],
            [[lb, f"{r.n:,}", f"{_f(r.coverage_nearest_mean)} / {_f(r.coverage_nearest_median)}",
              _f(r.coverage_intents_mean),
              f"{_f(r.specificity_margin_mean)} / {_f(r.specificity_margin_median)}",
              _f(r.max_summary_cos_mean),
              f"{_f(r.unsupported_lemma_share_mean)} / {_f(r.unsupported_lemma_share_median)}",
              _f(r.genericity), f"{_f(r.label_align_sim)} / {_f(r.label_align_acc)}",
              _f(r.coverage_nearest_mean_distinct)]
             for lb, r in zip(label, sub.itertuples())]))
        out.append("\n**Flags** (share of summaries)\n")
        out.append(_md(
            ["method (scope)", "not fluent", "low coverage", "low coverage without exact duplicates", "generic",
             "duplicate", "near-duplicate", "mentions", "margin <= 0", "unsupported"],
            [[lb, _pct(r.flag_not_fluent_share), _pct(r.flag_low_coverage_share),
              _pct(r.flag_low_coverage_share_distinct), _pct(r.flag_generic_share),
              _pct(r.flag_duplicate_share), _pct(r.flag_near_duplicate_share), _pct(r.flag_mentions_share),
              _pct(r.flag_low_margin_share), _pct(r.flag_unsupported_share)]
             for lb, r in zip(label, sub.itertuples())]))
        out.append("\n**Fluency checks** (share of summaries failing each)\n")
        out.append(_md(
            ["method (scope)", "no \"?\"", "echoes prompt", "copies a member", "markup/quotes", "non-Latin letters", "empty"],
            [[lb, _pct(r.flu_no_question_share), _pct(r.flu_echo_share), _pct(r.flu_copy_share),
              _pct(r.flu_markup_share), _pct(r.flu_script_share), _pct(r.flu_empty_share)]
             for lb, r in zip(label, sub.itertuples())]))
        if "kept" in sub and sub["kept"].notna().any():
            out.append("\n**Members at the step** (pooled over clusters, as a share of the members the summaries were "
                       "written from, counted on the partition after the step's Lloyd block. A member is *uncovered* "
                       "when its gold intent is one the summary does not cover; `lost if uncovered` / `lost if covered` "
                       "are the shares of uncovered / covered members that left their cluster)\n")
            out.append(_md(
                ["method (scope)", "kept", "lost", "gained", "members uncovered", "lost if uncovered",
                 "lost if covered"],
                [[lb, _pct(r.kept), _pct(r.lost), _pct(r.gained), _pct(r.member_uncovered), _pct(r.lost_if_uncovered),
                  _pct(r.lost_if_covered)]
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
