"""The synthesis (issue #50): when do generated summaries help or hurt clustering?

Joins, per (dataset, method, seed, summary step, cluster), the summary-quality scores (#34), the centroid-replacement
measures (#33), the cluster's outcome and the cost of its summary (#35), then writes the tables and figures that
`docs/results/synthesis.md` reads. See `kslimeans.synthesis` for the definitions.

Inputs: the run directories (any number; a new dataset or prompt needs only another directory and a #34 / #33 file that
covers it), #34's `summary_quality_per_cluster.csv` and #33's `replacement_diagnostic_clusters.csv` (neither is in git:
regenerate them with `scripts/summary_quality.py` and `scripts/replacement_diagnostic.py`, see their docs). Runs
without per-step assignments (BANKING77 before #30) are joined to #34 only and flagged approximate. The joined CSV goes
to `--joined-csv` (under the gitignored `results/`), the summary tables and figures to `--out-dir`
(`synthesis_*`). CPU only: the GPU is hidden, tokenizers load from the local Hugging Face cache, nothing is downloaded.

Usage:

    HF_HUB_OFFLINE=1 python scripts/synthesis.py results/clinc150_*/<stamp> results/prelim_bank77_*/<stamp> \\
        --cache-dir ../k-slimeans/cache --quality-clusters results/summary_quality/summary_quality_per_cluster.csv \\
        --diag-clusters ../k-slimeans/results/replacement_diagnostic/replacement_diagnostic_clusters.csv \\
        --diag-stale ../k-slimeans/results/replacement_diagnostic/replacement_diagnostic_stale.csv \\
        --out-dir docs/results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kslimeans import synthesis as syn

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
KMEANS_COLOR = "#666666"


def load_local_tokenizer(model_id: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, padding_side="left", local_files_only=True, trust_remote_code=trust_remote_code
    )


# ------------------------------------------------------------------------------------------------------- build


def build_joined(args) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """The joined per-cluster frame, the run-level frame and the join report rows."""
    quality = pd.read_csv(args.quality_clusters, low_memory=False)
    diag = pd.read_csv(args.diag_clusters)
    stale = pd.read_csv(args.diag_stale) if args.diag_stale else None

    cf_cache: dict = {}
    extras = [syn.run_extras(d, args.cache_dir, load_local_tokenizer, cf_cache) for d in args.run_dirs]
    cost = pd.concat([e["cost"] for e in extras], ignore_index=True)
    after = pd.concat([e["after"] for e in extras], ignore_index=True)
    runs = pd.concat([e["runs"] for e in extras], ignore_index=True)
    ref = pd.concat([e["ref"] for e in extras], ignore_index=True)

    wanted = set(runs["run"])
    missing = wanted - set(quality["run"])
    if missing:
        raise SystemExit(f"no #34 rows for run(s) {sorted(missing)}")
    quality = syn.prepare_quality(quality[quality["run"].isin(wanted)])
    diag_runs = {r.split("/")[0] for r in wanted}
    diag = diag[diag["run"].isin(diag_runs)]
    if stale is not None:
        stale = stale[stale["run"].isin(diag_runs)]

    joined, reports = syn.join_quality_diagnostic(quality, diag, stale)

    # Outcome columns from the logged partitions: only the rows that have per-step assignments.
    logged = joined[joined["has_diag"]]
    old = joined[~joined["has_diag"]]
    logged, rep = syn.attach_by_key(logged, after, syn.RUN_KEYS, "quality x after-partition", extra_unmatched_ok=True)
    reports.append(rep)
    bad = logged["majority_before"].astype("Int64") != logged["majority_intent"].astype("Int64")
    if bad.any():
        raise syn.JoinError(f"{int(bad.sum())} rows: the majority intent rebuilt from the logs differs from #34's")
    first = logged["step"] == 1
    step1, rep = syn.attach_by_key(logged[first], ref, syn.RUN_KEYS, "quality x step-1 references", extra_unmatched_ok=True)
    reports.append(rep)
    joined = pd.concat([step1, logged[~first], old], ignore_index=True)

    joined, rep = syn.attach_by_key(joined, cost, syn.RUN_KEYS, "quality x cost", extra_unmatched_ok=True)
    reports.append(rep)
    joined, rep = syn.attach_run_level(joined, runs.drop(columns="method_raw"), ["run", "seed"], "cluster rows x run metrics")
    reports.append(rep)
    return syn.add_derived(joined), runs, reports


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Per method and step: how often the replayed #33 state agrees with the logged partitions."""
    have = df[df["has_diag"]]
    cols = ["pre_matches_source", "end_matches_logged"]
    out = have.groupby(["method_id", "step"])[cols].mean().reset_index()
    return out.rename(columns={c: f"share_{c}" for c in cols})


# ------------------------------------------------------------------------------------------------------ tables


def fmt_ci(row, col: str, digits: int = 3) -> str:
    return syn.format_cell(row[col], row.get(f"{col}_lo", np.nan), row.get(f"{col}_hi", np.nan), digits)


def method_markdown(table: pd.DataFrame) -> str:
    header = ["method", "seeds", "nearest-summary share", "specificity margin", "unsupported share", "generic", "not fluent",
              "ACC", "NMI", "ARI", "dACC vs k-means", "dNMI vs k-means", "calls", "summarizer s", "input tok", "output tok"]  # fmt: skip
    rows = []
    for _, r in table.iterrows():
        def sd(col, digits=3):
            return "n/a" if pd.isna(r[col]) else f"{r[col]:.{digits}f} ± {r[col + '_sd']:.{digits}f}"

        rows.append([r["method_id"], str(int(r["n_seeds"])), f"{r['coverage_nearest']:.3f}", f"{r['specificity_margin']:.3f}",
                     f"{r['unsupported_lemma_share']:.3f}", f"{r['flag_generic']:.2f}", f"{r['flag_not_fluent']:.2f}",
                     sd("final_acc"), sd("final_nmi"), sd("final_ari"), fmt_ci(r, "d_acc"), fmt_ci(r, "d_nmi"),
                     "n/a" if pd.isna(r["n_calls"]) else f"{r['n_calls']:.0f}", "n/a" if pd.isna(r["summarizer_seconds"]) else f"{r['summarizer_seconds']:.0f}",
                     "n/a" if pd.isna(r["total_input_tokens"]) else f"{r['total_input_tokens']:,.0f}",
                     "n/a" if pd.isna(r["total_output_tokens"]) else f"{r['total_output_tokens']:,.0f}"])  # fmt: skip
    return syn.markdown_table(header, rows)


def strata_markdown(table: pd.DataFrame, outcomes: list[str]) -> str:
    out = []
    for (scope, strata), t in table.groupby(["scope", "strata"], sort=False):
        out.append(f"**{strata}, {scope}** (mean over seeds [95% interval over seeds]; `n` = clusters, pooled)\n")
        header = ["method", "stratum", "n", "seeds", *outcomes]
        rows = [[r["method_id"], str(r["stratum"]), str(int(r["n_clusters"])), str(int(r["n_seeds"])), *[fmt_ci(r, o) for o in outcomes]]
                for _, r in t.iterrows()]  # fmt: skip
        out.append(syn.markdown_table(header, rows) + "\n")
    return "\n".join(out)


def correlation_markdown(table: pd.DataFrame) -> str:
    out = []
    for (scope, controls, outcome), t in table.groupby(["scope", "controls", "outcome"], sort=False):
        out.append(f"**Spearman rho with {outcome}, {scope}, controlling for {controls}** (mean over seeds [95% interval over seeds])\n")
        methods = list(dict.fromkeys(t["method_id"]))
        rows = []
        for prop, p in t.groupby("property", sort=False):
            cells = {r["method_id"]: syn.format_cell(r["rho"], r["lo"], r["hi"], 2) for _, r in p.iterrows()}
            rows.append([prop, *[cells.get(m, "n/a") for m in methods]])
        out.append(syn.markdown_table(["property", *methods], rows) + "\n")
    return "\n".join(out)


def fluency_markdown(table: pd.DataFrame) -> str:
    out = []
    for (scope, outcome), t in table.groupby(["scope", "outcome"], sort=False):
        out.append(f"**Not fluent minus fluent, outcome {outcome}, {scope}** (mean over seeds [95% interval over seeds])\n")
        methods = list(dict.fromkeys(t["method_id"]))
        rows = []
        for adj, p in t.groupby("adjustment", sort=False):
            cells = {r["method_id"]: syn.format_cell(r["mean"], r["lo"], r["hi"]) for _, r in p.iterrows()}
            rows.append([adj, *[cells.get(m, "n/a") for m in methods]])
        out.append(syn.markdown_table(["adjustment", *methods], rows) + "\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------- figures


def color_map(methods: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}


def plot_pareto(table: pd.DataFrame, dataset: str, path: Path) -> None:
    """ACC and NMI against summarizer seconds, and ACC against mean nearest-summary share (mean ± sd over seeds).

    Drawn like `rq2_clinc150_pareto.png`: error bars, k-means as a dashed line, grid at alpha 0.3."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = color_map(list(table["method_id"]))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    panels = [("summarizer_seconds", "final_acc", "kmeans_acc", "ACC"), ("summarizer_seconds", "final_nmi", "kmeans_nmi", "NMI"),
              ("coverage_nearest", "final_acc", "kmeans_acc", "ACC")]  # fmt: skip
    for ax, (xcol, ycol, kcol, ylabel) in zip(axes, panels):
        for _, r in table.iterrows():
            if pd.isna(r[xcol]):
                continue
            ax.errorbar(r[xcol], r[ycol], yerr=r[f"{ycol}_sd"], xerr=None if xcol == "coverage_nearest" else r[f"{xcol}_sd"],
                        color=colors[r["method_id"]], marker="o", elinewidth=1, capsize=2, linestyle="none")  # fmt: skip
            ax.annotate(r["method_id"].replace("k-SLMmeans ", "").replace("k-NLPmeans", "LSA"), (r[xcol], r[ycol]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8, color=colors[r["method_id"]])  # fmt: skip
        ax.axhline(table[kcol].mean(), color=KMEANS_COLOR, linestyle="--", linewidth=1, label="k-means")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("summarizer seconds (mean ± sd over seeds; log scale, inflated, see #35)")
    axes[2].set_xlabel("mean nearest-summary share (coverage, #34)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"{dataset}: clustering quality against cost and summary quality, by method")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_strata(table: pd.DataFrame, strata: list[str], outcomes: list[str], scope: str, title: str, path: Path) -> None:
    """Outcome means by stratum, one line per method, bars are 95% intervals over seeds. Rows: outcomes, columns: strata."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(table["method_id"]))
    colors = color_map(methods)
    fig, axes = plt.subplots(len(outcomes), len(strata), figsize=(4.6 * len(strata), 3.6 * len(outcomes)), constrained_layout=True, squeeze=False)
    for j, strat in enumerate(strata):
        sub = table[(table["strata"] == strat) & (table["scope"] == scope)]
        labels = max((list(g["stratum"]) for _, g in sub.groupby("method_id")), key=len)  # tables are in bin order
        for i, outcome in enumerate(outcomes):
            ax = axes[i][j]
            for m_i, m in enumerate(methods):
                t = sub[sub["method_id"] == m].set_index("stratum").reindex(labels)
                x = np.arange(len(labels)) + (m_i - (len(methods) - 1) / 2) * 0.08
                y = t[outcome].to_numpy(float)
                err = np.vstack([y - t[f"{outcome}_lo"].to_numpy(float), t[f"{outcome}_hi"].to_numpy(float) - y])
                ax.errorbar(x, y, yerr=np.nan_to_num(err), color=colors[m], marker="o", ms=4, elinewidth=1, capsize=2, label=m)
            ax.set_xticks(range(len(labels)), labels, fontsize=8 if len(labels) <= 6 else 6.5,
                          rotation=0 if len(labels) <= 6 else 60, ha="center" if len(labels) <= 6 else "right")
            ax.axhline(0, color=KMEANS_COLOR, linewidth=0.6)
            ax.grid(alpha=0.3)
            ax.set_ylabel(outcome)
            if i == len(outcomes) - 1:
                ax.set_xlabel(strat)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fluency(table: pd.DataFrame, scope: str, path: Path) -> None:
    """The not-fluent minus fluent difference of each outcome, raw and adjusted, one marker per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = table[table["scope"] == scope]
    methods = list(dict.fromkeys(sub["method_id"]))
    colors = color_map(methods)
    adjustments = list(dict.fromkeys(sub["adjustment"]))
    outcomes = ["d_member_purity", "lost_share"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, outcome in zip(axes, outcomes):
        for m_i, m in enumerate(methods):
            t = sub[(sub["method_id"] == m) & (sub["outcome"] == outcome)].set_index("adjustment").reindex(adjustments)
            x = np.arange(len(adjustments)) + (m_i - (len(methods) - 1) / 2) * 0.1
            y = t["mean"].to_numpy(float)
            err = np.vstack([y - t["lo"].to_numpy(float), t["hi"].to_numpy(float) - y])
            ax.errorbar(x, y, yerr=np.nan_to_num(err), color=colors[m], marker="o", linestyle="none", elinewidth=1, capsize=2, label=m)
        ax.axhline(0, color=KMEANS_COLOR, linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(adjustments)), [a.replace(" x ", "\nx ").replace("within ", "within\n") for a in adjustments], fontsize=8)
        ax.set_ylabel(f"not fluent minus fluent: {outcome}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Does failing the fluency proxies predict the outcome? ({scope}; bars: 95% interval over seeds)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--cache-dir", required=True, help="the embeddings cache the runs were made with")
    parser.add_argument("--quality-clusters", required=True, help="#34's summary_quality_per_cluster.csv")
    parser.add_argument("--diag-clusters", required=True, help="#33's replacement_diagnostic_clusters.csv")
    parser.add_argument("--diag-stale", help="#33's replacement_diagnostic_stale.csv (optional)")
    parser.add_argument("--out-dir", required=True, help="summary tables and figures (synthesis_*), normally docs/results")
    parser.add_argument("--joined-csv", default="results/synthesis/synthesis_per_cluster.csv", help="the joined per-cluster CSV")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # analysis only: CPU, never the GPU
    start = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined, runs, reports = build_joined(args)

    joined_path = Path(args.joined_csv)
    joined_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(joined_path, index=False, float_format="%.6g")
    report = pd.DataFrame(reports)
    report.to_csv(out / "synthesis_join_report.csv", index=False)
    cons = consistency(joined)
    cons.to_csv(out / "synthesis_replay_consistency.csv", index=False, float_format="%.4g")
    print(report.to_string(index=False), file=sys.stderr)

    md = ["# Synthesis tables (generated by scripts/synthesis.py)\n", "## Join report\n",
          syn.markdown_table(list(report.columns), [[str(v) for v in r] for r in report.to_numpy()]) + "\n"]  # fmt: skip
    md += ["## Replay consistency (share of clusters where the replayed #33 state agrees with the logged partitions)\n",
           syn.markdown_table(list(cons.columns), [[str(v) if i < 2 else f"{v:.3f}" for i, v in enumerate(r)] for r in cons.to_numpy()]) + "\n"]  # fmt: skip

    for dataset, df in joined.groupby("dataset", sort=False):
        tag = f"synthesis_{dataset}"
        md.append(f"# {dataset}\n")
        ds_runs = runs[runs["run"].isin(df["run"].unique())]
        last = df[df["step"] == df.groupby("run")["step"].transform("max")]
        methods = syn.method_table(last, ds_runs)
        methods.to_csv(out / f"{tag}_methods.csv", index=False, float_format="%.6g")
        md += ["## Method level (quality at the last summary step; metrics and cost per run)\n", method_markdown(methods) + "\n"]
        plot_pareto(methods, dataset, out / f"{tag}_pareto.png")

        if df["has_diag"].any():
            res = syn.analyse_with_outcomes(df)
            for name, table in res.items():
                table.to_csv(out / f"{tag}_{name}.csv", index=False, float_format="%.6g")
            outs = list(syn.STRATA_OUTCOMES)
            md += ["## Q1 harm: strata of the summary and centroid properties\n", strata_markdown(res["harm_strata"], outs)]
            md += ["## Q1 harm: rank correlations\n", correlation_markdown(res["correlations"])]
            md += ["## Q2 fluency\n", fluency_markdown(res["fluency"]),
                   "**Partial rank correlation of the number of failed fluency proxies** (mean over seeds [95% interval over seeds])\n",
                   correlation_markdown(res["fluency_partial"])]  # fmt: skip
            md += ["## Q3 help: where purity rises\n", strata_markdown(res["help_strata"], outs)]
            if "help_reference" in res:
                md += ["## Q3 help: against the no-replacement counterfactual and the random-shuffle null (step 1)\n",
                       strata_markdown(res["help_reference"], list(syn.REF_OUTCOMES))]  # fmt: skip
                plot_strata(res["help_reference"], ["purity", "purity_x_size"], ["d_member_purity_vs_cf", "d_member_purity_vs_null"],
                            "step 1", f"{dataset}: member-purity change minus counterfactual and minus shuffle null, step 1",
                            out / f"{tag}_help_reference.png")  # fmt: skip
            md += ["## Size and the purity change (Spearman rho of cluster size)\n", correlation_markdown(res["size_corr"])]
            plot_strata(res["harm_strata"], ["coverage_nearest", "specificity_margin", "unsupported_lemma_share"], ["lost_share", "d_member_purity"],
                        "step 1", f"{dataset}: members lost and change in purity by summary property, step 1", out / f"{tag}_harm_strata.png")  # fmt: skip
            plot_strata(res["help_strata"], ["purity", "label_top1", "coverage_nearest"], ["d_member_purity", "helped"],
                        "step 1", f"{dataset}: change in purity and share helped by purity before the step, step 1", out / f"{tag}_help_strata.png")  # fmt: skip
            plot_fluency(res["fluency"], "step 1", out / f"{tag}_fluency.png")
        else:
            table = syn.cross_section(df)
            table.to_csv(out / f"{tag}_cross_section.csv", index=False, float_format="%.6g")
            md += ["## Cross-section: summary properties against purity (approximate final partition)\n", correlation_markdown(table.assign(scope="final (approx.)"))]

    (out / "synthesis_tables.md").write_text("\n".join(md))
    print(f"{len(joined):,} cluster rows in {time.perf_counter() - start:.0f}s; joined CSV: {joined_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
