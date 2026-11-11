"""Replacement diagnostic (issue #33): what happens when a summary replaces a numerical k-means centroid.

Replays the logged runs offline (`kslimeans.replay`): no language model and no encoding; the summary embeddings come
from `scripts/embed_logged_summaries.py`. Writes the verification table (replay vs logged labels, `n_iter` and
inertia, with the scikit-learn version) and a real-data check of the replay against a real `KMeans` fit; with
`--verify-only` nothing more. Otherwise, under `--out-dir` (small, committed):

    replacement_diagnostic_verification.csv   one row per run, method, seed and KMeans fit
    replacement_diagnostic_mechanism.csv      replay vs a real KMeans fit from the same start (one per method)
    replacement_diagnostic_steps.csv          one row per run, method, seed and summary step
    replacement_diagnostic_iterations.csv     one row per step, iteration and kind (replay / counterfactual)
    replacement_diagnostic_outcome.csv        final scores against k-means from the same initialization
    replacement_diagnostic_*.md / .png        tables, examples and figures

and under `--work-dir` (default `results/replacement_diagnostic`, gitignored; regenerate, do not commit):

    replacement_diagnostic_clusters.csv       one row per step and cluster (keys: run, method, seed, step, cluster)
    replacement_diagnostic_stale.csv          per cluster, does the displayed summary still describe the final partition
    <run>__<method>__seed<k>_movers.csv.gz    one row per point that moved

Usage:

    python scripts/replacement_diagnostic.py results/clinc150_lsa/<stamp> ... --cache-dir cache \
        --summary-embeddings results/replacement_diagnostic/summary_embeddings_cpu.npz --out-dir docs/results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from kslimeans.diagnostic import converged_comparison, staleness_rows, step_tables
from kslimeans.replay import (
    LloydReplayer,
    SummaryEmbeddings,
    compare_with_kmeans,
    fit_initial,
    replay_run,
    verify_replay,
)
from kslimeans.runlog import load_run, load_run_dataset
from kslimeans.surface import duplicate_groups

PREFIX = "replacement_diagnostic"
EXACT_REL_TOL = 1e-5


def _records(runlog):
    """(method, seed, record) of every record that logged summaries and per-step assignments."""
    for (method, seed), record in sorted(runlog.records.items()):
        if record.summaries_evolution and record.assignments_evolution is not None:
            yield method, seed, record


def analyse(run_dirs, cache_dir, embeddings, work_dir, verify_only, log=print):
    frames = {key: [] for key in ("verification", "mechanism", "steps", "iterations", "clusters", "stale", "outcome", "examples")}
    initial_fits = {}  # step 0 is shared by the methods of a seed (same data, k, iterations, seed)
    for run_dir in run_dirs:
        runlog = load_run(run_dir)
        run = runlog.config["name"]
        ds, X = load_run_dataset(runlog, cache_dir=cache_dir)
        gold = np.asarray(ds.labels)
        replayer = LloydReplayer(X)
        for method, seed, record in _records(runlog):
            t0 = time.time()
            emb = embeddings.steps(run, method, seed)
            k, iters = len(emb[0]), record.trace[0]["last_max_iter"]
            fit_key = (runlog.config["dataset"], k, iters, seed)
            if fit_key not in initial_fits:
                initial_fits[fit_key] = fit_initial(X, k, iters, seed)
            replay = replay_run(
                record, X, emb, replayer=replayer, counterfactual=not verify_only, initial=initial_fits[fit_key]
            )
            key = {"run": run, "method": method, "seed": seed}
            if seed == min(runlog.seeds):  # real-data guard of the private-API replay: one real KMeans fit per method
                start = replay.steps[0].start_centers
                frames["mechanism"].append(
                    {**key, **compare_with_kmeans(replayer, start, record.trace[1]["last_max_iter"])}
                )
            for check in verify_replay(record, replay):
                frames["verification"].append(
                    {
                        **key,
                        "fit": check.fit,
                        "n_label_mismatch": check.n_label_mismatch,
                        "n_iter": check.n_iter,
                        "logged_n_iter": check.logged_n_iter,
                        "rel_err_summary_inertia": check.rel_err_summary_inertia,
                        "rel_err_inertia": check.rel_err_inertia,
                        "sklearn": sklearn.__version__,
                    }
                )
            if not verify_only:
                base = runlog.records.get(("kmeans", seed))
                frames["outcome"].append(
                    {
                        **key,
                        **{m: record.metrics.get(m) for m in ("acc", "nmi", "ari")},
                        **{f"kmeans_{m}": base.metrics.get(m) if base else None for m in ("acc", "nmi", "ari")},
                    }
                )
                _tables(frames, key, record, replay, X, gold, work_dir, ds.texts, replayer)
            log(f"{run} {method} seed {seed}: {time.time() - t0:.0f}s")
    examples = frames.pop("examples")
    out = {name: pd.DataFrame(rows) for name, rows in frames.items()}
    out["examples"] = examples
    return out


def _examples(key, record, step, tables, texts, n_groups=3, n_texts=3):
    """Markdown lines: a few texts that moved out of duplicate-summary clusters at the replacement."""
    summaries = record.summaries_evolution[step.step]
    movers = [p for p in tables["points"] if p["direction_replacement"]]
    lines = [f"### {key['run']} seed {key['seed']} step {step.step + 1}\n"]
    for group in duplicate_groups(summaries)[:n_groups]:
        lines.append(f"Clusters {group} share the summary: \"{summaries[group[0]]}\"\n")
        out = [p for p in movers if p["from"] in group][:n_texts]
        if not out:
            lines.append("- no point moved out of these clusters\n")
        for p in out:
            lines.append(
                f"- \"{texts[p['point']]}\" (gold {p['gold']}, cluster {p['from']} -> {p['to_replacement']}, "
                f"{p['direction_replacement']}); new cluster's summary: \"{summaries[p['to_replacement']]}\""
            )
        lines.append("")
    return lines


def _tables(frames, key, record, replay, X, gold, work_dir, texts=None, replayer=None):
    points = []
    for step in replay.steps:
        j = step.step
        summaries = record.summaries_evolution[j]
        try:
            logged_after = record.partition_after(j)
        except IndexError:  # a final step that stopped on tol has no block
            logged_after = None
        tables = step_tables(step, X, gold, summaries, record.source_partition(j), logged_after=logged_after)
        if j == 0 and replayer is not None:  # the one step whose pre state is exact: run both sides to convergence
            tables["step"].update(converged_comparison(replayer, step, gold))
        ident = {**key, "step": j + 1}
        if texts is not None and key["seed"] == 0 and j == 0 and key["run"].endswith("qwen3-4b"):
            frames["examples"] += _examples(key, record, step, tables, texts)
        frames["steps"].append({**ident, **tables["step"]})
        frames["iterations"] += [{**ident, **row} for row in tables["iterations"]]
        frames["clusters"] += [{**ident, **row} for row in tables["clusters"]]
        points += [{**ident, **row} for row in tables["points"]]
        # Jaccard against the logged post-block partition; only the nearest-final-center rule uses replayed centers.
        final = logged_after if logged_after is not None else step.block.final_labels
        stale = staleness_rows(record.source_partition(j), final, step.start_centers, step.block.final_centers, summaries)
        frames["stale"] += [{**ident, **row} for row in stale]
    if points:
        work_dir.mkdir(parents=True, exist_ok=True)
        name = f"{key['run']}__{key['method']}__seed{key['seed']}_movers.csv.gz"
        pd.DataFrame(points).to_csv(work_dir / name, index=False)


def verification_summary(ver: pd.DataFrame) -> pd.DataFrame:
    """Per run and method: fits, exact fits, mismatched points and the worst relative errors."""
    ver = ver.assign(exact=(ver.n_label_mismatch == 0) & (ver.n_iter == ver.logged_n_iter))
    ver = ver.assign(is_block=ver.fit > 0)
    grouped = ver.groupby(["run", "method"])
    return grouped.agg(
        fits=("fit", "size"),
        exact_fits=("exact", "sum"),
        mismatched_points_median=("n_label_mismatch", "median"),
        mismatched_points_max=("n_label_mismatch", "max"),
        n_iter_differs=("n_iter", lambda s: int((s.values != ver.loc[s.index, "logged_n_iter"].values).sum())),
        max_rel_err_summary_inertia=("rel_err_summary_inertia", "max"),
        max_rel_err_inertia=("rel_err_inertia", "max"),
        sklearn=("sklearn", "first"),
    ).reset_index()


def label(run: str) -> str:
    return run.removeprefix("clinc150_")


def md_table(df: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    """GitHub markdown table of `df`; `formats` maps a column to a format spec (default: 3 significant digits)."""
    formats = formats or {}

    def cell(col, value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "n/a"
            return format(value, formats.get(col, ",.0f" if abs(value) >= 1000 else ".3g"))
        return str(value)

    df = df.rename(columns=str)
    header = "| " + " | ".join(df.columns) + " |"
    rule = "| " + " | ".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + " |"
    rows = ["| " + " | ".join(cell(c, v) for c, v in zip(df.columns, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def step_means(steps: pd.DataFrame, columns: list[str], step: int | None) -> pd.DataFrame:
    """Mean over seeds (and steps, when `step` is None) of `columns`, one row per run."""
    sel = steps if step is None else steps[steps.step == step]
    out = sel.groupby("run")[columns].mean().reset_index()
    out.insert(0, "model", out.pop("run").map(label))
    return out


def build_tables(frames: dict[str, pd.DataFrame]) -> str:
    steps, clusters, stale, outcome = frames["steps"], frames["clusters"], frames["stale"], frames["outcome"]
    parts = []

    objective = ["j_before", "jump_rel", "end_vs_before_rel", "end_vs_cf_rel", "n_iter"]
    movement = [
        "drift_median",
        "cosine_median",
        "own_nearest_share",
        "to_end_radii_median",
        "end_vs_pre_radii_median",
        "pulled_back_median",
    ]
    movers = [
        "frac_moved_replacement",
        "frac_moved_end",
        "ari_replacement_vs_pre",
        "ari_end_vs_pre",
        "jaccard_end_mean",
    ]
    direction = [
        "n_moved_replacement",
        "toward_replacement",
        "toward_shuffled",
        "away_replacement",
        "away_shuffled",
        "ends_in_own_majority_replacement",
        "ends_in_own_majority_shuffled",
        "share_in_own_majority_pre",
        "move_rate_boundary",
        "move_rate_core",
        "move_rate_flagged",
        "move_rate_unflagged",
        "share_movers_from_flagged",
        "share_points_in_flagged",
    ]
    collapse = ["n_dup_clusters", "n_meta_clusters", "n_empty_replacement", "n_empty_end", "n_reseeded"]
    for title, step in (
        ("Step 1 (shared step-0 partition; replayed from CPU embeddings)", 1),
        ("All five steps pooled (replayed; from step 2 on the previous block is replayed too)", None),
    ):
        parts.append(f"### {title}: means over seeds\n")
        for name, cols in (
            ("Objective (J_before; relative jump at replacement; relative change at block end vs before and vs the "
             "no-replacement counterfactual; iterations run)", objective),
            ("Centroid movement (per-cluster medians in pre RMS radii, cosine, share whose nearest pre center is its own, "
             "distance from summary to block-end center in radii, distance from block-end center to the pre center in radii, fraction by which the block shortens the summary-to-center distance)", movement),
            ("Partition change (fraction of points moved at the replacement and over the block; ARI and mean cluster "
             "Jaccard against the pre partition)", movers),
            ("Movers at the replacement (total, counts toward/away from the gold majority and ending in their own "
             "gold majority, each beside the mean of a shuffled-destination baseline; share of all points in their "
             "own cluster's majority; move rates by margin and flag)", direction),
            ("Duplicates and collapsed centroids (clusters per step)", collapse),
        ):
            parts.append(f"{name}\n\n" + md_table(step_means(steps, cols, step)) + "\n")

    # Outcome: block-end partition vs the no-replacement counterfactual, and the final scores vs k-means.
    sub = steps.assign(model=steps.run.map(label))
    parts.append("### Outcome\n")
    one = sub[sub.step == 1].assign(
        d_acc=lambda x: x.acc_logged_end - x.acc_cf,
        d_nmi=lambda x: x.nmi_logged_end - x.nmi_cf,
        d_ari=lambda x: x.ari_logged_end - x.ari_cf,
        d_acc_replayed=lambda x: x.acc_end - x.acc_cf,
        n_mismatch=lambda x: x.n_replay_vs_logged_end,
    )
    headline = one.groupby("model").agg(
        d_acc=("d_acc", "mean"),
        d_nmi=("d_nmi", "mean"),
        d_ari=("d_ari", "mean"),
        seeds_below=("d_acc", lambda v: f"{int((v < 0).sum())}/{len(v)}"),
        d_acc_replayed=("d_acc_replayed", "mean"),
        replay_vs_logged_points=("n_mismatch", "mean"),
    ).reset_index()
    parts.append(
        "**Step 1 headline.** Step 0 replays exactly, so at step 1 the pre state and the plain-Lloyd counterfactual "
        "are exact and `assignments_evolution[1]` is the real post-block partition. Mean over seeds of (score of the "
        "logged partition after the block) minus (score after plain Lloyd from the step-0 partition, same 20 "
        "iterations); `seeds_below` counts seeds with a lower ACC than the counterfactual; `d_acc_replayed` is the "
        "same from the CPU-embedding replay, as a check.\n\n"
        + md_table(headline, {"d_acc": "+.4f", "d_nmi": "+.4f", "d_ari": "+.4f", "d_acc_replayed": "+.4f"})
        + "\n"
    )
    seeds = one[["model", "seed", "d_acc", "d_nmi", "d_ari"]]
    parts.append(
        "Per seed, step 1, logged partition minus counterfactual.\n\n"
        + md_table(seeds, {"d_acc": "+.4f", "d_nmi": "+.4f", "d_ari": "+.4f"})
        + "\n"
    )
    if "conv_n_iter" in steps:
        conv = sub[sub.step == 1].assign(
            d_acc=lambda x: x.conv_acc - x.conv_acc_cf,
            d_nmi=lambda x: x.conv_nmi - x.conv_nmi_cf,
            d_ari=lambda x: x.conv_ari - x.conv_ari_cf,
            j_end_rel=lambda x: x.conv_j_end / x.conv_j_cf - 1,
        )
        c = conv.groupby("model").agg(
            n_iter=("conv_n_iter", "mean"),
            n_iter_cf=("conv_n_iter_cf", "mean"),
            capped=("conv_stop", lambda v: int((v == "max_iter").sum())),
            capped_cf=("conv_stop_cf", lambda v: int((v == "max_iter").sum())),
            j_end_rel=("j_end_rel", "mean"),
            d_acc=("d_acc", "mean"),
            d_nmi=("d_nmi", "mean"),
            d_ari=("d_ari", "mean"),
            seeds_below=("d_acc", lambda v: f"{int((v < 0).sum())}/{len(v)}"),
        ).reset_index()
        parts.append(
            "**Step 1, both sides run to sklearn's own stop** (cap 300 iterations; replayed from the CPU embeddings). "
            "A fixed 20-iteration budget can leave the replacement block less converged than the counterfactual "
            "(which starts converged), so this separates a worse end state from less convergence. `n_iter` and "
            "`n_iter_cf` are mean iterations to the stop, `capped*` the runs that hit the cap, `j_end_rel` the "
            "mean relative objective of the replacement run over the counterfactual's, and `d_*` the "
            "score differences (replacement minus counterfactual).\n\n"
            + md_table(c, {"d_acc": "+.4f", "d_nmi": "+.4f", "d_ari": "+.4f", "j_end_rel": "+.4f"})
            + "\n"
        )
    parts.append("**Per step, replayed.** From step 2 on every row starts from a replayed previous block.\n")
    d = sub.assign(
        d_acc=sub.acc_end - sub.acc_cf, d_nmi=sub.nmi_end - sub.nmi_cf, d_ari=sub.ari_end - sub.ari_cf
    ).groupby(["model", "step"])[["d_acc", "d_nmi", "d_ari"]].mean().reset_index()
    parts.append(
        "Mean over seeds of (replayed score after the block) minus (score after plain Lloyd from the replayed "
        "pre-replacement state, same iterations).\n\n" + md_table(d, {"d_acc": "+.4f", "d_nmi": "+.4f", "d_ari": "+.4f"}) + "\n"
    )
    o = outcome.assign(model=outcome.run.map(label))
    o = o.assign(d_acc=o.acc - o.kmeans_acc, d_nmi=o.nmi - o.kmeans_nmi, d_ari=o.ari - o.kmeans_ari)
    final = o.groupby("model")[["acc", "kmeans_acc", "d_acc", "d_nmi", "d_ari"]].mean().reset_index()
    parts.append(
        "Final scores against k-means from the same initialization (logged values; mean over seeds).\n\n"
        + md_table(final, {"acc": ".3f", "kmeans_acc": ".3f", "d_acc": "+.4f", "d_nmi": "+.4f", "d_ari": "+.4f"})
        + "\n"
    )
    s1 = steps[steps.step == 1].merge(outcome, on=["run", "method", "seed"])
    s1 = s1.assign(
        d_final_acc=s1.acc - s1.kmeans_acc,
        d_final_nmi=s1.nmi - s1.kmeans_nmi,
        d_step_acc=s1.acc_logged_end - s1.acc_cf,
    )
    corr = []
    for x in ("jump_rel", "drift_median", "n_empty_replacement", "frac_moved_replacement"):
        for y in ("d_step_acc", "d_final_acc", "d_final_nmi"):
            corr.append(
                {
                    "step-1 measure": x,
                    "outcome": y,
                    "spearman (n=%d)" % len(s1): s1[x].corr(s1[y], method="spearman"),
                }
            )
    parts.append(
        "Spearman correlation across all (model, seed) pairs at step 1; a descriptive number, not a test. The "
        "pooled correlation mixes between-model and within-model variation.\n\n"
        + md_table(pd.DataFrame(corr), {"spearman (n=%d)" % len(s1): "+.2f"})
        + "\n"
    )

    # Stale displayed summaries: last summary step of each (method, seed).
    last = stale[stale.step == stale.step.max()]
    rows = []
    for run, g in last.groupby("run"):
        g = g[g.has_text]
        row = {"model": label(run), "summaries": len(g), "mean_jaccard": g.jaccard_source_vs_final.mean()}
        for th in (0.3, 0.5, 0.7):
            row[f"jaccard<{th}"] = (g.jaccard_source_vs_final < th).mean()
        row["nearest_not_own"] = (~g.own_nearest_final).mean()
        for th in (0.3, 0.5, 0.7):
            row[f"stale@{th}"] = ((g.jaccard_source_vs_final < th) | ~g.own_nearest_final).mean()
        rows.append(row)
    parts.append(
        "### Do the displayed (final-step) summaries describe the final partition?\n\n"
        "Share of final summaries that are out of date: Jaccard between the members each summary was written from and "
        "the final members below the threshold, or its embedding nearest to another cluster's final center; "
        "`stale@t` is either condition. All seeds pooled; 0.5 is the documented convention, 0.3 and 0.7 show "
        "sensitivity.\n\n" + md_table(pd.DataFrame(rows), {c: ".3f" for c in ("mean_jaccard",)}) + "\n"
    )
    every = stale[stale.has_text].assign(model=stale.run.map(label))
    every = every.assign(stale_any=(every.jaccard_source_vs_final < 0.5) | ~every.own_nearest_final)
    parts.append(
        "Share stale@0.5 by summary step (summary of step j against the partition after its own block).\n\n"
        + md_table(every.pivot_table(index="model", columns="step", values="stale_any", aggfunc="mean").reset_index(),
                   {str(s): ".3f" for s in range(1, 8)})
        + "\n"
    )
    return "\n".join(parts)


PALETTE = {"lsa": "#0072B2", "qwen3-0.6b": "#E69F00", "qwen3-1.7b": "#009E73", "qwen3-4b": "#D55E00"}


def build_figures(frames: dict[str, pd.DataFrame], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it, steps, clusters = frames["iterations"], frames["steps"], frames["clusters"]
    runs = list(dict.fromkeys(it.run))
    n_steps = int(it.step.max())

    fig, grid = plt.subplots(2, n_steps, figsize=(3.1 * n_steps, 6.4))
    for col, step in enumerate(range(1, n_steps + 1)):
        for row, ax in enumerate(grid[:, col]):
            for run in runs:
                sel = it[(it.run == run) & (it.seed == 0) & (it.step == step)]
                before = steps[(steps.run == run) & (steps.seed == 0) & (steps.step == step)].j_before.iloc[0]
                color = PALETTE[label(run)]
                for kind, style in (("replay", "-"), ("counterfactual", "--")):
                    part = sel[sel.kind == kind]
                    part = part if row == 0 else part[part.t >= 2]
                    t = np.r_[-1, part.t] if row == 0 else part.t.to_numpy()
                    j = np.r_[before, part.objective] if row == 0 else part.objective.to_numpy()
                    ratio = j / before if row == 0 else 100 * (j / before - 1)
                    ax.plot(t, ratio, color=color, ls=style, lw=1.6 if kind == "replay" else 1, label=label(run) if kind == "replay" else None)
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(f"step {step}", fontsize=10)
            ax.set_xlabel("Lloyd iteration" + (" (-1: before)" if row == 0 else ""))
            ax.grid(alpha=0.25)
    grid[0, 0].set_ylabel("objective / objective before")
    grid[1, 0].set_ylabel("objective vs before (%), iteration >= 2")
    grid[0, 0].legend(fontsize=8, title="solid: replacement\ndashed: plain Lloyd", title_fontsize=7)
    fig.suptitle("Objective at each Lloyd iteration around each replacement (seed 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{PREFIX}_objective.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    pooled = clusters[(clusters.size_pre > 0) & np.isfinite(clusters.drift)]
    for ax, col, ylabel, log in (
        (axes[0], "drift", "summary-to-pre-center distance (RMS radii)", True),
        (axes[1], "cosine", "cosine(summary, pre center)", False),
    ):
        data = [pooled[pooled.run == run][col].to_numpy() for run in runs]
        box = ax.boxplot(data, tick_labels=[label(r) for r in runs], showfliers=False, patch_artist=True)
        for patch, run in zip(box["boxes"], runs):
            patch.set_facecolor(PALETTE[label(run)])
            patch.set_alpha(0.6)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Centroid movement at the replacement: all clusters, steps and seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{PREFIX}_movement.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--summary-embeddings", required=True, help=".npz from scripts/embed_logged_summaries.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--work-dir", default="results/replacement_diagnostic", help="gitignored per-point files")
    parser.add_argument("--verify-only", action="store_true", help="write only the verification table")
    args = parser.parse_args()

    embeddings = SummaryEmbeddings.load(args.summary_embeddings)
    print(f"embeddings: {embeddings.meta}", file=sys.stderr)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    frames = analyse(
        args.run_dirs, args.cache_dir, embeddings, Path(args.work_dir), args.verify_only, log=lambda m: print(m, file=sys.stderr)
    )
    frames["verification"].to_csv(out / f"{PREFIX}_verification.csv", index=False)
    summary = verification_summary(frames["verification"])
    print(summary.to_string(index=False))
    print(frames["mechanism"].to_string(index=False))
    frames["mechanism"].to_csv(out / f"{PREFIX}_mechanism.csv", index=False)
    if args.verify_only:
        print(f"{time.time() - start:.0f}s", file=sys.stderr)
        return
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for name in ("steps", "iterations", "outcome", "mechanism"):  # small enough to commit
        frames[name].to_csv(out / f"{PREFIX}_{name}.csv", index=False)
    for name in ("clusters", "stale"):  # per-cluster files are large: regenerate them, they stay out of git
        frames[name].to_csv(work / f"{PREFIX}_{name}.csv", index=False)
    (out / f"{PREFIX}_verification.md").write_text(md_table(summary, {c: ".2e" for c in summary.columns if "rel_err" in c}) + "\n")
    (out / f"{PREFIX}_tables.md").write_text(build_tables(frames))
    (out / f"{PREFIX}_examples.md").write_text("\n".join(frames["examples"]) + "\n")
    build_figures(frames, out)
    print(f"{time.time() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
