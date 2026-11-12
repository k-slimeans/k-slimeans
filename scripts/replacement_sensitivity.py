"""How sensitive is a replacement block to float noise in its start centers? (noise floor for issue #33 / #47)

Float32 cannot represent relative changes below about 6e-8 (half its spacing, 1.2e-7), so perturbations of 1e-9 and 1e-8
change few or no elements; `frac_elements_changed` reports how many were actually changed. The `tied` noise is the
realistic model, since identical texts encode identically; `iid` noise breaks ties that real encoder noise would not.

The summary embeddings of the original runs were computed on a GPU; re-embedding on a CPU changes them by ~1e-6 (L2,
per row), and a Lloyd block started from them can end in a different partition. This script measures that: for each
method, seed and summary step it replays the block from the logged-summary embeddings (the reference; with the GPU
encode this is an exact replay of the logged block), then from
copies perturbed by Gaussian noise of relative size `rel` (`start * (1 + rel * N(0, 1))`), with a fixed seed and several
draws, and reports how many labels change and how ACC / NMI / ARI of the post-block partition change.

Two noise kinds: `iid` (every row perturbed independently, so identical start centers become distinct) and `tied`
(identical rows get the same noise, so ties stay exact). Points are split into those whose nearest start center is one
of a group of identical centers ("tied points") and the rest.

Usage:

    python scripts/replacement_sensitivity.py results/clinc150_lsa/<stamp> ... --cache-dir cache \
        --summary-embeddings results/replacement_diagnostic/summary_embeddings_cuda.npz --out-dir docs/results

(`summary_embeddings_cpu.npz` is the fallback without a GPU; the reference is then not the logged block.)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from kslimeans.diagnostic import fraction_changed, partition_scores, perturb, tied_clusters
from kslimeans.replay import LloydReplayer, SummaryEmbeddings
from kslimeans.runlog import load_run, load_run_dataset

PREFIX = "replacement_diagnostic_sensitivity"
RELS = (1e-9, 1e-8, 1e-7, 1e-6)
TIED_RELS = (1e-7, 1e-6)


def run(run_dirs, cache_dir, embeddings, seeds, steps, draws, log):
    rows = []
    for run_dir in run_dirs:
        runlog = load_run(run_dir)
        name = runlog.config["name"]
        ds, X = load_run_dataset(runlog, cache_dir=cache_dir)
        gold = np.asarray(ds.labels)
        replayer = LloydReplayer(X)
        for (method, seed), record in sorted(runlog.records.items()):
            if not record.summaries_evolution or seed not in seeds:
                continue
            emb = embeddings.steps(name, method, seed)
            t0 = time.time()
            for step in steps:
                j = step - 1
                max_iter = record.trace[record.summary_source_steps[j] + 1]["last_max_iter"]
                start = emb[j]
                ref = replayer.run_block(start, max_iter)
                ref_scores = partition_scores(gold, ref.final_labels)
                tied_cl = tied_clusters(start)
                # Argmin ties go to the first of identical centers, so tied points sit in a tied cluster at t=0.
                tied_pts = tied_cl[ref.start.labels]
                for tied, rels in ((False, RELS), (True, TIED_RELS)):
                    for rel in rels:
                        for draw in range(draws):
                            rng = np.random.default_rng([seed, step, int(rel * 1e12), int(tied), draw])
                            noisy = perturb(start, rel, rng, tied)
                            frac_changed = fraction_changed(start, noisy)
                            assert rel < 1e-7 or frac_changed > 0, "perturbation left the start centers unchanged"
                            blk = replayer.run_block(noisy, max_iter)
                            changed = blk.final_labels != ref.final_labels
                            scores = partition_scores(gold, blk.final_labels)
                            rows.append(
                                {
                                    "run": name,
                                    "method": method,
                                    "seed": seed,
                                    "step": step,
                                    "noise": "tied" if tied else "iid",
                                    "rel": rel,
                                    "frac_elements_changed": frac_changed,
                                    "draw": draw,
                                    "n_points": len(gold),
                                    "n_tied_clusters": int(tied_cl.sum()),
                                    "n_changed": int(changed.sum()),
                                    "ari_vs_ref": float(adjusted_rand_score(ref.final_labels, blk.final_labels)),
                                    "d_acc": scores["acc"] - ref_scores["acc"],
                                    "d_nmi": scores["nmi"] - ref_scores["nmi"],
                                    "d_ari": scores["ari"] - ref_scores["ari"],
                                    "n_tied_points": int(tied_pts.sum()),
                                    "n_changed_tied_points": int(changed[tied_pts].sum()),
                                    "n_changed_other_points": int(changed[~tied_pts].sum()),
                                    "n_iter_ref": ref.n_iter,
                                    "n_iter": blk.n_iter,
                                }
                            )
            log(f"{name} {method} seed {seed}: {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per run, noise kind and size: mean labels changed (and share), mean |change| of each metric, mean ARI(ref)."""
    n = df.n_points
    g = df.assign(
        frac=df.n_changed / n,
        abs_acc=df.d_acc.abs(),
        abs_nmi=df.d_nmi.abs(),
        abs_ari=df.d_ari.abs(),
        frac_tied=df.n_changed_tied_points / df.n_tied_points.where(df.n_tied_points > 0),
        frac_other=df.n_changed_other_points / (n - df.n_tied_points),
    ).groupby(["run", "noise", "rel"])
    return g.agg(
        cells=("n_changed", "size"),
        n_tied_clusters=("n_tied_clusters", "mean"),
        n_changed=("n_changed", "mean"),
        frac_changed=("frac", "mean"),
        frac_changed_tied_points=("frac_tied", "mean"),
        frac_changed_other_points=("frac_other", "mean"),
        ari_vs_ref=("ari_vs_ref", "mean"),
        abs_d_acc=("abs_acc", "mean"),
        abs_d_nmi=("abs_nmi", "mean"),
        abs_d_ari=("abs_ari", "mean"),
        sd_d_acc=("d_acc", "std"),
        max_abs_d_acc=("abs_acc", "max"),
        frac_elements_changed=("frac_elements_changed", "mean"),
        sd_frac_changed=("frac", "std"),
    ).reset_index()


def summary_md(summary: pd.DataFrame) -> str:
    """Markdown table of `summarize` output."""
    cols = {
        "run": "model",
        "noise": "noise",
        "rel": "rel. noise",
        "frac_elements_changed": "elements changed",
        "n_tied_clusters": "tied clusters",
        "frac_changed": "labels changed",
        "frac_changed_tied_points": "of tied points",
        "frac_changed_other_points": "of other points",
        "ari_vs_ref": "ARI vs reference",
        "abs_d_acc": "mean abs dACC",
        "abs_d_nmi": "mean abs dNMI",
        "abs_d_ari": "mean abs dARI",
    }
    df = summary[list(cols)].rename(columns=cols)
    df["model"] = df["model"].str.removeprefix("clinc150_")
    lines = ["| " + " | ".join(df.columns) + " |", "| --- | --- | ---: " + "| ---: " * (len(df.columns) - 3) + "|"]
    for row in df.itertuples(index=False):
        cells = [row[0], row[1], f"{row[2]:.0e}", f"{row[3]:.2g}", f"{row[4]:.1f}"]
        cells += ["n/a" if pd.isna(v) else f"{v:.4f}" if i >= 8 else f"{v:.3f}" for i, v in enumerate(row[5:], start=5)]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--summary-embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 3, 5], help="1-based summary steps")
    parser.add_argument("--draws", type=int, default=2)
    args = parser.parse_args()

    start = time.time()
    df = run(
        args.run_dirs,
        args.cache_dir,
        SummaryEmbeddings.load(args.summary_embeddings),
        set(args.seeds),
        args.steps,
        args.draws,
        lambda m: print(m, file=sys.stderr, flush=True),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{PREFIX}.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / f"{PREFIX}_summary.csv", index=False)
    (out / f"{PREFIX}_summary.md").write_text(summary_md(summary))
    print(summary.to_string(index=False))
    print(f"{time.time() - start:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
