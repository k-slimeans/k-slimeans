"""Counterfactual summary replacements at step 1 (issue #47, stage 1): do the 4B's flagged summaries cause its scores?

Replays the first summary step of the logged runs offline (`kslimeans.replay`, no language model, no encoding: the
summary embeddings come from `scripts/embed_logged_summaries.py`) with rows of the start centers edited
(`kslimeans.counterfactual` defines the flags and the variants a, b, c, d, m, r, rd and ref). Every variant is
measured after the replacement and after the Lloyd block, for the logged block length and run to convergence.

Writes under `--out-dir` (small, committed), prefix `counterfactual_swaps_`:

    flags.csv        per seed and flag set: clusters flagged, points in them, overlaps
    fidelity.csv     per seed and model: the replayed step-1 partition against the logged one
    per_seed.csv     per seed, variant, flag set and setting (control draws averaged), the measures of both stages
    draws.csv        spread over the control draws (sd of end ACC/NMI/ARI/J)
    paired.csv       paired differences against (a) or (m) with 95% t-intervals over seeds
    gap_closed.csv   share of the gap to k-means (ref) and to the 1.7B (m) that (b), (c), (d) close
    stability.csv    the same differences under tie-preserving perturbations of the CPU embeddings
    tables.md        the tables above as markdown

and, under `--work-dir` (default `results/counterfactual_swaps`, gitignored), every row including each control draw.

Usage:

    CUDA_VISIBLE_DEVICES="" python scripts/counterfactual_swaps.py \
        --own results/clinc150_qwen3-4b/<stamp> --other results/clinc150_qwen3-1.7b/<stamp> --cache-dir cache \
        --summary-embeddings results/replacement_diagnostic/summary_embeddings_cpu.npz --out-dir docs/results
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from kslimeans.counterfactual import (
    FLAG_SETS,
    N_DRAWS,
    collapse_draws,
    evaluate_seed,
    flag_sets,
    gap_closed,
    paired_differences,
)
from kslimeans.diagnostic import partition_scores, perturb
from kslimeans.replay import LloydReplayer, SummaryEmbeddings, fit_initial
from kslimeans.runlog import load_run, load_run_dataset

PREFIX = "counterfactual_swaps"
METHOD = "k-SLMmeans"
PERTURB_REL = 1e-6
"""Relative size of the tie-preserving noise: the measured CPU/GPU encoder difference (#33 section 2)."""


def _seed_job(job: dict) -> dict:
    """Everything for one seed: flags, fidelity, the variants and the perturbed reruns. Runs in a worker process."""
    seed = job["seed"]
    own_log, other_log = load_run(job["own"]), load_run(job["other"])
    emb = SummaryEmbeddings.load(job["embeddings"])
    ds, X = load_run_dataset(own_log, cache_dir=job["cache_dir"])
    gold = np.asarray(ds.labels)
    own_run, other_run = own_log.config["name"], other_log.config["name"]
    rec_own, rec_other = own_log.records[(METHOD, seed)], other_log.records[(METHOD, seed)]
    own, other = emb.steps(own_run, METHOD, seed)[0], emb.steps(other_run, METHOD, seed)[0]
    by_step = {row["step"]: row for row in rec_own.trace}
    fixed_iter = by_step[1]["last_max_iter"]
    k = len(own)
    pre, _ = fit_initial(X, k, by_step[0]["last_max_iter"], seed)
    for rec in (rec_own, rec_other):  # the pairing the variants rest on: one shared, exactly replayed step-0 partition
        assert np.array_equal(pre.labels, rec.assignments_evolution[0]), "step 0 does not replay exactly"
        assert rec.summary_source_steps[0] == 0
    replayer = LloydReplayer(X)

    flags = flag_sets(rec_own.summaries_evolution[0], own, X, rec_own.source_partition(0))
    sizes = np.bincount(pre.labels, minlength=k)
    flag_rows = []
    for name, mask in flags.items():
        flag_rows.append(
            {
                "seed": seed,
                "flag": name,
                "n_flagged": int(mask.sum()),
                "points_in_flagged": int(sizes[mask].sum()),
                "n_dup_and_meta": int((flags["dup"] & flags["meta"]).sum()),
                "n_generic_not_union": int((flags["generic"] & ~flags["union"]).sum()),
                "n_distinct_summaries": len(set(rec_own.summaries_evolution[0])),
            }
        )

    fidelity = []
    for model, rec, start in (("own", rec_own, own), ("other", rec_other, other)):
        block = replayer.run_block(start, fixed_iter)
        logged = rec.assignments_evolution[1]
        fidelity.append(
            {
                "seed": seed,
                "model": model,
                "n_label_mismatch": int(np.count_nonzero(block.final_labels != logged)),
                "n_iter": block.n_iter,
                "logged_n_iter": by_step[1]["last_n_iter"] if model == "own" else None,
                "logged_acc": partition_scores(gold, logged)["acc"],
                "replayed_acc": partition_scores(gold, block.final_labels)["acc"],
                "logged_nmi": partition_scores(gold, logged)["nmi"],
                "replayed_nmi": partition_scores(gold, block.final_labels)["nmi"],
            }
        )

    kwargs = dict(fixed_iter=fixed_iter, control_seed=seed)
    rows = pd.DataFrame(evaluate_seed(replayer, pre, gold, own, other, flags, n_draws=job["draws"], **kwargs))
    rows.insert(0, "perturbation", 0)
    perturbed = []
    for p in range(1, job["perturbations"] + 1):
        rng = np.random.default_rng(1000 * seed + p)
        own_p, other_p = perturb(own, PERTURB_REL, rng, tied=True), perturb(other, PERTURB_REL, rng, tied=True)
        out = evaluate_seed(replayer, pre, gold, own_p, other_p, flags, n_draws=0, reference=False, **kwargs)
        out = pd.DataFrame(out)
        out.insert(0, "perturbation", p)
        perturbed.append(out)
    rows = pd.concat([rows, *perturbed], ignore_index=True)
    rows.insert(1, "seed", seed)
    return {"seed": seed, "rows": rows, "flags": flag_rows, "fidelity": fidelity}


def comparisons():
    """`(label, variant, flag, baseline_variant, baseline_flag)` of every reported difference."""
    out = [
        ("ref vs a: no replacement vs the 4B as logged", "ref", "", "a", ""),
        ("m vs a: 1.7B vs 4B as logged", "m", "", "a", ""),
    ]
    for flag in FLAG_SETS:
        out += [
            (f"b/{flag}: flagged keep k-means center", "b", flag, "a", ""),
            (f"c/{flag}: flagged take the 1.7B summary", "c", flag, "a", ""),
            (f"d/{flag}: control, random unflagged keep k-means center", "d", flag, "a", ""),
            (f"b-d/{flag}: (b) minus its control", "b", flag, "d", flag),
            (f"c-d/{flag}: (c) minus the control", "c", flag, "d", flag),
            (f"r/{flag}: 1.7B with the 4B's flagged summaries vs 1.7B as logged", "r", flag, "m", ""),
            (f"rd/{flag}: control, 4B summaries in random unflagged clusters of the 1.7B", "rd", flag, "m", ""),
            (f"r-rd/{flag}: (r) minus its control", "r", flag, "rd", flag),
        ]
    return out


def stability_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean differences of b, c, r (vs a, a, m) under each perturbation, against the unperturbed ones."""
    comps = [c for c in comparisons() if c[1] in ("b", "c", "r") and c[3] in ("a", "m")]
    out = []
    for p, sub in rows.groupby("perturbation"):
        sub, _ = collapse_draws(sub.drop(columns="perturbation"))
        paired = paired_differences(sub, comps, metrics=("end_acc", "end_nmi"))
        out.append(paired.assign(perturbation=p))
    paired = pd.concat(out)
    key = ["comparison", "setting", "metric"]
    ref = paired[paired.perturbation == 0].set_index(key)["mean"].rename("reference")
    pert = paired[paired.perturbation > 0].join(ref, on=key)
    pert["same_sign"] = np.sign(pert["mean"]) == np.sign(pert["reference"])
    grouped = pert.groupby(key)
    table = pd.concat(
        [
            ref,
            grouped["mean"].mean().rename("perturbed_mean"),
            grouped["mean"].min().rename("perturbed_min"),
            grouped["mean"].max().rename("perturbed_max"),
            grouped["same_sign"].sum().rename("n_same_sign"),
            grouped.size().rename("n_perturbations"),
        ],
        axis=1,
    )
    return table.reset_index()


def build(rows: pd.DataFrame, flag_rows, fidelity):
    ref_rows = rows[rows.perturbation == 0].drop(columns="perturbation")
    per_seed, spreads = collapse_draws(ref_rows)
    paired = paired_differences(per_seed, comparisons())
    gap = []
    for flag in FLAG_SETS:
        for variant in ("b", "c", "d"):
            for setting in ("fixed", "converged"):
                for metric in ("end_acc", "end_nmi"):
                    gap.append(
                        {
                            "flag": flag,
                            "variant": variant,
                            "setting": setting,
                            "metric": metric,
                            "closes_of_gap_to_kmeans": gap_closed(per_seed, variant, flag, "a", "ref", metric, setting),
                            "closes_of_gap_to_1.7b": gap_closed(per_seed, variant, flag, "a", "m", metric, setting),
                        }
                    )
    sd_cols = [c for c in spreads.columns if c.endswith("_sd")]
    controls = spreads[spreads.variant.isin(["d", "rd"])]
    draws = controls.groupby(["variant", "flag", "setting"])[sd_cols].mean().reset_index()
    return {
        "flags": pd.DataFrame(flag_rows),
        "fidelity": pd.DataFrame(fidelity),
        "per_seed": per_seed,
        "draws": draws,
        "paired": paired,
        "gap_closed": pd.DataFrame(gap),
        "stability": stability_table(rows),
    }


def tables_markdown(frames: dict[str, pd.DataFrame]) -> str:
    from replacement_diagnostic import md_table

    paired = frames["paired"]
    parts = []
    for setting in ("fixed", "converged"):
        parts.append(
            f"### Paired differences, {setting} block (mean over seeds [95% t-interval] and seeds above 0; "
            "positive = better than the baseline)\n"
        )
        sub = paired[(paired.setting == setting) & paired.metric.isin(["end_acc", "end_nmi", "end_ari", "end_j_rel"])]

        def cell(r):
            return f"{r['mean']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] {int(r.n_positive)}/{int(r.n_seeds)}"

        wide = sub.assign(cell=sub.apply(cell, axis=1)).pivot(index="comparison", columns="metric", values="cell")
        order = list(dict.fromkeys(sub.comparison))
        columns = [m for m in ("end_acc", "end_nmi", "end_ari", "end_j_rel") if m in wide.columns]
        wide = wide.loc[order, columns].reset_index()
        parts.append(md_table(wide))
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--own", required=True, help="run directory of the summarizer under test (the 4B)")
    parser.add_argument("--other", required=True, help="run directory of the other summarizer (the 1.7B)")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--summary-embeddings", required=True)
    parser.add_argument("--out-dir", default="docs/results")
    parser.add_argument("--work-dir", default="results/counterfactual_swaps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--draws", type=int, default=N_DRAWS, help="random control draws per seed and flag set")
    parser.add_argument("--perturbations", type=int, default=3, help="tie-preserving perturbed reruns of b, c, r")
    parser.add_argument("--workers", type=int, default=5, help="seeds replayed in parallel")
    args = parser.parse_args()

    # Fix the thread count of every worker before sklearn is imported there, so the run is repeatable.
    os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) // args.workers))
    jobs = [
        {
            "seed": s, "own": args.own, "other": args.other, "cache_dir": args.cache_dir,
            "embeddings": args.summary_embeddings, "draws": args.draws, "perturbations": args.perturbations,
        }
        for s in args.seeds
    ]  # fmt: skip
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_seed_job, jobs))
    print(f"replayed {len(jobs)} seeds in {(time.time() - t0) / 60:.1f} min")
    rows = pd.concat([r["rows"] for r in results], ignore_index=True)
    frames = build(rows, [f for r in results for f in r["flags"]], [f for r in results for f in r["fidelity"]])

    work_dir, out_dir = Path(args.work_dir), Path(args.out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(work_dir / f"{PREFIX}_rows.csv.gz", index=False)
    for name, frame in frames.items():
        frame.to_csv(out_dir / f"{PREFIX}_{name}.csv", index=False)
    (out_dir / f"{PREFIX}_tables.md").write_text(tables_markdown(frames))


if __name__ == "__main__":
    main()
