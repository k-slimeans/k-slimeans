"""Stage 2 of #47: what the guarded Qwen3-4B run changed, from the logged runs only (no model, no GPU, no encoding).

Compares the guarded 4B (`configs/clinc150_guard.yaml`) with the #31 runs of the same seeds. All of them share the
step-0 partition (`step0_sha1`). Writes, under `--out-dir` (small, committed):

    summary_guard_paired.csv        guarded 4B minus each reference (the #31 4B, k-means, LSA, the 1.7B), ACC/NMI/ARI,
                                    paired by seed, 95% t-interval (`kslimeans.aggregate.paired_differences`)
    summary_guard_gap_closed.csv    share of each reference's gap to the #31 4B that the guard closes (ratio of means)
    summary_guard_counts.csv        per seed and summary step: clusters guarded, by flag (duplicate, mentions, both), and
                                    the same flags counted on the #31 4B's own summaries (not applied)
    summary_guard_steps.csv         per seed and partition (fit): ACC/NMI/ARI of the guarded and the #31 4B, ARI between
                                    the two partitions, points that differ, summarizer calls and share of points moved
    summary_guard_steps_paired.csv  the per-fit score differences (guarded minus #31 4B) over seeds, with intervals
    summary_guard_check.txt         consistency checks (logged vs recomputed flags and calls, step-1 texts, step0_sha1,
                                    step 1 against stage 1's variant b)

Usage:

    python scripts/summary_guard_report.py --guard results/clinc150_guard_qwen3-4b/<stamp> \\
        --unguarded results/clinc150_qwen3-4b/<stamp> --small results/clinc150_qwen3-1.7b/<stamp> \\
        --lsa results/clinc150_lsa/<stamp> --cache-dir cache --out-dir docs/results

Indexing (see `kslimeans.runlog`): trace row `t` and `assignments_evolution[t]` belong to the same fit; fit 0 is the
initial k-means, fit `t >= 1` is the Lloyd block that followed summary step `t - 1`, whose summaries were written from
fit `t - 1`. The guard's flags at fit `t` are those of summary step `t - 1`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from kslimeans.aggregate import METRICS, method_label, paired_differences, per_seed_table, step0_sha1_summary
from kslimeans.diagnostic import cluster_flags, partition_scores
from kslimeans.rq2 import t_interval
from kslimeans.runlog import RunRecord, load_run, load_run_dataset, members_by_cluster

K = 150
GUARD_METHOD = "k-SLMmeans-guard"
BASE_METHOD = "k-SLMmeans"
STAGE1_TOL = 1e-9


def guard_counts(record: RunRecord) -> pd.DataFrame:
    """One row per summary step of a guarded record: clusters guarded and why.

    Columns: `seed, step` (the fit the step's block produced, 1-based), `n_guarded` (union), `n_duplicate`,
    `n_mentions`, `n_both`, `n_duplicate_only`, `n_mentions_only`, `share` (of all clusters) and `n_calls`. The counts
    come from the trace; `n_both = n_duplicate + n_mentions - n_guarded`. (`build` adds the #31 4B's flag counts.)
    """
    rows = []
    for row in record.trace:
        if row["step"] < 1:
            continue
        dup, mentions, union = row["n_guarded_duplicate"], row["n_guarded_mentions"], row["n_guarded"]
        rows.append(
            {
                "seed": record.seed,
                "step": row["step"],
                "n_guarded": union,
                "n_duplicate": dup,
                "n_mentions": mentions,
                "n_both": dup + mentions - union,
                "n_duplicate_only": union - mentions,
                "n_mentions_only": union - dup,
                "share": union / len(record.summaries_evolution[row["step"] - 1]),
                "n_calls": row["n_calls"],
            }
        )
    return pd.DataFrame(rows)


def recomputed_flags(summaries: list[str | None]) -> tuple[int, int, int, list[int]]:
    """`(n_duplicate, n_mentions, n_union, flagged ids)` of one step's summaries by the surface rules."""
    f = cluster_flags(summaries)
    return int(f["dup"].sum()), int(f["meta"].sum()), int(f["flagged"].sum()), np.flatnonzero(f["flagged"]).tolist()


def expected_calls(assignments_evolution: list[np.ndarray], n_steps: int, k: int = K) -> list[int]:
    """Summarizer calls per summary step implied by the partitions: non-empty clusters whose exact member set no
    earlier summary step of the run had (the memo cache keys on the cluster's document ids)."""
    seen: set[tuple[int, ...]] = set()
    calls = []
    for j in range(n_steps):
        fresh = {tuple(m.tolist()) for m in members_by_cluster(assignments_evolution[j], k) if len(m)} - seen
        calls.append(len(fresh))
        seen |= fresh
    return calls


def base_flag_counts(record: RunRecord) -> pd.DataFrame:
    """The same flags counted (not applied) on an unguarded record's summaries: `step`, `n_flagged_base`,
    `n_duplicate_base`, `n_mentions_base` per summary step, `step` numbered as in `guard_counts`."""
    rows = []
    for j, summaries in enumerate(record.summaries_evolution):
        dup, mentions, union, _ = recomputed_flags(summaries)
        rows.append(
            {
                "step": j + 1,
                "n_flagged_base": union,
                "n_duplicate_base": dup,
                "n_mentions_base": mentions,
            }
        )
    return pd.DataFrame(rows)


def step_table(guard: RunRecord, base: RunRecord, small: RunRecord, gold: np.ndarray) -> pd.DataFrame:
    """One row per fit `t` (0 = the shared initial fit) of one seed, guarded against the #31 4B.

    `ari_between` is the ARI of the two partitions; `ari_guard_small` and `ari_base_small` give a scale for it, the
    ARI of each 4B partition with the #31 1.7B's partition of the same fit.
    """
    rows = []
    lengths = {len(r.assignments_evolution) for r in (guard, base, small)} | {len(guard.trace), len(base.trace)}
    if len(lengths) != 1:
        raise ValueError(f"seed {guard.seed}: the runs have different numbers of fits {sorted(lengths)}")
    for t, (a, b, c) in enumerate(
        zip(guard.assignments_evolution, base.assignments_evolution, small.assignments_evolution, strict=True)
    ):
        sg, sb = partition_scores(gold, a), partition_scores(gold, b)
        g_trace, b_trace = guard.trace[t], base.trace[t]
        row = {
            "seed": guard.seed,
            "step": t,
            "ari_between": float(adjusted_rand_score(a, b)),
            "ari_guard_small": float(adjusted_rand_score(a, c)),
            "ari_base_small": float(adjusted_rand_score(b, c)),
            "n_differ": int((a != b).sum()),
            "n_calls_guard": g_trace["n_calls"],
            "n_calls_base": b_trace["n_calls"],
            "frac_changed_guard": g_trace["frac_changed"],
            "frac_changed_base": b_trace["frac_changed"],
        }
        for metric in METRICS:
            row[f"{metric}_guard"] = sg[metric]
            row[f"{metric}_base"] = sb[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def steps_paired(steps: pd.DataFrame) -> pd.DataFrame:
    """Per fit and metric: mean levels and the paired difference (guarded minus #31 4B) over seeds, 95% t-interval."""
    rows = []
    for t, group in steps.groupby("step"):
        for metric in METRICS:
            diff = group[f"{metric}_guard"] - group[f"{metric}_base"]
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
            _, lo, hi = t_interval(mean, sd, len(diff))
            rows.append(
                {
                    "step": t,
                    "metric": metric,
                    "mean_guard": float(group[f"{metric}_guard"].mean()),
                    "mean_base": float(group[f"{metric}_base"].mean()),
                    "mean_diff": mean,
                    "sd": sd,
                    "ci_low": lo,
                    "ci_high": hi,
                    "wins": int((diff > 0).sum()),
                    "n": len(diff),
                }
            )
    return pd.DataFrame(rows)


def gap_closed(paired: pd.DataFrame, base_label: str, references: list[str]) -> pd.DataFrame:
    """Share of the gap from the #31 4B to each reference that the guard closes, per metric.

    `paired` holds the guarded method's differences against each reference (`mean_diff` = guard minus reference).
    `gain = mean(guard - base)` and the gap is `mean(reference - base) = gain - mean(guard - reference)`. The share
    is `gain / gap`, a ratio of means; NaN when the reference is not above the base on average.
    """
    rows = []
    for metric in METRICS:
        gain = float(paired[(paired.reference == base_label) & (paired.metric == metric)].mean_diff.iloc[0])
        for ref in references:
            vs_ref = float(paired[(paired.reference == ref) & (paired.metric == metric)].mean_diff.iloc[0])
            gap = gain - vs_ref
            rows.append(
                {
                    "metric": metric,
                    "reference": ref,
                    "gap_to_base": gap,
                    "gain": gain,
                    "share_closed": gain / gap if gap > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build(
    guard_dir, base_dir, small_dir, lsa_dir, cache_dir, stage1_csv=None
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    guard_log, base_log, small_log = load_run(guard_dir), load_run(base_dir), load_run(small_dir)
    ds, _ = load_run_dataset(guard_log, cache_dir=cache_dir)
    gold = np.asarray(ds.labels)
    seeds = guard_log.seeds
    for name, log in (("unguarded 4B", base_log), ("1.7B", small_log)):
        if log.seeds != seeds:
            raise ValueError(f"seed sets differ: guarded {seeds}, {name} {log.seeds}")

    labels = {
        "guard": method_label(guard_log.config),
        "base": method_label(base_log.config),
        "small": method_label(small_log.config),
        "kmeans": "k-means",
        "lsa": "k-NLPmeans (LSA)",
    }
    per_seed = per_seed_table([lsa_dir, base_dir, small_dir, guard_dir])
    for label, group in per_seed.groupby("method"):
        if sorted(group.seed) != seeds:
            raise ValueError(f"seed sets differ: guarded {seeds}, {label} {sorted(group.seed)}")
    references = [labels["base"], labels["kmeans"], labels["lsa"], labels["small"]]
    paired = paired_differences(per_seed, references)
    paired = paired[paired.method == labels["guard"]].drop(columns="method").reset_index(drop=True)

    counts, steps, checks = [], [], []
    for seed in seeds:
        g, b = guard_log.record(GUARD_METHOD, seed), base_log.record(BASE_METHOD, seed)
        counts.append(guard_counts(g).merge(base_flag_counts(b), on="step"))
        steps.append(step_table(g, b, small_log.record(BASE_METHOD, seed), gold))
        n_steps = len(g.summaries_evolution)
        # Logged flags and calls against recomputation.
        logged = guard_counts(g)
        for j, summaries in enumerate(g.summaries_evolution):
            dup, men, union, ids = recomputed_flags(summaries)
            row = logged.iloc[j]
            ok = (dup, men, union) == (row.n_duplicate, row.n_mentions, row.n_guarded)
            checks.append(f"seed {seed} summary step {j}: trace counts match the surface rules: {ok}")
            if not ok:
                raise ValueError(f"seed {seed} step {j}: logged guard counts differ from the surface rules")
        for name, rec in (("guarded", g), ("#31 4B", b)):
            exp = expected_calls(rec.assignments_evolution, n_steps, k=len(g.summaries_evolution[0]))
            got = [r["n_calls"] for r in rec.trace[1:]]
            checks.append(f"seed {seed} {name}: calls per step {got}, implied by the partitions {exp}, equal: {exp == got}")
            if exp != got:
                raise ValueError(f"seed {seed} {name}: logged calls {got} differ from those implied by the partitions {exp}")
        same = g.summaries_evolution[0] == b.summaries_evolution[0]
        checks.append(f"seed {seed}: step-1 summaries identical in both runs: {same}")
        same0 = bool((g.assignments_evolution[0] == b.assignments_evolution[0]).all())
        checks.append(f"seed {seed}: step-0 partitions identical in both runs: {same0}")
        if not (same and same0):
            raise ValueError(f"seed {seed}: the runs do not share step 0 and the step-1 summaries")
    if stage1_csv is not None and Path(stage1_csv).exists():
        stage1 = pd.read_csv(stage1_csv, keep_default_na=False)
        b = stage1[(stage1.variant == "b") & (stage1.flag == "meta") & (stage1.setting == "fixed")].set_index("seed")
        first = pd.concat(steps, ignore_index=True).query("step == 1").set_index("seed")
        if sorted(b.index) != sorted(first.index):
            raise ValueError(f"seed sets differ: guarded {sorted(first.index)}, stage 1 {sorted(b.index)}")
        for metric in METRICS:
            err = float((first[f"{metric}_guard"] - b.loc[first.index, f"end_{metric}"]).abs().max())
            checks.append(f"step-1 {metric} of the guarded run vs stage 1 variant b (meta, fixed): max abs diff {err:.2e}")
            if not err <= STAGE1_TOL:
                raise ValueError(f"step-1 {metric} differs from stage 1 variant b by {err:.2e} (> {STAGE1_TOL:.0e})")
    checks.append(step0_sha1_summary([lsa_dir, base_dir, small_dir, guard_dir]))

    counts_df = pd.concat(counts, ignore_index=True)
    steps_df = pd.concat(steps, ignore_index=True)
    frames = {
        "paired": paired,
        "gap_closed": gap_closed(paired, labels["base"], [labels["kmeans"], labels["lsa"], labels["small"]]),
        "counts": counts_df,
        "steps": steps_df,
        "steps_paired": steps_paired(steps_df),
    }
    return frames, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--guard", required=True, help="guarded 4B run directory")
    parser.add_argument("--unguarded", required=True, help="#31 4B run directory")
    parser.add_argument("--small", required=True, help="#31 1.7B run directory")
    parser.add_argument("--lsa", required=True, help="#31 LSA run directory (carries the k-means rows)")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument(
        "--stage1", default="docs/results/counterfactual_swaps_per_seed.csv", help="stage 1 per-seed CSV, for a check"
    )
    parser.add_argument("--out-dir", default="docs/results")
    args = parser.parse_args()

    frames, checks = build(args.guard, args.unguarded, args.small, args.lsa, args.cache_dir, args.stage1)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out / f"summary_guard_{name}.csv", index=False, float_format="%.6g")
    (out / "summary_guard_check.txt").write_text("\n".join(checks) + "\n")
    print("\n".join(checks))


if __name__ == "__main__":
    main()
