"""Counterfactual summary replacements at step 1, offline (issue #47, stage 1).

Question: do the 4B's flagged summaries *cause* its lower scores? At step 1 every method of a seed starts from the
same step-0 partition (`step0_sha1`), so cluster `c` has the same members whatever summarized it, and the start centers
of the step-1 Lloyd block can be edited row by row and replayed exactly (`kslimeans.replay`): no language model.

Flags (defined before any outcome was computed; `flag_sets`). Every flag is a per-cluster property of one step's
summaries, here the 4B's step-1 summaries written from the step-0 partition:

- `dup`: (i) the summary is identical to another cluster's (`surface.duplicate_groups`; every copy is flagged);
- `meta`: (ii) it talks about the cluster (`surface.mentions_cluster`: "cluster" or "summar", any case);
- `union`: (i) or (ii);
- `generic`: (iii) #34's `flag_generic` (`summary_quality.generic_flags`): duplicate, near-duplicate (cosine of the
  summary embeddings at least `NEAR_DUPLICATE_COS`), mentions the cluster, or a non-positive specificity margin.
  It contains `union`.

Variants of a step-1 replacement, with `own` the summarizer under test (the 4B) and `other` the other one (the 1.7B).
Each is one row-edit of the start centers; the block that follows is the logged one (up to `fixed_iter` Lloyd
iterations, sklearn's stops applied) and, separately, a run to convergence:

- `a`: `own` as logged;
- `b`: flagged clusters keep their k-means center (`pre.centers[c]`), the others take `own`;
- `c`: flagged clusters take `other`'s summary embedding of the same cluster;
- `d`: control. The same number of *unflagged* clusters, drawn at random, keep their k-means center;
- `m`: `other` as logged; `r`: `other` with `own`'s flagged summaries put into the same clusters (reverse direction);
  `rd`: control, `own`'s summaries in the same number of random unflagged clusters of `other`;
- `ref`: no replacement, plain Lloyd from `pre.centers` (what #33 calls the counterfactual).

The swap is an intervention on one step of one seed. It says what the summaries do to that step's partition, not what
they do over a whole run.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .diagnostic import CONVERGENCE_CAP, cluster_flags, partition_scores
from .replay import Block, LloydReplayer, State, assigned_objective
from .rq2 import t_interval
from .summary_quality import generic_flags

FLAG_SETS = ("dup", "meta", "union", "generic")
N_DRAWS = 20
"""Random control draws per seed and flag set."""

END_METRICS = ("end_acc", "end_nmi", "end_ari", "end_j")


def flag_sets(
    summaries: Sequence[str | None], S: np.ndarray, X: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """The boolean per-cluster flag sets named in the module docstring, keyed by `FLAG_SETS`."""
    surface = cluster_flags(list(summaries))
    return {
        "dup": surface["dup"],
        "meta": surface["meta"],
        "union": surface["flagged"],
        "generic": generic_flags(summaries, S, X, labels)["generic"],
    }


def swap_rows(base: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of `base` whose rows where `mask` is true are `source`'s rows."""
    out = base.copy()
    out[mask] = source[mask]
    return out


def random_controls(flagged: np.ndarray, n_draws: int, seed: int) -> list[np.ndarray]:
    """`n_draws` boolean masks, each marking as many *unflagged* clusters as `flagged` marks, drawn without replacement.

    Raises `ValueError` when fewer clusters are unflagged than flagged (no such control exists).
    """
    n, pool = int(flagged.sum()), np.flatnonzero(~flagged)
    if n > len(pool):
        raise ValueError(f"{n} flagged clusters but only {len(pool)} unflagged ones to draw a control from")
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n_draws):
        mask = np.zeros(len(flagged), dtype=bool)
        mask[rng.choice(pool, size=n, replace=False)] = True
        masks.append(mask)
    return masks


def truncate(block: Block, max_iter: int, replayer: LloydReplayer) -> Block:
    """The block `run_block(start, max_iter)` would give, taken from a longer one run with the same stops.

    Exact because the stops only look back: a block that stops within `max_iter` is unchanged, and one that does not
    is cut at iteration `max_iter`. The cut iteration's objective is computed if the long block skipped it
    (`run_block(..., track_objective=False)`).
    """
    if block.n_iter <= max_iter:
        return block
    its = block.iterations[: max_iter + 1]
    last = its[-1]
    if np.isnan(last.objective):
        its[-1] = replace(last, objective=assigned_objective(replayer.X, last.centers, last.labels))
    return Block(iterations=its, stop="max_iter", final_labels=last.labels)


def _empty(labels: np.ndarray, k: int) -> int:
    return k - len(np.unique(labels))


def outcome_row(block: Block, pre: State, gold: np.ndarray) -> dict:
    """Measures of a replayed block: at the replacement (iteration 0) and at the block end.

    `repl_*` are the nearest-center assignment to the start centers: objective, scores against gold, clusters left
    without a point, points whose cluster differs from `pre`. `end_*` are the fit's result (`final_labels`; the
    objective is that of the final centers). `end_moved` counts points that differ from `pre` net over the block.
    """
    k = len(pre.centers)
    first, last = block.start, block.end
    repl, end = partition_scores(gold, first.labels), partition_scores(gold, block.final_labels)
    return {
        "repl_j": first.objective,
        "repl_acc": repl["acc"],
        "repl_nmi": repl["nmi"],
        "repl_ari": repl["ari"],
        "repl_empty": _empty(first.labels, k),
        "repl_moved": int(np.count_nonzero(first.labels != pre.labels)),
        "end_j": last.objective,
        "end_acc": end["acc"],
        "end_nmi": end["nmi"],
        "end_ari": end["ari"],
        "end_empty": _empty(block.final_labels, k),
        "end_moved": int(np.count_nonzero(block.final_labels != pre.labels)),
        "end_ari_vs_pre": float(adjusted_rand_score(pre.labels, block.final_labels)),
        "n_iter": block.n_iter,
        "stop": block.stop,
    }


def evaluate_start(
    replayer: LloydReplayer,
    pre: State,
    gold: np.ndarray,
    start: np.ndarray,
    *,
    fixed_iter: int,
    cap: int = CONVERGENCE_CAP,
) -> list[dict]:
    """Two rows, `setting` "fixed" (the logged block, `fixed_iter` iterations) and "converged" (to sklearn's stop)."""
    converged = replayer.run_block(start, cap, track_objective=False)
    fixed = truncate(converged, fixed_iter, replayer)
    return [{"setting": s, **outcome_row(b, pre, gold)} for s, b in (("fixed", fixed), ("converged", converged))]


def evaluate_reference(
    replayer: LloydReplayer, pre: State, gold: np.ndarray, *, fixed_iter: int, cap: int = CONVERGENCE_CAP
) -> list[dict]:
    """The no-replacement counterfactual: plain Lloyd from `pre.centers`, `fixed_iter` iterations without stops
    (as #33) and run to convergence."""
    fixed = replayer.run_block(pre.centers, fixed_iter, stop=False, track_objective=False)
    converged = replayer.run_block(pre.centers, cap, track_objective=False)
    return [{"setting": s, **outcome_row(b, pre, gold)} for s, b in (("fixed", fixed), ("converged", converged))]


def evaluate_seed(
    replayer: LloydReplayer,
    pre: State,
    gold: np.ndarray,
    own: np.ndarray,
    other: np.ndarray,
    flags: dict[str, np.ndarray],
    *,
    fixed_iter: int,
    cap: int = CONVERGENCE_CAP,
    n_draws: int = N_DRAWS,
    control_seed: int = 0,
    reference: bool = True,
) -> list[dict]:
    """Rows (one per variant, flag set, draw and setting) for one seed; the variants are in the module docstring.

    `own`, `other` are the two summarizers' step-1 start centers, `flags` the `own` flag sets (`flag_sets`).
    `n_draws=0` skips the random controls. Rows carry `variant`, `flag` ("" for the unflagged variants), `draw`
    (control draw, -1 otherwise), `n_replaced` (rows edited) and `setting`.
    """
    rows: list[dict] = []

    def add(variant, flag, draw, n_replaced, outcomes):
        rows.extend({"variant": variant, "flag": flag, "draw": draw, "n_replaced": n_replaced, **o} for o in outcomes)

    seen: dict[bytes, list[dict]] = {}  # identical start centers (flag sets that coincide, empty flag sets) run once

    def run(start):
        key = hashlib.blake2b(np.ascontiguousarray(start).tobytes(), digest_size=16).digest()
        if key not in seen:
            seen[key] = evaluate_start(replayer, pre, gold, start, fixed_iter=fixed_iter, cap=cap)
        return seen[key]

    add("a", "", -1, 0, run(own))
    add("m", "", -1, 0, run(other))
    if reference:
        add("ref", "", -1, 0, evaluate_reference(replayer, pre, gold, fixed_iter=fixed_iter, cap=cap))
    for name, flagged in flags.items():
        n = int(flagged.sum())
        add("b", name, -1, n, run(swap_rows(own, pre.centers, flagged)))
        add("c", name, -1, n, run(swap_rows(own, other, flagged)))
        add("r", name, -1, n, run(swap_rows(other, own, flagged)))
        # One control per draw serves both directions: the same unflagged clusters are edited in `own` and `other`.
        for draw, mask in enumerate(random_controls(flagged, n_draws, control_seed) if n_draws else []):
            add("d", name, draw, n, run(swap_rows(own, pre.centers, mask)))
            add("rd", name, draw, n, run(swap_rows(other, own, mask)))
    return rows


# ------------------------------------------------------------------ summaries over draws and seeds


def collapse_draws(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (seed, variant, flag, setting [, perturbation]): the mean over control draws, and their spread.

    Returns `(means, spreads)`; `spreads` holds the sample sd over draws of `END_METRICS` (columns `<metric>_sd`,
    NaN for variants without draws). Non-control variants have one row and pass through unchanged.
    """
    keys = [c for c in ("perturbation", "seed", "variant", "flag", "setting") if c in rows.columns]
    numeric = [c for c in rows.columns if c not in keys + ["draw", "stop"] and pd.api.types.is_numeric_dtype(rows[c])]
    grouped = rows.groupby(keys, sort=False)
    means = grouped[numeric].mean().reset_index()
    means["n_draws"] = grouped.size().to_numpy()
    spreads = grouped[list(END_METRICS)].std(ddof=1).rename(columns=lambda c: f"{c}_sd").reset_index()
    return means, spreads


def paired_differences(
    per_seed: pd.DataFrame,
    comparisons: Iterable[tuple[str, str, str, str, str]],
    metrics: Sequence[str] = END_METRICS,
    settings: Sequence[str] = ("fixed", "converged"),
) -> pd.DataFrame:
    """Per-seed paired differences of variants against a baseline variant, with Student-t 95% intervals over seeds.

    `comparisons` are `(label, variant, flag, baseline_variant, baseline_flag)`. One row per comparison, setting and
    metric: the mean difference (variant minus baseline) over seeds, its sample sd, 95% interval (`rq2.t_interval`),
    the number of seeds and how many differences are positive. `end_j` is reported as a relative difference
    (difference over the baseline's objective) so it compares across seeds.
    """
    out = []
    for label, variant, flag, base, base_flag in comparisons:
        for setting in settings:
            sub = per_seed[per_seed.setting == setting]
            v = sub[(sub.variant == variant) & (sub.flag == flag)].set_index("seed")
            b = sub[(sub.variant == base) & (sub.flag == base_flag)].set_index("seed")
            seeds = v.index.intersection(b.index)
            for metric in metrics:
                diff = v.loc[seeds, metric] - b.loc[seeds, metric]
                if metric == "end_j":
                    diff = diff / b.loc[seeds, metric]
                mean = float(diff.mean())
                sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
                se, lo, hi = t_interval(mean, sd, len(diff))
                out.append(
                    {
                        "comparison": label,
                        "variant": variant,
                        "flag": flag,
                        "baseline": base,
                        "setting": setting,
                        "metric": metric + ("_rel" if metric == "end_j" else ""),
                        "mean": mean,
                        "sd": sd,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_seeds": len(diff),
                        "n_positive": int((diff > 0).sum()),
                    }
                )
    return pd.DataFrame(out)


def gap_closed(
    per_seed: pd.DataFrame, variant: str, flag: str, base: str, target: str, metric: str, setting: str
) -> float:
    """Share of the gap from `base` to `target` that `variant` closes: `mean(variant - base) / mean(target - base)`.

    Means over seeds, so it is a ratio of means; NaN when the target is not above the baseline on average.
    """
    sub = per_seed[per_seed.setting == setting]

    def mean_of(name, fl=""):
        return sub[(sub.variant == name) & (sub.flag == fl)].set_index("seed")[metric]

    v, b, t = mean_of(variant, flag), mean_of(base), mean_of(target)
    gap = float((t - b).mean())
    return float((v - b).mean() / gap) if gap > 0 else float("nan")
