"""Replacement-diagnostic measures: what a summary replacing a numerical centroid does (issue #33).

Pure functions on partitions, centers and a replayed `StepReplay` (`kslimeans.replay`); `step_tables` turns one
replayed step into the per-step, per-iteration, per-cluster and per-point rows the script writes. Conventions:

- `step` in the output is 1-based (step 1 is the first replacement); `source_step` is the trace step the summary
  was written from. Cluster ids are the run's own; at step 1 they correspond across methods of a seed (shared
  `step0_sha1` partition).
- "pre" is the state just before the replacement: the previous fit's partition and centers. "Replacement" is the
  nearest-center assignment to the summary embeddings (iteration 0 of the block); "block end" is the fit's result.
- RMS radius of cluster `c` is `sqrt(mean ||x - center_c||^2)` over its pre members (core's `_cluster_rms_radius`);
  distances are in those units where stated, as in `schedules.centroid_drift`.
- Gold direction. The majority gold label of each cluster is taken from the PRE partition (ties go to the smallest
  label; an empty cluster has none, -1). A point that moved from cluster `a` to `b` moves `toward` its gold label if
  `majority[b] == gold` and `majority[a] != gold`, `away` if `majority[a] == gold` and `majority[b] != gold`;
  otherwise `neutral` (both or neither majority equals its gold). `ends_in_own_majority` is `majority[b] == gold`.
- Boundary vs core. The pre margin of a point is `d2 - d1`, the distance to its second-nearest pre center minus
  that to the nearest. A point is a boundary point if its margin is in the lowest quartile of the step's points,
  else a core point.
- Flags are the surface rules of `kslimeans.surface`: `dup` (the summary is identical to another cluster's),
  `meta` (it mentions "cluster"/"summar") and `flagged` (either).
- Out of date. A displayed summary of cluster `c` is stale when the Jaccard similarity between the members it was
  written from and the members of `c` in the final partition is below `STALE_JACCARD`, or when its embedding's
  nearest final center belongs to another cluster.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

from .core import _cluster_rms_radius
from .metrics import cluster_metrics
from .replay import Block, StepReplay
from .schedules import centroid_drift
from .surface import duplicate_groups, mentions_cluster

STALE_JACCARD = 0.5
"""A summary whose source and final member sets overlap less than this (Jaccard) counts as out of date."""
BOUNDARY_QUANTILE = 0.25
"""Points with a pre margin at or below this quantile of the step's margins are boundary points."""


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of row `i` of `a` with row `i` of `b` (0 where either is the zero vector)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    norms = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
    dots = np.einsum("ij,ij->i", a64, b64)
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def nearest_center(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the nearest row of `centers` for every row of `points`."""
    return np.argmin(euclidean_distances(points, centers), axis=1)


def pre_margins(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """`d2 - d1` per point: distance to the second-nearest center minus the nearest (0 if there is one center)."""
    d = euclidean_distances(X, centers)
    if d.shape[1] < 2:
        return np.zeros(len(X))
    two = np.partition(d, 1, axis=1)[:, :2]
    return (two[:, 1] - two[:, 0]).astype(np.float64)


def majority_labels(labels: np.ndarray, gold: np.ndarray, k: int) -> np.ndarray:
    """Most common gold label of each cluster (smallest on ties), -1 for an empty cluster."""
    out = np.full(k, -1, dtype=np.int64)
    n_gold = int(gold.max()) + 1
    table = np.zeros((k, n_gold), dtype=np.int64)
    np.add.at(table, (labels, gold), 1)
    nonempty = table.sum(axis=1) > 0
    out[nonempty] = table[nonempty].argmax(axis=1)  # argmax takes the first (smallest) label on ties
    return out


def gold_direction(from_c: np.ndarray, to_c: np.ndarray, gold: np.ndarray, majority: np.ndarray) -> np.ndarray:
    """`"toward"`, `"away"` or `"neutral"` for each mover (see the module docstring)."""
    into = majority[to_c] == gold
    outof = majority[from_c] == gold
    return np.where(into & ~outof, "toward", np.where(outof & ~into, "away", "neutral"))


def jaccard_per_cluster(labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> np.ndarray:
    """Jaccard similarity of cluster `c` in `labels_a` and in `labels_b`, NaN where both are empty."""
    a = np.bincount(labels_a, minlength=k).astype(np.float64)
    b = np.bincount(labels_b, minlength=k).astype(np.float64)
    inter = np.bincount(labels_a[labels_a == labels_b], minlength=k).astype(np.float64)
    union = a + b - inter
    return np.divide(inter, union, out=np.full(k, np.nan), where=union > 0)


def cluster_flags(summaries: list[str | None]) -> dict[str, np.ndarray]:
    """Boolean per-cluster arrays `dup`, `meta`, `flagged` for one step's summaries."""
    dup = np.zeros(len(summaries), dtype=bool)
    for group in duplicate_groups(summaries):
        dup[group] = True
    meta = np.array([mentions_cluster(s) for s in summaries], dtype=bool)
    return {"dup": dup, "meta": meta, "flagged": dup | meta}


def partition_scores(gold: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """ACC (Hungarian), NMI and ARI of `labels` against `gold`, via `metrics.cluster_metrics`."""
    dummy = np.zeros((1, 1))
    scores = cluster_metrics(gold, labels, dummy, dummy, dummy)
    return {key: float(scores[key]) for key in ("acc", "nmi", "ari")}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _block_rows(block: Block, pre_labels: np.ndarray, k: int, kind: str) -> list[dict]:
    rows = []
    for it in block.iterations:
        sizes = np.bincount(it.labels, minlength=k)
        prev_sizes = np.bincount(block.iterations[max(it.t - 1, 0)].labels, minlength=k)
        jac = jaccard_per_cluster(pre_labels, it.labels, k)
        rows.append(
            {
                "kind": kind,
                "t": it.t,
                "objective": it.objective,
                "n_reassigned": it.n_reassigned,
                "center_shift": it.center_shift,
                "n_vs_pre": int(np.count_nonzero(it.labels != pre_labels)),
                "ari_vs_pre": float(adjusted_rand_score(pre_labels, it.labels)),
                "jaccard_mean": float(np.nanmean(jac)),
                "jaccard_median": float(np.nanmedian(jac)),
                "n_empty": int(np.count_nonzero(sizes == 0)),
                "n_relocated": int(np.count_nonzero(prev_sizes == 0)) if it.t > 0 else 0,
            }
        )
    return rows


def step_tables(
    replay: StepReplay,
    X: np.ndarray,
    gold: np.ndarray,
    summaries: list[str | None],
    source_labels: np.ndarray,
    *,
    keep_points: bool = True,
) -> dict:
    """The diagnostic rows for one replayed step.

    Returns `{"step": dict, "iterations": list[dict], "clusters": list[dict], "points": list[dict]}`; the caller
    adds `method`, `seed` and `step` keys. `source_labels` is the partition the summaries were written from (the
    pre partition), `summaries` the step's texts. `points` (one row per mover at the replacement or at the end
    of the block) is empty with `keep_points=False`.
    """
    pre, block, cf = replay.pre, replay.block, replay.counterfactual
    k = len(replay.start_centers)
    n = len(X)
    start, end = block.start, block.end
    pre_labels = pre.labels
    flags = cluster_flags(summaries)
    sizes_pre = np.bincount(pre_labels, minlength=k)
    sizes_start = np.bincount(start.labels, minlength=k)
    sizes_end = np.bincount(block.final_labels, minlength=k)
    final_centers = block.final_centers

    # Centroid movement, in pre RMS radii.
    radius = _cluster_rms_radius(X, pre_labels, pre.centers)
    drift = centroid_drift(replay.start_centers, pre.centers, radius)
    cos = cosine_rows(replay.start_centers, pre.centers)
    nearest_pre = nearest_center(replay.start_centers, pre.centers)
    own_nearest = nearest_pre == np.arange(k)
    to_start = np.linalg.norm(replay.start_centers.astype(np.float64) - pre.centers, axis=1)
    to_end = np.linalg.norm(replay.start_centers.astype(np.float64) - final_centers, axis=1)
    end_vs_pre = centroid_drift(final_centers, pre.centers, radius)
    to_end_radii = centroid_drift(replay.start_centers, final_centers, radius)
    nonempty = sizes_pre > 0
    finite = nonempty & np.isfinite(drift)

    # Movers.
    moved0 = start.labels != pre_labels
    moved_end = block.final_labels != pre_labels
    margins = pre_margins(X, pre.centers)
    boundary = margins <= np.quantile(margins, BOUNDARY_QUANTILE)
    majority = majority_labels(pre_labels, gold, k)
    dir0 = gold_direction(pre_labels, start.labels, gold, majority)
    dir_end = gold_direction(pre_labels, block.final_labels, gold, majority)
    flagged_point = flags["flagged"][pre_labels]

    def rate(mask: np.ndarray, moved: np.ndarray) -> float:
        return _safe_div(float(np.count_nonzero(moved & mask)), float(np.count_nonzero(mask)))

    own0 = majority[start.labels] == gold
    own_end = majority[block.final_labels] == gold
    scores_pre, scores_end = partition_scores(gold, pre_labels), partition_scores(gold, block.final_labels)
    scores_cf = partition_scores(gold, cf.final_labels) if cf is not None else None
    cf_end = cf.end if cf is not None else None
    jac_end = jaccard_per_cluster(pre_labels, block.final_labels, k)
    empty0 = sizes_start == 0
    first_nonempty = np.full(k, -1, dtype=np.int64)
    for it in block.iterations:
        seen = (np.bincount(it.labels, minlength=k) > 0) & (first_nonempty < 0)
        first_nonempty[seen] = it.t

    moved_away = finite & (to_start > 0)
    pulled_back = (
        float(np.median(1 - to_end[moved_away] / to_start[moved_away])) if moved_away.any() else float("nan")
    )

    step_row = {
        "n_clusters": k,
        "n_iter": block.n_iter,
        "stop": block.stop,
        "j_before": pre.objective,
        "j_old_assign": replay.old_assignment_objective,
        "j_replaced": start.objective,
        "j_end": end.objective,
        "j_cf_end": cf_end.objective if cf_end else float("nan"),
        "jump_rel": _safe_div(start.objective - pre.objective, pre.objective),
        "end_vs_before_rel": _safe_div(end.objective - pre.objective, pre.objective),
        "end_vs_cf_rel": _safe_div(end.objective - cf_end.objective, cf_end.objective) if cf_end else float("nan"),
        "drift_median": float(np.median(drift[finite])) if finite.any() else float("nan"),
        "cosine_median": float(np.median(cos[nonempty])),
        "own_nearest_share": float(np.mean(own_nearest[nonempty])),
        "to_end_radii_median": float(np.median(to_end_radii[finite])) if finite.any() else float("nan"),
        "end_vs_pre_radii_median": float(np.median(end_vs_pre[finite])) if finite.any() else float("nan"),
        "pulled_back_median": pulled_back,
        "n_moved_replacement": int(moved0.sum()),
        "frac_moved_replacement": float(moved0.mean()),
        "n_moved_end": int(moved_end.sum()),
        "frac_moved_end": float(moved_end.mean()),
        "ari_replacement_vs_pre": float(adjusted_rand_score(pre_labels, start.labels)),
        "ari_end_vs_pre": float(adjusted_rand_score(pre_labels, block.final_labels)),
        "jaccard_end_mean": float(np.nanmean(jac_end)),
        "toward_replacement": int((dir0[moved0] == "toward").sum()),
        "away_replacement": int((dir0[moved0] == "away").sum()),
        "ends_in_own_majority_replacement": int(own0[moved0].sum()),
        "toward_end": int((dir_end[moved_end] == "toward").sum()),
        "away_end": int((dir_end[moved_end] == "away").sum()),
        "ends_in_own_majority_end": int(own_end[moved_end].sum()),
        "move_rate_boundary": rate(boundary, moved0),
        "move_rate_core": rate(~boundary, moved0),
        "margin_median_movers": float(np.median(margins[moved0])) if moved0.any() else float("nan"),
        "margin_median_stayers": float(np.median(margins[~moved0])) if (~moved0).any() else float("nan"),
        "move_rate_flagged": rate(flagged_point, moved0),
        "move_rate_unflagged": rate(~flagged_point, moved0),
        "share_movers_from_flagged": _safe_div(float((moved0 & flagged_point).sum()), float(moved0.sum())),
        "share_points_in_flagged": float(flagged_point.mean()),
        "n_dup_clusters": int(flags["dup"].sum()),
        "n_meta_clusters": int(flags["meta"].sum()),
        "n_flagged_clusters": int(flags["flagged"].sum()),
        "n_empty_pre": int((sizes_pre == 0).sum()),
        "n_empty_replacement": int(empty0.sum()),
        "n_empty_end": int((sizes_end == 0).sum()),
        "n_reseeded": int((empty0 & (sizes_end > 0)).sum()),
        "acc_pre": scores_pre["acc"],
        "nmi_pre": scores_pre["nmi"],
        "ari_pre": scores_pre["ari"],
        "acc_end": scores_end["acc"],
        "nmi_end": scores_end["nmi"],
        "ari_end": scores_end["ari"],
        "acc_cf": scores_cf["acc"] if scores_cf else float("nan"),
        "nmi_cf": scores_cf["nmi"] if scores_cf else float("nan"),
        "ari_cf": scores_cf["ari"] if scores_cf else float("nan"),
    }

    iterations = _block_rows(block, pre_labels, k, "replay")
    if cf is not None:
        iterations += _block_rows(cf, pre_labels, k, "counterfactual")
        iterations[len(block.iterations)]["kind"] = "counterfactual"

    out_c0 = np.bincount(pre_labels[moved0], minlength=k)
    in_c0 = np.bincount(start.labels[moved0], minlength=k)
    out_end = np.bincount(pre_labels[moved_end], minlength=k)
    in_end = np.bincount(block.final_labels[moved_end], minlength=k)
    clusters = [
        {
            "cluster": c,
            "size_pre": int(sizes_pre[c]),
            "size_replacement": int(sizes_start[c]),
            "size_end": int(sizes_end[c]),
            "dup": bool(flags["dup"][c]),
            "meta": bool(flags["meta"][c]),
            "flagged": bool(flags["flagged"][c]),
            "rms_radius": float(radius[c]),
            "drift": float(drift[c]),
            "cosine": float(cos[c]),
            "nearest_pre_center": int(nearest_pre[c]),
            "own_nearest": bool(own_nearest[c]),
            "to_end_radii": float(to_end_radii[c]),
            "end_vs_pre_radii": float(end_vs_pre[c]),
            "empty_replacement": bool(empty0[c]),
            "first_nonempty_t": int(first_nonempty[c]),
            "n_out_replacement": int(out_c0[c]),
            "n_in_replacement": int(in_c0[c]),
            "n_out_end": int(out_end[c]),
            "n_in_end": int(in_end[c]),
            "jaccard_end_vs_pre": float(jac_end[c]),
            "majority_gold_pre": int(majority[c]),
        }
        for c in range(k)
    ]

    points = []
    if keep_points:
        for i in np.flatnonzero(moved0 | moved_end):
            points.append(
                {
                    "point": int(i),
                    "gold": int(gold[i]),
                    "from": int(pre_labels[i]),
                    "to_replacement": int(start.labels[i]),
                    "to_end": int(block.final_labels[i]),
                    "margin": float(margins[i]),
                    "boundary": bool(boundary[i]),
                    "from_flagged": bool(flagged_point[i]),
                    "direction_replacement": str(dir0[i]) if moved0[i] else "",
                    "direction_end": str(dir_end[i]) if moved_end[i] else "",
                }
            )
    return {"step": step_row, "iterations": iterations, "clusters": clusters, "points": points}


def staleness_rows(
    source_labels: np.ndarray,
    final_labels: np.ndarray,
    summary_embeddings: np.ndarray,
    final_centers: np.ndarray,
    summaries: list[str | None],
) -> list[dict]:
    """Per cluster, whether the displayed summary still describes the final partition.

    `source_labels` is the partition the summary was written from, `final_labels` / `final_centers` the
    partition and centers it is displayed with, `summary_embeddings` the summary vectors. A cluster with an empty
    source set has no Jaccard (NaN) and is never stale by it; an empty summary is never stale by either rule.
    """
    k = len(summaries)
    jac = jaccard_per_cluster(source_labels, final_labels, k)
    nearest = nearest_center(summary_embeddings, final_centers)
    rows = []
    for c in range(k):
        has_text = bool(summaries[c])
        low_overlap = has_text and bool(jac[c] < STALE_JACCARD)
        wrong_nearest = has_text and int(nearest[c]) != c
        rows.append(
            {
                "cluster": c,
                "jaccard_source_vs_final": float(jac[c]),
                "nearest_final_center": int(nearest[c]),
                "has_text": has_text,
                "own_nearest_final": int(nearest[c]) == c,
                "stale_jaccard": low_overlap,
                "stale_nearest": wrong_nearest,
                "stale": low_overlap or wrong_nearest,
            }
        )
    return rows
