"""Cluster quality metrics.

`avg_closest_distance` and the accuracy/Hungarian-matching logic inside
`cluster_metrics` are ported from upstream unchanged; `cluster_metrics`
returns a dict instead of upstream's positional list, and adds ARI.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn import metrics as sk_metrics


def avg_closest_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Mean distance of the optimal one-to-one (Hungarian) matching between two point sets."""
    distance_matrix = cdist(emb_a, emb_b, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    return float(np.mean(distance_matrix[row_ind, col_ind]))


def cluster_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    centroid_true: np.ndarray,
    centroid_pred: np.ndarray,
    summary_true: np.ndarray,
    summary_pred: np.ndarray | None = None,
) -> dict:
    """Clustering accuracy/NMI/ARI plus four centroid<->summary distances.

    `acc` finds the best cluster-id <-> true-label permutation via the
    Hungarian algorithm, same as upstream. If `summary_pred` is omitted,
    `centroid_pred` stands in for it (also matching upstream).
    """
    nmi = sk_metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = sk_metrics.adjusted_rand_score(y_true, y_pred)

    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / y_pred.size

    if summary_pred is None:
        summary_pred = centroid_pred

    return {
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "cen_cen_dist": avg_closest_distance(centroid_pred, centroid_true),
        "cen_sum_dist": avg_closest_distance(centroid_pred, summary_true),
        "sum_cen_dist": avg_closest_distance(summary_pred, centroid_true),
        "sum_sum_dist": avg_closest_distance(summary_pred, summary_true),
    }


def oracle_centroids(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Per-class mean of `X`, one row per class that has at least one member.

    Upstream's equivalent left a placeholder for empty classes that later
    crashed `cdist` inside `avg_closest_distance`; skipping empty classes
    here keeps the result usable, at the cost of possibly returning fewer
    than `n_clusters` rows.
    """
    rows = [X[labels == c].mean(axis=0) for c in range(n_clusters) if np.any(labels == c)]
    return np.array(rows)
