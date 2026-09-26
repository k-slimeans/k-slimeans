"""Sanity checks for cluster_metrics / avg_closest_distance / oracle_centroids."""

from __future__ import annotations

import numpy as np
import pytest

from kslimeans.metrics import avg_closest_distance, cluster_metrics, oracle_centroids


def test_acc_nmi_ari_are_one_for_a_relabeled_permutation():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    perm = {c: (c + 2) % 5 for c in range(5)}
    y_pred = np.array([perm[c] for c in y_true])

    centroids = rng.normal(size=(5, 3))
    metrics = cluster_metrics(y_true, y_pred, centroids, centroids, centroids)

    assert metrics["acc"] == 1.0
    assert metrics["nmi"] == pytest.approx(1.0)
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["cen_cen_dist"] == pytest.approx(0.0)


def test_avg_closest_distance_zero_for_identical_point_sets():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert avg_closest_distance(X, X) == pytest.approx(0.0)


def test_avg_closest_distance_finds_the_optimal_matching():
    a = np.array([[0.0, 0.0], [10.0, 10.0]])
    b = np.array([[10.0, 10.0], [0.0, 0.0]])  # reversed order
    assert avg_closest_distance(a, b) == pytest.approx(0.0)


def test_oracle_centroids_skips_empty_classes():
    X = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
    labels = np.array([0, 0, 2])  # class 1 has no members
    centroids = oracle_centroids(X, labels, n_clusters=3)
    assert centroids.shape == (2, 2)
    assert np.array_equal(centroids[0], np.array([1.0, 1.0]))
    assert np.array_equal(centroids[1], np.array([4.0, 4.0]))
