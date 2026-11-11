"""`kslimeans.diagnostic`: the replacement measures on tiny hand-made cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from kslimeans.diagnostic import (
    cluster_flags,
    cosine_rows,
    gold_direction,
    jaccard_per_cluster,
    majority_labels,
    nearest_center,
    partition_scores,
    pre_margins,
    staleness_rows,
    step_tables,
)
from kslimeans.replay import LloydReplayer, replay_step, state_of

# Three well separated clusters of two points; cluster centers (0,0), (10,0), (0,10), every RMS radius 1.
X = np.array([[-1, 0], [1, 0], [9, 0], [11, 0], [0, 9], [0, 11]], dtype=np.float32)
PRE_LABELS = np.array([0, 0, 1, 1, 2, 2])
GOLD = np.array([0, 0, 1, 1, 2, 2])
PRE_CENTERS = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
# Cluster 0 keeps its center, cluster 1 moves up by 2 (two radii), cluster 2 collapses onto cluster 0's center.
SUMMARY_CENTERS = np.array([[0, 0], [10, 2], [0, 0]], dtype=np.float32)
SUMMARIES = ["a", "b", "a"]


@pytest.fixture(scope="module")
def tables():
    pre = state_of(X, PRE_LABELS, PRE_CENTERS)
    step = replay_step(LloydReplayer(X), pre, SUMMARY_CENTERS, 5)
    return step, step_tables(step, X, GOLD, SUMMARIES, PRE_LABELS)


def test_known_displacement_in_radii_and_cosine(tables):
    _, out = tables
    clusters = out["clusters"]
    assert [c["rms_radius"] for c in clusters] == pytest.approx([1.0, 1.0, 1.0])
    assert [c["drift"] for c in clusters] == pytest.approx([0.0, 2.0, 10.0])
    assert clusters[1]["cosine"] == pytest.approx(10 / np.sqrt(104))
    assert clusters[0]["cosine"] == pytest.approx(0.0)  # zero vector: defined as 0
    assert [c["own_nearest"] for c in clusters] == [True, True, False]
    assert clusters[2]["nearest_pre_center"] == 0
    assert out["step"]["own_nearest_share"] == pytest.approx(2 / 3)


def test_known_movers_and_gold_direction(tables):
    _, out = tables
    row = out["step"]
    assert row["n_moved_replacement"] == 2  # the two points of the collapsed cluster 2 go to cluster 0
    assert [c["n_out_replacement"] for c in out["clusters"]] == [0, 0, 2]
    assert [c["n_in_replacement"] for c in out["clusters"]] == [2, 0, 0]
    assert row["away_replacement"] == 2 and row["toward_replacement"] == 0
    assert row["ends_in_own_majority_replacement"] == 0
    assert {(p["from"], p["to_replacement"]) for p in out["points"] if p["direction_replacement"]} == {(2, 0)}


def test_known_empty_cluster_is_relocated(tables):
    step, out = tables
    row = out["step"]
    assert row["n_empty_replacement"] == 1 and row["n_empty_pre"] == 0
    cluster2 = out["clusters"][2]
    assert cluster2["empty_replacement"] and cluster2["size_replacement"] == 0
    assert cluster2["first_nonempty_t"] == 1  # sklearn re-seeds it in the first iteration
    assert row["n_reseeded"] == 1 and row["n_empty_end"] == 0
    kinds = {r["kind"] for r in out["iterations"]}
    assert kinds == {"replay", "counterfactual"}
    first = [r for r in out["iterations"] if r["kind"] == "replay" and r["t"] == 1][0]
    assert first["n_relocated"] == 1 and first["n_empty"] == 0


def test_objective_values_and_counterfactual(tables):
    step, out = tables
    row = out["step"]
    assert row["j_before"] == pytest.approx(6.0)  # six points at distance 1
    # old assignment: cluster 0 points to (0,0): 1+1; cluster 1 to (10,2): 1+4 twice; cluster 2 to (0,0): 81 + 121
    assert row["j_old_assign"] == pytest.approx(2 + 10 + 81 + 121)
    # re-assigned: cluster 2's points join cluster 0 (distance 9 and 11 from (0,0)): 2 + 10 + 81 + 121
    assert row["j_replaced"] == pytest.approx(2 + 10 + 81 + 121)
    assert row["j_cf_end"] == pytest.approx(6.0)  # plain Lloyd from the converged pre state does nothing
    assert row["j_end"] < row["j_replaced"]
    cf = [r for r in out["iterations"] if r["kind"] == "counterfactual"]
    assert all(r["n_reassigned"] == 0 for r in cf)
    assert [r["objective"] for r in cf] == pytest.approx([6.0] * len(cf))


def test_flags_and_flagged_movers(tables):
    _, out = tables
    assert [c["dup"] for c in out["clusters"]] == [True, False, True]
    row = out["step"]
    assert row["n_dup_clusters"] == 2 and row["n_flagged_clusters"] == 2
    assert row["share_movers_from_flagged"] == pytest.approx(1.0)  # both movers come from the duplicate cluster 2
    assert row["move_rate_flagged"] == pytest.approx(2 / 4)
    assert row["move_rate_unflagged"] == pytest.approx(0.0)


def test_cluster_flags_rules():
    flags = cluster_flags(["same", "same", "what is this cluster?", "", "", None, "a unique text"])
    assert flags["dup"].tolist() == [True, True, False, False, False, False, False]  # "" is not a duplicate
    assert flags["meta"].tolist() == [False, False, True, False, False, False, False]
    assert flags["flagged"].tolist() == [True, True, True, False, False, False, False]


def test_margin_is_second_minus_nearest_distance():
    margins = pre_margins(X, PRE_CENTERS)
    assert margins[1] == pytest.approx(9 - 1)  # (1,0): nearest (0,0) at 1, second (10,0) at 9
    assert margins[0] == pytest.approx(np.sqrt(101) - 1)
    assert (pre_margins(X, PRE_CENTERS[:1]) == 0).all()


def test_majority_labels_and_direction():
    labels = np.array([0, 0, 0, 1, 1, 2])
    gold = np.array([5, 5, 3, 3, 3, 4])
    majority = majority_labels(labels, gold, 4)
    assert majority.tolist() == [5, 3, 4, -1]  # cluster 3 is empty
    # point with gold 3 moves 0 -> 1: toward; gold 5 moves 0 -> 1: away; gold 9 moves 0 -> 1: neutral (neither)
    direction = gold_direction(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 0]), np.array([3, 5, 9, 5]), majority)
    assert direction.tolist() == ["toward", "away", "neutral", "neutral"]
    tie = majority_labels(np.array([0, 0]), np.array([7, 2]), 1)
    assert tie.tolist() == [2]  # ties go to the smallest label


def test_jaccard_and_cosine_and_nearest():
    a = np.array([0, 0, 1, 1, 2])
    b = np.array([0, 1, 1, 1, 2])
    jac = jaccard_per_cluster(a, b, 4)
    assert jac[:3] == pytest.approx([1 / 2, 2 / 3, 1.0])
    assert np.isnan(jac[3])
    assert cosine_rows(np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 1.0], [2.0, 2.0]])) == pytest.approx([0.0, 1.0])
    assert nearest_center(np.array([[0.9, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])).tolist() == [1]


def test_partition_scores_are_permutation_invariant():
    gold = np.array([0, 0, 1, 1, 2, 2])
    assert partition_scores(gold, np.array([2, 2, 0, 0, 1, 1])) == pytest.approx({"acc": 1.0, "nmi": 1.0, "ari": 1.0})
    assert partition_scores(gold, np.array([0, 1, 0, 1, 0, 1]))["acc"] < 1.0


def test_staleness_rows():
    source = np.array([0, 0, 1, 1, 2, 2])
    final = np.array([0, 0, 1, 2, 2, 2])  # cluster 1 lost a point to cluster 2
    final_centers = np.array([[0.0, 0.0], [5.0, 0.0], [9.0, 0.0]])
    summary_emb = np.array([[0.1, 0.0], [8.9, 0.0], [9.0, 0.0]])  # summary 1 sits nearest the final center of 2
    rows = staleness_rows(source, final, summary_emb, final_centers, ["x", "y", "z"])
    assert [r["jaccard_source_vs_final"] for r in rows] == pytest.approx([1.0, 0.5, 2 / 3])
    assert [r["own_nearest_final"] for r in rows] == [True, False, True]
    assert [r["stale_nearest"] for r in rows] == [False, True, False]
    assert [r["stale"] for r in rows] == [False, True, False]  # Jaccard 0.5 is not below the 0.5 threshold
    assert not any(r["stale"] for r in staleness_rows(source, final, summary_emb, final_centers, ["", "", ""]))
