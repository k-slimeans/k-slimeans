"""`kslimeans.counterfactual`: the swaps, the control and the paired intervals on tiny hand-checkable cases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kslimeans.counterfactual as cf
from kslimeans.counterfactual import (
    collapse_draws,
    evaluate_reference,
    evaluate_seed,
    flag_sets,
    gap_closed,
    outcome_row,
    paired_differences,
    random_controls,
    swap_rows,
    truncate,
)
from kslimeans.replay import LloydReplayer, state_of

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from synth import make_synthetic_data  # noqa: E402

# Three well separated clusters of two points; cluster centers (0,0), (10,0), (0,10).
X = np.array([[-1, 0], [1, 0], [9, 0], [11, 0], [0, 9], [0, 11]], dtype=np.float32)
PRE_LABELS = np.array([0, 0, 1, 1, 2, 2])
GOLD = PRE_LABELS.copy()
PRE_CENTERS = np.array([[0, 0], [10, 0], [0, 10]], dtype=np.float32)
PRE = state_of(X, PRE_LABELS, PRE_CENTERS)
# The "4B": clusters 1 and 2 are given a summary far from their points, cluster 0's is fine.
OWN = np.array([[0, 0], [50, 50], [60, 60]], dtype=np.float32)
# The "1.7B": every summary sits next to its cluster.
OTHER = np.array([[0.5, 0], [10.5, 0.5], [0.5, 10.5]], dtype=np.float32)
FLAGGED = np.array([False, True, True])


@pytest.fixture(scope="module")
def replayer():
    return LloydReplayer(X)


# ------------------------------------------------------------------ building blocks


def test_swap_rows_replaces_exactly_the_masked_rows_and_copies():
    base, source = np.zeros((3, 2)), np.arange(6, dtype=float).reshape(3, 2)
    out = swap_rows(base, source, np.array([False, True, False]))
    np.testing.assert_array_equal(out, [[0, 0], [2, 3], [0, 0]])
    np.testing.assert_array_equal(base, np.zeros((3, 2)))


def test_random_controls_match_the_count_and_never_pick_flagged_clusters():
    flagged = np.zeros(30, dtype=bool)
    flagged[[0, 3, 4, 10, 20]] = True
    masks = random_controls(flagged, 50, seed=7)
    assert len(masks) == 50
    assert all(m.sum() == 5 and not (m & flagged).any() for m in masks)
    assert len({m.tobytes() for m in masks}) > 1  # the draws differ
    again = random_controls(flagged, 50, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(masks, again))  # fixed RNG
    other = random_controls(flagged, 50, seed=8)
    assert not all(np.array_equal(a, b) for a, b in zip(masks, other))


def test_random_controls_need_enough_unflagged_clusters():
    with pytest.raises(ValueError, match="only 1 unflagged"):
        random_controls(np.array([False, True, True]), 3, seed=0)
    assert [m.sum() for m in random_controls(np.zeros(4, dtype=bool), 2, seed=0)] == [0, 0]


def test_truncate_equals_a_shorter_block():
    _, Xs = make_synthetic_data()
    rng = np.random.default_rng(1)
    start = Xs[rng.choice(len(Xs), 12, replace=False)] + rng.normal(scale=2.0, size=(12, Xs.shape[1])).astype(np.float32)
    replayer = LloydReplayer(Xs)
    long = replayer.run_block(start, 300, track_objective=False)
    for m in (1, 3, long.n_iter, long.n_iter + 5):
        cut, direct = truncate(long, m, replayer), replayer.run_block(start, m)
        assert cut.n_iter == direct.n_iter and cut.stop == direct.stop
        assert np.array_equal(cut.final_labels, direct.final_labels)
        assert cut.end.objective == pytest.approx(direct.end.objective, rel=1e-9)


def test_flag_sets_on_hand_made_summaries():
    # Cluster means (10,0), (0,10), (-10,0). Summary 1 points toward cluster 0's mean: a negative margin and no
    # exact or near duplicate. Summaries 0 and 2 share their text.
    Xm = np.array([[10, 1], [10, -1], [1, 10], [-1, 10], [-10, 1], [-10, -1]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2])
    S = np.array([[1, 0], [1, 0.5], [-1, 0]], dtype=np.float32)
    flags = flag_sets(["p", "q", "p"], S, Xm, labels)
    np.testing.assert_array_equal(flags["dup"], [True, False, True])
    np.testing.assert_array_equal(flags["meta"], [False, False, False])
    np.testing.assert_array_equal(flags["union"], [True, False, True])
    np.testing.assert_array_equal(flags["generic"], [True, True, True])  # union plus cluster 1's negative margin
    only_meta = flag_sets(["a", "this cluster is about x", "c"], np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32), Xm, labels)
    np.testing.assert_array_equal(only_meta["meta"], [False, True, False])
    np.testing.assert_array_equal(only_meta["union"], [False, True, False])
    np.testing.assert_array_equal(only_meta["generic"], [False, True, False])
    assert not only_meta["dup"].any()


def test_outcome_row_counts_on_the_collapsed_start(replayer):
    block = replayer.run_block(OWN, 1, stop=False)
    row = outcome_row(block, PRE, GOLD)
    # Every point is nearest to the summary at (0, 0), the only one near the data.
    assert row["repl_empty"] == 2 and row["repl_moved"] == 4
    assert row["repl_acc"] == pytest.approx(2 / 6)
    assert row["repl_j"] == pytest.approx(block.start.objective)
    assert row["n_iter"] == 1


# ------------------------------------------------------------------ the variants


def _rows(replayer):
    return evaluate_seed(replayer, PRE, GOLD, OWN, OTHER, {"f": FLAGGED}, fixed_iter=5, n_draws=0)


def test_variants_start_from_the_rows_the_definition_names(replayer, monkeypatch):
    starts = []

    def fake(replayer, pre, gold, start, *, fixed_iter, cap):
        starts.append(start.copy())
        return [{"setting": "fixed", "start": start.copy()}]

    monkeypatch.setattr(cf, "evaluate_start", fake)
    monkeypatch.setattr(cf, "evaluate_reference", lambda *a, **k: [{"setting": "fixed", "start": None}])
    # Four flagged clusters out of eight so that a control has several unflagged clusters to pick among.
    pre = state_of(np.zeros((8, 2), dtype=np.float32), np.arange(8), np.arange(16, dtype=np.float32).reshape(8, 2))
    own, other = pre.centers + 100, pre.centers + 200
    flagged = np.array([True, False, True, False, False, False, True, False])
    rows = evaluate_seed(replayer, pre, np.arange(8), own, other, {"f": flagged}, fixed_iter=5, n_draws=6, control_seed=1)
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)

    np.testing.assert_array_equal(by["a"][0]["start"], own)
    np.testing.assert_array_equal(by["m"][0]["start"], other)
    # (b) flagged clusters take pre.centers rows, the others keep the summaries.
    b = by["b"][0]["start"]
    np.testing.assert_array_equal(b[flagged], pre.centers[flagged])
    np.testing.assert_array_equal(b[~flagged], own[~flagged])
    # (c) flagged clusters take the other model's rows.
    c = by["c"][0]["start"]
    np.testing.assert_array_equal(c[flagged], other[flagged])
    np.testing.assert_array_equal(c[~flagged], own[~flagged])
    # reverse: the other model's start with the own model's flagged rows.
    r = by["r"][0]["start"]
    np.testing.assert_array_equal(r[flagged], own[flagged])
    np.testing.assert_array_equal(r[~flagged], other[~flagged])
    assert by["b"][0]["n_replaced"] == by["c"][0]["n_replaced"] == by["r"][0]["n_replaced"] == 3
    # (d) and the reverse control: as many clusters as flagged, none of them flagged, same clusters in both.
    assert len(by["d"]) == len(by["cd"]) == len(by["rd"]) == 6
    for d, cd, rd in zip(by["d"], by["cd"], by["rd"]):
        edited = np.any(d["start"] != own, axis=1)
        assert edited.sum() == 3 and not (edited & flagged).any()
        np.testing.assert_array_equal(d["start"][edited], pre.centers[edited])
        cedited = np.any(cd["start"] != own, axis=1)  # (c)'s control: the other model's rows, same clusters
        np.testing.assert_array_equal(cedited, edited)
        np.testing.assert_array_equal(cd["start"][cedited], other[cedited])
        redited = np.any(rd["start"] != other, axis=1)
        np.testing.assert_array_equal(redited, edited)
        np.testing.assert_array_equal(rd["start"][redited], own[redited])
        assert d["draw"] == rd["draw"] and d["flag"] == "f" and d["n_replaced"] == 3


def test_identical_starts_run_once(replayer, monkeypatch):
    calls = []

    def fake(replayer, pre, gold, start, *, fixed_iter, cap):
        calls.append(start)
        return [{"setting": "fixed"}]

    monkeypatch.setattr(cf, "evaluate_start", fake)
    monkeypatch.setattr(cf, "evaluate_reference", lambda *a, **k: [{"setting": "fixed"}])
    rows = evaluate_seed(
        replayer, PRE, GOLD, OWN, OTHER, {"x": FLAGGED, "y": FLAGGED.copy()}, fixed_iter=5, n_draws=0
    )
    assert len(calls) == 2 + 3  # a, m, and b, c, r once for the two coinciding flag sets
    assert {r["flag"] for r in rows if r["variant"] == "b"} == {"x", "y"}


def test_swaps_repair_the_replacement_on_the_toy_problem(replayer):
    rows = pd.DataFrame(_rows(replayer))
    fixed = rows[rows.setting == "fixed"].set_index("variant")

    def get(variant):
        return fixed.loc[variant]

    # (a) the two far summaries lose their points: both clusters empty, four points move, accuracy 2/6.
    assert get("a").repl_empty == 2 and get("a").repl_moved == 4 and get("a").repl_acc == pytest.approx(2 / 6)
    # (b), (c) put sensible centers back in clusters 1 and 2: nothing moves at the replacement.
    for variant in ("b", "c"):
        assert get(variant).repl_empty == 0 and get(variant).repl_moved == 0 and get(variant).repl_acc == 1.0
        assert get(variant).end_acc == 1.0
    # reverse: the other model is fine as logged; its clusters 1 and 2 taking the far summaries breaks them.
    assert get("m").repl_empty == 0 and get("r").repl_empty == 2
    # the reference never replaces anything.
    assert get("ref").repl_moved == 0 and get("ref").end_acc == 1.0


def test_reference_runs_both_settings(replayer):
    fixed, converged = evaluate_reference(replayer, PRE, GOLD, fixed_iter=3)
    assert (fixed["setting"], converged["setting"]) == ("fixed", "converged")
    assert fixed["n_iter"] == 3 and converged["n_iter"] <= 3
    assert fixed["end_j"] == pytest.approx(converged["end_j"])


# ------------------------------------------------------------------ paired intervals


def _per_seed(values: dict[tuple[str, str], list[float]], metric="end_acc") -> pd.DataFrame:
    rows = []
    for (variant, flag), vals in values.items():
        for seed, v in enumerate(vals):
            rows.append({"seed": seed, "variant": variant, "flag": flag, "setting": "fixed", metric: v})
    return pd.DataFrame(rows)


def test_paired_difference_and_t_interval_by_hand():
    # Differences 0.1 .. 0.5: mean 0.3, sd sqrt(0.025), se sd/sqrt(5); t(0.975, 4) = 2.7764451.
    per_seed = _per_seed({("a", ""): [0.5] * 5, ("b", "f"): [0.6, 0.7, 0.8, 0.9, 1.0]})
    out = paired_differences(per_seed, [("b", "b", "f", "a", "")], metrics=("end_acc",), settings=("fixed",)).iloc[0]
    sd = np.sqrt(0.025)
    half = 2.7764451051977987 * sd / np.sqrt(5)
    assert out["mean"] == pytest.approx(0.3)
    assert out["sd"] == pytest.approx(sd)
    assert out["ci_lo"] == pytest.approx(0.3 - half) and out["ci_hi"] == pytest.approx(0.3 + half)
    assert out["n_seeds"] == 5 and out["n_positive"] == 5 and out["metric"] == "end_acc"


def test_paired_difference_pairs_by_seed_and_reports_the_objective_relatively():
    per_seed = _per_seed({("a", ""): [100.0, 200.0, 400.0], ("b", "f"): [90.0, 220.0, 400.0]}, metric="end_j")
    out = paired_differences(per_seed, [("b", "b", "f", "a", "")], metrics=("end_j",), settings=("fixed",)).iloc[0]
    assert out["metric"] == "end_j_rel"
    assert out["mean"] == pytest.approx((-0.1 + 0.1 + 0.0) / 3)
    assert out["n_positive"] == 1
    # A comparison whose seeds do not all overlap is paired on the common seeds only.
    shorter = per_seed[~((per_seed.variant == "b") & (per_seed.seed == 2))]
    assert paired_differences(shorter, [("b", "b", "f", "a", "")], metrics=("end_j",), settings=("fixed",)).iloc[0].n_seeds == 2


def test_single_seed_has_no_interval():
    per_seed = _per_seed({("a", ""): [0.5], ("b", "f"): [0.6]})
    out = paired_differences(per_seed, [("b", "b", "f", "a", "")], metrics=("end_acc",), settings=("fixed",)).iloc[0]
    assert out["mean"] == pytest.approx(0.1) and np.isnan(out["ci_lo"]) and np.isnan(out["sd"])


def test_gap_closed_is_the_share_of_the_gap_to_the_target():
    per_seed = _per_seed(
        {("a", ""): [0.5, 0.5], ("ref", ""): [0.6, 0.6], ("b", "f"): [0.55, 0.55], ("m", ""): [0.4, 0.4]}
    )
    assert gap_closed(per_seed, "b", "f", "a", "ref", "end_acc", "fixed") == pytest.approx(0.5)
    assert np.isnan(gap_closed(per_seed, "b", "f", "a", "m", "end_acc", "fixed"))  # the target is below the baseline


def test_collapse_draws_averages_the_controls_and_reports_their_spread():
    rows = []
    for draw, acc in enumerate([0.2, 0.4, 0.6]):
        rows.append(
            {"seed": 0, "variant": "d", "flag": "f", "draw": draw, "setting": "fixed", "stop": "max_iter",
             "end_acc": acc, "end_nmi": acc, "end_ari": acc, "end_j": 10.0, "n_replaced": 2}  # fmt: skip
        )
    rows.append(
        {"seed": 0, "variant": "a", "flag": "", "draw": -1, "setting": "fixed", "stop": "max_iter",
         "end_acc": 0.9, "end_nmi": 0.9, "end_ari": 0.9, "end_j": 9.0, "n_replaced": 0}  # fmt: skip
    )
    means, spreads = collapse_draws(pd.DataFrame(rows))
    d = means[means.variant == "d"].iloc[0]
    assert d.end_acc == pytest.approx(0.4) and d.n_draws == 3 and d.n_replaced == 2
    assert means[means.variant == "a"].iloc[0].n_draws == 1
    sd = spreads.set_index("variant")
    assert sd.loc["d", "end_acc_sd"] == pytest.approx(0.2)
    assert np.isnan(sd.loc["a", "end_acc_sd"])
