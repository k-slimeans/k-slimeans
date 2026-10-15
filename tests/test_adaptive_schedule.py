"""Summary anchors on `State`, `AdaptiveSchedule`, and the budget-matched `RandomSubsetSchedule`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from kslimeans.core import _cluster_rms_radius, _members_by_cluster, summary_kmeans
from kslimeans.schedules import (
    AdaptiveSchedule,
    FixedSchedule,
    RandomSubsetSchedule,
    State,
    Step,
    SummarySchedule,
    build_schedule,
    centroid_drift,
    membership_churn,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_equivalence import ClosestToMeanSummarizer, LookupEmbedder, synthetic  # noqa: E402,F401
from synth import N_BLOBS  # noqa: E402


class _Recording(SummarySchedule):
    """Plays a fixed list of steps (or defers to `inner`) and records every `State` it is shown."""

    def __init__(self, steps: list[Step] | None = None, inner: SummarySchedule | None = None, first: int = 1):
        self.steps, self.inner, self.first, self.states = steps, inner, first, []

    def first_iters(self) -> int:
        return self.inner.first_iters() if self.inner else self.first

    def next_step(self, state: State) -> Step | None:
        self.states.append(state)
        if self.inner:
            return self.inner.next_step(state)
        return self.steps[state.step] if state.step < len(self.steps) else None


def _run(synthetic, schedule, *, tol=-1.0, init="k-means++"):
    texts, X, embed = synthetic
    return summary_kmeans(
        texts,
        N_BLOBS,
        ClosestToMeanSummarizer(embed),
        LookupEmbedder(embed),
        schedule,
        text_features=X,
        tol=tol,
        random_state=0,
        init=init,
        verbose=False,
    )


def _mask(*on: int) -> list[bool]:
    return [c in on for c in range(N_BLOBS)]


def _members(labels: np.ndarray, k: int) -> list[np.ndarray]:
    return [np.flatnonzero(labels == c) for c in range(k)]


def _assert_members_equal(a, b) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# --- anchors and radii on State --------------------------------------------------------------------


def test_step_zero_anchors_are_the_initial_fit(synthetic):
    rec = _Recording([Step(True, 1)])
    _run(synthetic, rec, init="random")
    s0 = rec.states[0]
    assert s0.last_summarized.tolist() == [-1] * N_BLOBS
    assert np.array_equal(s0.anchor_centers, s0.centers)
    _assert_members_equal(s0.anchor_members, _members(s0.labels, N_BLOBS))


def test_full_step_anchors_are_the_pre_summary_state(synthetic):
    rec = _Recording([Step(True, 1), Step(False, 1)])
    _run(synthetic, rec, init="random")
    pre, post = rec.states[0], rec.states[1]
    assert post.last_summarized.tolist() == [1] * N_BLOBS
    assert np.array_equal(post.anchor_centers, pre.centers)
    assert not np.array_equal(post.anchor_centers, post.centers)
    _assert_members_equal(post.anchor_members, _members(pre.labels, N_BLOBS))
    # A plain continuation carries the anchors forward untouched.
    after_plain = rec.states[2]
    assert after_plain.last_summarized.tolist() == [1] * N_BLOBS
    assert np.array_equal(after_plain.anchor_centers, pre.centers)


def test_partial_step_moves_only_masked_on_anchors(synthetic):
    rec = _Recording([Step(True, 1), Step(True, 1, _mask(0, 2)), Step(True, 1, _mask(1))])
    _run(synthetic, rec, init="random")
    s0, s1, s2, s3 = rec.states

    assert s2.last_summarized.tolist() == [2, 1, 2, 1, 1, 1]
    assert np.array_equal(s2.anchor_centers[[0, 2]], s1.centers[[0, 2]])
    assert np.array_equal(s2.anchor_centers[[1, 3, 4, 5]], s0.centers[[1, 3, 4, 5]])
    pre = _members(s1.labels, N_BLOBS)
    old = _members(s0.labels, N_BLOBS)
    _assert_members_equal(s2.anchor_members, [pre[0], old[1], pre[2], old[3], old[4], old[5]])

    assert s3.last_summarized.tolist() == [2, 3, 2, 1, 1, 1]
    assert np.array_equal(s3.anchor_centers[1], s2.centers[1])
    assert np.array_equal(s3.anchor_centers[[0, 2]], s2.anchor_centers[[0, 2]])


def test_all_false_mask_leaves_anchors_alone(synthetic):
    rec = _Recording([Step(True, 1), Step(True, 1, _mask())])
    _run(synthetic, rec)
    assert rec.states[2].last_summarized.tolist() == rec.states[1].last_summarized.tolist()
    assert np.array_equal(rec.states[2].anchor_centers, rec.states[1].anchor_centers)


def test_cluster_rms_radius_matches_manual(synthetic):
    _, X, _ = synthetic
    rec = _Recording([Step(True, 1), Step(False, 1)])
    _run(synthetic, rec)
    for state in rec.states:
        expected = np.zeros(N_BLOBS)
        for c in range(N_BLOBS):
            members = X[state.labels == c].astype(np.float64)
            if len(members):
                expected[c] = np.sqrt(np.mean(np.sum((members - state.centers[c].astype(np.float64)) ** 2, axis=1)))
        assert np.allclose(state.cluster_rms_radius, expected, rtol=1e-9)


def test_cluster_rms_radius_chunking_and_empty_cluster():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    labels = rng.integers(0, 3, size=50)  # cluster 3 stays empty
    centers = rng.normal(size=(4, 4)).astype(np.float32)
    whole = _cluster_rms_radius(X, labels, centers)
    assert np.allclose(whole, _cluster_rms_radius(X, labels, centers, chunk_rows=7))
    assert whole[3] == 0.0


def test_members_by_cluster_matches_flatnonzero():
    labels = np.array([2, 0, 2, 3, 0, 0])
    _assert_members_equal(_members_by_cluster(labels, 5), _members(labels, 5))


# --- pure functions ------------------------------------------------------------------------------------


def test_centroid_drift_values_and_zero_radius():
    centers = np.array([[3.0, 4.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    anchors = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0], [9.0, 9.0]])
    radius = np.array([2.0, 5.0, 0.0, 0.0])
    drift = centroid_drift(centers, anchors, radius)
    assert drift[0] == pytest.approx(2.5)  # ||(3,4)|| / 2
    assert drift[1] == 0.0
    assert drift[2] == np.inf  # moved, zero radius
    assert drift[3] == 0.0  # did not move, zero radius


def test_membership_churn_jaccard_values():
    labels = np.array([0, 0, 1, 1])
    same = membership_churn(labels, (np.array([0, 1]), np.array([2, 3])), 2)
    assert same.tolist() == [0.0, 0.0]
    swapped = membership_churn(labels, (np.array([0, 2]), np.array([1, 3])), 2)
    assert swapped == pytest.approx([2 / 3, 2 / 3])


def test_membership_churn_empty_clusters():
    empty = np.array([], dtype=np.int64)
    # Empty now and at the anchor: churn 0. Non-empty at the anchor, empty now: churn 1.
    assert membership_churn(np.array([0, 0]), (np.array([0, 1]), empty), 2).tolist() == [0.0, 0.0]
    assert membership_churn(np.array([1, 1]), (np.array([0, 1]), empty), 2)[0] == 1.0


# --- AdaptiveSchedule decisions on hand-built states -------------------------------------------------


def _hand_state(
    *,
    step: int = 1,
    labels=(0, 0, 1, 1, 2, 2),
    centers=((0.0,), (10.0,), (20.0,)),
    anchor_centers=None,
    anchor_members=None,
    last_summarized=(1, 1, 1),
    radius=(1.0, 1.0, 1.0),
) -> State:
    labels = np.asarray(labels)
    centers = np.asarray(centers)
    return State(
        step=step,
        n_summaries=step,
        lloyd_iters_used=0,
        labels=labels,
        prev_labels=None,
        frac_changed=None,
        centers=centers,
        inertia=0.0,
        prev_inertia=None,
        last_n_iter=0,
        last_max_iter=0,
        summary_shift=None,
        last_summarized=np.asarray(last_summarized),
        anchor_centers=centers if anchor_centers is None else np.asarray(anchor_centers),
        anchor_members=tuple(_members(labels, len(centers)))
        if anchor_members is None
        else tuple(np.asarray(m, dtype=np.int64) for m in anchor_members),
        cluster_rms_radius=np.asarray(radius),
    )


def test_never_summarized_triggers_everything_as_a_full_step():
    step = AdaptiveSchedule(3, 40).next_step(_hand_state(step=0, last_summarized=(-1, -1, -1)))
    assert step == Step(True, 10) and step.summarize_mask is None


def test_nothing_triggers_is_a_plain_continuation():
    assert AdaptiveSchedule(3, 40).next_step(_hand_state()) == Step(False, 10)


def test_partial_mask_only_for_triggered_clusters():
    state = _hand_state(last_summarized=(1, -1, 1))
    step = AdaptiveSchedule(3, 40).next_step(state)
    assert step.summarize and step.summarize_mask == (False, True, False)


def test_drift_threshold_is_strict():
    state = _hand_state(anchor_centers=((-0.25,), (10.0,), (20.0,)))  # drift 0.25 on cluster 0
    assert AdaptiveSchedule(3, 40, tau_drift=0.25).next_step(state) == Step(False, 10)
    step = AdaptiveSchedule(3, 40, tau_drift=0.2).next_step(state)
    assert step.summarize_mask == (True, False, False)


def test_churn_threshold_is_strict():
    state = _hand_state(anchor_members=([0, 1, 2, 3], [2, 3], [4, 5]))  # cluster 0: A={0..3}, M={0,1} -> 0.5
    assert AdaptiveSchedule(3, 40, tau_churn=0.5).next_step(state) == Step(False, 10)
    step = AdaptiveSchedule(3, 40, tau_churn=0.4).next_step(state)
    assert step.summarize_mask == (True, False, False)


def test_zero_taus_trigger_on_any_change_but_not_on_none():
    sched = AdaptiveSchedule(3, 40, tau_drift=0.0, tau_churn=0.0)
    assert sched.next_step(_hand_state()) == Step(False, 10)
    moved = _hand_state(anchor_centers=((0.0,), (10.0,), (19.0,)))
    assert sched.next_step(moved).summarize_mask == (False, False, True)


def test_zero_radius_cluster_drifts_infinitely_when_moved():
    state = _hand_state(anchor_centers=((0.0,), (10.0,), (19.0,)), radius=(1.0, 1.0, 0.0))
    assert AdaptiveSchedule(3, 40, tau_drift=1e9).next_step(state).summarize_mask == (False, False, True)


def test_s_max_semantics():
    # A cluster summarized at step 1 may skip s_max - 1 consecutive candidate steps.
    # s_max=1: re-summarized at every candidate step (candidate 2 here).
    assert AdaptiveSchedule(5, 60, s_max=1).next_step(_hand_state(step=1)).summarize_mask is None
    # s_max=3: candidates 2 and 3 are skipped, candidate 4 (state.step=3) forces a re-summary.
    sched = AdaptiveSchedule(5, 60, s_max=3)
    assert sched.next_step(_hand_state(step=1)) == Step(False, 10)
    assert sched.next_step(_hand_state(step=2)) == Step(False, 10)
    assert sched.next_step(_hand_state(step=3)) == Step(True, 10)
    # Only the overdue cluster is forced.
    mixed = _hand_state(step=3, last_summarized=(1, 2, 3))
    assert sched.next_step(mixed).summarize_mask == (True, False, False)


def test_empty_clusters_never_trigger():
    state = _hand_state(
        labels=(0, 0, 1, 1), centers=((0.0,), (10.0,), (20.0,)), last_summarized=(1, 1, -1),
        anchor_centers=((0.0,), (10.0,), (0.0,)), anchor_members=([0, 1], [2, 3], [])
    )
    sched = AdaptiveSchedule(3, 40, tau_drift=0.0, tau_churn=0.0, s_max=1)
    step = sched.next_step(state)
    assert step.summarize_mask == (True, True, False)


def test_returns_none_after_n_summaries_candidates():
    sched = AdaptiveSchedule(3, 40)
    assert sched.next_step(_hand_state(step=2, last_summarized=(-1, -1, -1))) is not None
    assert sched.next_step(_hand_state(step=3)) is None


@pytest.mark.parametrize("n_summaries,max_iter", [(5, 100), (3, 30), (1, 10), (4, 47)])
@pytest.mark.parametrize("initial_iter", [True, False])
@pytest.mark.parametrize("final_iter", [True, False])
def test_block_structure_matches_fixed_schedule(n_summaries, max_iter, initial_iter, final_iter):
    try:
        fixed = FixedSchedule(n_summaries, max_iter, initial_iter, final_iter)
    except ValueError:
        with pytest.raises(ValueError):
            AdaptiveSchedule(n_summaries, max_iter, initial_iter, final_iter)
        return
    adaptive = AdaptiveSchedule(n_summaries, max_iter, initial_iter, final_iter)
    random = RandomSubsetSchedule(n_summaries, max_iter, initial_iter, final_iter, counts=[1] * n_summaries)
    assert adaptive.first_iters() == random.first_iters() == fixed.first_iters()
    for step in range(n_summaries + 1):
        want = fixed.next_step(_hand_state(step=step, last_summarized=(-1, -1, -1)))
        assert adaptive.next_step(_hand_state(step=step, last_summarized=(-1, -1, -1))) == want
        quiet = adaptive.next_step(_hand_state(step=step))
        assert (quiet is None) == (want is None)
        if want is not None:
            assert quiet == Step(False, want.kmeans_iters)


def test_adaptive_validation():
    with pytest.raises(ValueError):
        AdaptiveSchedule(tau_drift=-0.1)
    with pytest.raises(ValueError):
        AdaptiveSchedule(tau_churn=-0.1)
    with pytest.raises(ValueError):
        AdaptiveSchedule(s_max=0)
    AdaptiveSchedule(tau_drift=0, tau_churn=0, s_max=1)


def test_adaptive_needs_anchors():
    bare = State(
        step=1, n_summaries=1, lloyd_iters_used=0, labels=np.zeros(2, dtype=int), prev_labels=None,
        frac_changed=None, centers=np.zeros((1, 1)), inertia=0.0, prev_inertia=None, last_n_iter=0,
        last_max_iter=0, summary_shift=None,
    )
    with pytest.raises(ValueError, match="anchors"):
        AdaptiveSchedule(3, 40).next_step(bare)


# --- AdaptiveSchedule end to end ---------------------------------------------------------------------------


def test_zero_taus_end_to_end(synthetic):
    result = _run(synthetic, AdaptiveSchedule(3, 40, tau_drift=0.0, tau_churn=0.0))
    assert len(result.trace) == 4
    assert result.trace[1]["n_summarized"] == N_BLOBS  # first candidate: nothing summarized yet
    assert result.n_calls == sum(r["n_calls"] for r in result.trace)
    assert result.assignments.shape == (300,)


def test_huge_taus_summarize_only_the_first_candidate(synthetic):
    result = _run(synthetic, AdaptiveSchedule(3, 40, tau_drift=1e9, tau_churn=1e9))
    first_step_calls = result.trace[1]["n_calls"]
    assert first_step_calls > 0 and result.n_calls == first_step_calls
    assert [r["summarize"] for r in result.trace] == [False, True, False, False]
    assert len(result.centroids_evolution) == 1


def test_s_max_one_end_to_end_matches_fixed(synthetic):
    adaptive = _run(synthetic, AdaptiveSchedule(3, 40, tau_drift=1e9, tau_churn=1e9, s_max=1))
    fixed = _run(synthetic, FixedSchedule(3, 40))
    keys = ("step", "summarize", "n_calls", "n_summarized", "last_n_iter")
    assert [{k: r[k] for k in keys} for r in adaptive.trace] == [{k: r[k] for k in keys} for r in fixed.trace]
    assert np.array_equal(adaptive.assignments, fixed.assignments)


# --- RandomSubsetSchedule -------------------------------------------------------------------------------------


def _chosen(step):
    return None if step.summarize_mask is None else [i for i, m in enumerate(step.summarize_mask) if m]


def test_random_subset_picks_exact_count_of_distinct_clusters():
    sched = RandomSubsetSchedule(3, 40, counts=[2, 1, 2], seed=7)
    for step in range(3):
        out = sched.next_step(_hand_state(step=step, last_summarized=(-1, -1, -1)))
        chosen = _chosen(out)
        assert len(chosen) == sched.counts[step] == len(set(chosen))
        assert out.kmeans_iters == 10


def test_random_subset_is_reproducible_and_seed_dependent():
    k = 40
    state = _hand_state(
        step=0, labels=np.arange(k), centers=np.zeros((k, 1)), last_summarized=[-1] * k, radius=[1.0] * k
    )
    a = RandomSubsetSchedule(2, 20, counts=[5, 5], seed=1)
    assert a.next_step(state) == RandomSubsetSchedule(2, 20, counts=[5, 5], seed=1).next_step(state)
    assert a.next_step(state) != RandomSubsetSchedule(2, 20, counts=[5, 5], seed=2).next_step(state)
    later = _hand_state(
        step=1, labels=np.arange(k), centers=np.zeros((k, 1)), last_summarized=[-1] * k, radius=[1.0] * k
    )
    assert a.next_step(state).summarize_mask != a.next_step(later).summarize_mask


def test_random_subset_zero_count_and_full_count():
    sched = RandomSubsetSchedule(3, 40, counts=[0, 3, 5])
    never = (-1, -1, -1)
    assert sched.next_step(_hand_state(step=0, last_summarized=never)) == Step(False, 10)
    full = sched.next_step(_hand_state(step=1, last_summarized=never))
    assert full == Step(True, 10) and full.summarize_mask is None
    assert sched.next_step(_hand_state(step=2, last_summarized=never)).summarize_mask is None  # count > n_clusters
    assert sched.next_step(_hand_state(step=3)) is None


def test_random_subset_never_picks_empty_clusters():
    labels = (0, 0, 1, 1)  # clusters 2 and 3 empty
    centers = ((0.0,), (1.0,), (2.0,), (3.0,))
    for seed in range(20):
        state = _hand_state(
            step=0, labels=labels, centers=centers, last_summarized=(-1,) * 4, radius=(1.0,) * 4
        )
        assert _chosen(RandomSubsetSchedule(1, 10, counts=[1], seed=seed).next_step(state)) in ([0], [1])
        # Asking for more than exist (but fewer than k) selects every non-empty cluster, no more.
        assert _chosen(RandomSubsetSchedule(1, 10, counts=[3], seed=seed).next_step(state)) == [0, 1]


def test_random_subset_changed_only_draws_among_dirty_clusters():
    # 5 clusters: 0 never summarized, 1 churned (member 2 moved to 3), 2 clean, 3 churned, 4 empty.
    labels = (0, 0, 1, 3, 3, 2, 2)
    anchor_members = ([0, 1], [2, 3], [5, 6], [4], [])
    state = _hand_state(
        step=0,
        labels=labels,
        centers=((0.0,), (1.0,), (2.0,), (3.0,), (4.0,)),
        anchor_members=anchor_members,
        last_summarized=(-1, 1, 1, 1, 1),
        radius=(1.0,) * 5,
    )
    seen = set()
    for seed in range(30):
        out = RandomSubsetSchedule(1, 10, counts=[2], seed=seed).next_step(state)
        chosen = _chosen(out)
        assert len(chosen) == 2 and set(chosen) <= {0, 1, 3}
        seen.add(tuple(chosen))
    assert seen == {(0, 1), (0, 3), (1, 3)}  # the clean (2) and the empty (4) cluster are never drawn
    # More than the dirty clusters: all of them, as a partial step; nothing dirty: plain continuation.
    assert _chosen(RandomSubsetSchedule(1, 10, counts=[9]).next_step(state)) == [0, 1, 3]
    clean = _hand_state(step=0)
    assert RandomSubsetSchedule(1, 10, counts=[2]).next_step(clean) == Step(False, 5)
    # changed_only=False keeps the old behavior: any non-empty cluster, including clean ones.
    seen = {
        tuple(_chosen(RandomSubsetSchedule(1, 10, counts=[2], seed=s, changed_only=False).next_step(state)))
        for s in range(60)
    }
    assert any(2 in c for c in seen) and not any(4 in c for c in seen)


def test_random_subset_changed_only_is_full_step_only_when_every_cluster_is_dirty():
    never = _hand_state(step=0, last_summarized=(-1, -1, -1))
    assert RandomSubsetSchedule(1, 10, counts=[3]).next_step(never).summarize_mask is None


def test_random_subset_changed_only_needs_anchors():
    bare = State(
        step=0, n_summaries=0, lloyd_iters_used=0, labels=np.zeros(2, dtype=int), prev_labels=None,
        frac_changed=None, centers=np.zeros((1, 1)), inertia=0.0, prev_inertia=None, last_n_iter=0,
        last_max_iter=0, summary_shift=None,
    )
    with pytest.raises(ValueError, match="anchors"):
        RandomSubsetSchedule(1, 10, counts=[1]).next_step(bare)
    RandomSubsetSchedule(1, 10, counts=[1], changed_only=False).next_step(bare)


def test_random_subset_validation():
    with pytest.raises(ValueError, match="counts"):
        RandomSubsetSchedule(3, 40)
    with pytest.raises(ValueError, match="length"):
        RandomSubsetSchedule(3, 40, counts=[1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        RandomSubsetSchedule(3, 40, counts=[1, -1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        RandomSubsetSchedule(3, 40, counts=[1, 1.5, 1])
    RandomSubsetSchedule(3, 40, counts=np.array([1, 0, 2]))


def test_random_subset_end_to_end_budget_is_exact(synthetic):
    counts = [2, 0, 3]
    rec = _Recording(inner=RandomSubsetSchedule(3, 40, counts=counts, seed=0))
    result = _run(synthetic, rec)
    assert [r["n_summarized"] for r in result.trace] == [0, 2, 0, 3]
    assert [r["summarize"] for r in result.trace] == [False, True, False, True]
    assert result.n_calls <= sum(counts)  # only first-step clusters are dirty, so calls == counts[0] here


def test_random_subset_matches_an_adaptive_runs_per_step_calls(synthetic):
    """Fed an adaptive run's per-step `n_calls`, the control spends exactly those calls at every step.

    The adaptive run also *selects* clusters that turn out to be memo-cache hits (`n_summarized` > `n_calls`);
    the control draws only among changed clusters, so it selects fewer but spends the same calls."""
    for tau in (0.0, 0.3):
        adaptive = _run(synthetic, AdaptiveSchedule(4, 5, tau_drift=tau, tau_churn=tau), init="random")
        counts = [r["n_calls"] for r in adaptive.trace[1:]]
        assert sum(counts) > counts[0] > 0  # a non-trivial budget with a partial step after the first
        for seed in range(3):
            control = _run(synthetic, RandomSubsetSchedule(4, 5, counts=counts, seed=seed), init="random")
            assert [r["n_calls"] for r in control.trace] == [r["n_calls"] for r in adaptive.trace]
            assert control.n_calls == adaptive.n_calls
            assert all(c["n_summarized"] <= a["n_summarized"] for c, a in zip(control.trace, adaptive.trace))


def test_random_subset_seed_validation():
    for bad in (-1, 1.5, True, None):
        with pytest.raises(ValueError, match="seed"):
            RandomSubsetSchedule(1, 10, counts=[1], seed=bad)


# --- build_schedule -----------------------------------------------------------------------------------------


def test_build_schedule_adaptive_and_random_subset():
    adaptive = build_schedule({"type": "adaptive", "n_summaries": 4, "max_iter": 40, "tau_drift": 0.5, "s_max": 2})
    assert isinstance(adaptive, AdaptiveSchedule)
    assert (adaptive.n_summaries, adaptive.tau_drift, adaptive.tau_churn, adaptive.s_max) == (4, 0.5, 0.4, 2)
    rnd = build_schedule({"type": "random_subset", "n_summaries": 2, "counts": [1, 2], "seed": 3})
    assert isinstance(rnd, RandomSubsetSchedule)
    assert (rnd.counts, rnd.seed) == ((1, 2), 3)


def test_build_schedule_rejects_unknown_params_and_types():
    with pytest.raises(TypeError):
        build_schedule({"type": "adaptive", "tau": 1})
    with pytest.raises(TypeError):
        build_schedule({"type": "random_subset", "counts": [1], "bogus": 1})
    with pytest.raises(ValueError, match="adaptive"):
        build_schedule({"type": "nope"})
