"""`Step.summarize_mask`: per-cluster summary steps, exact call counting, and full-mask equivalence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import kslimeans.core as core
from kslimeans.core import summary_kmeans
from kslimeans.schedules import FixedSchedule, State, Step, SummarySchedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_equivalence import (  # noqa: E402,F401
    ClosestToMeanSummarizer,
    LookupEmbedder,
    _assert_matches,
    _load_fixture,
    synthetic,
)
from synth import N_BLOBS  # noqa: E402


class _AllTrueMask(SummarySchedule):
    """Wraps a schedule, adding an explicit all-True mask to every summary step."""

    def __init__(self, inner: SummarySchedule) -> None:
        self.inner = inner

    def first_iters(self) -> int:
        return self.inner.first_iters()

    def next_step(self, state: State) -> Step | None:
        step = self.inner.next_step(state)
        if step is None:
            return None
        return Step(step.summarize, step.kmeans_iters, [True] * N_BLOBS if step.summarize else None)


class _Scripted(SummarySchedule):
    """Plays a fixed list of steps and records every `State` it is shown."""

    def __init__(self, steps: list[Step], first: int = 20) -> None:
        self.steps, self.first, self.states = steps, first, []

    def first_iters(self) -> int:
        return self.first

    def next_step(self, state: State) -> Step | None:
        self.states.append(state)
        return self.steps[state.step] if state.step < len(self.steps) else None


class _SpySummarizer(ClosestToMeanSummarizer):
    """Records `cluster.index` of every cluster sent to `summarize_batch`, one list per call."""

    def __init__(self, embed) -> None:
        super().__init__(embed)
        self.batches: list[list[int]] = []

    def summarize_batch(self, clusters):
        self.batches.append([c.index for c in clusters])
        return super().summarize_batch(clusters)

    @property
    def seen(self) -> list[int]:
        return [i for batch in self.batches for i in batch]


def _run(synthetic, schedule, *, tol=-1.0, force_context_length=0, summarizer=None):
    texts, X, embed = synthetic
    return summary_kmeans(
        texts,
        N_BLOBS,
        summarizer or ClosestToMeanSummarizer(embed),
        LookupEmbedder(embed),
        schedule,
        text_features=X,
        force_context_length=force_context_length,
        tol=tol,
        random_state=0,
        init="k-means++",
        verbose=False,
    )


def _mask(*on: int) -> list[bool]:
    return [c in on for c in range(N_BLOBS)]


# --- Step -------------------------------------------------------------------------------------


def test_step_normalizes_mask_to_tuple_of_bools():
    a = Step(True, 5, np.array([True, False, True]))
    b = Step(True, 5, [True, False, True])
    c = Step(True, 5, (True, False, True))
    assert a.summarize_mask == (True, False, True)
    assert all(type(m) is bool for m in a.summarize_mask)
    assert a == b == c
    assert Step(True, 5).summarize_mask is None
    assert Step(True, 5) == Step(True, 5, None)


# --- equivalence of the all-True mask ------------------------------------------------------------


@pytest.mark.parametrize(
    "initial_iter,final_iter", [(True, True), (True, False), (False, True), (False, False)]
)
def test_all_true_mask_matches_fixture_for_each_flag_combo(synthetic, initial_iter, final_iter):
    fixture = _load_fixture(f"synthetic_i{int(initial_iter)}_f{int(final_iter)}.npz")
    schedule = _AllTrueMask(FixedSchedule(3, 30, initial_iter, final_iter))
    result = _run(synthetic, schedule)
    _assert_matches(result, fixture)
    assert result.converged is False


def test_all_true_mask_tol_break_matches_fixture(synthetic):
    fixture = _load_fixture("synthetic_tol_break.npz")
    result = _run(synthetic, _AllTrueMask(FixedSchedule(10, 100)), tol=1e-4)
    assert result.converged is True
    _assert_matches(result, fixture)


def test_all_true_mask_force_context_length_matches_fixture(synthetic):
    fixture = _load_fixture("synthetic_force_context.npz")
    result = _run(synthetic, _AllTrueMask(FixedSchedule(3, 30)), force_context_length=5)
    _assert_matches(result, fixture)


def test_all_true_mask_matches_none_mask_trace(synthetic):
    plain = _run(synthetic, FixedSchedule(3, 30))
    masked = _run(synthetic, _AllTrueMask(FixedSchedule(3, 30)))
    # sklearn's float32 inertia isn't bit-reproducible run to run (even at step 0), so skip those keys.
    exact = ("step", "frac_changed", "summary_shift", "last_n_iter", "summarize", "n_calls", "n_summarized")
    assert [{k: r[k] for k in exact} for r in masked.trace] == [{k: r[k] for k in exact} for r in plain.trace]
    for a, b in zip(masked.centroids_evolution, plain.centroids_evolution):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert masked.n_calls == plain.n_calls


# --- partial masks ------------------------------------------------------------------------------


def _partial_schedule() -> _Scripted:
    return _Scripted(
        [
            Step(True, 5, _mask(0, 2)),
            Step(True, 5, _mask(1)),
            Step(True, 5, None),
        ]
    )


def test_partial_masks_set_start_centroids_per_row(synthetic):
    texts, X, embed = synthetic
    schedule = _partial_schedule()
    result = _run(synthetic, schedule)

    masks = [_mask(0, 2), _mask(1), [True] * N_BLOBS]
    assert len(result.centroids_evolution) == 3
    for i, (start, mask) in enumerate(zip(result.centroids_evolution, masks)):
        prev_centers = schedule.states[i].centers  # the k-means centers the step started from
        for c in range(N_BLOBS):
            if mask[c]:
                summary = result.summaries_evolution[i][c]
                assert np.array_equal(start[c], embed([summary])[0])
            else:
                assert np.array_equal(start[c], prev_centers[c])


def test_spy_summarizer_only_sees_masked_on_clusters(synthetic):
    _, _, embed = synthetic
    spy = _SpySummarizer(embed)
    _run(synthetic, _Scripted([Step(True, 5, _mask(0, 2)), Step(True, 5, _mask(1))]), summarizer=spy)
    assert [sorted(b) for b in spy.batches] == [[0, 2], [1]]


def test_carry_forward_summaries_and_embeddings(synthetic):
    _, _, embed = synthetic
    result = _run(synthetic, _Scripted([Step(True, 5, _mask(0, 2)), Step(True, 5, _mask(1))]))

    first, second = result.summaries_evolution
    assert [s is None for s in first] == [c not in (0, 2) for c in range(N_BLOBS)]
    assert [s is None for s in second] == [c not in (0, 1, 2) for c in range(N_BLOBS)]
    assert second[0] == first[0] and second[2] == first[2]
    assert result.summaries == second

    emb = result.summary_embeddings
    assert emb.shape[0] == N_BLOBS
    never = [c for c in range(N_BLOBS) if c not in (0, 1, 2)]
    assert np.isnan(emb[never]).all()
    for c in (0, 1, 2):
        assert np.array_equal(emb[c], embed([result.summaries[c]])[0])


def test_summary_shift_covers_masked_on_rows_only(synthetic):
    _, _, embed = synthetic
    schedule = _Scripted([Step(True, 5, _mask(0, 2))])
    result = _run(synthetic, schedule)
    ref = schedule.states[0].centers
    expected = sum(
        float(np.linalg.norm(ref[c] - embed([result.summaries[c]])[0])) for c in (0, 2)
    )
    np.testing.assert_allclose(result.trace[1]["summary_shift"], expected, rtol=1e-6)


def test_ref_updates_for_masked_on_rows_only(synthetic):
    # Step 2 re-summarizes cluster 0, whose ref row is its step-1 summary embedding, not its k-means
    # center: an unchanged membership gives shift exactly 0 only if `ref` was updated at step 1.
    schedule = _Scripted([Step(True, 5, _mask(0)), Step(True, 5, _mask(0))])
    result = _run(synthetic, schedule)
    assert schedule.states[1].labels is not None
    same = np.array_equal(schedule.states[1].labels == 0, schedule.states[0].labels == 0)
    assert same, "precondition: cluster 0 membership must be stable on the blobs"
    assert result.trace[2]["summary_shift"] == 0.0


# --- call counting ------------------------------------------------------------------------------


def test_n_calls_counts_cache_misses_only(synthetic):
    _, _, embed = synthetic
    spy = _SpySummarizer(embed)
    schedule = _Scripted([Step(True, 20, _mask(0, 2)), Step(True, 20, _mask(0, 2))])
    result = _run(synthetic, schedule, summarizer=spy)

    # Blobs are well separated, so memberships are already stable after the first fit.
    for c in (0, 2):
        assert np.array_equal(
            schedule.states[2].labels == c, schedule.states[0].labels == c
        ), "precondition: membership of clusters 0 and 2 must not change"
    assert [r["n_calls"] for r in result.trace] == [0, 2, 0]
    assert [r["n_summarized"] for r in result.trace] == [0, 2, 2]
    assert result.n_calls == 2
    assert spy.seen == [0, 2]  # the second step never reached the summarizer


def test_trace_and_state_call_counts(synthetic):
    _, _, embed = synthetic
    spy = _SpySummarizer(embed)
    schedule = _Scripted(
        [Step(False, 3), Step(True, 5, _mask(0, 2)), Step(True, 5, _mask(1)), Step(True, 5)]
    )
    result = _run(synthetic, schedule, summarizer=spy)

    trace = result.trace
    assert [r["n_summarized"] for r in trace] == [0, 0, 2, 1, N_BLOBS]
    assert trace[0]["n_calls"] == trace[1]["n_calls"] == 0
    assert trace[2]["n_calls"] == 2
    assert trace[3]["n_calls"] == 1
    # State shown before each step holds the cumulative total after the fits so far.
    assert [s.n_calls for s in schedule.states] == list(np.cumsum([r["n_calls"] for r in trace]))
    assert result.n_calls == sum(r["n_calls"] for r in trace) == len(spy.seen)
    assert len(result.summaries_evolution) == 3


def test_n_calls_includes_tol_break_step(synthetic):
    spy = _SpySummarizer(synthetic[2])
    # A huge tol stops at the first (full) summary step, after it has already summarized every cluster.
    result = _run(synthetic, FixedSchedule(3, 30), tol=1e9, summarizer=spy)
    assert result.converged
    assert len(result.trace) == 1
    assert sum(r["n_calls"] for r in result.trace) == 0
    assert result.n_calls == N_BLOBS == len(spy.seen)


def test_fixed_schedule_counts_one_call_per_nonempty_cluster(synthetic):
    result = _run(synthetic, FixedSchedule(3, 30))
    assert result.trace[1]["n_calls"] == N_BLOBS
    assert result.trace[1]["n_summarized"] == N_BLOBS
    assert result.n_calls == sum(r["n_calls"] for r in result.trace)


# --- tol rule -----------------------------------------------------------------------------------


def test_partial_step_with_zero_shift_does_not_stop(synthetic):
    schedule = _Scripted([Step(True, 20, _mask(0)), Step(True, 20, _mask(0)), Step(True, 5, _mask(1))])
    result = _run(synthetic, schedule, tol=1e-4)
    assert result.trace[2]["summary_shift"] == 0.0 and result.trace[2]["n_calls"] == 0
    assert result.converged is False
    assert len(result.trace) == 4


# --- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [[True] * (N_BLOBS - 1), [True] * (N_BLOBS + 1), []])
def test_wrong_length_mask_raises(synthetic, bad):
    with pytest.raises(ValueError, match="length"):
        _run(synthetic, _Scripted([Step(True, 5, bad)]))


def test_mask_on_non_summary_step_raises_at_construction():
    with pytest.raises(ValueError, match="summarize"):
        Step(False, 5, _mask(0))


# --- aliasing, all-False masks, partial -> full, empty clusters --------------------------------------


@pytest.fixture
def recorded_inits(monkeypatch):
    """Patch `core.KMeans` to record a private copy of every array `init` it is constructed with."""
    inits: list[np.ndarray] = []

    class _Recording(core.KMeans):
        def __init__(self, n_clusters=8, *, init="k-means++", max_iter=300, random_state=None):
            if isinstance(init, np.ndarray):
                inits.append(np.array(init, copy=True))
            super().__init__(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state)

    monkeypatch.setattr(core, "KMeans", _Recording)
    return inits


def test_centroids_evolution_and_embeddings_are_not_aliased(synthetic, recorded_inits):
    _, _, embed = synthetic
    steps = [Step(True, 5), Step(True, 5, _mask(0, 2)), Step(True, 5, _mask(1)), Step(True, 5)]
    result = _run(synthetic, _Scripted(steps))

    assert len(recorded_inits) == len(result.centroids_evolution) == 4
    for start, init in zip(result.centroids_evolution, recorded_inits):
        assert np.array_equal(start, init)  # nothing mutated `start` after KMeans received it

    # Rows of the final summary_embeddings equal the embedding computed when each cluster was last
    # summarized: here every cluster is re-summarized at the final full step.
    assert np.array_equal(result.summary_embeddings, recorded_inits[3])
    # Distinct arrays: a dropped `.copy()` would make consecutive entries share memory.
    for a, b in zip(result.centroids_evolution, result.centroids_evolution[1:]):
        assert not np.shares_memory(a, b)


def test_summary_embeddings_rows_keep_their_own_step(synthetic, recorded_inits):
    steps = [Step(True, 5, _mask(0, 2)), Step(True, 5, _mask(1))]
    result = _run(synthetic, _Scripted(steps))
    emb = result.summary_embeddings
    assert np.array_equal(emb[[0, 2]], recorded_inits[0][[0, 2]])
    assert np.array_equal(emb[1], recorded_inits[1][1])


class _SpyEmbedder(LookupEmbedder):
    def __init__(self, embed) -> None:
        super().__init__(embed)
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return super().encode(texts)


def test_all_false_mask_is_a_free_summary_step(synthetic):
    texts, X, embed = synthetic
    spy_emb, spy_sum = _SpyEmbedder(embed), _SpySummarizer(embed)
    schedule = _Scripted([Step(True, 5, _mask())])
    result = summary_kmeans(
        texts, N_BLOBS, spy_sum, spy_emb, schedule,
        text_features=X, tol=1e-4, random_state=0, verbose=False,
    )  # fmt: skip

    assert spy_emb.calls == [] and spy_sum.batches == []
    assert result.n_calls == 0 and result.trace[1]["n_calls"] == 0
    assert result.trace[1]["n_summarized"] == 0
    assert result.trace[1]["summary_shift"] == 0.0
    assert result.converged is False
    assert len(result.trace) == 2  # tol did not stop the run
    assert schedule.states[1].n_summaries == 1
    assert result.summaries == [None] * N_BLOBS
    assert result.summary_embeddings is None
    assert np.array_equal(result.centroids_evolution[0], schedule.states[0].centers)


def test_partial_then_full_shift_uses_partially_updated_ref(synthetic):
    _, _, embed = synthetic
    schedule = _Scripted([Step(True, 5, _mask(0, 2)), Step(True, 5)])
    result = _run(synthetic, schedule)

    step0_centers = schedule.states[0].centers
    partial_summaries = result.summaries_evolution[0]
    ref = step0_centers.copy()
    for c in (0, 2):
        ref[c] = embed([partial_summaries[c]])[0]
    new = embed(result.summaries_evolution[1])
    expected = float(np.linalg.norm(ref - new, axis=1).sum())
    np.testing.assert_allclose(result.trace[2]["summary_shift"], expected, rtol=1e-5)
    assert not np.isnan(result.summary_embeddings).any()
    assert result.trace[2]["n_summarized"] == N_BLOBS


def test_masked_on_empty_cluster_gets_empty_summary_for_free(synthetic, monkeypatch):
    _, _, embed = synthetic
    last = N_BLOBS - 1

    class _EmptyLast(core.KMeans):
        """Folds the last cluster into cluster 0 on the first fit, leaving it empty."""

        fits = 0

        def fit_predict(self, X, *args, **kwargs):
            labels = super().fit_predict(X, *args, **kwargs)
            type(self).fits += 1
            return np.where(labels == last, 0, labels) if type(self).fits == 1 else labels

    monkeypatch.setattr(core, "KMeans", _EmptyLast)
    spy = _SpySummarizer(embed)
    result = _run(synthetic, _Scripted([Step(True, 5, _mask(last))]), summarizer=spy)

    assert result.summaries[last] == ""
    assert spy.batches == [] and result.n_calls == 0
    assert result.trace[1]["n_summarized"] == 1
    assert np.array_equal(result.summary_embeddings[last], np.zeros_like(result.summary_embeddings[last]))


class _DriftingEmbedder(LookupEmbedder):
    """Offsets every `encode` output by a call-index-dependent amount, so re-embedding an unchanged
    summary at a later step gives different values (a memo hit alone would hide an in-place overwrite)."""

    def __init__(self, embed) -> None:
        super().__init__(embed)
        self.n_encodes = 0

    def encode(self, texts):
        out = super().encode(texts) + np.float32(1e-3 * self.n_encodes)
        self.n_encodes += 1
        return out


def test_earlier_evolution_entries_survive_later_partial_steps(synthetic, recorded_inits):
    texts, X, embed = synthetic
    schedule = _Scripted(
        [Step(True, 5), Step(True, 5, _mask(0, 2)), Step(True, 5, _mask(1)), Step(True, 5, _mask(0))]
    )
    result = summary_kmeans(
        texts, N_BLOBS, ClosestToMeanSummarizer(embed), _DriftingEmbedder(embed), schedule,
        text_features=X, tol=-1.0, random_state=0, verbose=False,
    )  # fmt: skip

    assert len(recorded_inits) == len(result.centroids_evolution) == 4
    for i, (start, init) in enumerate(zip(result.centroids_evolution, recorded_inits)):
        assert np.array_equal(start, init), f"centroids_evolution[{i}] was mutated after the fact"
        assert not np.shares_memory(start, result.summary_embeddings)
    # Cluster 0 was re-embedded last (step 4), cluster 1 at step 3, cluster 2 at step 2.
    assert np.array_equal(result.summary_embeddings[0], recorded_inits[3][0])
    assert np.array_equal(result.summary_embeddings[1], recorded_inits[2][1])
    assert np.array_equal(result.summary_embeddings[2], recorded_inits[1][2])
