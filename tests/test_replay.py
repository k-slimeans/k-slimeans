"""`kslimeans.replay`: the one-iteration-at-a-time Lloyd replay equals real KMeans fits and `summary_kmeans`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans

from kslimeans.core import _objective, summary_kmeans
from kslimeans.replay import (
    LloydReplayer,
    SummaryEmbeddings,
    assigned_objective,
    fit_initial,
    replay_run,
    replay_step,
    state_of,
    summary_embeddings,
    verify_replay,
)
from kslimeans.runlog import RunRecord
from kslimeans.schedules import FixedSchedule
from kslimeans.summarizers.base import Cluster, Summarizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_equivalence import LookupEmbedder  # noqa: E402
from synth import closest_to_mean, make_lookup_embedder, make_synthetic_data  # noqa: E402

K = 12


@pytest.fixture(scope="module")
def data():
    texts, X = make_synthetic_data()
    return texts, X, make_lookup_embedder(texts, X)


def _kmeans(X, start, max_iter, tol=1e-4):
    return KMeans(n_clusters=len(start), init=start, max_iter=max_iter, tol=tol).fit(X)


def _start(X, k=K, seed=1):
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), k, replace=False)] + rng.normal(scale=2.0, size=(k, X.shape[1])).astype(np.float32)


def _assert_same_fit(block, km):
    assert np.array_equal(block.final_labels, km.labels_)
    assert block.n_iter == km.n_iter_
    np.testing.assert_allclose(block.final_centers, km.cluster_centers_, rtol=0, atol=1e-5)
    assert block.end.objective == pytest.approx(km.inertia_, rel=1e-5)


@pytest.mark.parametrize("max_iter", [1, 2, 3, 5, 300])
def test_block_equals_single_fit(data, max_iter):
    _, X, _ = data
    start = _start(X)
    _assert_same_fit(LloydReplayer(X).run_block(start, max_iter), _kmeans(X, start, max_iter))


def test_every_prefix_of_a_block_equals_a_shorter_fit(data):
    _, X, _ = data
    start = _start(X)
    block = LloydReplayer(X).run_block(start, 6, stop=False)
    for t in range(1, 7):
        km = _kmeans(X, start, t, tol=0)
        assert np.array_equal(block.iterations[t].labels, km.labels_)
        np.testing.assert_allclose(block.iterations[t].centers, km.cluster_centers_, rtol=0, atol=1e-5)


def test_tol_stop(data):
    _, X, _ = data
    start = _start(X)
    loose = 0.5
    block = LloydReplayer(X, tol=loose).run_block(start, 100)
    assert block.stop == "tol"
    assert block.n_iter < 100
    _assert_same_fit(block, _kmeans(X, start, 100, tol=loose))


def test_strict_convergence_stop(data):
    _, X, _ = data
    start = _start(X)
    block = LloydReplayer(X, tol=1e-12).run_block(start, 300)
    assert block.stop == "strict"
    assert block.n_iter < 300
    _assert_same_fit(block, _kmeans(X, start, 300, tol=1e-12))


def test_fixed_length_ignores_the_stops(data):
    _, X, _ = data
    start = _start(X)
    block = LloydReplayer(X, tol=1e-12).run_block(start, 40, stop=False)
    assert block.n_iter == 40 and block.stop == "fixed"


def test_identical_centers_leave_clusters_empty_and_are_relocated_like_sklearn(data):
    _, X, _ = data
    start = _start(X)
    start[1] = start[0]
    start[2] = start[0]
    block = LloydReplayer(X).run_block(start, 20)
    sizes = np.bincount(block.start.labels, minlength=K)
    assert sizes[0] > 0 and sizes[1] == 0 and sizes[2] == 0  # argmin ties go to the first of the identical centers
    centers = block.iterations[1].centers
    assert not np.array_equal(centers[1], centers[0]) and not np.array_equal(centers[2], centers[0])  # re-seeded
    _assert_same_fit(block, _kmeans(X, start, 20))


def test_far_away_center_is_empty_then_relocated(data):
    _, X, _ = data
    start = _start(X)
    start[5] = 1e3
    block = LloydReplayer(X).run_block(start, 20)
    assert np.bincount(block.start.labels, minlength=K)[5] == 0
    _assert_same_fit(block, _kmeans(X, start, 20))


def test_objective_from_labels_matches_core_objective(data):
    _, X, _ = data
    block = LloydReplayer(X).run_block(_start(X), 8)
    for it in block.iterations:
        assert it.objective == pytest.approx(_objective(X, it.centers), rel=1e-12)
    assert block.start.objective == pytest.approx(assigned_objective(X, block.start.centers, block.start.labels))


def test_iteration_bookkeeping(data):
    _, X, _ = data
    block = LloydReplayer(X).run_block(_start(X), 6, stop=False)
    assert block.start.n_reassigned == 0 and block.start.center_shift == 0
    for prev, it in zip(block.iterations, block.iterations[1:]):
        assert it.n_reassigned == int(np.count_nonzero(it.labels != prev.labels))
        assert it.center_shift == pytest.approx(float(np.linalg.norm(it.centers - prev.centers)), rel=1e-4)
        assert it.objective <= prev.objective + 1e-6  # Lloyd never increases the objective


class DuplicatingSummarizer(Summarizer):
    """closest-to-mean summaries, except clusters 1 and 2 reuse cluster 0's text (identical centers)."""

    needs_sentences = False

    def __init__(self, embed):
        self._embed = embed
        self._first = None

    def summarize_batch(self, clusters: list[Cluster]) -> list[str]:
        out = [closest_to_mean(list(c.texts), self._embed) for c in clusters]
        for c, text in zip(clusters, out):
            if c.index == 0:
                self._first = text
        return [self._first if c.index in (1, 2) and self._first else text for c, text in zip(clusters, out)]

    def summarize(self, cluster: Cluster) -> str:
        return self.summarize_batch([cluster])[0]


def _logged_run(data, summarizer_cls, **schedule):
    texts, X, embed = data
    result = summary_kmeans(
        texts,
        K,
        summarizer_cls(embed),
        LookupEmbedder(embed),
        FixedSchedule(**{"n_summaries": 3, "max_iter": 60, "initial_iter": True, "final_iter": True, **schedule}),
        text_features=X,
        tol=-1.0,
        random_state=3,
        verbose=False,
    )
    record = RunRecord(
        method="m",
        seed=3,
        trace=result.trace,
        summaries_evolution=result.summaries_evolution,
        assignments_evolution=result.assignments_evolution,
        summary_source_steps=result.summary_source_steps,
    )
    return result, record


@pytest.mark.parametrize("summarizer_cls", [DuplicatingSummarizer, None])
def test_replay_run_reproduces_summary_kmeans(data, summarizer_cls):
    from test_core_equivalence import ClosestToMeanSummarizer

    texts, X, embed = data
    result, record = _logged_run(data, summarizer_cls or ClosestToMeanSummarizer)
    emb = summary_embeddings(record, LookupEmbedder(embed))
    for a, b in zip(emb, result.centroids_evolution):
        assert np.array_equal(a, b)

    replay = replay_run(record, X, emb)
    assert len(replay.steps) == 3
    checks = verify_replay(record, replay)
    assert len(checks) == 4
    for check in checks:
        assert check.n_label_mismatch == 0 and check.n_iter == check.logged_n_iter
        assert check.rel_err_inertia < 1e-5
        if check.rel_err_summary_inertia is not None:
            assert check.rel_err_summary_inertia < 1e-9
    if summarizer_cls is DuplicatingSummarizer:
        first = replay.steps[0]
        assert (np.bincount(first.block.start.labels, minlength=K) == 0).sum() >= 2  # collapsed centers
    assert np.array_equal(replay.steps[-1].block.final_labels, result.assignments)


def test_counterfactual_is_plain_lloyd_from_the_pre_state(data):
    texts, X, embed = data
    result, record = _logged_run(data, DuplicatingSummarizer)
    replay = replay_run(record, X, summary_embeddings(record, LookupEmbedder(embed)))
    for step in replay.steps:
        cf = step.counterfactual
        assert cf.n_iter == step.block.n_iter
        km = _kmeans(X, step.pre.centers, cf.n_iter, tol=0)
        assert np.array_equal(cf.final_labels, km.labels_)
        np.testing.assert_allclose(cf.final_centers, km.cluster_centers_, rtol=0, atol=1e-5)
        assert cf.start.objective == pytest.approx(step.pre.objective, rel=1e-9)


def test_pre_state_of_step_one_is_the_initial_fit(data):
    texts, X, embed = data
    result, record = _logged_run(data, DuplicatingSummarizer)
    replay = replay_run(record, X, summary_embeddings(record, LookupEmbedder(embed)))
    assert np.array_equal(replay.steps[0].pre.labels, result.assignments_evolution[0])
    assert replay.steps[1].pre.labels is replay.steps[0].block.final_labels


def test_swap_hook_replays_a_block_from_swapped_rows(data):
    texts, X, embed = data
    _, record = _logged_run(data, DuplicatingSummarizer)
    emb = summary_embeddings(record, LookupEmbedder(embed))
    kept = [0, 3, 7]

    def keep_kmeans_center(j, pre, summary_emb):
        out = summary_emb.copy()
        if j == 0:
            out[kept] = pre.centers[kept]
        return out

    swapped = replay_run(record, X, emb, swap=keep_kmeans_center, counterfactual=False)
    first = swapped.steps[0]
    assert np.array_equal(first.start_centers[kept], first.pre.centers[kept])
    others = [c for c in range(K) if c not in kept]
    assert np.array_equal(first.start_centers[others], emb[0][others])
    expected = _kmeans(X, first.start_centers, record.trace[1]["last_max_iter"])
    assert np.array_equal(first.block.final_labels, expected.labels_)
    assert first.counterfactual is None
    assert len(swapped.steps) == 3  # later steps continue from the swapped block


def test_swapping_every_row_back_equals_the_counterfactual(data):
    texts, X, embed = data
    _, record = _logged_run(data, DuplicatingSummarizer)
    emb = summary_embeddings(record, LookupEmbedder(embed))
    replay = replay_run(record, X, emb, swap=lambda j, pre, e: pre.centers)
    first = replay.steps[0]
    assert np.array_equal(first.block.final_labels[:0], first.counterfactual.final_labels[:0])
    assert first.block.start.objective == pytest.approx(first.pre.objective, rel=1e-9)
    assert first.old_assignment_objective == pytest.approx(first.pre.objective, rel=1e-9)


def test_replay_step_accepts_arbitrary_start_centers(data):
    _, X, _ = data
    pre_state, n_iter = fit_initial(X, K, 20, 0)
    replayer = LloydReplayer(X)
    step = replay_step(replayer, pre_state, _start(X, seed=9), 10)
    assert step.block.n_iter <= 10 and step.counterfactual.n_iter == step.block.n_iter
    assert step.old_assignment_objective >= step.block.start.objective - 1e-6  # nearest assignment is no worse
    assert state_of(X, pre_state.labels, pre_state.centers).objective == pre_state.objective


def test_summary_embeddings_round_trip(tmp_path):
    arrays = {("run", "k-SLMmeans", 0, 0): np.ones((2, 3), np.float32), ("run", "k-SLMmeans", 0, 1): np.zeros((2, 3), np.float32)}
    path = tmp_path / "e.npz"
    SummaryEmbeddings(arrays, {"device": "cpu"}).save(path)
    loaded = SummaryEmbeddings.load(path)
    assert loaded.meta == {"device": "cpu"}
    steps = loaded.steps("run", "k-SLMmeans", 0)
    assert len(steps) == 2 and steps[0].dtype == np.float32 and steps[1].sum() == 0
    with pytest.raises(KeyError):
        loaded.steps("run", "k-SLMmeans", 1)
