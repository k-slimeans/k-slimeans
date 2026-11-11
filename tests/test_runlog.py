"""`kslimeans.runlog`: parse run directories and rebuild the summarizer inputs of a logged step."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from kslimeans.cache import cached_embeddings
from kslimeans.core import summary_kmeans
from kslimeans.data import Dataset
from kslimeans.results import RunDir
from kslimeans.runlog import (
    CacheMissError,
    embed_summaries,
    load_run_embedder,
    MissingAssignmentsError,
    load_run,
    load_run_dataset,
    load_run_sentences,
    members_by_cluster,
    sent_clusters,
    step_inputs,
)
from kslimeans.schedules import FixedSchedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_equivalence import ClosestToMeanSummarizer, LookupEmbedder, synthetic  # noqa: E402,F401
from synth import N_BLOBS, make_lookup_embedder  # noqa: E402

CONFIG = {"name": "synth_run", "dataset": "synth", "embedder": "fake", "seeds": [3], "cache_dir": "cache"}


class StepSpy:
    """Per summary step, the clusters the summarizer was actually called on.

    `summary_kmeans` encodes the step's k summaries exactly once, right after summarizing, so each `encode`
    call closes a step (a step whose clusters were all memoized has no `summarize_batch` call: an empty list).
    """

    def __init__(self):
        self.steps: list[list] = []
        self._current: list = []


class SpyEmbedder(LookupEmbedder):
    def __init__(self, embed, spy):
        super().__init__(embed)
        self.spy = spy

    def encode(self, texts):
        self.spy.steps.append(self.spy._current)
        self.spy._current = []
        return super().encode(texts)


class RecordingSummarizer(ClosestToMeanSummarizer):
    def __init__(self, embed, spy):
        super().__init__(embed)
        self.spy = spy

    def summarize_batch(self, clusters):
        self.spy._current.extend(clusters)
        return super().summarize_batch(clusters)


@pytest.fixture(scope="module")
def noisy():
    """Overlapping blobs: with short Lloyd blocks the labels change at every KMeans fit."""
    rng = np.random.default_rng(1)
    centers = rng.normal(scale=2.0, size=(N_BLOBS, 8))
    X = (np.repeat(centers, 40, axis=0) + rng.normal(scale=2.0, size=(240, 8))).astype(np.float32)
    texts = [f"doc{i}" for i in range(len(X))]
    return texts, X, make_lookup_embedder(texts, X)


def _run_and_log(data, tmp_path, *, tol=-1.0, force_context_length=10, seed=3, max_iter=12):
    texts, X, embed = data
    spy = StepSpy()
    result = summary_kmeans(
        texts,
        N_BLOBS,
        RecordingSummarizer(embed, spy),
        SpyEmbedder(embed, spy),
        FixedSchedule(3, max_iter, True, True),
        text_features=X,
        force_context_length=force_context_length,
        tol=tol,
        random_state=seed,
        verbose=False,
    )
    run = RunDir(tmp_path, "synth_run", {**CONFIG, "seeds": [seed], "force_context_length": force_context_length})
    run.log_environment({"git_commit": "abc"})
    run.log_metrics({"seed": seed, "method": "kmeans", "acc": 0.5})
    run.log_metrics({"seed": seed, "method": "k-test", "acc": 0.6})
    run.log_trace({"seed": seed, "method": "k-test"}, result.trace)
    run.log_summaries(
        {
            "seed": seed,
            "method": "k-test",
            "summaries": result.summaries,
            "assignments": result.assignments,
            "summaries_evolution": result.summaries_evolution,
            "assignments_evolution": result.assignments_evolution,
            "summary_source_steps": result.summary_source_steps,
        }
    )
    return run.path, result, spy


def test_step_inputs_equal_what_the_summarizer_received(noisy, tmp_path):
    texts, X, _ = noisy
    path, result, spy = _run_and_log(noisy, tmp_path, max_iter=5)
    record = load_run(path).record("k-test", 3)

    # Guard against a vacuous test: every fit changes the labels, so an off-by-one in the partition
    # indexing would hand the summarizer different members.
    fits = result.assignments_evolution
    assert all(not np.array_equal(a, b) for a, b in zip(fits, fits[1:]))
    assert record.n_summary_steps == len(spy.steps) == 3

    n_checked = 0
    for j, received in enumerate(spy.steps):
        inputs = step_inputs(record, j, texts, X, force_context_length=10)
        assert [c.index for c in inputs] == list(range(N_BLOBS))
        assert all(len(c.doc_ids) <= 10 for c in inputs)
        # What the summarizer saw at this step, independent of how the partitions are indexed.
        sent = sent_clusters(record, j, texts, X, force_context_length=10)
        assert sent == [c.index for c in received]
        for c in received:
            mine = inputs[c.index]
            assert (mine.doc_ids, mine.texts) == (c.doc_ids, c.texts)
        n_checked += len(received)
    assert n_checked >= 3 * N_BLOBS - 2  # nearly nothing memoized with moving labels

    for j in range(record.n_summary_steps):
        source = result.summary_source_steps[j]
        np.testing.assert_array_equal(record.source_partition(j), fits[source])
        np.testing.assert_array_equal(record.partition_after(j), fits[source + 1])


def test_sent_clusters_skips_memoized_clusters(synthetic, tmp_path):
    # Well-separated blobs: after the first step the partition no longer changes, so the memo skips everything.
    texts, X, _ = synthetic
    path, _, spy = _run_and_log(synthetic, tmp_path)
    record = load_run(path).record("k-test", 3)

    first = step_inputs(record, 0, texts, X, 10)
    second = step_inputs(record, 1, texts, X, 10)
    unchanged = [c.index for c, d in zip(first, second) if c.doc_ids == d.doc_ids]
    assert unchanged, "fixture should leave at least one cluster unchanged"

    assert sent_clusters(record, 0, texts, X, 10) == list(range(N_BLOBS))
    for j, received in enumerate(spy.steps):
        assert sent_clusters(record, j, texts, X, 10) == [c.index for c in received]
    skipped = set(range(N_BLOBS)) - set(sent_clusters(record, 1, texts, X, 10))
    assert skipped == set(unchanged)
    assert len(spy.steps[1]) == N_BLOBS - len(unchanged)


def test_step_inputs_without_trimming(synthetic, tmp_path):
    texts, X, _ = synthetic
    path, result, spy = _run_and_log(synthetic, tmp_path, force_context_length=0)
    record = load_run(path).record("k-test", 3)
    inputs = step_inputs(record, 0, texts, X, force_context_length=0)
    assert [(c.index, c.doc_ids) for c in inputs] == [(c.index, c.doc_ids) for c in spy.steps[0]]


def test_step_inputs_refuses_a_run_without_trace_rows(synthetic, tmp_path):
    texts, X, _ = synthetic
    path, _, _ = _run_and_log(synthetic, tmp_path)
    record = load_run(path).record("k-test", 3)
    record.trace = []
    with pytest.raises(ValueError, match="no trace rows"):
        step_inputs(record, 0, texts, X, 10)


def test_record_fields_and_baseline(synthetic, tmp_path):
    path, result, _ = _run_and_log(synthetic, tmp_path)
    log = load_run(path)

    assert log.environment == {"git_commit": "abc"}
    assert log.config["dataset"] == "synth"
    assert log.methods == ["kmeans", "k-test"]
    assert log.seeds == [3]

    base = log.record("kmeans", 3)
    assert base.metrics == {"acc": 0.5}
    assert base.summaries_evolution == [] and base.summaries is None
    assert base.assignments_evolution is None and base.trace == []
    with pytest.raises(MissingAssignmentsError):
        base.source_partition(0)

    rec = log.record("k-test", 3)
    assert [row["step"] for row in rec.trace] == list(range(len(result.trace)))
    assert "seed" not in rec.trace[0] and "method" not in rec.metrics
    assert rec.summaries == result.summaries
    assert rec.summary_source_steps == result.summary_source_steps
    assert rec.assignments.dtype == np.int64
    np.testing.assert_array_equal(rec.assignments, result.assignments)
    assert all(a.dtype == np.int64 for a in rec.assignments_evolution)
    assert [r.seed for r in log.method_records("k-test")] == [3]
    with pytest.raises(KeyError):
        log.record("k-test", 99)


def test_trace_is_sorted_by_step(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    rows = [{"seed": 0, "method": "m", "step": s} for s in (2, 0, 1)]
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [r["step"] for r in load_run(run).record("m", 0).trace] == [0, 1, 2]


def test_tol_stop_has_no_partition_after(synthetic, tmp_path):
    path, result, _ = _run_and_log(synthetic, tmp_path, tol=1e9)
    record = load_run(path).record("k-test", 3)
    assert result.converged and record.n_summary_steps == 1
    assert record.source_partition(0) is record.assignments_evolution[0]
    with pytest.raises(IndexError, match="tol"):
        record.partition_after(0)
    with pytest.raises(IndexError):
        record.source_partition(1)


def test_old_format_row_loads_with_missing_fields_none(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(CONFIG))
    (run / "metrics.jsonl").write_text(json.dumps({"seed": 0, "method": "k-SLMmeans", "acc": 0.1}) + "\n")
    row = {
        "seed": 0,
        "method": "k-SLMmeans",
        "summaries": ["a", "b"],
        "assignments": [0, 1, 1],
        "summaries_evolution": [["a", "b"], ["a", None]],
    }
    (run / "summaries.jsonl").write_text(json.dumps(row) + "\n")

    log = load_run(run)
    rec = log.record("k-SLMmeans", 0)
    assert log.environment is None
    assert rec.trace == []
    assert rec.assignments_evolution is None and rec.summary_source_steps is None
    assert rec.summaries_evolution == [["a", "b"], ["a", None]]
    np.testing.assert_array_equal(rec.assignments, [0, 1, 1])
    with pytest.raises(MissingAssignmentsError, match="did not log"):
        rec.source_partition(0)
    with pytest.raises(MissingAssignmentsError):
        step_inputs(rec, 0, ["x", "y", "z"], np.zeros((3, 2)), 0)


def test_masked_step_inputs_are_refused(synthetic, tmp_path):
    texts, X, _ = synthetic
    path, _, _ = _run_and_log(synthetic, tmp_path)
    record = load_run(path).record("k-test", 3)
    record.trace[record.summary_source_steps[0] + 1]["n_summarized"] = 2
    with pytest.raises(ValueError, match="not logged"):
        step_inputs(record, 0, texts, X, 0)


def test_members_by_cluster_ascending():
    members = members_by_cluster(np.array([1, 0, 1, 2, 0]), 4)
    assert [m.tolist() for m in members] == [[1, 4], [0, 2], [3], []]


@pytest.fixture
def cached_run(synthetic, tmp_path, monkeypatch):
    """A run whose dataset resolves to the synthetic corpus; embeddings and sentences are not cached yet."""
    texts, X, embed = synthetic
    ds = Dataset("synth", texts, np.zeros(len(texts), dtype=int), N_BLOBS, "p", "t")
    monkeypatch.setattr("kslimeans.experiment.load_dataset", lambda name, **kw: ds)
    path, _, _ = _run_and_log(synthetic, tmp_path)
    return load_run(path), ds, X, LookupEmbedder(embed)


def test_load_run_dataset_hit_never_encodes(cached_run, tmp_path, monkeypatch):
    log, ds, X, fake = cached_run
    cache_dir = tmp_path / "cache"
    cached_embeddings("synth", ds.texts, fake, str(cache_dir))
    monkeypatch.setattr("kslimeans.cache.get_embedder", lambda name: pytest.fail("encode attempted"))

    got_ds, got_X = load_run_dataset(log, cache_dir)
    assert got_ds is ds
    np.testing.assert_array_equal(got_X, X)


def test_load_run_dataset_miss_raises_and_names_the_path(cached_run, tmp_path, monkeypatch):
    log, *_ = cached_run
    monkeypatch.setattr("kslimeans.cache.get_embedder", lambda name: pytest.fail("encode attempted"))
    with pytest.raises(CacheMissError, match=r"synth__fake_embeddings\.npz") as err:
        load_run_dataset(log, tmp_path / "empty")
    assert "cache_dir" in str(err.value) and "allow_encode" in str(err.value)


def test_load_run_dataset_stale_cache_counts_as_miss(cached_run, tmp_path):
    log, ds, _, fake = cached_run
    cache_dir = tmp_path / "cache"
    cached_embeddings("synth", ds.texts[::-1], fake, str(cache_dir))
    with pytest.raises(CacheMissError):
        load_run_dataset(log, cache_dir)


def test_load_run_dataset_allow_encode_encodes_with_the_named_embedder(cached_run, tmp_path, monkeypatch):
    log, ds, X, fake = cached_run
    monkeypatch.setattr("kslimeans.cache.get_embedder", lambda name: fake if name == "fake" else pytest.fail(name))
    cache_dir = tmp_path / "new_cache"
    _, got_X = load_run_dataset(log, cache_dir, allow_encode=True)
    np.testing.assert_array_equal(got_X, X)
    assert (cache_dir / "synth__fake_embeddings.npz").exists()
    load_run_dataset(log, cache_dir)  # now a hit


def test_load_run_sentences_miss_raises_and_allow_encode_builds(cached_run, tmp_path, monkeypatch):
    log, ds, _, _ = cached_run
    sentinel = object()
    built = []
    monkeypatch.setattr("kslimeans.runlog.has_cached_sentences", lambda *a: False)
    monkeypatch.setattr("kslimeans.runlog.cached_sentences", lambda *a: built.append(a) or sentinel)
    with pytest.raises(CacheMissError, match=r"synth__fake_sentences\.npz"):
        load_run_sentences(log, ds, tmp_path / "empty")
    assert built == []
    assert load_run_sentences(log, ds, tmp_path / "empty", allow_encode=True) is sentinel
    assert built and built[0][2] == "fake"


def test_embed_summaries_reproduces_core_start_centroids(noisy, tmp_path):
    _, _, embed = noisy
    path, result, spy = _run_and_log(noisy, tmp_path, max_iter=5)
    record = load_run(path).record("k-test", 3)
    assert len(result.centroids_evolution) == record.n_summary_steps == 3

    calls = []

    class CountingEmbedder(LookupEmbedder):
        def encode(self, texts):
            calls.append(list(texts))
            return super().encode(texts)

    for j, start in enumerate(result.centroids_evolution):
        got = embed_summaries(CountingEmbedder(embed), record.summaries_evolution[j])
        np.testing.assert_array_equal(got, start)
        assert got.dtype == start.dtype
    assert [len(c) for c in calls] == [N_BLOBS] * 3  # one encode call over all k summaries per step


def test_embed_summaries_passes_empty_text_through_and_refuses_none(noisy):
    _, X, embed = noisy
    got = embed_summaries(LookupEmbedder(embed), ["doc0", ""])
    np.testing.assert_array_equal(got[0], X[0])
    np.testing.assert_array_equal(got[1], np.zeros(X.shape[1]))
    with pytest.raises(ValueError, match="None"):
        embed_summaries(LookupEmbedder(embed), ["doc0", None])


def test_load_run_embedder_uses_the_experiment_resolver(noisy, tmp_path, monkeypatch):
    path, _, _ = _run_and_log(noisy, tmp_path)
    sentinel = object()
    monkeypatch.setattr("kslimeans.experiment.get_embedder", lambda name: (name, sentinel))
    ds = Dataset("synth", noisy[0], np.zeros(len(noisy[0]), dtype=int), N_BLOBS, "p", "t")
    assert load_run_embedder(load_run(path), ds) == ("fake", sentinel)
