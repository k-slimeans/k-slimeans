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
    MissingAssignmentsError,
    load_run,
    load_run_dataset,
    members_by_cluster,
    step_inputs,
)
from kslimeans.schedules import FixedSchedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core_equivalence import ClosestToMeanSummarizer, LookupEmbedder, synthetic  # noqa: E402,F401
from synth import N_BLOBS  # noqa: E402

CONFIG = {"name": "synth_run", "dataset": "synth", "embedder": "fake", "seeds": [3], "cache_dir": "cache"}


class RecordingSummarizer(ClosestToMeanSummarizer):
    """Remembers the clusters of every `summarize_batch` call."""

    def __init__(self, embed):
        super().__init__(embed)
        self.batches = []

    def summarize_batch(self, clusters):
        self.batches.append(list(clusters))
        return super().summarize_batch(clusters)


def _run_and_log(synthetic, tmp_path, *, tol=-1.0, force_context_length=10, seed=3):
    texts, X, embed = synthetic
    summarizer = RecordingSummarizer(embed)
    result = summary_kmeans(
        texts,
        N_BLOBS,
        summarizer,
        LookupEmbedder(embed),
        FixedSchedule(3, 12, True, True),
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
    return run.path, result, summarizer


def test_step_inputs_equal_what_the_summarizer_received(synthetic, tmp_path):
    texts, X, _ = synthetic
    path, result, summarizer = _run_and_log(synthetic, tmp_path)
    record = load_run(path).record("k-test", 3)

    assert record.n_summary_steps == len(result.summaries_evolution) == 3
    batches = iter(summarizer.batches)
    seen = set()  # the loop memoizes by doc_ids, so a repeated cluster is not sent again
    n_checked = 0
    for j in range(record.n_summary_steps):
        source = result.summary_source_steps[j]
        np.testing.assert_array_equal(record.source_partition(j), result.assignments_evolution[source])
        np.testing.assert_array_equal(record.partition_after(j), result.assignments_evolution[source + 1])

        inputs = step_inputs(record, j, texts, X, force_context_length=10)
        assert [c.index for c in inputs] == list(range(N_BLOBS))
        assert all(len(c.doc_ids) <= 10 for c in inputs)
        sent = [c for c in inputs if c.doc_ids and c.doc_ids not in seen]
        if sent:
            received = next(batches)
            assert [(c.index, c.doc_ids, c.texts) for c in sent] == [(c.index, c.doc_ids, c.texts) for c in received]
            n_checked += len(sent)
        seen.update(c.doc_ids for c in inputs)
    assert next(batches, None) is None
    assert n_checked >= N_BLOBS


def test_step_inputs_without_trimming(synthetic, tmp_path):
    texts, X, _ = synthetic
    path, result, summarizer = _run_and_log(synthetic, tmp_path, force_context_length=0)
    record = load_run(path).record("k-test", 3)
    inputs = step_inputs(record, 0, texts, X, force_context_length=0)
    assert [(c.index, c.doc_ids) for c in inputs] == [
        (c.index, c.doc_ids) for c in summarizer.batches[0]
    ]


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


def test_load_run_dataset_uses_cache_and_never_encodes(synthetic, tmp_path, monkeypatch):
    texts, X, embed = synthetic
    ds = Dataset("synth", texts, np.zeros(len(texts), dtype=int), N_BLOBS, "p", "t")
    monkeypatch.setattr("kslimeans.experiment.load_dataset", lambda name, **kw: ds)
    cache_dir = tmp_path / "cache"
    cached_embeddings("synth", texts, LookupEmbedder(embed), str(cache_dir))
    path, _, _ = _run_and_log(synthetic, tmp_path)

    # An unknown embedder name would try to load a real model, so a cache miss fails loudly here.
    monkeypatch.setattr("kslimeans.cache.get_embedder", lambda name: pytest.fail("encode attempted"))
    got_ds, got_X = load_run_dataset(load_run(path), cache_dir)
    assert got_ds is ds
    np.testing.assert_array_equal(got_X, X)
