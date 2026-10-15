"""Fast: config validation + a synthetic end-to-end run_offline. Slow: real Bank77 numbers.

The synthetic run monkeypatches `load_dataset`, `get_embedder`, and
`build_summarizer` inside `kslimeans.experiment` with cheap fakes -- a
lookup-table embedder and a summarizer that skips sentence splitting
entirely (`needs_sentences = False`) -- so the test never touches
sentence-transformers, spaCy, or disk-cached models.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from kslimeans import experiment
from kslimeans.config import load_config
from kslimeans.data import Dataset
from kslimeans.summarizers.base import Cluster, Summarizer

N_DOCS = 24
N_CLUSTERS = 4
DIM = 6


class _FakeEmbedder:
    """Looks a text up in a fixed table; anything else gets a deterministic random vector."""

    name = "fake"

    def __init__(self, lookup: dict[str, np.ndarray]) -> None:
        self._lookup = lookup

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        rows = []
        for t in texts:
            vec = self._lookup.get(t)
            if vec is None:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                vec = rng.normal(size=DIM).astype(np.float32)
            rows.append(vec)
        return np.stack(rows).astype(np.float32)


class _FakeSummarizer(Summarizer):
    """Deterministic, sentence-free stand-in for a real (extractive or SLM) summarizer."""

    needs_sentences = False

    def summarize(self, cluster: Cluster) -> str:
        return f"summary-of-{len(cluster.doc_ids)}-docs-{cluster.index}"


def _make_synthetic_dataset() -> tuple[Dataset, np.ndarray]:
    """N_CLUSTERS well-separated Gaussian blobs, matching tests/fixtures/synth.py's approach."""
    rng = np.random.default_rng(0)
    per = N_DOCS // N_CLUSTERS
    centers = rng.normal(scale=5.0, size=(N_CLUSTERS, DIM))
    X = np.empty((N_DOCS, DIM), dtype=np.float32)
    labels = np.empty(N_DOCS, dtype=np.int64)
    for c in range(N_CLUSTERS):
        blob = centers[c] + rng.normal(scale=0.5, size=(per, DIM))
        X[c * per : (c + 1) * per] = blob.astype(np.float32)
        labels[c * per : (c + 1) * per] = c
    texts = [f"doc{i}" for i in range(N_DOCS)]
    return Dataset("synth", texts, labels, N_CLUSTERS, prompt="", text_type=""), X


def _base_config(tmp_path, **overrides) -> dict:
    cfg = {
        "name": "synth_run",
        "dataset": "synth",
        "embedder": "fake",
        "seeds": [0, 1],
        "baselines": {"kmeans": {"n_init": 2}},
        "method_name": "k-Fakemeans",
        "force_context_length": 0,
        "tol": 1.0e-4,
        "schedule": {
            "type": "fixed",
            "n_summaries": 2,
            "max_iter": 10,
            "initial_iter": True,
            "final_iter": True,
        },
        "summarizer": {"type": "lsa", "top_k": 3},
        "cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
    }
    cfg.update(overrides)
    return cfg


def test_unknown_key_raises(tmp_path):
    cfg = _base_config(tmp_path, bogus_key="oops")
    with pytest.raises(ValueError, match="bogus_key"):
        experiment.run_offline(cfg)


def test_missing_required_key_raises(tmp_path):
    cfg = _base_config(tmp_path)
    del cfg["dataset"]
    with pytest.raises(ValueError, match="dataset"):
        experiment.run_offline(cfg)


def test_run_offline_writes_expected_files(tmp_path, monkeypatch):
    ds, X = _make_synthetic_dataset()
    lookup = {t: X[i] for i, t in enumerate(ds.texts)}
    fake_embedder = _FakeEmbedder(lookup)

    monkeypatch.setattr(experiment, "load_dataset", lambda name, **kwargs: ds)
    monkeypatch.setattr(experiment, "get_embedder", lambda name: fake_embedder)
    monkeypatch.setattr(
        experiment, "build_summarizer", lambda cfg, prompt=None, text_type=None: _FakeSummarizer()
    )

    cfg = _base_config(tmp_path)
    run_dir = experiment.run_offline(cfg)

    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    summaries_path = run_dir / "summaries.jsonl"
    summary_csv = run_dir / "summary.csv"
    assert config_path.exists()
    assert metrics_path.exists()
    assert summaries_path.exists()
    assert summary_csv.exists()

    written_cfg = yaml.safe_load(config_path.read_text())
    assert written_cfg["name"] == "synth_run"

    metrics_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    # 2 seeds x (1 baseline + 1 method) = 4 rows.
    assert len(metrics_rows) == 4
    assert {row["method"] for row in metrics_rows} == {"kmeans", "k-Fakemeans"}
    for row in metrics_rows:
        assert 0.0 <= row["nmi"] <= 1.0
        assert 0.0 <= row["acc"] <= 1.0
        assert "runtime_s" in row and row["runtime_s"] >= 0.0

    method_rows = [row for row in metrics_rows if row["method"] == "k-Fakemeans"]
    assert len(method_rows) == 2
    for row in method_rows:
        assert isinstance(row["n_summary_steps"], int)
        assert isinstance(row["converged"], bool)

    trace_rows = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    assert all({"seed", "method", "step"} <= set(row) for row in trace_rows)
    assert {row["method"] for row in trace_rows} == {"k-Fakemeans"}  # baselines have no trace
    for row in method_rows:
        steps = [r["step"] for r in trace_rows if r["seed"] == row["seed"]]
        # One row per KMeans fit: the initial fit plus one per summary step, minus a tol early stop.
        assert steps == list(range(1 + row["n_summary_steps"] - int(row["converged"])))

    summaries_rows = [json.loads(line) for line in summaries_path.read_text().splitlines()]
    assert len(summaries_rows) == 2  # one per seed; baselines don't produce summaries
    for row in summaries_rows:
        assert row["method"] == "k-Fakemeans"
        assert len(row["summaries"]) == N_CLUSTERS
        assert len(row["assignments"]) == N_DOCS

    csv_text = summary_csv.read_text()
    assert "kmeans" in csv_text
    assert "k-Fakemeans" in csv_text


@pytest.mark.slow
def test_bank77_config_reproduces_reference_nmi(tmp_path):
    cfg = load_config("configs/bank77_knlpmeans.yaml", [f"results_dir={tmp_path}"])
    run_dir = experiment.run_offline(cfg)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    kmeans_nmi = next(r["nmi"] for r in rows if r["method"] == "kmeans")
    knlp_nmi = next(r["nmi"] for r in rows if r["method"] == "k-NLPmeans")

    assert round(kmeans_nmi, 4) == 0.6541
    assert round(knlp_nmi, 4) == 0.6518
