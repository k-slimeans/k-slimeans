"""`scripts/run_streaming.py`'s config typo-guard and `method_name` support.

`run_streaming.run_streaming(cfg)` mirrors `kslimeans.experiment.run_offline(cfg)`:
config validation (`check_keys`) is the very first thing it does, so these tests can
exercise it with a deliberately incomplete config and never touch a real dataset,
embedder, or summarizer. The script lives outside the installed `kslimeans` package, so
it's loaded from its file path via `importlib` (same approach as
`tests/fixtures/make_fixtures.py`'s `_load_module`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from kslimeans.aggregate import aggregate_runs
from kslimeans.data import Dataset
from kslimeans.embeddings import Embedder
from kslimeans.summarizers.base import Summarizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))
from synth import N_BLOBS, closest_to_mean, make_lookup_embedder, make_synthetic_minibatch_data  # noqa: E402
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_streaming.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_streaming_module():
    return _load_module("run_streaming_under_test", SCRIPT_PATH)


def _minimal_config(**overrides) -> dict:
    cfg = {
        "name": "stream_run",
        "dataset": {"name": "stackexchange", "limit": 100},
        "years": [2023],
        "embedder": "distilbert",
        "seeds": [0],
        "max_batch_size": 1000,
        "eval_assign": "lloyd1",
        "baselines": {"kmeans": {}},
        "force_context_length": 0,
        "tol": 1.0e-4,
        "schedule": {"type": "fixed", "n_summaries": 5, "max_iter": 120},
        "summarizer": {"type": "lsa", "top_k": 5},
        "cache_dir": "cache",
        "results_dir": "results",
    }
    cfg.update(overrides)
    return cfg


def test_unknown_key_raises_before_touching_any_dataset(run_streaming_module):
    cfg = _minimal_config(bogus_key="oops")
    with pytest.raises(ValueError, match="bogus_key"):
        run_streaming_module.run_streaming(cfg)


def test_missing_required_key_raises(run_streaming_module):
    cfg = _minimal_config()
    del cfg["max_batch_size"]
    with pytest.raises(ValueError, match="max_batch_size"):
        run_streaming_module.run_streaming(cfg)


def test_method_name_is_allowed_but_not_required(run_streaming_module):
    from kslimeans.config import check_keys

    assert "method_name" in run_streaming_module._ALLOWED_KEYS
    assert "method_name" not in run_streaming_module._REQUIRED_KEYS

    allowed, required = run_streaming_module._ALLOWED_KEYS, run_streaming_module._REQUIRED_KEYS
    cfg = _minimal_config()
    assert "method_name" not in cfg
    check_keys(cfg, allowed, required)  # must not raise: method_name is optional

    check_keys(_minimal_config(method_name="k-NLPmeans"), allowed, required)  # explicit value is fine too


def test_empty_years_list_is_not_treated_as_all_years(run_streaming_module, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("load_dataset must not be called for years: []")

    monkeypatch.setattr(run_streaming_module.kslimeans, "load_dataset", fail)
    cfg = _minimal_config(years=[], results_dir=str(tmp_path / "results"), cache_dir=str(tmp_path / "cache"))
    try:
        run_streaming_module.run_streaming(cfg)
    except AssertionError:
        raise
    except Exception:
        pass  # an empty run may fail while summarising its (empty) metrics; it must not load data


# --- summary-distance metrics (issue #29) -------------------------------------------

class _LookupEmbedder(Embedder):
    name = "fake"

    def __init__(self, embed):
        self._embed = embed

    def encode(self, texts):
        return self._embed(list(texts))


class _ClosestToMeanSummarizer(Summarizer):
    needs_sentences = False

    def __init__(self, embed):
        self._embed = embed

    def summarize(self, cluster):
        return closest_to_mean(list(cluster.texts), self._embed)


@pytest.fixture
def fake_stream(run_streaming_module, monkeypatch, tmp_path):
    """Patches the script's data/embedder/summarizer lookups with tiny synthetic fakes."""
    texts, X = make_synthetic_minibatch_data()
    embed = make_lookup_embedder(texts, X)
    base, extra = divmod(len(texts), N_BLOBS)
    labels = np.concatenate([np.full(base + (b < extra), b) for b in range(N_BLOBS)])
    ds = Dataset(name="synth", texts=texts, labels=labels, n_clusters=N_BLOBS, prompt="", text_type="doc")

    module = run_streaming_module
    monkeypatch.setattr(module.kslimeans, "load_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(module, "dataset_embeddings", lambda *a, **k: X)
    monkeypatch.setattr(module, "build_summarizer", lambda *a, **k: _ClosestToMeanSummarizer(embed))
    monkeypatch.setattr(module, "get_embedder", lambda name: _LookupEmbedder(embed))
    return _minimal_config(
        years=[2023],
        max_batch_size=120,  # 320 docs -> 3 batches
        baselines={"kmeans": {"n_init": 1}},
        schedule={"type": "fixed", "n_summaries": 2, "max_iter": 20},
        results_dir=str(tmp_path / "results"),
        cache_dir=str(tmp_path / "cache"),
    )


def _rows(run_path: Path) -> dict:
    lines = (run_path / "metrics.jsonl").read_text().splitlines()
    return {row["method"]: row for row in map(json.loads, lines)}


def test_streaming_summary_distances_are_not_degenerate(run_streaming_module, fake_stream):
    rows = _rows(run_streaming_module.run_streaming(fake_stream))
    method = rows[next(m for m in rows if m != "kmeans")]
    # Summaries are real documents, so they sit away from the oracle centroids.
    assert method["sum_cen_dist"] > 0
    assert method["sum_sum_dist"] == method["sum_cen_dist"]
    assert method["sum_cen_dist"] != method["cen_cen_dist"]
    # Reference is the oracle, not the method's own centroids (upstream gave 0 here).
    assert method["cen_sum_dist"] > 0
    assert method["cen_sum_dist"] == pytest.approx(method["cen_cen_dist"])


def test_streaming_baselines_have_missing_summary_distances(run_streaming_module, fake_stream):
    baseline = _rows(run_streaming_module.run_streaming(fake_stream))["kmeans"]
    assert baseline["sum_cen_dist"] is None
    assert baseline["sum_sum_dist"] is None
    assert baseline["cen_cen_dist"] > 0
    assert baseline["cen_sum_dist"] == pytest.approx(baseline["cen_cen_dist"])


def test_aggregator_handles_streaming_run_with_missing_distances(run_streaming_module, fake_stream):
    run_path = run_streaming_module.run_streaming(fake_stream)
    table = aggregate_runs([run_path])
    assert "k-means" in set(table["method"])
    assert table[["acc_mean", "nmi_mean", "ari_mean"]].notna().all().all()
    summary = (run_path / "summary.csv").read_text()
    assert "dist" not in summary and "nan" not in summary.lower()


def test_final_batch_summary_embeddings_drops_none_and_handles_empty(run_streaming_module):
    embedder = _LookupEmbedder(lambda texts: np.array([[float(len(t))] for t in texts], dtype=np.float32))
    embed = run_streaming_module._final_batch_summary_embeddings
    assert embed(["a", None, "abc"], embedder).tolist() == [[1.0], [3.0]]
    assert embed([None, None], embedder) is None
    assert embed(None, embedder) is None


def test_streaming_metrics_without_summaries_are_none_not_centroid_fallback(run_streaming_module):
    labels = np.array([0, 0, 1, 1])
    oracle = np.array([[0.0, 0.0], [4.0, 0.0]])
    centroids = np.array([[1.0, 0.0], [4.0, 0.0]])
    metrics = run_streaming_module._streaming_metrics(labels, labels, oracle, centroids, None)
    assert metrics["sum_cen_dist"] is None and metrics["sum_sum_dist"] is None
    assert metrics["cen_cen_dist"] == pytest.approx(0.5)
    with_summaries = run_streaming_module._streaming_metrics(labels, labels, oracle, centroids, oracle + 1)
    assert with_summaries["sum_cen_dist"] == pytest.approx(np.sqrt(2))
def test_gpu_check_is_an_optional_key(run_streaming_module):
    assert "gpu_check" in run_streaming_module._ALLOWED_KEYS
    assert "gpu_check" not in run_streaming_module._REQUIRED_KEYS


def test_bad_gpu_check_block_raises_before_touching_any_dataset(run_streaming_module):
    cfg = _minimal_config(gpu_check={"require_idle": "sometimes"})
    with pytest.raises(ValueError, match="require_idle"):
        run_streaming_module.run_streaming(cfg)
