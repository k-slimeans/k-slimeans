"""arXiv loaders, precomputed-embedding plumbing (`dataset_embeddings`) and the yearless streaming path.

A tiny fake `arxiv/` directory is built per the contract written by `scripts/prepare_arxiv.py`;
no network, GPU or model is involved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kslimeans import cache
from kslimeans.cache import dataset_embeddings
from kslimeans.data import Dataset, load_dataset
from kslimeans.embeddings import Embedder
from kslimeans.summarizers.base import Summarizer

REPO_ROOT = Path(__file__).resolve().parent.parent
N_ROWS, DIM = 40, 8
LABELS = ["cs.LG", "hep-th", "math.AG", "stat.ML"]
SUB_ROWS = np.arange(1, N_ROWS, 3, dtype=np.int64)  # 13 rows


def _write_arxiv(root: Path, complete: bool = True, emb_rows: int = N_ROWS) -> Path:
    arxiv = root / "arxiv"
    arxiv.mkdir()
    df = pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(N_ROWS)],
            "text": [f"abstract number {i}" for i in range(N_ROWS)],
            # the first rows only use cs.LG / hep-th so `limit` changes the label set
            "label": [LABELS[min(i // 5, 1) if i < 10 else i % 4] for i in range(N_ROWS)],
            "archive": "x",
            "update_date": [f"{2020 + i % 3}-01-0{1 + i % 9}" for i in range(N_ROWS)],
        }
    )
    df.to_parquet(arxiv / "arxiv.parquet", row_group_size=7)
    sub = df.iloc[SUB_ROWS].copy()
    sub["row"] = SUB_ROWS
    sub.to_parquet(arxiv / "arxiv_sub200k.parquet", row_group_size=4)

    mm = np.lib.format.open_memmap(arxiv / "embeddings_minilm_fp16.npy", mode="w+", dtype=np.float16, shape=(emb_rows, DIM))
    mm[:] = np.random.default_rng(0).normal(size=(emb_rows, DIM)).astype(np.float16)
    mm.flush()
    (arxiv / "manifest.json").write_text(json.dumps({"embedding": {"complete": complete}}))
    return arxiv


@pytest.fixture
def arxiv_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("kslimeans.data.DATA_DIR", tmp_path)
    return _write_arxiv(tmp_path)


def test_full_loader(arxiv_dir):
    ds = load_dataset("arxiv")
    assert ds.texts[3] == "abstract number 3" and len(ds.texts) == N_ROWS
    assert ds.label_names == sorted(LABELS)
    assert ds.n_clusters == 4
    assert sorted(set(ds.labels.tolist())) == [0, 1, 2, 3]
    assert [ds.label_names[i] for i in ds.labels[:3]] == ["cs.LG"] * 3
    assert ds.embeddings_rows is None and ds.embeddings_embedder == "minilm"
    assert ds.embeddings_path == arxiv_dir / "embeddings_minilm_fp16.npy"
    assert ds.text_type == "Topic:" and "arXiv" in ds.prompt


def test_limit_makes_labels_contiguous(arxiv_dir):
    ds = load_dataset("arxiv", limit=10)
    assert ds.texts == [f"abstract number {i}" for i in range(10)]
    assert ds.label_names == ["cs.LG", "hep-th"]
    assert ds.n_clusters == 2 and set(ds.labels.tolist()) == {0, 1}
    assert ds.name == "arxiv_limit10"
    assert np.array_equal(ds.embeddings_rows, np.arange(10))


def test_year_filters_then_limits(arxiv_dir):
    ds = load_dataset("arxiv", year=2021, limit=5)
    assert ds.texts == [f"abstract number {i}" for i in (1, 4, 7, 10, 13)]
    assert np.array_equal(ds.embeddings_rows, [1, 4, 7, 10, 13])
    assert len(load_dataset("arxiv", year=2022).texts) == sum(1 for i in range(N_ROWS) if i % 3 == 2)


def test_subsample_loader_carries_rows(arxiv_dir):
    ds = load_dataset("arxiv_sub200k")
    assert np.array_equal(ds.embeddings_rows, SUB_ROWS)
    assert ds.texts == [f"abstract number {i}" for i in SUB_ROWS]
    lim = load_dataset("arxiv_sub200k", limit=4)
    assert np.array_equal(lim.embeddings_rows, SUB_ROWS[:4]) and len(lim.texts) == 4
    yr = load_dataset("arxiv_sub200k", year=2021)  # every subsample row has i % 3 == 1, i.e. year 2021
    assert np.array_equal(yr.embeddings_rows, SUB_ROWS)


@pytest.mark.parametrize("name", ["arxiv", "arxiv_sub200k"])
@pytest.mark.parametrize("kwargs", [{"year": 1999}, {"limit": 0}])
def test_empty_selection_raises(arxiv_dir, name, kwargs):
    with pytest.raises(ValueError, match="no rows"):
        load_dataset(name, **kwargs)


def test_subsample_year_and_limit_combine(arxiv_dir):
    ds = load_dataset("arxiv_sub200k", year=2021, limit=3)
    assert np.array_equal(ds.embeddings_rows, SUB_ROWS[:3])
    assert ds.texts == [f"abstract number {i}" for i in SUB_ROWS[:3]]
    assert ds.name == "arxiv_sub200k_2021_limit3"
    with pytest.raises(ValueError, match="no rows"):
        load_dataset("arxiv_sub200k", year=2020, limit=3)


def test_incomplete_manifest_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("kslimeans.data.DATA_DIR", tmp_path)
    _write_arxiv(tmp_path, complete=False)
    with pytest.raises(ValueError, match="prepare_arxiv"):
        load_dataset("arxiv")


def test_shape_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("kslimeans.data.DATA_DIR", tmp_path)
    _write_arxiv(tmp_path, emb_rows=N_ROWS - 1)
    with pytest.raises(ValueError, match="prepare_arxiv"):
        load_dataset("arxiv_sub200k")


def test_missing_files_raise(tmp_path, monkeypatch):
    monkeypatch.setattr("kslimeans.data.DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_arxiv"):
        load_dataset("arxiv")


def test_dataset_embeddings_selects_rows_as_float32(arxiv_dir):
    raw = np.load(arxiv_dir / "embeddings_minilm_fp16.npy")
    sub = dataset_embeddings(load_dataset("arxiv_sub200k"), "minilm")
    assert sub.dtype == np.float32
    np.testing.assert_array_equal(sub, raw[SUB_ROWS].astype(np.float32))
    full = dataset_embeddings(load_dataset("arxiv", limit=6), "minilm")
    np.testing.assert_array_equal(full, raw[:6].astype(np.float32))


def test_dataset_embeddings_wrong_embedder_raises(arxiv_dir):
    with pytest.raises(ValueError, match="minilm.*distilbert"):
        dataset_embeddings(load_dataset("arxiv_sub200k"), "distilbert")


def test_dataset_embeddings_delegates_to_cached_embeddings(monkeypatch):
    calls = []
    sentinel = np.ones((2, 3))
    monkeypatch.setattr(cache, "cached_embeddings", lambda *a, **k: calls.append((a, k)) or sentinel)
    ds = Dataset("toy", ["a", "b"], np.array([0, 1]), 2, "p", "t")
    out = dataset_embeddings(ds, "distilbert", "some_cache")
    assert out is sentinel
    assert calls == [(("toy", ["a", "b"], "distilbert", "some_cache"), {})]


class _StubEmbedder(Embedder):
    def __init__(self):
        self.name = "minilm"

    def encode(self, texts):
        out = np.random.default_rng(len(texts)).normal(size=(len(texts), DIM)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class _StubSummarizer(Summarizer):
    def summarize(self, cluster):
        return "stub summary"


def test_run_streaming_without_years(arxiv_dir, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_streaming_arxiv", REPO_ROOT / "scripts" / "run_streaming.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "build_summarizer", lambda *a, **k: _StubSummarizer())
    monkeypatch.setattr("kslimeans.minibatch.get_embedder", lambda name: _StubEmbedder())
    cfg = {
        "name": "arxiv_stream",
        "dataset": {"name": "arxiv_sub200k", "limit": 12},
        "embedder": "minilm",
        "seeds": [0],
        "max_batch_size": 6,
        "eval_assign": "nearest",
        "baselines": {"kmeans": {}},
        "force_context_length": 0,
        "tol": 1.0e-4,
        "schedule": {"type": "fixed", "n_summaries": 1, "max_iter": 3},
        "summarizer": {"type": "lsa", "top_k": 5},
        "cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
    }
    run_dir = module.run_streaming(cfg)
    lines = [json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]
    assert {row["year"] for row in lines} == {"all"}
    assert {row["method"] for row in lines} == {"kmeans", "k-NLPmeans"}
