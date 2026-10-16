"""`scripts/prepare_arxiv.py` on tiny synthetic raw shards (no network, no GPU)."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "prepare_arxiv.py"


@pytest.fixture(scope="module")
def pa_mod():
    spec = importlib.util.spec_from_file_location("prepare_arxiv_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


# (id, categories, abstract, comments)
SHARD_0 = [
    ("a1", "cs.LG stat.ML", words(60, "a"), None),
    ("a2", "hep-th", "  " + words(55, "b").replace(" ", "\n ") + " ", "10 pages"),
    ("a3", "math.AG", words(60, "c") + " This paper has been withdrawn by the author.", None),
    ("a4", "cs.CL", words(60, "d"), "Withdrawn"),
    ("a5", "cs.CL", words(49, "e"), None),
    ("a6", "cs.CL", words(50, "f"), None),
    ("a7", "cs.CV", words(60, "a").upper(), None),  # duplicate of a1 (case)
    ("a8", "cs.CV", words(60, "g") + " The withdrawn drug was studied.", None),
]
SHARD_1 = [
    ("b1", "cs.LG", words(55, "b"), None),  # duplicate of a2 across shards (whitespace)
    ("b2", "math.AG math.CO", words(60, "h"), "This paper has been withdrawn"),
    ("b3", "astro-ph", words(60, "i") + " The paper was withdrawn by the author", None),
    ("b4", "cs.LG", words(60, "j"), None),
    ("b5", "cs.LG", words(60, "j") + " ", None),  # duplicate of b4
    ("b6", "math.AG", words(70, "k"), None),
]
EXPECTED_IDS = ["a1", "a2", "a6", "a8", "b4", "b6"]


def write_shard(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids, cats, abstracts, comments = zip(*rows)
    date = datetime(2024, 1, 2)
    table = pa.table(
        {
            "id": list(ids),
            "title": ["t"] * len(rows),
            "categories": list(cats),
            "abstract": list(abstracts),
            "comments": list(comments),
            "update_date": pa.array([date] * len(rows), pa.timestamp("ms")),
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def data_dir(tmp_path, pa_mod):
    raw = tmp_path / "raw" / "data"
    write_shard(raw / "train-00000-of-00011.parquet", SHARD_0)
    write_shard(raw / "train-00001-of-00011.parquet", SHARD_1)
    return tmp_path


def backdate(path: Path) -> int:
    """Set an old mtime so a rewrite is detectable despite coarse filesystem timestamps."""
    os.utime(path, ns=(10**18, 10**18))
    return path.stat().st_mtime_ns


def manifest(data_dir: Path) -> dict:
    return json.loads((data_dir / "manifest.json").read_text())


def fake_encoder(calls: list | None = None, fail_on_call: int | None = None):
    def encode(texts: list[str]) -> np.ndarray:
        if calls is not None:
            calls.append(len(texts))
            if fail_on_call == len(calls):
                raise RuntimeError("interrupted")
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, len(t) % 384] = 1.0
            out[i, sum(t.encode()) % 384] += 0.5
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    return encode


def test_filter_rules_labels_and_manifest(pa_mod, data_dir):
    pa_mod.filter_shards(data_dir)
    t = pq.read_table(data_dir / "arxiv.parquet").to_pydict()

    assert t["id"] == EXPECTED_IDS
    assert t["label"] == ["cs.LG", "hep-th", "cs.CL", "cs.CV", "cs.LG", "math.AG"]
    assert t["archive"] == ["cs", "hep-th", "cs", "cs", "cs", "math"]
    assert t["update_date"] == ["2024-01-02"] * 6
    assert t["text"][1] == words(55, "b")  # whitespace normalized
    assert "withdrawn drug" in t["text"][3]

    m = manifest(data_dir)
    f = m["filter"]
    assert f["rows_in"] == 14
    assert f["dropped_withdrawn"] == 4  # a3, a4, b2, b3
    assert f["dropped_short"] == 1  # a5
    assert f["dropped_duplicate"] == 3  # a7, b1, b5
    assert f["rows_out"] == 6
    assert f["rows_in"] == (
        f["dropped_withdrawn"] + f["dropped_short"] + f["dropped_duplicate"] + f["rows_out"]
    )
    assert m["labels"] == {"cs.CL": 1, "cs.CV": 1, "cs.LG": 2, "hep-th": 1, "math.AG": 1}
    assert m["n_labels"] == 5
    assert m["n_archives"] == 3
    assert m["source"]["revision"] == pa_mod.REVISION
    assert not list(data_dir.glob("*.tmp"))


@pytest.mark.parametrize(
    "abstract,comments,expected",
    [
        ("This paper has been withdrawn.", "", True),
        ("It was withdrawn by the author", "", True),
        ("The withdrawn drug", "", False),
        ("fine", "Withdrawn", True),
        ("fine", "  (This article is withdrawn)", True),
        ("fine", "withdrawn by author", True),
        ("fine", "10 pages; not withdrawn", False),
        ("fine", "", False),
    ],
)
def test_is_withdrawn(pa_mod, abstract, comments, expected):
    assert pa_mod.is_withdrawn(abstract, comments) is expected


def test_filter_skipped_when_complete(pa_mod, data_dir):
    pa_mod.run(data_dir, ("filter",))
    path = data_dir / "arxiv.parquet"
    before = backdate(path)
    pa_mod.run(data_dir, ("filter",))
    assert path.stat().st_mtime_ns == before
    pa_mod.run(data_dir, ("filter",), force="filter")
    assert path.stat().st_mtime_ns != before
    assert pq.read_table(path).column("id").to_pylist() == EXPECTED_IDS


def test_embed_resume_equals_single_pass(pa_mod, data_dir, tmp_path_factory):
    pa_mod.filter_shards(data_dir)
    emb_path = data_dir / "embeddings_minilm_fp16.npy"

    calls: list[int] = []
    with pytest.raises(RuntimeError, match="interrupted"):
        pa_mod.embed(data_dir, fake_encoder(calls, fail_on_call=2), chunk_rows=2)
    progress = json.loads((data_dir / "progress.json").read_text())
    assert progress["rows_done"] == 2 and progress["n_rows"] == 6
    assert "embedding" not in manifest(data_dir)

    resumed_calls: list[int] = []
    info = pa_mod.embed(data_dir, fake_encoder(resumed_calls), chunk_rows=2)
    assert resumed_calls == [2, 2]  # rows 2..6 only
    assert info["complete"] is True
    assert manifest(data_dir)["embedding"]["complete"] is True
    resumed = np.load(emb_path)

    # Reference: an uninterrupted pass in a separate dir.
    ref_dir = tmp_path_factory.mktemp("ref")
    (ref_dir / "arxiv.parquet").write_bytes((data_dir / "arxiv.parquet").read_bytes())
    pa_mod.embed(ref_dir, fake_encoder(), chunk_rows=4)
    reference = np.load(ref_dir / "embeddings_minilm_fp16.npy")

    assert resumed.dtype == np.float16
    assert resumed.shape == (6, 384)
    np.testing.assert_array_equal(resumed, reference)
    assert np.abs(resumed).sum() > 0


def test_embed_max_rows_is_incomplete_and_stage_skips_when_done(pa_mod, data_dir):
    pa_mod.filter_shards(data_dir)
    pa_mod.embed(data_dir, fake_encoder(), chunk_rows=2, max_rows=3)
    assert json.loads((data_dir / "progress.json").read_text())["rows_done"] == 3
    assert manifest(data_dir)["embedding"]["complete"] is False

    calls: list[int] = []
    pa_mod.run(data_dir, ("embed",), chunk_rows=2, encode=fake_encoder(calls))
    assert sum(calls) == 3  # resumed from row 3, mid-chunk
    assert manifest(data_dir)["embedding"]["complete"] is True

    calls.clear()
    pa_mod.run(data_dir, ("embed",), chunk_rows=2, encode=fake_encoder(calls))
    assert calls == []


def test_subsample_deterministic_sorted_and_rows_match(pa_mod, data_dir):
    pa_mod.filter_shards(data_dir)
    pa_mod.subsample(data_dir, size=4, batch_size=2)
    idx = np.load(data_dir / "subsample_200k.npy")
    again = np.sort(np.random.default_rng(0).choice(6, 4, replace=False))

    assert idx.dtype == np.int64
    np.testing.assert_array_equal(idx, again)
    assert len(np.unique(idx)) == 4 and (np.diff(idx) > 0).all()

    full = pq.read_table(data_dir / "arxiv.parquet").to_pydict()
    sub = pq.read_table(data_dir / "arxiv_sub200k.parquet").to_pydict()
    assert sub["row"] == idx.tolist()
    for col in ("id", "text", "label", "archive", "update_date"):
        assert sub[col] == [full[col][i] for i in idx]
    assert manifest(data_dir)["subsample"] == {"size": 4, "seed": 0, "file": "subsample_200k.npy"}


def test_subsample_size_capped_and_skipped(pa_mod, data_dir):
    pa_mod.run(data_dir, ("filter", "subsample"))  # default 200k > N=6
    assert len(np.load(data_dir / "subsample_200k.npy")) == 6
    path = data_dir / "arxiv_sub200k.parquet"
    before = backdate(path)
    pa_mod.run(data_dir, ("filter", "subsample"))
    assert path.stat().st_mtime_ns == before


def test_force_filter_clears_downstream(pa_mod, data_dir):
    pa_mod.run(data_dir, ("filter", "embed", "subsample"), encode=fake_encoder())
    assert (data_dir / "embeddings_minilm_fp16.npy").exists()
    pa_mod._clear(data_dir, "filter")
    assert not (data_dir / "embeddings_minilm_fp16.npy").exists()
    assert not (data_dir / "arxiv_sub200k.parquet").exists()
    assert "embedding" not in manifest(data_dir)


def test_filter_rewrite_without_force_invalidates_downstream(pa_mod, data_dir):
    pa_mod.run(data_dir, ("filter", "embed", "subsample"), encode=fake_encoder())
    (data_dir / "arxiv.parquet").unlink()  # e.g. deleted by hand; filter reruns, no --force

    calls: list = []
    pa_mod.run(data_dir, ("filter", "embed", "subsample"), encode=fake_encoder(calls))

    assert sum(calls) == len(EXPECTED_IDS)  # embed recomputed every row
    assert manifest(data_dir)["embedding"]["complete"]
    assert (data_dir / "arxiv_sub200k.parquet").exists()


def test_filter_rewrite_clears_stale_outputs(pa_mod, data_dir):
    pa_mod.run(data_dir, ("filter", "embed", "subsample"), encode=fake_encoder())
    pa_mod.filter_shards(data_dir)
    for name in ("embeddings_minilm_fp16.npy", "progress.json", "subsample_200k.npy", "arxiv_sub200k.parquet"):
        assert not (data_dir / name).exists()
    assert "embedding" not in manifest(data_dir) and "subsample" not in manifest(data_dir)


def test_force_stage_must_be_in_stages(pa_mod, data_dir):
    pa_mod.run(data_dir, ("filter", "embed"), encode=fake_encoder())
    emb = data_dir / "embeddings_minilm_fp16.npy"
    with pytest.raises(ValueError, match="--force embed"):
        pa_mod.run(data_dir, ("filter",), force="embed")
    assert emb.exists() and manifest(data_dir)["embedding"]["complete"]


def test_subsample_done_checks_seed_and_file(pa_mod, data_dir):
    pa_mod.run(data_dir, ("filter", "subsample"))
    assert pa_mod.subsample_done(data_dir, 6)
    info = manifest(data_dir)["subsample"]
    for key, value in (("seed", 1), ("file", "other.npy")):
        (data_dir / "manifest.json").write_text(json.dumps({**manifest(data_dir), "subsample": {**info, key: value}}))
        assert not pa_mod.subsample_done(data_dir, 6)
        (data_dir / "manifest.json").write_text(json.dumps({**manifest(data_dir), "subsample": info}))
    assert not pa_mod.subsample_done(data_dir, 6, seed=1)


def test_subsample_crash_leaves_stage_not_done(pa_mod, data_dir, monkeypatch):
    pa_mod.run(data_dir, ("filter", "subsample"))
    assert pa_mod.subsample_done(data_dir, 6)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pa_mod.np, "save", boom)
    with pytest.raises(OSError):
        pa_mod.subsample(data_dir, size=4)
    monkeypatch.undo()

    # neither the old nor the new size may look done next to half-written outputs
    assert not pa_mod.subsample_done(data_dir, 6)
    assert not pa_mod.subsample_done(data_dir, 4)
    pa_mod.run(data_dir, ("filter", "subsample"), subsample_size=4)
    assert pa_mod.subsample_done(data_dir, 4)
    assert len(np.load(data_dir / "subsample_200k.npy")) == 4


def test_force_download_rejected(pa_mod, data_dir):
    with pytest.raises(ValueError, match="--force must be one of"):
        pa_mod.run(data_dir, ("download",), force="download")
