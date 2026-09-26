"""`cached_embeddings`/`cached_sentences`: cache hit/miss behavior, keyed by a content
fingerprint over `texts` (not just a document count), so editing a data file in place
without changing its row count doesn't silently reuse stale embeddings.

Uses a fake, deterministic, call-counting embedder (in the style of
`tests/fixtures/synth.py`'s fakes) so these stay fast and never touch a real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kslimeans.cache import cached_embeddings, cached_sentences
from kslimeans.embeddings import Embedder
from kslimeans.sentences import SentenceCorpus

DIM = 4


class _CountingEmbedder(Embedder):
    """Deterministic function of each text's content (not just its position), so
    editing a text's content -- even keeping the same document count -- changes what
    it encodes to; counts how many times `encode` is actually called."""

    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        rows = [np.full(DIM, float(sum(map(ord, t))), dtype=np.float32) for t in texts]
        return np.stack(rows)


def _fake_corpus(texts: list[str]) -> SentenceCorpus:
    n = len(texts)
    return SentenceCorpus(
        sentences=[f"{t} sentence" for t in texts],
        embeddings=np.array([[float(sum(map(ord, t)))] for t in texts], dtype=np.float32),
        doc_ptr=np.arange(n + 1, dtype=np.int64),
    )


# ----------------------------------------------------------------------------
# cached_embeddings
# ----------------------------------------------------------------------------


def test_cached_embeddings_hit_skips_recompute(tmp_path):
    embedder = _CountingEmbedder()
    texts = ["alpha", "beta", "gamma"]

    first = cached_embeddings("ds", texts, embedder, cache_dir=str(tmp_path))
    second = cached_embeddings("ds", texts, embedder, cache_dir=str(tmp_path))

    assert embedder.calls == 1
    assert np.array_equal(first, second)


def test_cached_embeddings_recomputes_on_document_count_mismatch(tmp_path):
    embedder = _CountingEmbedder()
    cached_embeddings("ds", ["alpha", "beta"], embedder, cache_dir=str(tmp_path))
    cached_embeddings("ds", ["alpha", "beta", "gamma"], embedder, cache_dir=str(tmp_path))

    assert embedder.calls == 2


def test_cached_embeddings_recomputes_when_content_changes_but_count_stays_the_same(tmp_path):
    """The bug this fixes: editing a document in place (same row count) used to be
    silently invisible to the cache, since it only ever checked `len(texts)`."""
    embedder = _CountingEmbedder()
    result1 = cached_embeddings("ds", ["alpha", "beta"], embedder, cache_dir=str(tmp_path))
    result2 = cached_embeddings("ds", ["alpha", "EDITED"], embedder, cache_dir=str(tmp_path))

    assert embedder.calls == 2
    assert not np.array_equal(result1, result2)


def test_cached_embeddings_treats_a_pre_fingerprint_cache_file_as_stale(tmp_path):
    """A cache file written before this fix has no `fingerprint` key at all; it must be
    recomputed even though its document count still matches."""
    texts = ["alpha", "beta"]
    path = Path(tmp_path) / "ds__fake_embeddings.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = np.zeros((len(texts), DIM), dtype=np.float32)
    np.savez(path, embeddings=stale)  # old format: no "fingerprint" key

    embedder = _CountingEmbedder()
    result = cached_embeddings("ds", texts, embedder, cache_dir=str(tmp_path))

    assert embedder.calls == 1
    assert not np.array_equal(result, stale)


def test_cached_embeddings_writes_a_fingerprint_that_a_later_call_reads_back(tmp_path):
    embedder = _CountingEmbedder()
    cached_embeddings("ds", ["alpha", "beta"], embedder, cache_dir=str(tmp_path))

    data = np.load(Path(tmp_path) / "ds__fake_embeddings.npz")
    assert "fingerprint" in data.files


# ----------------------------------------------------------------------------
# cached_sentences (SentenceCorpus.build monkeypatched, so no real spaCy needed)
# ----------------------------------------------------------------------------


@pytest.fixture
def fake_build(monkeypatch):
    calls: list[list[str]] = []

    def build(docs, embedder):
        calls.append(list(docs))
        return _fake_corpus(docs)

    monkeypatch.setattr(SentenceCorpus, "build", staticmethod(build))
    return calls


def test_cached_sentences_hit_skips_rebuild(tmp_path, fake_build):
    embedder = _CountingEmbedder()
    texts = ["doc one here", "doc two here"]

    first = cached_sentences("ds", texts, embedder, cache_dir=str(tmp_path))
    second = cached_sentences("ds", texts, embedder, cache_dir=str(tmp_path))

    assert len(fake_build) == 1
    assert second.sentences == first.sentences
    assert np.array_equal(second.embeddings, first.embeddings)


def test_cached_sentences_recomputes_on_document_count_mismatch(tmp_path, fake_build):
    embedder = _CountingEmbedder()
    cached_sentences("ds", ["doc one here"], embedder, cache_dir=str(tmp_path))
    cached_sentences("ds", ["doc one here", "doc two here"], embedder, cache_dir=str(tmp_path))

    assert len(fake_build) == 2


def test_cached_sentences_recomputes_when_content_changes_but_count_stays_the_same(tmp_path, fake_build):
    embedder = _CountingEmbedder()
    cached_sentences("ds", ["doc one here", "doc two here"], embedder, cache_dir=str(tmp_path))
    cached_sentences("ds", ["doc one here", "doc TWO edited"], embedder, cache_dir=str(tmp_path))

    assert len(fake_build) == 2
    assert fake_build[1] == ["doc one here", "doc TWO edited"]


def test_cached_sentences_treats_a_pre_fingerprint_cache_file_as_stale(tmp_path, fake_build):
    texts = ["doc one here", "doc two here"]
    base = str(Path(tmp_path) / "ds__fake_sentences")
    _fake_corpus(texts).save(base)  # old format: save() with no fingerprint

    cached_sentences("ds", texts, embedder=_CountingEmbedder(), cache_dir=str(tmp_path))

    assert len(fake_build) == 1  # rebuilt despite a matching doc_ptr count


def test_cached_sentences_writes_a_fingerprint_that_a_later_call_reads_back(tmp_path, fake_build):
    texts = ["doc one here", "doc two here"]
    cached_sentences("ds", texts, embedder=_CountingEmbedder(), cache_dir=str(tmp_path))

    base = str(Path(tmp_path) / "ds__fake_sentences")
    assert SentenceCorpus.load_fingerprint(base) is not None
