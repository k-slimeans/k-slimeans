"""Cluster member selection must use GLOBAL document indices throughout.

This is a regression test for an upstream bug: in the `force_context_length
> 0` NLP path, upstream stored the cluster-LOCAL k-means++ selection
indices and then used them directly to index the GLOBAL per-document
sentence store, silently pulling sentences from the wrong documents
whenever a cluster was actually trimmed. `core._select_cluster` converts
the local k-means++ positions back to global document indices before
anything downstream sees them.
"""

from __future__ import annotations

import numpy as np

from kslimeans.core import _select_cluster
from kslimeans.sentences import SentenceCorpus


def _make_corpus(n_docs: int) -> SentenceCorpus:
    sentences = []
    doc_ptr = [0]
    for i in range(n_docs):
        sentences.extend([f"doc{i}-s0", f"doc{i}-s1"])
        doc_ptr.append(len(sentences))
    embeddings = np.arange(len(sentences) * 4, dtype=np.float32).reshape(len(sentences), 4)
    return SentenceCorpus(sentences=sentences, embeddings=embeddings, doc_ptr=np.array(doc_ptr, dtype=np.int64))


def test_gathered_sentences_belong_to_the_sampled_global_doc_ids():
    n_docs = 12
    texts = [f"doc{i}" for i in range(n_docs)]
    corpus = _make_corpus(n_docs)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_docs, 2)).astype(np.float32)
    labels = np.zeros(n_docs, dtype=int)  # every doc in cluster 0

    cluster = _select_cluster(0, labels, X, texts, force_context_length=5, random_state=0, sentences=corpus)

    assert len(cluster.doc_ids) == 5
    expected = [s for d in cluster.doc_ids for s in (f"doc{d}-s0", f"doc{d}-s1")]
    assert cluster.sentences == expected


def test_gathered_sentences_stay_within_the_selected_clusters_docs():
    n_docs = 12
    texts = [f"doc{i}" for i in range(n_docs)]
    corpus = _make_corpus(n_docs)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_docs, 2)).astype(np.float32)
    # cluster 1 == docs 5..11; cluster 0 == docs 0..4
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    cluster = _select_cluster(1, labels, X, texts, force_context_length=3, random_state=0, sentences=corpus)

    assert len(cluster.doc_ids) == 3
    assert all(5 <= d <= 11 for d in cluster.doc_ids), cluster.doc_ids
    expected = [s for d in cluster.doc_ids for s in (f"doc{d}-s0", f"doc{d}-s1")]
    assert cluster.sentences == expected


def test_small_cluster_keeps_every_member_when_not_over_the_limit():
    n_docs = 6
    texts = [f"doc{i}" for i in range(n_docs)]
    corpus = _make_corpus(n_docs)
    X = np.zeros((n_docs, 2), dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1])

    cluster = _select_cluster(0, labels, X, texts, force_context_length=10, random_state=0, sentences=corpus)

    assert cluster.doc_ids == (0, 2, 4)
    assert cluster.sentences == ["doc0-s0", "doc0-s1", "doc2-s0", "doc2-s1", "doc4-s0", "doc4-s1"]


def test_empty_cluster_has_no_members_and_no_sentences():
    n_docs = 4
    texts = [f"doc{i}" for i in range(n_docs)]
    corpus = _make_corpus(n_docs)
    X = np.zeros((n_docs, 2), dtype=np.float32)
    labels = np.zeros(n_docs, dtype=int)

    cluster = _select_cluster(1, labels, X, texts, force_context_length=0, random_state=0, sentences=corpus)

    assert cluster.doc_ids == ()
    assert cluster.texts == []
    assert cluster.sentences is None
