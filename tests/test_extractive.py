"""Sanity checks for the extractive summarizers -- selection behavior, not exact scores."""

from __future__ import annotations

import numpy as np
import pytest

from kslimeans.summarizers.base import Cluster
from kslimeans.summarizers.extractive import (
    ExtractiveSummarizer,
    mmr_rerank,
    summarize_centroid,
    summarize_lsa,
    summarize_textrank,
)


def _toy_sentences() -> tuple[list[str], np.ndarray]:
    sentences = [f"sentence {i}" for i in range(6)]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 8)).astype(np.float32)
    return sentences, X


def test_mmr_rerank_returns_k_distinct_sentences():
    sentences, X = _toy_sentences()
    scores = np.array([0.9, 0.1, 0.5, 0.7, 0.3, 0.6])
    chosen = mmr_rerank(sentences, scores, X, k=4)
    assert len(chosen) == 4
    picked = [text for text, _ in chosen]
    assert len(set(picked)) == 4


def test_mmr_rerank_empty_input():
    assert mmr_rerank([], np.array([]), np.zeros((0, 4)), k=5) == []


def test_mmr_rerank_caps_k_at_n():
    sentences, X = _toy_sentences()
    chosen = mmr_rerank(sentences, np.arange(6.0), X, k=100)
    assert len(chosen) == len(sentences)


@pytest.mark.parametrize("scorer", [summarize_centroid, summarize_lsa, summarize_textrank])
def test_scorers_return_top_k_sentences_from_the_input(scorer):
    sentences, X = _toy_sentences()
    result = scorer(sentences, X, top_k=3)
    assert len(result) == 3
    for text, score in result:
        assert text in sentences
        assert isinstance(score, float)


@pytest.mark.parametrize("method", ["lsa", "centroid", "textrank", "lexrank"])
def test_extractive_summarizer_joins_chosen_sentences(method):
    sentences, X = _toy_sentences()
    summarizer = ExtractiveSummarizer(method=method, top_k=3)
    cluster = Cluster(
        index=0, doc_ids=(0, 1), texts=["doc a", "doc b"], sentences=sentences, sentence_embeddings=X
    )
    summary = summarizer.summarize(cluster)
    lines = summary.split("\n")
    assert len(lines) == 3
    assert all(line in sentences for line in lines)
    assert len(set(lines)) == 3


def test_extractive_summarizer_requires_sentences():
    summarizer = ExtractiveSummarizer(method="lsa")
    cluster = Cluster(index=0, doc_ids=(0,), texts=["doc a"], sentences=None, sentence_embeddings=None)
    with pytest.raises(ValueError):
        summarizer.summarize(cluster)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        ExtractiveSummarizer(method="not-a-real-method")
