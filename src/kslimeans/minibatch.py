"""Mini-batch / streaming variant of the summary-centroids clustering loop.

Ports upstream's `miniBatchSummarymeans`: split the corpus into
sequential batches, run `summary_kmeans` (see `core.py`) on each batch
using the *previous* batch's centroids as `init`, then fold the new
batch's centroids into a running per-cluster weighted average. This is
"k-NLPmeans"/"k-LLMmeans" made streaming, the same way `MiniBatchKMeans`
is streaming k-means: `sequentialMiniBatchKmeans` below is the plain
(non-summary) baseline for comparison, and `assign_to_centroids`
reproduces upstream's streaming evaluation step.

Batches are computed exactly like `np.array_split` would split a range
of `n` items into `num_batches` nearly-equal pieces (upstream's own
`divmod`-based formula, reproduced in `_batch_bounds`): batch sizes
differ by at most one, with the larger batches first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin

from .core import summary_kmeans
from .embeddings import Embedder, get_embedder
from .schedules import FixedSchedule, SummarySchedule
from .sentences import SentenceCorpus
from .summarizers import Summarizer, build_summarizer
from .summarizers.extractive import ExtractiveSummarizer


@dataclass
class MiniBatchResult:
    """Everything `mini_batch_summary_kmeans` produces."""

    summaries: list[list[str] | None]  # one summary_kmeans result's summaries, per batch
    centroids: np.ndarray  # final running centroids after the last batch
    batch_bounds: list[tuple[int, int]]  # [start, stop) document ranges, in corpus order


def _batch_bounds(n: int, num_batches: int) -> list[tuple[int, int]]:
    """`n` items split into `num_batches` batches, upstream's `divmod` formula."""
    k, m = divmod(n, num_batches)
    return [(i * k + min(i, m), (i + 1) * k + min(i + 1, m)) for i in range(num_batches)]


def _merge_centroids(
    centroids: np.ndarray, counts: np.ndarray, batch_centroids: np.ndarray, batch_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fold one batch's centroids into the running per-cluster weighted average.

    `centroids[i] = (centroids[i]*counts[i] + batch_centroids[i]*batch_counts[i]) /
    (counts[i] + batch_counts[i])`, upstream's exact expression, except a
    cluster that has had zero members in every batch so far (both counts
    zero) keeps its previous centroid unchanged instead of upstream's `0/0
    = nan`.
    """
    new_counts = counts + batch_counts
    merged = np.array(
        [
            centroids[i]
            if new_counts[i] == 0
            else (centroids[i] * counts[i] + batch_centroids[i] * batch_counts[i]) / new_counts[i]
            for i in range(len(centroids))
        ]
    )
    return merged, new_counts


def mini_batch_summary_kmeans(
    texts: list[str],
    n_clusters: int,
    summarizer: Summarizer,
    embedder: Embedder | str,
    schedule: SummarySchedule,
    *,
    text_features: np.ndarray,
    sentences: SentenceCorpus | None = None,
    max_batch_size: int = 5000,
    init="k-means++",
    force_context_length: int = 0,
    tol: float = 1e-4,
    random_state=None,
    verbose: bool = True,
) -> MiniBatchResult:
    """Run `summary_kmeans` over sequential batches, carrying centroids forward.

    `text_features` is required: upstream's `miniBatchSummarymeans` sliced
    it unconditionally and crashed with a confusing error (or silently
    wrong results) if it was omitted, so we raise a clear one instead.

    `sentences`, if omitted, is built once from the whole corpus (only if
    `summarizer.needs_sentences`) and then sliced per batch -- upstream's
    `miniBatchSummarymeans` instead called `.items()` on `text_sentences`
    unconditionally, which crashed outright when it was `None`.

    The running centroid update reproduces upstream's per-cluster
    weighted average by member count, `centroids[i] = (centroids[i] *
    counts[i] + batch_centroids[i] * batch_counts[i]) / (counts[i] +
    batch_counts[i])`, *except* upstream divided by zero whenever a
    cluster had received no members in any batch seen so far (both counts
    zero); here that cluster's centroid is simply carried forward
    unchanged instead of becoming `nan`.
    """
    if not isinstance(texts, list):
        raise TypeError(f"Expected a list for variable texts, but got {type(texts).__name__}")
    if text_features is None:
        raise ValueError(
            "mini_batch_summary_kmeans requires text_features (precomputed embeddings for "
            "the whole corpus) -- pass e.g. text_features=embedder.encode(texts)"
        )

    text_features = np.asarray(text_features)
    n = len(texts)

    if isinstance(embedder, str):
        embedder = get_embedder(embedder)

    if summarizer.needs_sentences and sentences is None:
        sentences = SentenceCorpus.build(texts, embedder)

    num_batches = int(np.ceil(n / max_batch_size))
    bounds = _batch_bounds(n, num_batches)

    centroids = init
    counts: np.ndarray | None = None
    summaries: list[list[str] | None] = []

    for start, stop in bounds:
        batch_texts = texts[start:stop]
        batch_X = text_features[start:stop]
        batch_sentences = sentences.subset(start, stop) if sentences is not None else None

        result = summary_kmeans(
            batch_texts,
            n_clusters,
            summarizer,
            embedder,
            schedule,
            text_features=batch_X,
            sentences=batch_sentences,
            init=centroids,
            force_context_length=force_context_length,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

        batch_centroids = result.centroids
        batch_counts = np.array(
            [np.sum(np.asarray(result.assignments) == i) for i in range(n_clusters)]
        )

        if counts is None:
            centroids = batch_centroids
            counts = batch_counts
        else:
            centroids, counts = _merge_centroids(centroids, counts, batch_centroids, batch_counts)

        summaries.append(result.summaries)

    return MiniBatchResult(summaries=summaries, centroids=centroids, batch_bounds=bounds)


def miniBatchKNLPmeans(
    text_data: list[str],
    num_clusters: int,
    max_batch_size: int = 5000,
    init="k-means++",
    force_context_length: int = 0,
    max_llm_iter: int = 5,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state=None,
    emb_type: str = "distilbert",
    text_features: np.ndarray | None = None,
    nlp: str = "lsa",
    top_k: int = 5,
    text_sentences: SentenceCorpus | None = None,
) -> MiniBatchResult:
    """Extractive-summary mini-batch k-means, matching upstream `miniBatchKNLPmeans`'s argument names."""
    schedule = FixedSchedule(max_llm_iter, max_iter)
    summarizer = ExtractiveSummarizer(method=nlp, top_k=top_k)
    return mini_batch_summary_kmeans(
        text_data,
        num_clusters,
        summarizer,
        emb_type,
        schedule,
        text_features=text_features,
        sentences=text_sentences,
        max_batch_size=max_batch_size,
        init=init,
        force_context_length=force_context_length,
        tol=tol,
        random_state=random_state,
    )


def miniBatchKLLMmeans(
    text_data: list[str],
    num_clusters: int,
    summarizer: Summarizer | None = None,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "",
    text_type: str = "",
    max_batch_size: int = 5000,
    init="k-means++",
    force_context_length: int = 0,
    max_llm_iter: int = 5,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state=None,
    emb_type: str = "distilbert",
    text_features: np.ndarray | None = None,
) -> MiniBatchResult:
    """SLM-summary mini-batch k-means, matching upstream `miniBatchKLLMmeans`'s argument names."""
    if summarizer is None:
        summarizer = build_summarizer(
            {"type": "hf", "model_id": model}, prompt=prompt or None, text_type=text_type or None
        )

    schedule = FixedSchedule(max_llm_iter, max_iter)
    return mini_batch_summary_kmeans(
        text_data,
        num_clusters,
        summarizer,
        emb_type,
        schedule,
        text_features=text_features,
        max_batch_size=max_batch_size,
        init=init,
        force_context_length=force_context_length,
        tol=tol,
        random_state=random_state,
    )


def sequentialMiniBatchKmeans(
    text_features: np.ndarray, num_clusters: int, random_state, max_batch_size: int, max_iter: int = 100
) -> MiniBatchKMeans:
    """Streaming baseline: `MiniBatchKMeans.partial_fit` over the same batches, exact port.

    `max_iter` is accepted for signature parity with upstream but unused
    there too -- `MiniBatchKMeans.partial_fit` always takes one pass per
    call, regardless of `max_iter`.
    """
    text_features = np.asarray(text_features)
    n = len(text_features)
    num_batches = int(np.ceil(n / max_batch_size))
    bounds = _batch_bounds(n, num_batches)

    mbk = MiniBatchKMeans(
        n_clusters=num_clusters, random_state=random_state, batch_size=max_batch_size, init="k-means++"
    )
    for start, stop in bounds:
        mbk = mbk.partial_fit(text_features[start:stop])

    return mbk


def assign_to_centroids(X: np.ndarray, centroids: np.ndarray, how: str = "lloyd1") -> np.ndarray:
    """Assign `X` to fixed `centroids` for streaming evaluation, without moving them.

    `how="lloyd1"` reproduces upstream's streaming evaluation exactly:
    `KMeans(n_clusters=k, init=centroids, max_iter=1).fit_predict(X)` --
    one full-data Lloyd step from `centroids`, which can shift centroids
    slightly and is therefore *not* pure nearest-centroid assignment.
    `how="nearest"` is plain nearest-centroid assignment
    (`sklearn.metrics.pairwise_distances_argmin`), added for comparison.
    """
    centroids = np.asarray(centroids)
    if how == "lloyd1":
        return KMeans(n_clusters=len(centroids), init=centroids, max_iter=1).fit_predict(X)
    elif how == "nearest":
        return pairwise_distances_argmin(X, centroids)
    else:
        raise ValueError(f"Unknown eval_assign '{how}', expected one of ('lloyd1', 'nearest')")
