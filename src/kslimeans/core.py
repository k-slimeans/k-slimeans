"""The summary-centroids clustering loop.

`summary_kmeans` generalizes upstream's `kSummariesmeans`: alternate
between k-means Lloyd iterations and "summarize each cluster, re-embed the
summary, use it as the next centroid" steps, with a `SummarySchedule`
deciding when each kind of step happens. With a `FixedSchedule` (the
paper's schedule) this function is bit-identical to upstream.

`kNLPmeans` and `kLLMmeans` are thin wrappers matching the paper's
upstream function names and argument names, for anyone porting old
experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import pairwise_distances_argmin
from tqdm import tqdm

from .embeddings import Embedder, get_embedder
from .schedules import FixedSchedule, State, SummarySchedule
from .sentences import SentenceCorpus
from .summarizers import Cluster, Summarizer, build_summarizer


@dataclass
class Result:
    """Everything `summary_kmeans` produces."""

    assignments: np.ndarray
    summaries: list[str] | None
    summary_embeddings: np.ndarray | None
    centroids: np.ndarray
    summaries_evolution: list[list[str]] = field(default_factory=list)
    centroids_evolution: list[np.ndarray] = field(default_factory=list)
    converged: bool = False
    trace: list[dict] = field(default_factory=list)
    """One plain-Python (JSON-serializable) row per KMeans fit, in order, including the initial fit
    (step 0); see `_trace_row` for the keys. A summary step that stops on `tol` fits no KMeans, so
    it gets no row: `len(trace) == 1 + len(centroids_evolution)`."""

    def as_tuple(self):
        """Upstream's 6-tuple return order, for drop-in compatibility."""
        return (
            self.assignments,
            self.summaries,
            self.summary_embeddings,
            self.centroids,
            self.summaries_evolution,
            self.centroids_evolution,
        )


def _objective(X: np.ndarray, centers: np.ndarray, chunk_rows: int = 65536) -> float:
    """K-means objective of `centers` on `X`: sum of squared distances to the nearest center.

    The same quantity as sklearn's `inertia_`, but RNG-free. Nearest centers come from
    sklearn's argmin; the squared distances are then summed exactly in float64, `chunk_rows`
    rows at a time so a large `X` is never copied whole.
    """
    centers64 = centers.astype(np.float64)
    total = 0.0
    for lo in range(0, len(X), chunk_rows):
        chunk = X[lo : lo + chunk_rows]
        diff = chunk.astype(np.float64) - centers64[pairwise_distances_argmin(chunk, centers)]
        total += float(np.sum(diff**2))
    return total


def _trace_row(state: State, summarize: bool, summary_inertia: float | None) -> dict:
    """One `Result.trace` row for the fit that just produced `state`.

    `summarize` says whether the step that produced this fit was a summary step;
    `summary_inertia` is the objective of the summary-embedding centroids before that
    step's Lloyd iterations (`None` on non-summary rows), so
    `summary_inertia - prev_inertia` is the jump the summary step caused.
    """
    return {
        "step": state.step,
        "inertia": float(state.inertia),
        "prev_inertia": None if state.prev_inertia is None else float(state.prev_inertia),
        "frac_changed": state.frac_changed,
        "summary_shift": state.summary_shift,
        "last_n_iter": int(state.last_n_iter),
        "last_max_iter": int(state.last_max_iter),
        "summarize": summarize,
        "summary_inertia": summary_inertia,
    }


def _select_cluster(
    cluster_id: int,
    labels: np.ndarray,
    X: np.ndarray,
    texts: list[str],
    force_context_length: int,
    random_state,
    sentences: SentenceCorpus | None,
) -> Cluster:
    """Pick the members of cluster `cluster_id` to summarize.

    Ascending global document indices, optionally trimmed to
    `force_context_length` members via k-means++ seeding (same as
    upstream). Upstream had a bug here: in its `force_context_length > 0`
    NLP path it stored the cluster-LOCAL k-means++ positions and then used
    them to index the GLOBAL per-document sentence store, which silently
    grabbed the wrong documents' sentences whenever a cluster was actually
    trimmed. `idx[sel]` below converts those local positions back to
    global document indices before anything downstream sees them.
    """
    idx = np.flatnonzero(labels == cluster_id)
    if force_context_length > 0 and len(idx) > force_context_length:
        _, sel = kmeans_plusplus(X[idx], n_clusters=force_context_length, random_state=random_state)
        doc_ids = idx[sel]
    else:
        doc_ids = idx

    cluster_texts = [texts[i] for i in doc_ids]
    cluster_sentences = None
    cluster_sentence_embeddings = None
    if sentences is not None and len(doc_ids) > 0:
        cluster_sentences, cluster_sentence_embeddings = sentences.gather(doc_ids)

    return Cluster(
        index=cluster_id,
        doc_ids=tuple(int(i) for i in doc_ids),
        texts=cluster_texts,
        sentences=cluster_sentences,
        sentence_embeddings=cluster_sentence_embeddings,
    )


def _summarize_clusters(
    clusters: list[Cluster], summarizer: Summarizer, cache: dict[tuple[int, ...], str]
) -> list[str]:
    """Summary per cluster, in cluster-id order. Empty clusters get "" for free."""
    summaries: list[str | None] = [None] * len(clusters)
    to_summarize: list[tuple[int, Cluster]] = []
    for i, cluster in enumerate(clusters):
        if len(cluster.doc_ids) == 0:
            summaries[i] = ""
        elif cluster.doc_ids in cache:
            summaries[i] = cache[cluster.doc_ids]
        else:
            to_summarize.append((i, cluster))

    if to_summarize:
        computed = summarizer.summarize_batch([cluster for _, cluster in to_summarize])
        for (i, cluster), summary in zip(to_summarize, computed):
            summaries[i] = summary
            cache[cluster.doc_ids] = summary

    return summaries  # type: ignore[return-value]


def summary_kmeans(
    texts: list[str],
    n_clusters: int,
    summarizer: Summarizer,
    embedder: Embedder | str,
    schedule: SummarySchedule,
    *,
    text_features: np.ndarray | None = None,
    sentences: SentenceCorpus | None = None,
    force_context_length: int = 0,
    tol: float = 1e-4,
    random_state=None,
    init="k-means++",
    verbose: bool = True,
) -> Result:
    """Run the summary-centroids clustering loop.

    `embedder` may be an `Embedder` instance or a name resolved via
    `get_embedder`. `text_features`, if given, is used instead of
    embedding `texts` from scratch. `sentences`, if given, is used instead
    of building a `SentenceCorpus` for summarizers that need one.
    """
    if not isinstance(texts, list):
        raise TypeError(f"Expected a list for variable texts, but got {type(texts).__name__}")

    if isinstance(embedder, str):
        embedder = get_embedder(embedder)

    X = embedder.encode(texts) if text_features is None else np.asarray(text_features)

    if summarizer.needs_sentences and sentences is None:
        sentences = SentenceCorpus.build(texts, embedder)

    km = KMeans(n_clusters=n_clusters, init=init, max_iter=schedule.first_iters(), random_state=random_state)
    labels = km.fit_predict(X)
    ref = km.cluster_centers_

    summaries: list[str] | None = None
    s_emb: np.ndarray | None = None
    summaries_evolution: list[list[str]] = []
    centroids_evolution: list[np.ndarray] = []
    converged = False
    cache: dict[tuple[int, ...], str] = {}

    state = State(
        step=0,
        n_summaries=0,
        lloyd_iters_used=km.n_iter_,
        labels=labels,
        prev_labels=None,
        frac_changed=None,
        centers=ref,
        inertia=km.inertia_,
        prev_inertia=None,
        last_n_iter=km.n_iter_,
        last_max_iter=schedule.first_iters(),
        summary_shift=None,
    )

    trace = [_trace_row(state, False, None)]

    pbar = tqdm(desc="Iterating KMeans", disable=not verbose)
    try:
        while (step := schedule.next_step(state)) is not None:
            pbar.update(1)
            shift = None
            summary_inertia = None
            if step.summarize:
                clusters = [
                    _select_cluster(c, labels, X, texts, force_context_length, random_state, sentences)
                    for c in range(n_clusters)
                ]
                summaries = _summarize_clusters(clusters, summarizer, cache)
                summaries_evolution.append(summaries)

                s_emb = embedder.encode(summaries)
                shift = float(np.linalg.norm(ref - s_emb, axis=1).sum())
                if shift < tol:
                    converged = True
                    if verbose:
                        print(f"Converged after {state.step + 1} iterations.")
                    break

                ref = s_emb
                centroids_evolution.append(s_emb)
                start = s_emb
                summary_inertia = _objective(X, start)
            else:
                start = km.cluster_centers_

            km = KMeans(n_clusters=n_clusters, init=start, max_iter=step.kmeans_iters)
            prev_labels = labels
            labels = km.fit_predict(X)
            state = State(
                step=state.step + 1,
                n_summaries=state.n_summaries + int(step.summarize),
                lloyd_iters_used=state.lloyd_iters_used + km.n_iter_,
                labels=labels,
                prev_labels=prev_labels,
                frac_changed=float(np.mean(labels != prev_labels)),
                centers=km.cluster_centers_,
                inertia=km.inertia_,
                prev_inertia=state.inertia,
                last_n_iter=km.n_iter_,
                last_max_iter=step.kmeans_iters,
                summary_shift=shift,
            )
            trace.append(_trace_row(state, step.summarize, summary_inertia))
    finally:
        pbar.close()

    return Result(
        assignments=labels,
        summaries=summaries,
        summary_embeddings=s_emb,
        centroids=km.cluster_centers_,
        summaries_evolution=summaries_evolution,
        centroids_evolution=centroids_evolution,
        converged=converged,
        trace=trace,
    )


def kNLPmeans(
    text_data: list[str],
    num_clusters: int,
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
    initial_iter: bool = True,
    final_iter: bool = True,
) -> Result:
    """Extractive-summary k-means, matching upstream `kNLPmeans`'s argument names."""
    from .summarizers.extractive import ExtractiveSummarizer

    schedule = FixedSchedule(max_llm_iter, max_iter, initial_iter, final_iter)
    summarizer = ExtractiveSummarizer(method=nlp, top_k=top_k)
    return summary_kmeans(
        text_data,
        num_clusters,
        summarizer,
        emb_type,
        schedule,
        text_features=text_features,
        sentences=text_sentences,
        force_context_length=force_context_length,
        tol=tol,
        random_state=random_state,
        init=init,
    )


def kLLMmeans(
    text_data: list[str],
    num_clusters: int,
    summarizer: Summarizer | None = None,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    prompt: str = "",
    text_type: str = "",
    init="k-means++",
    force_context_length: int = 0,
    max_llm_iter: int = 5,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state=None,
    emb_type: str = "distilbert",
    text_features: np.ndarray | None = None,
    initial_iter: bool = True,
    final_iter: bool = True,
) -> Result:
    """SLM-summary k-means, matching upstream `kLLMmeans`'s argument names."""
    if summarizer is None:
        summarizer = build_summarizer(
            {"type": "hf", "model_id": model}, prompt=prompt or None, text_type=text_type or None
        )

    schedule = FixedSchedule(max_llm_iter, max_iter, initial_iter, final_iter)
    return summary_kmeans(
        text_data,
        num_clusters,
        summarizer,
        emb_type,
        schedule,
        text_features=text_features,
        force_context_length=force_context_length,
        tol=tol,
        random_state=random_state,
        init=init,
    )
