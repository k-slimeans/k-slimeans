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
    """Everything `summary_kmeans` produces.

    With per-cluster summarize masks (`Step.summarize_mask`), `summaries` and every entry of
    `summaries_evolution` hold each cluster's latest summary, `None` for a cluster not yet summarized.
    `summary_embeddings` likewise holds each cluster's latest summary embedding, with a NaN row for a
    never-summarized cluster.
    """

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
    n_calls: int = 0
    """Total summarizer calls: clusters actually sent to `summarize_batch` (memo-cache hits and empty
    clusters are free), including those of a summary step that stopped on `tol` and so has no trace row."""

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


def _cluster_rms_radius(
    X: np.ndarray, labels: np.ndarray, centers: np.ndarray, chunk_rows: int = 8192
) -> np.ndarray:
    """Per-cluster RMS distance of its members to its center: `sqrt(mean_i ||X_i - center||^2)`.

    0 for an empty cluster. Squared distances are accumulated in float64, `chunk_rows` rows at a
    time, so a large `X` is never copied whole and no n x k matrix is built.
    """
    k = len(centers)
    centers64 = centers.astype(np.float64)
    sq_sum = np.zeros(k, dtype=np.float64)
    for lo in range(0, len(X), chunk_rows):
        lab = labels[lo : lo + chunk_rows]
        diff = X[lo : lo + chunk_rows].astype(np.float64) - centers64[lab]
        sq_sum += np.bincount(lab, weights=np.einsum("ij,ij->i", diff, diff), minlength=k)
    counts = np.bincount(labels, minlength=k)
    return np.sqrt(sq_sum / np.maximum(counts, 1))


def _members_by_cluster(labels: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    """Ascending document indices of every cluster, from one stable sort (== `np.flatnonzero(labels == c)`)."""
    order = np.argsort(labels, kind="stable")
    bounds = np.cumsum(np.bincount(labels, minlength=n_clusters))[:-1]
    return np.split(order, bounds)


def _trace_row(
    state: State, summarize: bool, summary_inertia: float | None, n_calls: int, n_summarized: int
) -> dict:
    """One `Result.trace` row for the fit that just produced `state`.

    `summarize` says whether the step that produced this fit was a summary step;
    `summary_inertia` is the objective of the start centroids before that step's Lloyd
    iterations (`None` on non-summary rows), so `summary_inertia - prev_inertia` is the jump
    the summary step caused. `n_calls` is the summarizer calls that step made and
    `n_summarized` the number of clusters whose start centroid came from a summary (k on a
    full step); both are 0 on non-summary rows and step 0.
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
        "n_calls": n_calls,
        "n_summarized": n_summarized,
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
) -> tuple[list[str], int]:
    """Summary per cluster, in input order, and the number of clusters actually sent to the summarizer.

    Empty clusters get "" and memo-cache hits reuse the cached summary; neither counts as a call.
    """
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

    return summaries, len(to_summarize)  # type: ignore[return-value]


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

    summaries: list[str | None] | None = None
    s_emb: np.ndarray | None = None
    summaries_evolution: list[list[str]] = []
    centroids_evolution: list[np.ndarray] = []
    converged = False
    cache: dict[tuple[int, ...], str] = {}
    total_calls = 0

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
        n_calls=0,
        last_summarized=np.full(n_clusters, -1, dtype=np.int64),
        anchor_centers=ref,
        anchor_members=tuple(_members_by_cluster(labels, n_clusters)),
        cluster_rms_radius=_cluster_rms_radius(X, labels, ref),
    )

    trace = [_trace_row(state, False, None, 0, 0)]

    pbar = tqdm(desc="Iterating KMeans", disable=not verbose)
    try:
        while (step := schedule.next_step(state)) is not None:
            pbar.update(1)
            shift = None
            summary_inertia = None
            step_calls = 0
            n_summarized = 0
            mask = step.summarize_mask
            summarized_now: list[int] = []  # clusters whose summary anchors move to the pre-summary state
            if mask is not None and len(mask) != n_clusters:
                raise ValueError(f"summarize_mask has length {len(mask)}, expected n_clusters={n_clusters}")
            if step.summarize and (mask is None or all(mask)):
                # Full step (the paper's behavior): every cluster, one encode call over all k summaries.
                clusters = [
                    _select_cluster(c, labels, X, texts, force_context_length, random_state, sentences)
                    for c in range(n_clusters)
                ]
                summaries, step_calls = _summarize_clusters(clusters, summarizer, cache)
                total_calls += step_calls
                n_summarized = n_clusters
                summarized_now = list(range(n_clusters))
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
            elif step.summarize:
                # Partial step: only masked-on clusters are selected, summarized and encoded. The
                # `tol` stop is skipped: a shift over a subset isn't comparable to `tol`, and one
                # unchanged (memo-cache hit) cluster would otherwise end the run.
                active = [c for c in range(n_clusters) if mask[c]]
                clusters = [
                    _select_cluster(c, labels, X, texts, force_context_length, random_state, sentences)
                    for c in active
                ]
                new_summaries, step_calls = _summarize_clusters(clusters, summarizer, cache)
                total_calls += step_calls
                n_summarized = len(active)
                summarized_now = active

                latest: list[str | None] = [None] * n_clusters if summaries is None else list(summaries)
                for c, summary in zip(active, new_summaries):
                    latest[c] = summary
                summaries = latest
                summaries_evolution.append(list(latest))

                # Masked-off rows keep the current k-means center. `start` copies the centers (so it
                # keeps KMeans' own dtype, which is also what KMeans casts any `init` to); the float32
                # summary embeddings are written into it exactly, with no precision lost.
                start = km.cluster_centers_.copy()
                ref = ref.copy()
                if active:
                    new_emb = embedder.encode(new_summaries)
                    shift = float(np.linalg.norm(ref[active] - new_emb, axis=1).sum())
                    ref[active] = new_emb
                    start[active] = new_emb
                    if s_emb is None:
                        s_emb = np.full((n_clusters, new_emb.shape[1]), np.nan, dtype=new_emb.dtype)
                    else:
                        s_emb = s_emb.copy()
                    s_emb[active] = new_emb
                else:
                    shift = 0.0
                centroids_evolution.append(start)
                summary_inertia = _objective(X, start)
            else:
                start = km.cluster_centers_

            km = KMeans(n_clusters=n_clusters, init=start, max_iter=step.kmeans_iters)
            prev_labels = labels
            labels = km.fit_predict(X)
            last_summarized = state.last_summarized
            anchor_centers = state.anchor_centers
            anchor_members = state.anchor_members
            if summarized_now:
                # Anchors are the PRE-summary state (`state` is still the previous step's snapshot).
                last_summarized = last_summarized.copy()
                last_summarized[summarized_now] = state.step + 1
                anchor_centers = anchor_centers.copy()
                anchor_centers[summarized_now] = state.centers[summarized_now]
                pre_members = _members_by_cluster(state.labels, n_clusters)
                renewed = set(summarized_now)
                anchor_members = tuple(
                    pre_members[c] if c in renewed else m for c, m in enumerate(anchor_members)
                )
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
                n_calls=total_calls,
                last_summarized=last_summarized,
                anchor_centers=anchor_centers,
                anchor_members=anchor_members,
                cluster_rms_radius=_cluster_rms_radius(X, labels, km.cluster_centers_),
            )
            trace.append(_trace_row(state, step.summarize, summary_inertia, step_calls, n_summarized))
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
        n_calls=total_calls,
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
