"""Extractive summarizers: pick a handful of representative sentences.

`mmr_rerank` and the four `summarize_*` scoring functions are ported
verbatim from upstream (same math, same defaults, including
`TruncatedSVD(random_state=0)`), so that they select exactly the same
sentences given the same sentence embeddings.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .base import Cluster, Summarizer

_VALID_METHODS = ("lsa", "centroid", "textrank", "lexrank")


def mmr_rerank(
    sentences: list[str], scores: np.ndarray, X: np.ndarray, k: int = 5, lambda_: float = 0.7
) -> list[tuple[str, float]]:
    n = len(sentences)
    if n == 0:
        return []
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    scores = np.asarray(scores, dtype=float).ravel()
    smin, smax = float(scores.min()), float(scores.max())
    if smax > smin:
        scores = (scores - smin) / (smax - smin)
    else:
        scores = np.ones_like(scores)
    chosen: list[int] = []
    while len(chosen) < min(k, n):
        if not chosen:
            j = int(np.argmax(scores))
        else:
            red = sim[:, chosen].max(axis=1)
            obj = lambda_ * scores - (1 - lambda_) * red
            obj[chosen] = -1e9
            j = int(np.argmax(obj))
        chosen.append(j)
    return [(sentences[j], float(scores[j])) for j in chosen]


def summarize_centroid(sents: list[str], X: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    centroid = np.asarray(X.mean(axis=0)).ravel()
    scores = np.asarray(X @ centroid).ravel()
    return mmr_rerank(sents, scores, X, k=top_k)


def summarize_textrank(
    sents: list[str], X: np.ndarray, top_k: int = 5, knn: int = 4, alpha: float = 0.85
) -> list[tuple[str, float]]:
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        nbr_idx = np.argsort(S[i])[::-1][:knn]
        for j in nbr_idx:
            if S[i, j] > 0:
                G.add_edge(i, j, weight=float(S[i, j]))
    if G.number_of_edges() == 0:
        pr_scores = np.ones(n) / n
    else:
        pr = nx.pagerank(G, alpha=alpha, weight="weight")
        pr_scores = np.array([pr[i] for i in range(n)])
    return mmr_rerank(sents, pr_scores, X, k=top_k)


def summarize_lsa(
    sents: list[str], X: np.ndarray, top_k: int = 5, n_components: int | None = None
) -> list[tuple[str, float]]:
    n_sent = X.shape[0]
    if n_components is None:
        n_components = max(2, min(8, n_sent - 1))
    n_components = max(2, min(n_components, n_sent - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    Z = svd.fit_transform(X)
    scores = np.linalg.norm(Z, axis=1)
    return mmr_rerank(sents, scores, X, k=top_k)


_METHODS = {
    "centroid": summarize_centroid,
    "textrank": summarize_textrank,
    "lexrank": summarize_textrank,
    "lsa": summarize_lsa,
}


class ExtractiveSummarizer(Summarizer):
    """Summarizes a cluster by extracting `top_k` representative sentences."""

    needs_sentences = True

    def __init__(self, method: str = "lsa", top_k: int = 5) -> None:
        method = method.lower()
        if method not in _VALID_METHODS:
            raise ValueError(f"Unknown extractive method '{method}', expected one of {_VALID_METHODS}")
        self.method = method
        self.top_k = top_k

    def summarize(self, cluster: Cluster) -> str:
        if cluster.sentences is None or cluster.sentence_embeddings is None:
            raise ValueError(
                "ExtractiveSummarizer needs cluster.sentences/sentence_embeddings "
                "(needs_sentences=True); got a cluster with none attached."
            )
        scorer = _METHODS[self.method]
        top = scorer(cluster.sentences, cluster.sentence_embeddings, self.top_k)
        top_sentences = [text for text, _ in top]
        return "\n".join(top_sentences)
