"""The `Cluster` input and `Summarizer` interface shared by every backend.

A summarizer turns one cluster's member texts (and, if it needs them,
their sentences) into a single string that stands in for the cluster's
new centroid once it is re-embedded. Anything specific to *how* a
summarizer phrases its output -- an LLM prompt, a `text_type` prefix, a
choice of extractive method -- belongs in that summarizer's constructor,
not on `Cluster`, so that `Cluster` stays a plain description of "these
are the selected members."
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cluster:
    """The members selected to summarize one cluster."""

    index: int  # cluster id
    doc_ids: tuple[int, ...]  # GLOBAL document indices selected for summarization
    texts: list[str]  # texts of those documents, same order as doc_ids
    sentences: list[str] | None = None  # filled only if the summarizer needs sentences
    sentence_embeddings: np.ndarray | None = None  # parallel to `sentences`


class Summarizer(ABC):
    """Produces one summary string per cluster."""

    needs_sentences: bool = False

    @abstractmethod
    def summarize(self, cluster: Cluster) -> str:
        """Summarize a single cluster."""

    def summarize_batch(self, clusters: list[Cluster]) -> list[str]:
        """Summarize several clusters. Batched backends (e.g. an SLM) should override this."""
        return [self.summarize(cluster) for cluster in clusters]
