"""kslimeans: pluggable summary-centroids text clustering (k-NLPmeans / k-LLMmeans).

Re-implements the method from "k-LLMmeans" (arXiv:2502.09667): alternate
k-means with steps that summarize each cluster and re-embed the summary
as the next centroid. `summary_kmeans` is the generalized loop; `schedules`
controls when summarization happens, and `summarizers` controls how a
cluster gets turned into a summary string.
"""

from .core import Result, kLLMmeans, kNLPmeans, summary_kmeans
from .data import Dataset, load_dataset
from .embeddings import Embedder, get_embedder
from .minibatch import (
    MiniBatchResult,
    assign_to_centroids,
    miniBatchKLLMmeans,
    miniBatchKNLPmeans,
    mini_batch_summary_kmeans,
    sequentialMiniBatchKmeans,
)
from .schedules import FixedSchedule, State, Step, SummarySchedule, build_schedule
from .sentences import SentenceCorpus, build_sentence_corpus, split_into_sentences
from .summarizers import Cluster, ExtractiveSummarizer, Summarizer, build_summarizer

__all__ = [
    "Result",
    "summary_kmeans",
    "kNLPmeans",
    "kLLMmeans",
    "mini_batch_summary_kmeans",
    "miniBatchKNLPmeans",
    "miniBatchKLLMmeans",
    "sequentialMiniBatchKmeans",
    "assign_to_centroids",
    "MiniBatchResult",
    "Dataset",
    "load_dataset",
    "Embedder",
    "get_embedder",
    "FixedSchedule",
    "State",
    "Step",
    "SummarySchedule",
    "build_schedule",
    "SentenceCorpus",
    "build_sentence_corpus",
    "split_into_sentences",
    "Cluster",
    "Summarizer",
    "ExtractiveSummarizer",
    "build_summarizer",
]
