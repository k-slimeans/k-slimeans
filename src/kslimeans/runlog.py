"""Read a run directory back: per-(method, seed) records with summaries, partitions and per-step inputs.

`load_run(run_dir)` parses what `results.RunDir` wrote (`config.yaml`, `environment.json`, `metrics.jsonl`,
`trace.jsonl`, `summaries.jsonl`) into a `RunLog` holding one `RunRecord` per (method, seed). Analyses of a
logged run (replacement diagnostics, summary quality, cost checks, prompt tests) start from here.

Indexing contract (the same one `core.Result` and `results.py` document). The trace has one row per KMeans fit,
step 0 being the initial fit, and `assignments_evolution[t]` are the labels after fit `t`. Summary step `j` (entry
`j` of `summaries_evolution`) was written from `assignments_evolution[summary_source_steps[j]]`, and the partition
after its Lloyd block is `assignments_evolution[summary_source_steps[j] + 1]`; that one is absent for a final
summary step that stopped on `tol` (it fits no KMeans).

Runs made before #30 lack `assignments_evolution`, `summary_source_steps`, `environment.json` and possibly the
trace. They load with those fields set to `None` (or `[]` for the trace); nothing is reconstructed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .cache import (
    cached_sentences,
    dataset_embeddings,
    embeddings_cache_path,
    has_cached_embeddings,
    has_cached_sentences,
    sentences_cache_base,
)
from .core import _members_by_cluster, _select_cluster
from .data import Dataset
from .embeddings import Embedder
from .experiment import _resolve_dataset, _resolve_embedder
from .sentences import SentenceCorpus
from .summarizers import Cluster

_IDENTITY_KEYS = ("seed", "method")


# Where the real caches usually live: the `cache/` of the checkout the runs were made from (worktrees have none).
REAL_CACHE_HINT = "the main checkout's cache/ directory"


class CacheMissError(FileNotFoundError):
    """The run's embeddings or sentences are not in the cache and encoding was not allowed."""


class MissingAssignmentsError(ValueError):
    """The run did not log per-step assignments (it predates `assignments_evolution`)."""


@dataclass
class RunRecord:
    """Everything logged for one (method, seed) of a run.

    Baselines (e.g. `kmeans`) have only `metrics`: `summaries_evolution` is `[]` and the other fields `None`.
    """

    method: str
    seed: int
    metrics: dict = field(default_factory=dict)
    """The `metrics.jsonl` row without its `seed` / `method` keys."""
    trace: list[dict] = field(default_factory=list)
    """`trace.jsonl` rows in step order without `seed` / `method`; `[]` if the run has none."""
    summaries: list[str | None] | None = None
    """Final summaries (`Result.summaries`), one per cluster."""
    assignments: np.ndarray | None = None
    """Final labels, one per document."""
    summaries_evolution: list[list[str | None]] = field(default_factory=list)
    """Per summary step, the k summaries; `None` marks a cluster a masked schedule had not summarized yet."""
    assignments_evolution: list[np.ndarray] | None = None
    """Labels after every KMeans fit (int arrays), or `None` when not logged."""
    summary_source_steps: list[int] | None = None
    """Per summary step, the trace step it was written from, or `None` when not logged."""

    @property
    def n_summary_steps(self) -> int:
        return len(self.summaries_evolution)

    def _check_logged(self, j: int) -> None:
        if self.assignments_evolution is None or self.summary_source_steps is None:
            raise MissingAssignmentsError(
                f"({self.method!r}, seed {self.seed}) did not log assignments_evolution / summary_source_steps"
            )
        if not 0 <= j < len(self.summary_source_steps):
            raise IndexError(f"summary step {j} out of range for {len(self.summary_source_steps)} steps")

    def source_partition(self, j: int) -> np.ndarray:
        """Labels summary step `j` was written from: `assignments_evolution[summary_source_steps[j]]`."""
        self._check_logged(j)
        return self.assignments_evolution[self.summary_source_steps[j]]

    def partition_after(self, j: int) -> np.ndarray:
        """Labels after summary step `j`'s Lloyd block: `assignments_evolution[summary_source_steps[j] + 1]`.

        Raises `IndexError` for a final step that stopped on `tol` (no KMeans was fit after it).
        """
        self._check_logged(j)
        after = self.summary_source_steps[j] + 1
        if after >= len(self.assignments_evolution):
            raise IndexError(f"summary step {j} stopped on tol: no partition after it")
        return self.assignments_evolution[after]


@dataclass
class RunLog:
    """A parsed run directory."""

    path: Path
    config: dict
    environment: dict | None
    records: dict[tuple[str, int], RunRecord]

    @property
    def methods(self) -> list[str]:
        """Method names in first-seen order."""
        return list(dict.fromkeys(method for method, _ in self.records))

    @property
    def seeds(self) -> list[int]:
        return sorted({seed for _, seed in self.records})

    def record(self, method: str, seed: int) -> RunRecord:
        try:
            return self.records[(method, seed)]
        except KeyError:
            raise KeyError(f"no record for method {method!r}, seed {seed} in {self.path}") from None

    def method_records(self, method: str) -> list[RunRecord]:
        """Records of `method`, ordered by seed."""
        return sorted((r for r in self.records.values() if r.method == method), key=lambda r: r.seed)


def members_by_cluster(labels: np.ndarray, k: int) -> list[np.ndarray]:
    """Ascending document indices of each of the `k` clusters (core's `_members_by_cluster`)."""
    return _members_by_cluster(np.asarray(labels), k)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _IDENTITY_KEYS}


def load_run(run_dir: str | Path) -> RunLog:
    """Parse the run directory `run_dir`. Only `config.yaml` is required."""
    path = Path(run_dir)
    with open(path / "config.yaml") as f:
        config = yaml.safe_load(f)
    env_path = path / "environment.json"
    environment = json.loads(env_path.read_text()) if env_path.exists() else None

    records: dict[tuple[str, int], RunRecord] = {}

    def get(row: dict) -> RunRecord:
        key = (row["method"], int(row["seed"]))
        if key not in records:
            records[key] = RunRecord(method=key[0], seed=key[1])
        return records[key]

    for row in _read_jsonl(path / "metrics.jsonl"):
        get(row).metrics = _strip(row)
    for row in _read_jsonl(path / "trace.jsonl"):
        get(row).trace.append(_strip(row))
    for record in records.values():
        record.trace.sort(key=lambda r: r["step"])
    for row in _read_jsonl(path / "summaries.jsonl"):
        record = get(row)
        record.summaries = row.get("summaries")
        if row.get("assignments") is not None:
            record.assignments = np.asarray(row["assignments"], dtype=np.int64)
        record.summaries_evolution = row.get("summaries_evolution") or []
        if row.get("assignments_evolution") is not None:
            record.assignments_evolution = [np.asarray(a, dtype=np.int64) for a in row["assignments_evolution"]]
        record.summary_source_steps = row.get("summary_source_steps")
    return RunLog(path=path, config=config, environment=environment, records=records)


def step_inputs(
    record: RunRecord,
    j: int,
    texts: list[str],
    X: np.ndarray,
    force_context_length: int,
    sentences: SentenceCorpus | None = None,
) -> list[Cluster]:
    """The per-cluster summarizer inputs summary step `j` received, one `Cluster` per cluster in order.

    Built by core's `_select_cluster` from `record.source_partition(j)`, with `record.seed` as its
    `random_state` (what `run_offline` passes `summary_kmeans`), so member selection, including the
    k-means++ trimming to `force_context_length`, is the one the run made. `X` must be the embeddings the run
    used (`load_run_dataset`), `force_context_length` the config's, and `sentences` the run's corpus for
    summarizers that need one.

    What this is and is not:

    - These are the members `summary_kmeans` selected, not necessarily what the summarizer was called on.
      The loop sends no empty cluster and no cluster whose `doc_ids` it already summarized earlier in the
      run (it reuses that summary); `sent_clusters` says which were sent. `summaries_evolution[j][c]` still
      belongs to cluster `c`'s members here either way, but call and token counts must skip unsent clusters.
    - The members are those before any token-budget trimming. `HFSummarizer` fits the texts to its prompt
      budget afterwards (`hf.fit_texts_to_budget`, seeded with `TRUNCATION_SEED`), so a token recount over
      these texts overcounts truncated clusters.

    Only full summary steps are supported: a masked step summarized a subset of clusters that is not logged.
    This is verified from the step's trace row (`n_summarized`); a run with no trace rows at all raises, since
    a full step cannot be confirmed. A final step that stopped on `tol` has no row of its own and is accepted
    (masked steps never stop on `tol`).
    """
    labels = record.source_partition(j)
    k = len(record.summaries_evolution[j])
    if not record.trace:
        raise ValueError(f"({record.method!r}, seed {record.seed}) has no trace rows: cannot confirm step {j} was full")
    # Trace row `t` is the fit after step `t - 1`, so summary step `j`'s own row has step `source + 1`.
    own = [r for r in record.trace if r["step"] == record.summary_source_steps[j] + 1]
    if own and own[0]["n_summarized"] != k:
        raise ValueError(f"summary step {j} summarized {own[0]['n_summarized']} of {k} clusters; inputs not logged")
    return [_select_cluster(c, labels, X, texts, force_context_length, record.seed, sentences) for c in range(k)]


def sent_clusters(
    record: RunRecord, j: int, texts: list[str], X: np.ndarray, force_context_length: int
) -> list[int]:
    """Indices of the clusters `summary_kmeans` actually sent to the summarizer at summary step `j`.

    Reproduces the loop's memo: a cluster is skipped when it has no members or when its selected `doc_ids`
    were already summarized at an earlier step of this run (or, impossibly within one step, earlier in this one).
    Arguments are those of `step_inputs`; every step up to `j` is rebuilt to know what was already seen.
    """
    seen: set[tuple[int, ...]] = set()
    sent: list[int] = []
    for step in range(j + 1):
        inputs = step_inputs(record, step, texts, X, force_context_length)
        sent = [c.index for c in inputs if c.doc_ids and c.doc_ids not in seen]
        seen.update(c.doc_ids for c in inputs)
    return sent


def _check_cache(hit: bool, what: str, path: str, allow_encode: bool) -> None:
    if not hit and not allow_encode:
        raise CacheMissError(
            f"no valid cached {what} at {path}; loading would encode them. Pass cache_dir pointing at the real "
            f"cache ({REAL_CACHE_HINT}), or allow_encode=True to encode here."
        )


def load_run_dataset(
    runlog: RunLog, cache_dir: str | Path | None = None, allow_encode: bool = False
) -> tuple[Dataset, np.ndarray]:
    """The run's `Dataset` and its embeddings, resolved the way `experiment.run_offline` does.

    `cache_dir` defaults to the config's (a path relative to the current directory, so a worktree without a
    `cache/` misses). Unless `allow_encode`, a missing or stale cache entry raises `CacheMissError` instead of
    loading the embedder and encoding the whole dataset. Datasets with precomputed vectors need no cache.
    """
    cfg = runlog.config
    cache = str(cache_dir or cfg.get("cache_dir", "cache"))
    ds = _resolve_dataset(cfg["dataset"])
    if ds.embeddings_path is None:
        _check_cache(
            has_cached_embeddings(ds.name, ds.texts, cfg["embedder"], cache),
            "embeddings",
            str(embeddings_cache_path(ds.name, cfg["embedder"], cache)),
            allow_encode,
        )
    return ds, dataset_embeddings(ds, cfg["embedder"], cache)


def load_run_sentences(
    runlog: RunLog, ds: Dataset, cache_dir: str | Path | None = None, allow_encode: bool = False
) -> SentenceCorpus:
    """The run's cached `SentenceCorpus`, for extractive summarizers (`step_inputs(..., sentences=...)`).

    Raises `CacheMissError` on a missing or stale cache entry unless `allow_encode`.
    """
    cfg = runlog.config
    cache = str(cache_dir or cfg.get("cache_dir", "cache"))
    _check_cache(
        has_cached_sentences(ds.name, ds.texts, cfg["embedder"], cache),
        "sentences",
        sentences_cache_base(ds.name, cfg["embedder"], cache) + ".npz",
        allow_encode,
    )
    return cached_sentences(ds.name, ds.texts, cfg["embedder"], cache)


def load_run_embedder(runlog: RunLog) -> Embedder:
    """The embedder the run used, resolved by `experiment._resolve_embedder` (`get_embedder(name)`).

    Loads the model, so call it only when something must really be encoded.
    """
    return _resolve_embedder(runlog.config["embedder"])


def embed_summaries(embedder: Embedder, summaries: list[str]) -> np.ndarray:
    """Embed one full summary step's `k` summaries as `summary_kmeans` does: one `encode` call over all of them.

    The result is the step's start centroids (`Result.centroids_evolution[j]`, for a step that did not stop on
    `tol`). `""` (an empty cluster) is passed to the embedder like any other text, exactly as core does. A `None`
    (a cluster a masked schedule had not summarized) never occurs in a full step, and core never embeds it, so it
    raises `ValueError` here instead of reaching the embedder.
    """
    if any(summary is None for summary in summaries):
        raise ValueError("summaries contain None (a cluster never summarized); only full steps can be embedded")
    return embedder.encode(list(summaries))
