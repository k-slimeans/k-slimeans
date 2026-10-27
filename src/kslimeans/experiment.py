"""Config-driven offline experiment runner.

`run_offline(cfg)` is the one function every teammate calls (directly, or
via `scripts/run_offline.py`): load a dataset, embed it once (cached to
disk), build the summarizer once (SLM loading is expensive), then for
every seed run each configured baseline plus the summary-centroids method,
logging metrics/summaries/per-step traces into a fresh `results/<name>/<timestamp>/` run
directory (see `results.RunDir`). See `configs/*.yaml` for example configs.

`_ALLOWED_KEYS`/`_REQUIRED_KEYS` feed `config.check_keys`, a flat typo-guard, not a
schema validator: unknown top-level keys raise immediately instead of silently being
ignored, but nested config blocks (`schedule`, `summarizer`, ...) are validated by the
functions that build from them (`build_schedule`, `build_summarizer`). `scripts/run_streaming.py`
uses the same helper for streaming configs.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .cache import cached_sentences, dataset_embeddings
from .config import check_keys, default_method_name
from .core import summary_kmeans
from .data import Dataset, load_dataset
from .embeddings import Embedder, get_embedder
from .environment import collect_environment, peak_gpu_memory_gb, reset_peak_gpu_memory
from .metrics import cluster_metrics, oracle_centroids
from .results import RunDir
from .schedules import build_schedule
from .summarizers import build_summarizer

_ALLOWED_KEYS = frozenset(
    {
        "name",
        "dataset",
        "embedder",
        "seeds",
        "baselines",
        "method_name",
        "force_context_length",
        "tol",
        "schedule",
        "summarizer",
        "cache_dir",
        "results_dir",
    }
)
_REQUIRED_KEYS = frozenset({"name", "dataset", "embedder", "schedule", "summarizer"})


def _check_keys(cfg: dict) -> None:
    check_keys(cfg, _ALLOWED_KEYS, _REQUIRED_KEYS)


def _resolve_dataset(spec) -> Dataset:
    """`spec` is a dataset name, or a dict of `{name: ..., **loader kwargs}`."""
    if isinstance(spec, str):
        return load_dataset(spec)
    if isinstance(spec, dict):
        spec = dict(spec)
        try:
            name = spec.pop("name")
        except KeyError:
            raise ValueError("'dataset' dict must have a 'name' key") from None
        return load_dataset(name, **spec)
    raise TypeError(f"'dataset' must be a string or a dict, got {type(spec).__name__}")


def _resolve_embedder(spec) -> Embedder:
    return get_embedder(spec) if isinstance(spec, str) else spec


def _run_baseline(name: str, kwargs: dict, X: np.ndarray, n_clusters: int, seed: int):
    """Fit one sklearn baseline. Adding one (e.g. GaussianMixture)? Add a branch here."""
    if name == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=seed, **kwargs).fit(X)
        return model.labels_, model.cluster_centers_
    raise ValueError(f"Unknown baseline '{name}', expected one of ('kmeans',)")


_SUMMARY_COLUMNS = (
    "acc",
    "nmi",
    "ari",
    "n_calls",
    "input_tokens",
    "output_tokens",
    "summarizer_seconds",
    "n_empty_summaries",
    "n_truncated_clusters",
    "n_dropped_texts",
    "peak_gpu_mem_gb",
    "peak_gpu_reserved_gb",
    "runtime_s",
)
"""`summary.csv` columns (mean/std over seeds); the cost ones are empty for baseline rows."""


def _labels_sha1(labels: np.ndarray) -> str:
    """sha1 of `labels` as contiguous int64 bytes."""
    return hashlib.sha1(np.ascontiguousarray(labels, dtype=np.int64).tobytes()).hexdigest()


def _step0_labels(X: np.ndarray, n_clusters: int, first_iters: int, seed: int) -> np.ndarray:
    """Labels of the first-stage fit `summary_kmeans` makes, for the baseline's shared-init hash.

    Same constructor arguments as core's first stage (`run_offline` always gives `summary_kmeans`
    k-means++), so the k-means++ draw (which doesn't depend on `max_iter`) is the one the summary
    method starts from. A baseline configured with another `init` still gets this hash; it then
    certifies the seed's k-means++ draw, not that baseline's own start.
    """
    km = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=first_iters, random_state=seed)
    return km.fit_predict(X)


def run_offline(cfg: dict) -> Path:
    """Run the offline experiment described by `cfg`. Returns the run directory."""
    _check_keys(cfg)

    ds = _resolve_dataset(cfg["dataset"])
    embedder = _resolve_embedder(cfg["embedder"])
    cache_dir = cfg.get("cache_dir", "cache")
    results_dir = cfg.get("results_dir", "results")
    seeds = cfg.get("seeds") or [0]
    baselines = cfg.get("baselines") or {}
    tol = cfg.get("tol", 1e-4)
    force_context_length = cfg.get("force_context_length", 0)
    method_name = cfg.get("method_name") or default_method_name(cfg["summarizer"])

    X = dataset_embeddings(ds, embedder, cache_dir)
    summarizer = build_summarizer(cfg["summarizer"], prompt=ds.prompt, text_type=ds.text_type)
    sentences = cached_sentences(ds.name, ds.texts, embedder, cache_dir) if summarizer.needs_sentences else None
    schedule = build_schedule(cfg["schedule"])
    oracle = oracle_centroids(X, ds.labels, ds.n_clusters)

    run = RunDir(results_dir, cfg["name"], cfg)
    run.log_environment(collect_environment(summarizer, getattr(embedder, "name", type(embedder).__name__), ds))

    for seed in seeds:
        for baseline_name, baseline_kwargs in baselines.items():
            start = time.perf_counter()
            assignments, centers = _run_baseline(baseline_name, baseline_kwargs or {}, X, ds.n_clusters, seed)
            elapsed = time.perf_counter() - start
            metrics = cluster_metrics(ds.labels, assignments, oracle, centers, oracle)
            step0 = _step0_labels(X, ds.n_clusters, schedule.first_iters(), seed)
            run.log_metrics(
                {
                    "seed": seed,
                    "method": baseline_name,
                    **metrics,
                    "runtime_s": elapsed,
                    "step0_sha1": _labels_sha1(step0),
                }
            )

        reset_peak_gpu_memory()
        start = time.perf_counter()
        result = summary_kmeans(
            ds.texts,
            ds.n_clusters,
            summarizer,
            embedder,
            schedule,
            text_features=X,
            sentences=sentences,
            force_context_length=force_context_length,
            tol=tol,
            random_state=seed,
        )
        elapsed = time.perf_counter() - start
        peak_alloc_gb, peak_reserved_gb = peak_gpu_memory_gb()
        # Clusters a masked schedule never summarized have NaN rows; the Hungarian matching in
        # `avg_closest_distance` handles the resulting rectangular matrix.
        summary_pred = result.summary_embeddings
        if summary_pred is not None:
            summary_pred = summary_pred[np.isfinite(summary_pred).all(axis=1)]
            if len(summary_pred) == 0:
                summary_pred = None
        metrics = cluster_metrics(ds.labels, result.assignments, oracle, result.centroids, oracle, summary_pred)
        run.log_metrics(
            {
                "seed": seed,
                "method": method_name,
                **metrics,
                "runtime_s": elapsed,
                "n_summary_steps": len(result.summaries_evolution),
                "converged": result.converged,
                "n_calls": result.n_calls,
                **result.cost,
                "peak_gpu_mem_gb": peak_alloc_gb,
                "peak_gpu_reserved_gb": peak_reserved_gb,
                "step0_sha1": _labels_sha1(result.assignments_evolution[0]),
            }
        )
        run.log_trace({"seed": seed, "method": method_name}, result.trace)
        run.log_summaries(
            {
                "seed": seed,
                "method": method_name,
                "summaries": result.summaries,
                "assignments": result.assignments,
                "summaries_evolution": result.summaries_evolution,
                "assignments_evolution": result.assignments_evolution,
                "summary_source_steps": result.summary_source_steps,
            }
        )

    summary = run.finish(metrics=[c for c in _SUMMARY_COLUMNS if any(c in row for row in run.rows)])
    print(summary)
    print(f"\nRun directory: {run.path}")
    return run.path
