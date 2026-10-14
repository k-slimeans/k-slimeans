"""Config-driven offline experiment runner.

`run_offline(cfg)` is the one function every teammate calls (directly, or
via `scripts/run_offline.py`): load a dataset, embed it once (cached to
disk), build the summarizer once (SLM loading is expensive), then for
every seed run each configured baseline plus the summary-centroids method,
logging metrics/summaries into a fresh `results/<name>/<timestamp>/` run
directory (see `results.RunDir`). See `configs/*.yaml` for example configs.

`_ALLOWED_KEYS`/`_REQUIRED_KEYS` feed `config.check_keys`, a flat typo-guard, not a
schema validator: unknown top-level keys raise immediately instead of silently being
ignored, but nested config blocks (`schedule`, `summarizer`, ...) are validated by the
functions that build from them (`build_schedule`, `build_summarizer`). `scripts/run_streaming.py`
uses the same helper for streaming configs.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .cache import cached_embeddings, cached_sentences
from .config import check_keys, default_method_name
from .core import summary_kmeans
from .data import Dataset, load_dataset
from .embeddings import Embedder, get_embedder
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

    X = cached_embeddings(ds.name, ds.texts, embedder, cache_dir)
    summarizer = build_summarizer(cfg["summarizer"], prompt=ds.prompt, text_type=ds.text_type)
    sentences = cached_sentences(ds.name, ds.texts, embedder, cache_dir) if summarizer.needs_sentences else None
    schedule = build_schedule(cfg["schedule"])
    oracle = oracle_centroids(X, ds.labels, ds.n_clusters)

    run = RunDir(results_dir, cfg["name"], cfg)

    for seed in seeds:
        for baseline_name, baseline_kwargs in baselines.items():
            start = time.perf_counter()
            assignments, centers = _run_baseline(baseline_name, baseline_kwargs or {}, X, ds.n_clusters, seed)
            elapsed = time.perf_counter() - start
            metrics = cluster_metrics(ds.labels, assignments, oracle, centers, oracle)
            run.log_metrics({"seed": seed, "method": baseline_name, **metrics, "runtime_s": elapsed})

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
        metrics = cluster_metrics(
            ds.labels, result.assignments, oracle, result.centroids, oracle, result.summary_embeddings
        )
        run.log_metrics(
            {
                "seed": seed,
                "method": method_name,
                **metrics,
                "runtime_s": elapsed,
                "n_summary_steps": len(result.summaries_evolution),
                "converged": result.converged,
            }
        )
        run.log_summaries(
            {
                "seed": seed,
                "method": method_name,
                "summaries": result.summaries,
                "assignments": result.assignments,
                "summaries_evolution": result.summaries_evolution,
            }
        )

    summary = run.finish()
    print(summary)
    print(f"\nRun directory: {run.path}")
    return run.path
