"""Run the mini-batch streaming benchmark over one or more StackExchange years.

Mirrors `sequential_experiments.ipynb`: per year, per seed, runs the
sklearn baselines (`MiniBatchKMeans`, `sequentialMiniBatchKmeans`,
plain `KMeans`) and the summary-centroids streaming method
(`mini_batch_summary_kmeans`), logging metrics/summaries for each.

`_ALLOWED_KEYS`/`_REQUIRED_KEYS` feed `kslimeans.config.check_keys`, the same flat
typo-guard `kslimeans.experiment.run_offline` uses for offline configs: an unknown
top-level key (e.g. a typo'd `--set`) raises immediately instead of being silently
ignored.

A config without a `summarizer` block (or with `summarizer: null`) is baseline-only: just the sklearn
baselines run, and the schedule, embedder object, summarizer and abstract texts are never loaded. Add
`load_texts: false` to the `dataset` block to skip reading the texts too (arXiv loaders only; the memory
that saves at full scale is the point of `configs/arxiv_full_baselines.yaml`).

Usage:

    python scripts/run_streaming.py configs/stackexchange_stream.yaml \\
        --set years=[2023] --set dataset.limit=5000 --set max_batch_size=1000
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import kslimeans
from kslimeans.cache import cached_sentences, dataset_embeddings
from kslimeans.config import check_keys, default_method_name, load_config
from kslimeans.embeddings import get_embedder
from kslimeans.gpu_monitor import GpuSession
from kslimeans.metrics import cluster_metrics, oracle_centroids
from kslimeans.minibatch import assign_to_centroids, mini_batch_summary_kmeans, sequentialMiniBatchKmeans
from kslimeans.results import RunDir
from kslimeans.schedules import build_schedule
from kslimeans.summarizers import build_summarizer

_ALLOWED_KEYS = frozenset(
    {
        "name",
        "dataset",
        "years",
        "embedder",
        "seeds",
        "max_batch_size",
        "eval_assign",
        "baselines",
        "method_name",
        "force_context_length",
        "tol",
        "schedule",
        "summarizer",
        "summary_every",
        "cache_dir",
        "results_dir",
        "gpu_check",
    }
)
# Keys only the summary method reads: required when the config has a `summarizer`, ignored without one.
_SUMMARY_KEYS = frozenset({"summarizer", "summary_every", "schedule", "force_context_length", "tol", "eval_assign"})
_REQUIRED_KEYS = _ALLOWED_KEYS - {"method_name", "years", "gpu_check", "summary_every"} - _SUMMARY_KEYS


def _required_keys(cfg: dict) -> frozenset[str]:
    """The required keys for `cfg`: the summary method's keys too, unless it is baseline-only."""
    if cfg.get("summarizer") is None:
        return _REQUIRED_KEYS
    return _REQUIRED_KEYS | (_SUMMARY_KEYS - {"summary_every"})


def _peak_rss_mb() -> float:
    """The process's peak resident set size so far (a high-water mark, so it never goes down), in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB


def _run_baseline(name: str, kw: dict, X, n_clusters: int, seed: int, max_batch_size: int):
    """One sklearn baseline's (assignments, centroids). `name` selects the algorithm."""
    if name == "minibatchkmeans":
        model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed, batch_size=max_batch_size, init="k-means++", **kw
        )
        return model.fit_predict(X), model.cluster_centers_
    elif name == "seqminibatchkmeans":
        model = sequentialMiniBatchKmeans(X, n_clusters, random_state=seed, max_batch_size=max_batch_size, **kw)
        return model.predict(X), model.cluster_centers_
    elif name == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=seed, **kw)
        return model.fit_predict(X), model.cluster_centers_
    else:
        raise ValueError(f"Unknown baseline '{name}', expected one of "
                          "('minibatchkmeans', 'seqminibatchkmeans', 'kmeans')")


def _final_batch_summary_embeddings(summaries: list[str | None] | None, embedder) -> np.ndarray | None:
    """Embed the last batch's per-cluster summaries, or None if it has none.

    Entries that are None (a cluster never summarized) are dropped, like the NaN rows the offline
    runner drops; an empty-cluster summary ("") is embedded, as `summary_kmeans` does.
    """
    kept = [s for s in summaries or [] if s is not None]
    return embedder.encode(kept) if kept else None


def _streaming_metrics(labels, assignments, oracle, centroids, summary_pred) -> dict:
    """`cluster_metrics` against the oracle centroids, as `kslimeans.experiment.run_offline` calls it.

    `summary_pred` is the method's summary embeddings, or None for a method without summaries (the
    baselines, or a k-SLMmeans run whose last batch has none). Then `sum_cen_dist` and `sum_sum_dist`
    are None (null in `metrics.jsonl`) instead of falling back to the centroids.
    """
    metrics = cluster_metrics(labels, assignments, oracle, centroids, oracle, summary_pred)
    if summary_pred is None:
        metrics["sum_cen_dist"] = metrics["sum_sum_dist"] = None
    return metrics


def run_streaming(cfg: dict) -> Path:
    """Run the streaming benchmark described by `cfg`. Returns the run directory.

    Wraps the run in the config's `gpu_check` policy (pre-run idle check, `gpu_samples.csv` / `gpu.json`).
    """
    check_keys(cfg, _ALLOWED_KEYS, _required_keys(cfg))
    with GpuSession(cfg.get("gpu_check")) as gpu:
        return _run_streaming(cfg, gpu)


def _run_streaming(cfg: dict, gpu: GpuSession) -> Path:
    run = RunDir(cfg["results_dir"], cfg["name"], cfg)
    gpu.attach(run.path)

    max_batch_size = cfg["max_batch_size"]
    dataset_name = cfg["dataset"]["name"]
    limit = cfg["dataset"].get("limit")
    with_summary = cfg.get("summarizer") is not None
    load_texts = cfg["dataset"].get("load_texts", True)
    if not load_texts and with_summary:
        raise ValueError("dataset.load_texts: false is for baseline-only configs; the summary method reads the texts")
    dataset_kwargs = {} if load_texts else {"load_texts": False}
    schedule = build_schedule(cfg["schedule"]) if with_summary else None
    method_name = (cfg.get("method_name") or default_method_name(cfg["summarizer"])) if with_summary else None

    summarizer = None  # built once we have a Dataset's prompt/text_type -- see below

    # `years` is optional: a dataset with no year (e.g. arxiv) streams once, unfiltered (year=None).
    for year in cfg["years"] if "years" in cfg else [None]:
        ds = kslimeans.load_dataset(dataset_name, year=year, limit=limit, **dataset_kwargs)
        tag = "all" if year is None else year  # logged/grouped label (a None key would be dropped by groupby)
        print(f"=== year {tag}: {len(ds.texts)} posts, {ds.n_clusters} clusters ===")

        if with_summary and summarizer is None:
            # Built once (not per year): SLM summarizers are expensive to load.
            summarizer = build_summarizer(cfg["summarizer"], prompt=ds.prompt, text_type=ds.text_type)

        t0 = time.perf_counter()
        X = dataset_embeddings(ds, cfg["embedder"], cache_dir=cfg["cache_dir"])
        embed_time = time.perf_counter() - t0
        print(f"  embeddings: {embed_time:.1f}s")

        sentences = None
        sentence_time = 0.0
        if with_summary and summarizer.needs_sentences:
            t0 = time.perf_counter()
            sentences = cached_sentences(ds.name, ds.texts, cfg["embedder"], cache_dir=cfg["cache_dir"])
            sentence_time = time.perf_counter() - t0
            print(f"  sentence corpus: {sentence_time:.1f}s")

        oracle = oracle_centroids(X, ds.labels, ds.n_clusters)

        for seed in cfg["seeds"]:
            for name, kw in cfg["baselines"].items():
                t0 = time.perf_counter()
                assignments, centroids = _run_baseline(name, kw, X, ds.n_clusters, seed, max_batch_size)
                runtime = time.perf_counter() - t0

                metrics = _streaming_metrics(ds.labels, assignments, oracle, centroids, None)
                peak_rss_mb = _peak_rss_mb()
                run.log_metrics(
                    {
                        "year": tag,
                        "seed": seed,
                        "method": name,
                        "runtime": runtime,
                        "peak_rss_mb": peak_rss_mb,
                        **metrics,
                    }
                )
                print(f"  [{tag} seed={seed}] {name}: {metrics} ({runtime:.1f}s, peak RSS {peak_rss_mb:.0f} MB)")

            if not with_summary:
                continue

            embedder = get_embedder(cfg["embedder"])
            t0 = time.perf_counter()
            result = mini_batch_summary_kmeans(
                ds.texts,
                ds.n_clusters,
                summarizer,
                embedder,
                schedule,
                text_features=X,
                sentences=sentences,
                max_batch_size=max_batch_size,
                force_context_length=cfg["force_context_length"],
                tol=cfg["tol"],
                random_state=seed,
                verbose=False,
                summary_every=cfg.get("summary_every", 1),
            )
            assignments = assign_to_centroids(X, result.centroids, cfg["eval_assign"])
            runtime = time.perf_counter() - t0

            # Deliberately differs from upstream, whose notebook passed the predicted centroids as both
            # the "true" and the predicted summaries (so cen_sum_dist was 0): the reference is the oracle
            # and summary_pred is the final batch's summaries, embedded -- see README "Streaming".
            # A batch that `summary_every` skipped has no summaries; the last summarized batch stands in.
            last_summaries = next((s for s in reversed(result.summaries) if s is not None), None)
            summary_pred = _final_batch_summary_embeddings(last_summaries, embedder)
            metrics = _streaming_metrics(ds.labels, assignments, oracle, result.centroids, summary_pred)
            run.log_metrics(
                {
                    "year": tag,
                    "seed": seed,
                    "method": method_name,
                    "runtime": runtime,
                    "embed_time": embed_time,
                    "sentence_time": sentence_time,
                    "peak_rss_mb": _peak_rss_mb(),
                    **metrics,
                }
            )
            run.log_summaries(
                {"year": tag, "seed": seed, "method": method_name, "summaries": result.summaries}
            )
            print(f"  [{tag} seed={seed}] {method_name}: {metrics} ({runtime:.1f}s)")

    table = run.finish(group_by=["year", "method"])
    print()
    print(table)
    print(f"\nResults written to {run.path}")
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a YAML run config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="dotted-key override, e.g. --set years=[2023] --set dataset.limit=5000",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_streaming(cfg)


if __name__ == "__main__":
    main()
