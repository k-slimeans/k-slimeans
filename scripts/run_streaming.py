"""Run the mini-batch streaming benchmark over one or more StackExchange years.

Mirrors `sequential_experiments.ipynb`: per year, per seed, runs the
sklearn baselines (`MiniBatchKMeans`, `sequentialMiniBatchKmeans`,
plain `KMeans`) and the summary-centroids streaming method
(`mini_batch_summary_kmeans`), logging metrics/summaries for each.

`_ALLOWED_KEYS`/`_REQUIRED_KEYS` feed `kslimeans.config.check_keys`, the same flat
typo-guard `kslimeans.experiment.run_offline` uses for offline configs: an unknown
top-level key (e.g. a typo'd `--set`) raises immediately instead of being silently
ignored.

Usage:

    python scripts/run_streaming.py configs/stackexchange_stream.yaml \\
        --set years=[2023] --set dataset.limit=5000 --set max_batch_size=1000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sklearn.cluster import KMeans, MiniBatchKMeans

import kslimeans
from kslimeans.cache import cached_sentences, dataset_embeddings
from kslimeans.config import check_keys, default_method_name, load_config
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
        "cache_dir",
        "results_dir",
    }
)
_REQUIRED_KEYS = _ALLOWED_KEYS - {"method_name", "years"}


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


def run_streaming(cfg: dict) -> Path:
    """Run the streaming benchmark described by `cfg`. Returns the run directory."""
    check_keys(cfg, _ALLOWED_KEYS, _REQUIRED_KEYS)

    run = RunDir(cfg["results_dir"], cfg["name"], cfg)

    schedule = build_schedule(cfg["schedule"])
    max_batch_size = cfg["max_batch_size"]
    dataset_name = cfg["dataset"]["name"]
    limit = cfg["dataset"].get("limit")
    method_name = cfg.get("method_name") or default_method_name(cfg["summarizer"])

    summarizer = None  # built once we have a Dataset's prompt/text_type -- see below

    # `years` is optional: a dataset with no year (e.g. arxiv) streams once, unfiltered (year=None).
    for year in cfg["years"] if "years" in cfg else [None]:
        ds = kslimeans.load_dataset(dataset_name, year=year, limit=limit)
        tag = "all" if year is None else year  # logged/grouped label (a None key would be dropped by groupby)
        print(f"=== year {tag}: {len(ds.texts)} posts, {ds.n_clusters} clusters ===")

        if summarizer is None:
            # Built once (not per year): SLM summarizers are expensive to load.
            summarizer = build_summarizer(cfg["summarizer"], prompt=ds.prompt, text_type=ds.text_type)

        t0 = time.perf_counter()
        X = dataset_embeddings(ds, cfg["embedder"], cache_dir=cfg["cache_dir"])
        embed_time = time.perf_counter() - t0
        print(f"  embeddings: {embed_time:.1f}s")

        sentences = None
        sentence_time = 0.0
        if summarizer.needs_sentences:
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

                metrics = cluster_metrics(ds.labels, assignments, oracle, centroids, centroids)
                run.log_metrics({"year": tag, "seed": seed, "method": name, "runtime": runtime, **metrics})
                print(f"  [{tag} seed={seed}] {name}: {metrics} ({runtime:.1f}s)")

            t0 = time.perf_counter()
            result = mini_batch_summary_kmeans(
                ds.texts,
                ds.n_clusters,
                summarizer,
                cfg["embedder"],
                schedule,
                text_features=X,
                sentences=sentences,
                max_batch_size=max_batch_size,
                force_context_length=cfg["force_context_length"],
                tol=cfg["tol"],
                random_state=seed,
                verbose=False,
            )
            assignments = assign_to_centroids(X, result.centroids, cfg["eval_assign"])
            runtime = time.perf_counter() - t0

            metrics = cluster_metrics(ds.labels, assignments, oracle, result.centroids, result.centroids)
            run.log_metrics(
                {
                    "year": tag,
                    "seed": seed,
                    "method": method_name,
                    "runtime": runtime,
                    "embed_time": embed_time,
                    "sentence_time": sentence_time,
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
