"""Re-summarize one fixed workload (the inputs of a logged summary step) and report what came out.

A "workload" is the list of clusters a logged run sent to its summarizer at one summary step
(`runlog.step_inputs` restricted to `runlog.sent_clusters`). Summarizing it again with another prompt, model or
batch size changes one thing at a time, which a whole run cannot do (its clusterings drift with the summaries).
Two users: the prompt test (#46: do the summaries change?) and the cost controls (#35: how long does a model take
at a given batch size on identical work?). `scripts/fixed_workload.py` is the CLI, `scripts/compare_workloads.py`
the side-by-side table; this module holds the parts that need no model, so tests can drive them with fakes.

A workload output directory holds `config.yaml` (what was run, see `scripts/fixed_workload.py`),
`environment.json`, `outputs.jsonl` (`repeat`, `cluster`, `summary`), `cost.jsonl` (one row per repeat) and
`surface.json`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .environment import peak_gpu_memory_gb, reset_peak_gpu_memory
from .runlog import RunLog, RunRecord, sent_clusters, step_inputs
from .summarizers import Cluster, Summarizer
from .surface import duplicate_groups, surface_counts


def pick_record(runlog: RunLog, method: str | None, seed: int) -> RunRecord:
    """The record of `(method, seed)`; `method=None` picks the run's only method that logged summaries."""
    if method is None:
        summarizing = [m for m in runlog.methods if any(r.summaries_evolution for r in runlog.method_records(m))]
        if len(summarizing) != 1:
            raise ValueError(f"{runlog.path} has {len(summarizing)} summarizing methods {summarizing}; pass --method")
        method = summarizing[0]
    record = runlog.record(method, seed)
    if not record.summaries_evolution:
        raise ValueError(f"({method!r}, seed {seed}) logged no summaries")
    return record


def build_workload(
    record: RunRecord, step: int, texts: list[str], X: np.ndarray, force_context_length: int
) -> list[Cluster]:
    """The clusters the logged run actually sent to its summarizer at summary step `step`, in cluster order."""
    inputs = step_inputs(record, step, texts, X, force_context_length)
    return [inputs[c] for c in sent_clusters(record, step, texts, X, force_context_length)]


@dataclass
class RepeatResult:
    repeat: int
    summaries: list[str]
    """One per workload cluster, in workload order."""
    seconds: float
    """Wall time of all batches (the sum of `batch_seconds`)."""
    batch_seconds: list[float]
    cost: dict[str, int]
    """Change of `Summarizer.stats()` over this repeat (tokens, truncation); `{}` for summarizers with none."""
    peak_gpu_gb: tuple[float | None, float | None] = (None, None)
    """`(max allocated, max reserved)` during this repeat, `(None, None)` without CUDA."""


@dataclass
class WorkloadRun:
    cluster_ids: list[int]
    repeats: list[RepeatResult] = field(default_factory=list)


def summarize_workload(
    summarizer: Summarizer,
    clusters: list[Cluster],
    repeat: int = 1,
    batch_size: int | None = None,
    warmup: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> WorkloadRun:
    """Summarize `clusters` `repeat` times and time every batch.

    The batches are cut here, `batch_size` clusters per `summarize_batch` call (default: the summarizer's own
    `batch_size`, else everything in one call). `HFSummarizer.summarize_batch` cuts its input the same way, so
    this is the call pattern it would have made, with a clock around each generate. `warmup` first runs the first
    batch once, unrecorded, so repeat 0 does not pay one-off CUDA start-up.
    """
    size = batch_size or getattr(summarizer, "batch_size", None) or len(clusters) or 1
    chunks = [clusters[i : i + size] for i in range(0, len(clusters), size)]
    if warmup and chunks:
        summarizer.summarize_batch(chunks[0])

    run = WorkloadRun(cluster_ids=[c.index for c in clusters])
    for r in range(repeat):
        before = summarizer.stats()
        reset_peak_gpu_memory()
        summaries: list[str] = []
        batch_seconds: list[float] = []
        for chunk in chunks:
            start = clock()
            summaries.extend(summarizer.summarize_batch(chunk))
            batch_seconds.append(clock() - start)
        after = summarizer.stats()
        run.repeats.append(
            RepeatResult(
                repeat=r,
                summaries=summaries,
                seconds=sum(batch_seconds),
                batch_seconds=batch_seconds,
                cost={key: value - before.get(key, 0) for key, value in after.items()},
                peak_gpu_gb=peak_gpu_memory_gb(),
            )
        )
    return run


def workload_surface(summaries: list[str]) -> dict:
    """The §2.4 surface counts of one step's summaries (`kslimeans.surface` rules)."""
    c = surface_counts([[summaries]])
    groups = duplicate_groups(summaries)
    n = c.n
    return {
        "n": n,
        "mentions": c.mentions,
        "mentions_share": c.mentions / n if n else 0.0,
        "duplicate_groups": len(groups),
        "clusters_in_duplicate_groups": sum(len(g) for g in groups),
        "largest_duplicate_group": c.final_largest_group,
        "mean_words": c.mean_words,
        "ends_with_question": c.ends_with_question,
        "ends_with_question_share": c.ends_with_question / n if n else 0.0,
    }


def write_workload(
    out_dir: Path, run: WorkloadRun, logged_summaries: list[str | None] | None = None
) -> dict:
    """Write `outputs.jsonl`, `cost.jsonl` and `surface.json` into `out_dir`; returns the surface dict.

    The surface counts are those of repeat 0. `identical_across_repeats` says whether decoding was
    deterministic over the repeats, and `logged_match` the share of repeat 0's outputs equal to the logged run's
    summaries of the same clusters (`logged_summaries` is the step's full list, indexed by cluster id).
    """
    with open(out_dir / "outputs.jsonl", "w") as f:
        for rep in run.repeats:
            for cluster, summary in zip(run.cluster_ids, rep.summaries):
                f.write(json.dumps({"repeat": rep.repeat, "cluster": cluster, "summary": summary}) + "\n")
    with open(out_dir / "cost.jsonl", "w") as f:
        for rep in run.repeats:
            row = {
                "repeat": rep.repeat,
                "n_calls": len(rep.summaries),
                "n_batches": len(rep.batch_seconds),
                "seconds": rep.seconds,
                "batch_seconds": rep.batch_seconds,
                **rep.cost,
                "peak_gpu_mem_gb": rep.peak_gpu_gb[0],
                "peak_gpu_reserved_gb": rep.peak_gpu_gb[1],
            }
            f.write(json.dumps(row) + "\n")

    first = run.repeats[0].summaries if run.repeats else []
    surface = workload_surface(first)
    surface["repeats"] = len(run.repeats)
    surface["identical_across_repeats"] = all(rep.summaries == first for rep in run.repeats)
    if logged_summaries is not None and first:
        same = sum(s == logged_summaries[c] for c, s in zip(run.cluster_ids, first))
        surface["logged_match"] = same / len(first)
    with open(out_dir / "surface.json", "w") as f:
        json.dump(surface, f, indent=2)
        f.write("\n")
    return surface


def load_workload_dir(path: str | Path) -> dict:
    """What `write_workload` and the CLI left in `path`: `config`, `surface` and `cost` (list of rows)."""
    path = Path(path)
    with open(path / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(path / "surface.json") as f:
        surface = json.load(f)
    with open(path / "cost.jsonl") as f:
        cost = [json.loads(line) for line in f if line.strip()]
    return {"path": path, "config": config, "surface": surface, "cost": cost}
