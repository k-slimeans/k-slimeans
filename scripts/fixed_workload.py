"""Re-summarize one fixed workload from a logged run: the inputs of one summary step, with any summarizer setup.

The workload is what `RUN_DIR` sent to its summarizer at summary step `--step` of `--method`'s seed `--seed`
(default seed 0, step 0: the first summary step, written from the step-0 partition every method shares for a
seed). It is rebuilt from the run's logged partition and cached embeddings (nothing is encoded), then summarized
`--repeat` times with the logged run's summarizer settings, changed by `--summarizer-config` (a YAML with a
`summarizer:` block, or just the block) and `--set key=value` (summarizer keys: `model_id`, `prompt`,
`batch_size`, `max_new_tokens`, `max_input_tokens`, ...). Without either, the logged settings are reused, so the
dataset's own prompt applies. Real models run: this needs the GPU, checked and monitored as in `run_offline` (`gpu_check`, from the
`--summarizer-config` file or the logged config; `gpu.json` and `gpu_samples.csv` land in the output directory).

Writes `<results-dir>/<name>/<stamp>/` (see `kslimeans.workload`): `config.yaml`, `environment.json`,
`outputs.jsonl`, `cost.jsonl` (seconds per repeat and per batch, tokens, truncation, peak GPU memory) and
`surface.json` (share mentioning the cluster, duplicate groups, mean words, share ending in "?").

Usage:

    python scripts/fixed_workload.py results/clinc150_qwen3-4b/<stamp> --name old_4b
    python scripts/fixed_workload.py results/clinc150_qwen3-4b/<stamp> --name v2_4b \\
        --summarizer-config configs/clinc150_prompt_v2.yaml --set model_id=Qwen/Qwen3-4B --set batch_size=1
    python scripts/fixed_workload.py RUN_DIR --name time_bs4 --set batch_size=4 --repeat 3   # 1 warm-up pass by default

Compare the output directories with `scripts/compare_workloads.py`.
"""

from __future__ import annotations

import argparse
import copy

import yaml

from kslimeans.config import apply_overrides
from kslimeans.environment import collect_environment
from kslimeans.gpu_monitor import GpuSession
from kslimeans.results import RunDir
from kslimeans.runlog import load_run, load_run_dataset, load_run_embedder
from kslimeans.summarizers import build_summarizer
from kslimeans.workload import build_workload, pick_record, summarize_workload, write_workload


def resolve_summarizer(logged_config: dict, config_path: str | None, overrides: list[str]) -> tuple[dict, dict | None]:
    """The summarizer config to run and the `gpu_check` spec of the config it came from (or None).

    From `config_path` (a full config or just a summarizer block) if given, else the logged run's config.
    """
    if config_path is None:
        full, block = logged_config, logged_config["summarizer"]
    else:
        with open(config_path) as f:
            full = yaml.safe_load(f)
        block = full["summarizer"] if "summarizer" in full else full
    return apply_overrides(copy.deepcopy(block), overrides), full.get("gpu_check")


def embedder_name(runlog, ds) -> str:
    """The embedder's resolved name, as `experiment.py` logs it: a named spec is its own name (no model is
    loaded for it); a corpus-fitted spec (TF-IDF) is built to read the name that encodes its settings."""
    spec = runlog.config["embedder"]
    if isinstance(spec, str):
        return spec
    embedder = load_run_embedder(runlog, ds)
    return getattr(embedder, "name", type(embedder).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", metavar="RUN_DIR", help="logged run (results/<name>/<timestamp>)")
    parser.add_argument("--name", required=True, help="output goes to <results-dir>/<name>/<timestamp>")
    parser.add_argument("--method", help="method whose partitions to use (default: the run's only summarizing one)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=0, help="summary step (default 0: the first)")
    parser.add_argument("--summarizer-config", metavar="YAML", help="summarizer settings; default: the logged run's")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value",
                        help="override a summarizer key, e.g. --set batch_size=1 (repeatable)")
    parser.add_argument("--repeat", type=int, default=1, help="summarize the workload this many times (default 1)")
    parser.add_argument("--warmup", type=int, metavar="N",
                        help="untimed passes over the first batch before the repeats (default: 1 if --repeat > 1, "
                             "else 0, so a single-repeat pilot pays for no warm-up)")
    parser.add_argument("--cache-dir", help="embedding cache (default: the logged run's cache_dir)")
    parser.add_argument("--results-dir", default="results/fixed_workload")
    args = parser.parse_args()
    warmup = args.warmup if args.warmup is not None else int(args.repeat > 1)

    runlog = load_run(args.run_dir)
    cfg = runlog.config
    summarizer_cfg, gpu_spec = resolve_summarizer(cfg, args.summarizer_config, args.overrides)
    if summarizer_cfg.get("type") != "hf":
        parser.error("only hf summarizers are supported (the workload carries no sentences)")
    if args.summarizer_config and not any(o.partition("=")[0] == "batch_size" for o in args.overrides):
        parser.error(
            f"--summarizer-config needs an explicit --set batch_size=N (its file says {summarizer_cfg.get('batch_size')}"
            "): a batch size that is too large for the model overflows the 12 GB card, and under WSL2 that hangs "
            "the job instead of raising (4B: 1, 1.7B: 4, 0.6B: 8)"
        )
    print(f"Summarizer: model_id={summarizer_cfg.get('model_id')} batch_size={summarizer_cfg.get('batch_size')}")

    record = pick_record(runlog, args.method, args.seed)
    ds, X = load_run_dataset(runlog, cache_dir=args.cache_dir)
    force_context_length = cfg.get("force_context_length", 0)
    clusters = build_workload(record, args.step, ds.texts, X, force_context_length)

    config = {
        "name": args.name,
        "kind": "fixed_workload",
        "source": {
            "run_dir": str(args.run_dir),
            "method": record.method,
            "seed": record.seed,
            "step": args.step,
            "n_clusters": len(record.summaries_evolution[args.step]),
            "n_sent": len(clusters),
            "step0_sha1": record.metrics.get("step0_sha1"),
        },
        "dataset": cfg["dataset"],
        "embedder": cfg["embedder"],
        "force_context_length": force_context_length,
        "summarizer": summarizer_cfg,
        "prompt": summarizer_cfg.get("prompt") or ds.prompt,
        "text_type": summarizer_cfg.get("text_type") or ds.text_type,
        "repeat": args.repeat,
        "warmup": warmup,
    }
    if warmup == 0:
        config["note"] = "no warm-up pass: repeat 0 includes one-off CUDA start-up (kernel loading, allocator growth)"

    # Like run_offline: the `gpu_check` policy (idle check on entry, before the model loads; monitor from attach).
    with GpuSession(gpu_spec) as gpu:
        summarizer = build_summarizer(summarizer_cfg, prompt=ds.prompt, text_type=ds.text_type)
        run = RunDir(args.results_dir, args.name, config)
        run.log_environment(collect_environment(summarizer, embedder_name(runlog, ds), ds))
        gpu.attach(run.path)
        result = summarize_workload(summarizer, clusters, repeat=args.repeat, warmup=warmup)
    surface = write_workload(run.path, result, record.summaries_evolution[args.step])

    print(f"Run directory: {run.path}")
    if warmup == 0:
        print(f"Note: {config['note']}")
    print(f"{len(clusters)} clusters x {args.repeat} repeat(s); "
          f"{surface['mentions']}/{surface['n']} mention the cluster, "
          f"{surface['duplicate_groups']} duplicate group(s), mean {surface['mean_words']:.1f} words")


if __name__ == "__main__":
    main()
