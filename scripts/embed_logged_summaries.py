"""Embed every logged summary step of run directories, for the replacement diagnostic and its replays.

Each summary step is encoded as `summary_kmeans` encodes it: one `encode` call over all k summaries of the step, so
the batch composition matches the original run. The result (`.npz`, keyed by run name, method, seed and step, with
the device and library versions in its metadata) is what `scripts/replacement_diagnostic.py --summary-embeddings`
reads; nothing else encodes.

Usage:

    CUDA_VISIBLE_DEVICES="" python scripts/embed_logged_summaries.py results/clinc150_lsa/<stamp> ... \
        --out results/replacement_diagnostic/summary_embeddings_cpu.npz --device cpu
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import sys
from pathlib import Path

import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

from kslimeans.replay import SummaryEmbeddings
from kslimeans.runlog import embed_summaries, load_run, load_run_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--out", required=True, help="output .npz")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    arrays = {}
    runs = []
    embedder_name = None
    for run_dir in args.run_dirs:
        runlog = load_run(run_dir)
        name = runlog.config["name"]
        embedder_name = embedder_name or runlog.config["embedder"]
        if runlog.config["embedder"] != embedder_name:
            sys.exit(f"{run_dir} uses embedder {runlog.config['embedder']!r}, others use {embedder_name!r}")
        embedder = load_run_embedder(runlog)
        embedder._model.to(args.device)
        for (method, seed), record in sorted(runlog.records.items()):
            for j, summaries in enumerate(record.summaries_evolution):
                arrays[(name, method, seed, j)] = embed_summaries(embedder, summaries)
        runs.append({"name": name, "path": str(runlog.path)})
        print(f"{name}: {len(runlog.records)} records embedded on {args.device}", file=sys.stderr)

    meta = {
        "device": args.device,
        "embedder": embedder_name,
        "encode_batch_size": inspect.signature(SentenceTransformer.encode).parameters["batch_size"].default,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    SummaryEmbeddings(arrays, meta).save(args.out)
    print(f"wrote {len(arrays)} steps to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
