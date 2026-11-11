"""Estimate the summarizer calls and GPU-hours of a streaming k-SLMmeans run, before running it (#48).

`mini_batch_summary_kmeans` runs a whole `summary_kmeans` schedule on every batch, so the number of
summarizer calls is set by four numbers: the batch size, `k`, the summary steps per batch and how often
a batch is summarized (`summary_every`). This script turns those into calls and GPU-hours:

    batches          = ceil(n_rows / max_batch_size)
    summarized       = ceil(batches / summary_every)
    calls (upper)    = summarized * k * n_summaries      (a memo-cache hit or an empty cluster is free)
    input tokens     = members * (mean abstract tokens + 1) + prompt overhead, per call
    GPU seconds      = calls * input tokens * seconds per input token

Seconds per input token are derived from the CLINC150 cost table (`docs/results/clinc150_summarizers.md`
section 3): `summarizer_seconds / input_tokens` of each Qwen3 size at its logged batch size (8 / 4 / 1).
Those runs shared the GPU and ran about 1.7x slow, so the default divides by `--gpu-slowdown 1.7`; pass
`--gpu-slowdown 1` for the as-logged (pessimistic) numbers. The prompt-token inputs (mean abstract
tokens, prompt overhead) are measured on a sample of the real abstracts with the Qwen3 tokenizer:

    HF_HUB_OFFLINE=1 python scripts/estimate_stream_cost.py --measure --lsa --save-stats docs/results/arxiv_token_stats.json

and read back (no tokenizer, no data needed) by every later run:

    python scripts/estimate_stream_cost.py --stats docs/results/arxiv_token_stats.json

Nothing here loads a model or touches a GPU. The CPU k-means term is an upper bound at an assumed
throughput (`--cpu-gflops`), not a measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROWS = 3_035_644  # arXiv abstracts after filtering (#17)
DEFAULT_K = 171  # primary-category labels
DEFAULT_STATS = Path(__file__).resolve().parent.parent / "docs" / "results" / "arxiv_token_stats.json"

# Per-seed means over seeds 0-4 of the CLINC150 Qwen3 runs (docs/results/clinc150_summarizers.md, section 3):
# summarizer seconds, input tokens, calls, and the batch size each size ran at.
CLINC150_LOGGED = {
    "Qwen3-0.6B": {"batch_size": 8, "summarizer_s": 895.8, "input_tokens": 1_113_037, "calls": 707},
    "Qwen3-1.7B": {"batch_size": 4, "summarizer_s": 1238.6, "input_tokens": 1_101_755, "calls": 702},
    "Qwen3-4B": {"batch_size": 1, "summarizer_s": 1855.0, "input_tokens": 1_120_190, "calls": 714},
}
MODEL_ORDER = tuple(CLINC150_LOGGED)
DEFAULT_GPU_SLOWDOWN = 1.7  # the CLINC150 runs shared the GPU; see section 3 of the report
DEFAULT_CPU_GFLOPS = 30.0  # assumed effective CPU throughput of sklearn's Lloyd step (not measured)
EMBED_DIM = 384
MAX_INPUT_TOKENS = 4096  # HFSummarizer default; prompts are trimmed to this


@dataclass(frozen=True)
class ScheduleOption:
    """One way to run the streaming method: batch size, summary steps per batch, batches summarized."""

    label: str
    max_batch_size: int
    n_summaries: int
    summary_every: int = 1


@dataclass(frozen=True)
class TokenStats:
    """Prompt-size inputs, measured on a sample of the abstracts with the Qwen3 tokenizer."""

    mean_abstract_tokens: float
    prompt_overhead_tokens: float  # chat template + system prompt + task prompt + prefill, no abstracts

    @classmethod
    def from_json(cls, path: str | Path) -> "TokenStats":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(data["mean_abstract_tokens"]), float(data["prompt_overhead_tokens"]))


def n_batches(n_rows: int, max_batch_size: int) -> int:
    """Batches `mini_batch_summary_kmeans` splits `n_rows` into."""
    return math.ceil(n_rows / max_batch_size)


def summarized_batches(batches: int, summary_every: int) -> int:
    """Batches 0, m, 2m, ... of `batches` are summarized (the first always is)."""
    return math.ceil(batches / summary_every)


def max_calls(n_rows: int, k: int, option: ScheduleOption) -> int:
    """Upper bound on summarizer calls: every cluster of every summary step of every summarized batch.

    Memo-cache hits (a cluster whose members did not change since an earlier step of the same batch) and
    empty clusters are not calls, so the real count is a little lower.
    """
    batches = n_batches(n_rows, option.max_batch_size)
    return summarized_batches(batches, option.summary_every) * k * option.n_summaries


def tokens_per_call(stats: TokenStats, k: int, max_batch_size: int, force_context_length: int) -> float:
    """Mean input tokens of one call: the prompt overhead plus the abstracts sent for a cluster.

    A cluster sends `force_context_length` members (all of them if it is 0 or the cluster is smaller;
    the mean cluster has `max_batch_size / k`), one newline each, and the prompt is trimmed to
    `MAX_INPUT_TOKENS`. Using the mean cluster size ignores that small clusters send fewer, so this leans
    high for small batches.
    """
    members = max_batch_size / k
    if force_context_length > 0:
        members = min(members, force_context_length)
    return min(stats.prompt_overhead_tokens + members * (stats.mean_abstract_tokens + 1), MAX_INPUT_TOKENS)


def seconds_per_input_token(model: str, gpu_slowdown: float = DEFAULT_GPU_SLOWDOWN) -> float:
    """Summarizer seconds per input token for `model`, from its CLINC150 run, deflated by `gpu_slowdown`."""
    logged = CLINC150_LOGGED[model]
    return logged["summarizer_s"] / logged["input_tokens"] / gpu_slowdown


def gpu_hours(calls: float, tokens: float, model: str, gpu_slowdown: float = DEFAULT_GPU_SLOWDOWN) -> float:
    """GPU-hours of `calls` summarizer calls of `tokens` input tokens each."""
    return calls * tokens * seconds_per_input_token(model, gpu_slowdown) / 3600


def kmeans_cpu_hours(n_rows: int, k: int, max_iter: int, cpu_gflops: float = DEFAULT_CPU_GFLOPS) -> float:
    """Upper bound on the CPU k-means time: every batch runs all `max_iter` Lloyd iterations.

    Each iteration costs 2 * rows * k * dim flops, so over all batches the total depends on the number of
    rows, not on how they are batched. Warm-started batches stop on `tol` well before `max_iter`.
    """
    return max_iter * n_rows * k * EMBED_DIM * 2 / (cpu_gflops * 1e9) / 3600


def default_options() -> list[ScheduleOption]:
    """The schedule options compared in the status report: (a) current, (b) one step, (c) every m-th, (d) larger."""
    return [
        ScheduleOption("(a) current: 10k batches, 5 steps, every batch", 10_000, 5, 1),
        ScheduleOption("(b) one summary step per batch: 10k batches", 10_000, 1, 1),
        ScheduleOption("(c1) 10k batches, 5 steps, every 20th batch", 10_000, 5, 20),
        ScheduleOption("(c2) 10k batches, 1 step, every 4th batch", 10_000, 1, 4),
        ScheduleOption("(d1) 100k batches, 5 steps, every batch", 100_000, 5, 1),
        ScheduleOption("(d2) 100k batches, 2 steps, every batch", 100_000, 2, 1),
        ScheduleOption("(d3) 100k batches, 1 step, every batch", 100_000, 1, 1),
        ScheduleOption("(d4) 150k batches, 2 steps, every batch", 150_000, 2, 1),
        ScheduleOption("(d5) 250k batches, 2 steps, every batch", 250_000, 2, 1),
    ]


def parse_option(spec: str) -> ScheduleOption:
    """`BATCH:STEPS` or `BATCH:STEPS:EVERY`, e.g. `150000:2` or `10000:1:4`."""
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"option must be BATCH:STEPS[:EVERY] with positive integers, got {spec!r}")
    batch, steps, every = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) == 3 else 1
    every_text = "" if every == 1 else f", every {every}th batch"
    return ScheduleOption(f"{batch:,} batches, {steps} step{'s' * (steps != 1)}{every_text}", batch, steps, every)


def estimate_table(
    options: list[ScheduleOption],
    n_rows: int,
    k: int,
    stats: TokenStats,
    force_context_length: int,
    gpu_slowdown: float = DEFAULT_GPU_SLOWDOWN,
) -> list[dict]:
    """One dict per option: batches, summarized batches, calls (upper bound), tokens per call, GPU-hours per size."""
    rows = []
    for option in options:
        batches = n_batches(n_rows, option.max_batch_size)
        calls = max_calls(n_rows, k, option)
        tokens = tokens_per_call(stats, k, option.max_batch_size, force_context_length)
        rows.append(
            {
                "label": option.label,
                "batches": batches,
                "summarized_batches": summarized_batches(batches, option.summary_every),
                "calls": calls,
                "tokens_per_call": tokens,
                "gpu_hours": {m: gpu_hours(calls, tokens, m, gpu_slowdown) for m in MODEL_ORDER},
            }
        )
    return rows


def format_table(rows: list[dict], budget_hours: float | None = None) -> str:
    """Markdown table of `estimate_table`; a GPU-hours cell within `budget_hours` is bold."""
    header = ["Option", "batches", "summarized", "calls (max)", "input tok/call"] + [
        f"{m} GPU-h" for m in MODEL_ORDER
    ]
    lines = ["| " + " | ".join(header) + " |", "| --- " + "| ---: " * (len(header) - 1) + "|"]
    for row in rows:
        cells = []
        for m in MODEL_ORDER:
            hours = row["gpu_hours"][m]
            text = f"{hours:.1f}"
            cells.append(f"**{text}**" if budget_hours is not None and hours <= budget_hours else text)
        lines.append(
            f"| {row['label']} | {row['batches']:,} | {row['summarized_batches']:,} | {row['calls']:,} "
            f"| {row['tokens_per_call']:,.0f} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def lsa_projection(n_rows: int, lsa: dict, embed_rows_per_s: float, spacy_procs: int = 1) -> dict:
    """Project the LSA (k-NLPmeans) preprocessing to `n_rows` abstracts from a sample's `lsa` stats.

    Sentence segmentation is `n_rows / (spacy_docs_per_s * procs)`. The sentence embedding time is the
    per-token estimate: abstracts embed at `embed_rows_per_s` (about 725 rows/s on the RTX 3060, #17), so
    the token throughput is that times the mean MiniLM tokens per abstract, and the sentences carry
    `minilm_sentence_tokens_per_abstract` tokens per abstract. The per-row figure (every sentence costing
    as much as a whole abstract) is the pessimistic bound. Memory is the float32 sentence embeddings
    (`SentenceCorpus.embeddings`) and the sentence strings.
    """
    sentences = n_rows * lsa["sentences_per_abstract"]
    tokens_per_s = embed_rows_per_s * lsa["minilm_abstract_tokens"]
    return {
        "n_sentences": sentences,
        "spacy_hours": n_rows / (lsa["spacy_docs_per_s"] * spacy_procs) / 3600,
        "embed_hours_by_tokens": n_rows * lsa["minilm_sentence_tokens_per_abstract"] / tokens_per_s / 3600,
        "embed_hours_by_rows": sentences / embed_rows_per_s / 3600,
        "embeddings_gb": sentences * EMBED_DIM * 4 / 1e9,
        "sentence_text_gb": sentences * lsa["mean_sentence_chars"] / 1e9,
    }


def measure_sample(sample_size: int, seed: int, force_context_length: int, with_lsa: bool) -> dict:
    """Token (and, with `with_lsa`, sentence) statistics of `sample_size` random abstracts of the 200k subsample.

    Needs the Qwen3 tokenizer (and MiniLM's tokenizer and spaCy `en_core_web_sm` for LSA) in the local
    Hugging Face cache; run it with `HF_HUB_OFFLINE=1`. The prompt is rendered exactly as `HFSummarizer`
    sends it (`fit_and_render`, arXiv prompt, `Topic:` prefill, thinking off).
    """
    import time

    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from kslimeans.data import _ARXIV_PROMPT, _ARXIV_TEXT_TYPE, DATA_DIR
    from kslimeans.summarizers.hf import DEFAULT_SYSTEM_PROMPT, fit_and_render

    table = pq.read_table(DATA_DIR / "arxiv" / "arxiv_sub200k.parquet", columns=["text"])
    texts = table.column("text").to_pylist()
    rng = np.random.default_rng(seed)
    sample = [texts[i] for i in rng.choice(len(texts), size=sample_size, replace=False)]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
    n_tokens = np.array([len(ids) for ids in tokenizer(sample, add_special_tokens=False)["input_ids"]])

    def render(members: list[str]) -> tuple[str, int, bool]:
        return fit_and_render(
            tokenizer,
            members,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            prompt=_ARXIV_PROMPT,
            text_type=_ARXIV_TEXT_TYPE,
            chat_template_kwargs={"enable_thinking": False},
            fold_system=False,
            max_input_tokens=MAX_INPUT_TOKENS,
        )

    def count(rendered: str) -> int:
        return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])

    overhead = count(render([])[0])
    n_prompts = 300
    members = force_context_length or 10
    draws = [render([sample[i] for i in rng.choice(sample_size, size=members, replace=False)]) for _ in range(n_prompts)]
    prompt_tokens = np.array([count(rendered) for rendered, _, _ in draws])

    stats = {
        "sample_size": sample_size,
        "seed": seed,
        "source": "arxiv_sub200k.parquet",
        "tokenizer": "Qwen/Qwen3-0.6B (shared by the 1.7B and 4B)",
        "mean_words": float(np.mean([len(t.split()) for t in sample])),
        "mean_abstract_tokens": float(n_tokens.mean()),
        "abstract_tokens_p5_p50_p95_p99": [float(x) for x in np.percentile(n_tokens, [5, 50, 95, 99])],
        "prompt_overhead_tokens": float(overhead),
        "check_prompts": {
            "n": n_prompts,
            "members": members,
            "mean_tokens": float(prompt_tokens.mean()),
            "max_tokens": int(prompt_tokens.max()),
            "n_truncated": int(sum(1 for _, _, truncated in draws if truncated)),
        },
    }

    if with_lsa:
        import spacy

        from kslimeans.sentences import build_sentence_corpus

        minilm = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        spacy.load("en_core_web_sm")  # fail early if missing; also warms the import
        t0 = time.perf_counter()
        sentences, index_map = build_sentence_corpus(sample)
        seconds = time.perf_counter() - t0
        sentence_tokens = [len(ids) for ids in minilm(sentences, truncation=True, max_length=256)["input_ids"]]
        abstract_tokens = [len(ids) for ids in minilm(sample, truncation=True, max_length=256)["input_ids"]]
        per_doc = np.bincount([d for d, _ in index_map], minlength=sample_size)
        stats["lsa"] = {
            "spacy_model": "en_core_web_sm, nlp.pipe(disable=[ner, lemmatizer]), 1 process",
            "spacy_docs_per_s": sample_size / seconds,
            "sentences_per_abstract": float(per_doc.mean()),
            "sentences_per_abstract_p5_p50_p95": [float(x) for x in np.percentile(per_doc, [5, 50, 95])],
            "mean_sentence_chars": float(np.mean([len(s) for s in sentences])),
            "minilm_sentence_tokens_per_abstract": float(np.sum(sentence_tokens) / sample_size),
            "minilm_abstract_tokens": float(np.mean(abstract_tokens)),
        }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--force-context-length", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=120, help="Lloyd iterations per batch, for the CPU bound")
    parser.add_argument("--gpu-slowdown", type=float, default=DEFAULT_GPU_SLOWDOWN)
    parser.add_argument("--cpu-gflops", type=float, default=DEFAULT_CPU_GFLOPS)
    parser.add_argument("--budget-hours", type=float, default=4.0, help="GPU-hours within this are bold")
    parser.add_argument(
        "--option", action="append", default=[], metavar="BATCH:STEPS[:EVERY]",
        help="a schedule option to tabulate instead of the default set; repeatable",
    )
    parser.add_argument("--stats", default=str(DEFAULT_STATS), help="JSON written by --save-stats")
    parser.add_argument("--measure", action="store_true", help="measure the token stats on a sample (needs data + tokenizer)")
    parser.add_argument("--lsa", action="store_true", help="with --measure: also spaCy sentences per abstract and throughput")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-stats", help="with --measure: write the measured stats to this JSON file")
    parser.add_argument("--embed-rows-per-s", type=float, default=725.0, help="MiniLM abstracts/s (manifest.json)")
    parser.add_argument("--spacy-procs", type=int, default=1)
    args = parser.parse_args()

    if args.measure:
        measured = measure_sample(args.sample_size, args.seed, args.force_context_length, args.lsa)
        print(json.dumps(measured, indent=2))
        if args.save_stats:
            Path(args.save_stats).write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
        stats = TokenStats(measured["mean_abstract_tokens"], measured["prompt_overhead_tokens"])
        lsa = measured.get("lsa")
    else:
        stats = TokenStats.from_json(args.stats)
        lsa = json.loads(Path(args.stats).read_text(encoding="utf-8")).get("lsa")

    print(
        f"\nn_rows={args.n_rows:,} k={args.k} force_context_length={args.force_context_length}; "
        f"{stats.mean_abstract_tokens:.0f} tokens/abstract + {stats.prompt_overhead_tokens:.0f} prompt overhead; "
        f"GPU slowdown divisor {args.gpu_slowdown}\n"
    )
    options = [parse_option(o) for o in args.option] or default_options()
    rows = estimate_table(options, args.n_rows, args.k, stats, args.force_context_length, args.gpu_slowdown)
    print(format_table(rows, args.budget_hours))
    print(
        f"\nCPU k-means upper bound (all {args.max_iter} Lloyd iterations on every batch, "
        f"{args.cpu_gflops:g} GFLOP/s assumed): "
        f"{kmeans_cpu_hours(args.n_rows, args.k, args.max_iter, args.cpu_gflops):.2f} h"
    )
    print("\nSeconds per input token (idle-GPU equivalent): " + ", ".join(
        f"{m} {seconds_per_input_token(m, args.gpu_slowdown) * 1000:.3f} ms" for m in MODEL_ORDER
    ))
    if lsa:
        proj = lsa_projection(args.n_rows, lsa, args.embed_rows_per_s, args.spacy_procs)
        print(f"\nLSA preprocessing at {args.n_rows:,} abstracts: " + json.dumps({k: round(v, 2) for k, v in proj.items()}))


if __name__ == "__main__":
    main()
