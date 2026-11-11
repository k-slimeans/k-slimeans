# Cost counter verification (#35, part 1)

This checks what the logged cost counters count, so the cost section of `clinc150_summarizers.md` (§3) can be built on. It uses no GPU and loads no model: the four logged CLINC150 runs are rebuilt from their logged partitions and summaries, and prompts and outputs are re-tokenized with the Qwen3 tokenizer from the local Hugging Face cache (`local_files_only`; the three sizes share one tokenizer).

- **Part 2** (a GPU idle check and a monitor logged into each run directory) and **part 3** (re-timing at a fixed batch size, and measuring the inflation) are separate work. Nothing here re-times anything: the timing findings below are about what the existing seconds cover, not about how fast the model is.
- **Result: no counting bug.** Every counter that can be rebuilt matched exactly, except the output tokens, which match within 0.1% (cause not isolated, see below). Every number in §3 of `clinc150_summarizers.md` equals the mean and standard deviation recomputed from the runs' `metrics.jsonl`; the report prints that recomputation, formatted like §3, under each method, so the comparison can be repeated. No published number changes.

## 1. What was checked

`scripts/verify_cost.py RUN_DIR... --cache-dir CACHE` runs the checks in `src/kslimeans/costcheck.py` and writes a markdown report (pass/fail per check, with the maximum discrepancy):

```
HF_HUB_OFFLINE=1 python scripts/verify_cost.py results/clinc150_{lsa,qwen3-0.6b,qwen3-1.7b,qwen3-4b}/<stamp> \
    --cache-dir <main checkout>/cache --out report.md
```

It took 48 s for the four runs (about 16 s per run with a tokenizer, under 1 s for LSA), all on the CPU. Indexing follows `runlog`: summary step `j` is trace row `summary_source_steps[j] + 1`.

| Check | How |
| --- | --- |
| Calls per step and per run | The logged `n_calls` against `len(sent_clusters(...))`: clusters that are non-empty and not memo-cache hits. Trace rows against the per-run total in `metrics.jsonl`; the total may exceed the trace only by a final step that stopped on `tol` (none of these runs did). |
| Input tokens, truncated clusters, dropped texts | Each sent prompt is rebuilt with the run's prompt (the dataset's), `text_type` prefill, system prompt, chat-template kwargs (thinking off), `max_words_per_text` (when the config sets it, applied before the budget trim, as in the live call) and `max_input_tokens`, through `hf.fit_and_render`, the function `HFSummarizer._render` itself now calls. The prompt is re-tokenized as `HFSummarizer` counts it (`add_special_tokens=False`). All 25 steps of every run were checked, not a sample. |
| Output tokens | Each sent summary is re-tokenized. The logged count is "up to and including the first EOS" of the raw generation, which cannot be recovered from decoded, stripped text, so the expectation is the re-tokenized length plus one EOS per summary. |
| `embed_*` counters | When a run logs the embedder's counters (TF-IDF runs, #54), each must equal the sum of the trace rows. Runs without them skip this check. |
| Empty summaries | The logged `n_empty_summaries` against the blank summaries among the clusters sent. |
| Seconds | Trace sums against the per-run `summarizer_seconds`; `summarizer_seconds <= runtime_s`; the remainder `runtime_s - summarizer_seconds`. |

## 2. Results

All checks pass on all four runs (5 seeds each, 25 summary steps per run). Maximum discrepancies:

| Run | calls (per step / per run) | input tokens | truncated / dropped | empty | output tokens, per seed (logged minus expected) | output, largest per-step difference | trace sums = metrics |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| LSA | 0 / 0 | n/a | n/a | 0 | n/a | n/a | exact |
| Qwen3-0.6B (batch 8) | 0 / 0 | 0 | 0 / 0 | 0 | +6, +3, -9, -3, -5 (of ~10,200; max 0.085%) | 5 tokens | exact |
| Qwen3-1.7B (batch 4) | 0 / 0 | 0 | 0 / 0 | 0 | 0, 0, -1, 0, -4 (of ~18,300; max 0.022%) | 2 tokens | exact |
| Qwen3-4B (batch 1) | 0 / 0 | 0 | 0 / 0 | 0 | +5, +15, +6, +13, -1 (of ~15,600; max 0.096%) | 6 tokens | exact |

- **Calls:** the logged per-step `n_calls` equals the number of clusters sent in all 100 steps (including LSA), and the per-run totals equal both the trace sums and the rebuild. The empty-cluster and memo rules in §3 of the published report are what the code does.
- **Input tokens:** equal to the token, at batch sizes 8, 4 and 1. Left padding therefore does not leak into the count (`attention_mask.sum()` counts real tokens only). Truncated clusters and dropped texts also match exactly, so the seeded member trim (#28) is reproduced.
- **Output tokens:** the difference is at most 15 tokens per seed (0.1%). Its sign is mixed for the 0.6B and 1.7B and mostly positive for the 4B (+5, +15, +6, +13, −1); the cause is not isolated. Two hypotheses, which the decoded text alone cannot test:
  - A generation that ran to `max_new_tokens` (64) has no EOS, so the logged count would be one lower than the expectation. By re-tokenized length, 17–22 summaries per seed for the 0.6B, 0–5 for the 1.7B and 10–18 for the 4B reach 63 tokens or more.
  - `decode(skip_special_tokens=True)` followed by `.strip()` may drop leading and trailing whitespace tokens (the prefill continues after `Sentence:`), and re-encoding a decoded string need not reproduce the generated token boundaries, so the re-tokenized length could be either side.
  - EOS counting itself is as documented: `count_generated_tokens` counts up to and including the first EOS in `generation_config.eos_token_id` (`<|im_end|>` and `<|endoftext|>`), and the padding after it is excluded, so batched runs are not inflated.
  - **What the output check guarantees.** It fails when a seed's logged output tokens differ from the expectation by more than 0.3% (`--output-tol`, default 0.003), about 30–50 tokens per seed here, e.g. one token missing on every 15th to 25th call. A counter that missed the EOS on every call would be off by 4–7% and is caught by a wide margin (a test covers it). A drift below 0.3% passes.
- **Seconds:** per-step `summarizer_seconds` sum to the per-run value within 1e-6 s, and `summarizer_seconds <= runtime_s` everywhere.

### Total time minus summarizer time

`total s` in the published table is `runtime_s`: wall time around the whole `summary_kmeans` call. The remainder after `summarizer_seconds` is small and about the same for every model:

| Run | `runtime_s - summarizer_seconds` per seed |
| --- | --- |
| LSA | 7.6–9.9 s |
| Qwen3-0.6B | 7.1–8.8 s |
| Qwen3-1.7B | 7.7–10.5 s |
| Qwen3-4B | 5.3–8.0 s |

It is the initial and the five post-summary k-means fits, embedding the summaries (one `encode` call per step), the objective and trace bookkeeping, and the member selection. The corpus embeddings are not in it: they come from the cache and are computed before the clock starts. So for the language models the summarizer is 99.1–99.7% of `total s`, and for LSA about 62%.

## 3. What `summarizer_seconds` covers

It is measured in `core._summarize_clusters`, with `time.perf_counter()` read immediately before and after the single `summarizer.summarize_batch(...)` call of a step (the counter reads `summarizer.stats()` sit outside the bracket). It therefore covers, for `HFSummarizer`, everything inside `summarize_batch`:

- chat-template rendering and token counting, including the repeated tokenizations of the #28 binary search for an over-long cluster (`_render`);
- batch tokenization with padding and the host-to-device copy;
- `model.generate` (prefill and the decode loop);
- the token counters (`attention_mask.sum()`, `count_generated_tokens`);
- `batch_decode` and `.strip()`.

It does not cover loading the model, member selection (`_select_cluster`, including k-means++ trimming when `force_context_length > 0`), the memo lookup, embedding the summaries, k-means, or anything else in the loop.

**The reading is taken after the GPU work has finished.** It is a wall-clock reading, not a CUDA event, so this matters, and it holds for four independent reasons:

- `generate` blocks the host on the device every decoding step: in the installed `transformers` 5.17.0, `StopCheck.__call__` returns `bool(unfinished_sequences.max() == 0)`. The deferred variant that avoids this is only enabled for `mps` (`DeferredStopCheck.is_supported`).
- `int(enc["attention_mask"].sum())` and `count_generated_tokens(...)`'s `int(lengths.sum())` read device tensors.
- `batch_decode` moves the generated tokens to the CPU.
- The resulting strings are used before the clock stops.

The first timed call also includes any one-time lazy initialization (CUDA kernels, allocator growth) because the model is loaded at construction, outside the bracket.

**Tokenization is a negligible part of it.** Rendering and tokenizing all prompts of a seed, plus re-tokenizing the outputs, took 2.7–3.3 s of CPU per seed here, against 850–2,050 s of `summarizer_seconds` (0.15–0.4%), so the seconds are dominated by `generate`.

## 4. What this check cannot catch

The verifier shares code with the live path: it calls `fit_and_render` (the trim and the rendering), `step_inputs` (member selection) and `sent_clusters` (the memo rule). So it cannot detect:

- a bug inside the trim, the rendering or the member selection themselves, since both sides would be wrong the same way;
- a `config.yaml` that differs from what actually ran: the prompt, `text_type`, `max_input_tokens` and the rest are taken from the logged config;
- output-token drift below the tolerance (0.3%).

It does catch bugs in counter wiring (which quantity is counted, and where), in aggregation (per step into trace rows into per-run totals) and in call accounting (empty clusters, memo hits, the step that stopped on `tol`). The prompt rebuild is also an independent re-tokenization with the real tokenizer, so a stats counter that disagreed with the tokenizer would show.

For the counters that are not counts (`output_tokens`, `summarizer_seconds`, and the embedder's `embed_*` counters), the check on a run that ended on `tol` is one-sided: the stopped step is in the per-run total only, so the total must be at least the trace sum, but its exact size cannot be rebuilt (seconds) or is only an expectation (output tokens, which have their own check). Call, input-token, truncation, dropped-text and empty-summary counters are rebuilt exactly for that step as well.

## 5. Known timing caveats (already documented, not re-measured here)

- **Shared GPU:** the wall-clock times are inflated, by roughly 1.7× according to a single BANKING77 check (185 s against 105–115 s), most likely by a Windows-side process using the same card. That 1.7× is an estimate from one BANKING77 run, not a measurement of these CLINC150 runs. Because `summarizer_seconds` is wall-clock and synchronizes with the device, anything else using the GPU or CPU at the time is inside it.
- **Batch sizes 8 / 4 / 1** (0.6B / 1.7B / 4B) differ by model, so per-call seconds mix model size with batching. Peak memory depends on the batch size and on `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` too.
- The token, call and truncation counts do not depend on either caveat; they were verified above.

Part 2 (the GPU idle check) and part 3 (re-timing at a fixed batch size, on a card verified idle) are tracked separately in #35 and are not done here.
