# arXiv at full scale: status for the November 16 submission (#48)

This is a progress status, not a results report. The only scores in it are from the earlier 20k smoke run. Everything else is an estimate, and each estimate says what it is based on. Nothing at full scale has been run: the baselines, the pilot and the LSA preprocessing all wait for approval.

## TL;DR

- **Baselines at full scale** (3,035,644 abstracts, k = 171, seeds 0-2) are ready to run as a CPU job, `configs/arxiv_full_baselines.yaml`. Expected: about 6-7 GB peak RAM, about 1 hour (worst case about 2.5 hours). Full-batch k-means fits the 15 GB box only with `copy_x: false`; with sklearn's default copy it would need 11-13 GB and is not safe.
- **LSA (k-NLPmeans) should use the 200k subsample, not the full corpus.** At full scale the sentence embeddings alone are 29 GB in float32 and segmentation is about 10 hours on one CPU core.
- **The current streaming schedule is out of reach**: up to 259,920 summarizer calls, about 74 GPU-hours with Qwen3-0.6B and 103 with Qwen3-1.7B.
- **Recommended for the full summary run (#49): Qwen3-1.7B, 150,000-row batches (21 batches), two summary steps per batch.** At most 7,182 calls, about 2.9 GPU-hours on an idle card (4.9 if the card is as slow as during the CLINC150 runs).
- **Pilot** (`configs/arxiv_sub200k_pilot.yaml`): the same model and steps on the 200k subsample with 20,000-row batches, at most 3,360 calls, about 1.3 GPU-hours idle (2.3 as logged). It measures the real per-call cost on arXiv prompts, which this estimate has only from CLINC150 and one smoke run.

## 1. What exists

Data (#17, PR #27), built by `scripts/prepare_arxiv.py` from the `librarian-bots/arxiv-metadata-snapshot` revision recorded in `data/arxiv/manifest.json`:

| | |
| --- | --- |
| Abstracts | 3,035,644 after filtering (3,205,432 read; 5,587 withdrawn, 163,016 shorter than 50 words, 1,185 duplicates dropped) |
| Labels | 171 primary categories (38 archives); the 200k subsample (seed 0) contains 168 of them |
| Vectors | `all-MiniLM-L6-v2`, 384-d, normalized, stored as a float16 memmap (2.3 GB); 4.7 GB as float32 |
| Text | mean 152 words, 210 Qwen3 tokens per abstract (sample of 2,000 from the subsample) |

The 20k smoke run (first 20,000 rows of the 200k subsample, 149 clusters, batches of 5,000, Qwen3-0.6B at batch size 8, seed 0, `eval_assign: lloyd1`):

| Method | ACC | NMI | ARI | time |
| --- | ---: | ---: | ---: | ---: |
| MiniBatchKMeans | 0.2327 | 0.5269 | 0.1238 | 0.9 s |
| Sequential mini-batch k-means | 0.2553 | 0.5231 | 0.1391 | 0.3 s |
| k-means | 0.2462 | 0.5362 | 0.1315 | 1.3 s |
| k-SLMmeans (Qwen3-0.6B) | 0.2595 | 0.5304 | 0.1358 | 2,938 s (49 min) |

One seed, a non-random prefix of the subsample (file order), so these say only that the pipeline works and the methods are in the same range; they support no claim about which is better. The run predates the cost fields (#30) and the summary-distance fix (#29), so its call count and distance columns are not usable. Its runtime is used below as one calibration point.

## 2. Full-scale baselines

`configs/arxiv_full_baselines.yaml`: the three `run_streaming.py` baselines, seeds 0-2, `max_batch_size: 10000` (the batch size of `seqminibatchkmeans` and of `minibatchkmeans`), no summarizer, no GPU.

### 2.1 Changes this needed

- A config without a `summarizer` block is baseline-only: no schedule, embedder object, summarizer or sentence corpus is built, and the summary-method keys are not required. With a `summarizer` nothing changes (the same keys are required, the typo guard is intact).
- `dataset.load_texts: false` skips the abstracts. At full scale they are about 3 GB of Python strings that the baselines never read. The arXiv loaders return `""` for every text and keep the labels and vector rows.
- Every metrics row now logs `peak_rss_mb`, the process high-water mark after that method (so it never decreases within a run).
- New `summary_every` for the summary method, see section 4.

### 2.2 Peak RAM

The box has 15 GB (about 12 GB available when idle; the Windows side and other jobs take the rest). Estimates, not measurements at full scale:

| Piece | Size | Notes |
| --- | ---: | --- |
| Interpreter, numpy, sklearn, torch imports | ~1.0 GB | measured: 0.95 GB RSS after imports |
| Embeddings as float32 | 4.7 GB | 3,035,644 x 384 x 4 B; materialized once from the float16 memmap |
| Labels, oracle centroids, metrics | < 0.1 GB | |
| Load transient | + up to 2.3 GB | file-backed memmap pages, reclaimable, released after the conversion |

**The vectors cannot stay memory-mapped.** `cache.dataset_embeddings` reads the float16 memmap and converts it to a float32 array in one step, which is the 4.7 GB. That is necessary: sklearn accepts float32 or float64 only, and given float16 it upcasts to float64 (9.3 GB). The memmap is used only so the conversion streams from disk.

| Baseline | Peak RAM (estimate) | Basis |
| --- | ---: | --- |
| `seqminibatchkmeans` | ~6 GB | X + one 10k-row batch; `partial_fit` copies nothing else |
| `minibatchkmeans` | ~6-7 GB | X + 30k-row init sample + sklearn's chunked distance buffers (capped by `working_memory`, 1 GiB) |
| `kmeans`, default | ~11-13 GB | X + a second 4.7 GB centered copy (`copy_x=True`) + buffers. Not safe in 12 GB. |
| `kmeans`, `copy_x: false` | ~7 GB | centers X in place and restores it; exact up to float32 rounding (~1e-7 per value) |

Checked at small scale only (`tracemalloc`, synthetic 40k x 384 float32): default `KMeans` allocates one extra copy of X, `copy_x=False` allocates none beyond the buffers, and the mini-batch variants stay near X. The config therefore uses `kmeans: {copy_x: false}`, so no baseline is dropped or chunked. The first thing to look at when it runs is `peak_rss_mb` after the first `kmeans` row; if it passes about 10 GB, stop. A slice check (3,000 rows, `load_texts: false`, `copy_x: false`) ran through the real memmap and loader without problems.

The metrics code is not a memory risk: `cluster_metrics` loops over the 3M assignments in Python (seconds) and builds a 171 x 171 confusion matrix.

### 2.3 Runtime

Extrapolated from the smoke run's baseline timings (20k rows, k = 149) by work: scaling by rows x k is 152 x 1.15 = 174x. The box was shared with other jobs while this was prepared (load average 17), so a clean-box run may be faster.

| Baseline | Smoke | Full scale, per seed | Reasoning |
| --- | ---: | ---: | --- |
| `seqminibatchkmeans` | 0.34 s | ~1-3 min | one pass, no convergence loop: 0.34 s x 174 = 1 min, plus labelling |
| `minibatchkmeans` | 0.89 s | ~2-13 min | the work per step is fixed; the step count depends on early stopping (upper bound `max_iter` = 100 epochs = 30,356 steps at about 25 ms) |
| `kmeans` | 1.29 s | ~4-30 min | one Lloyd iteration is 394 GFLOP; same iteration count as the smoke run gives 3.7 min, a few times more iterations is typical at 150x the rows |

Three seeds: about 45-60 minutes nominal, about 2.5 hours worst case, plus a few minutes to read 2.3 GB and convert it. Run seed 0 first (`--set seeds=[0]`), check the RSS and time, then the other two.

## 3. LSA (k-NLPmeans) feasibility

k-NLPmeans needs every abstract split into sentences, and every sentence embedded (`SentenceCorpus.build`). Measured on 2,000 random abstracts of the subsample (`scripts/estimate_stream_cost.py --measure --lsa`, spaCy `en_core_web_sm` on one CPU core, MiniLM tokenizer), and extrapolated:

| Quantity | Sample | 200k subsample | 3,035,644 (full) |
| --- | ---: | ---: | ---: |
| Sentences per abstract (kept, >= 5 tokens) | 6.31 (p5-p95: 3-11) | 1.26 M sentences | 19.2 M sentences |
| spaCy segmentation | 73-90 abstracts/s | 37-46 min | 9.4-11.5 h |
| MiniLM sentence embedding, by tokens | 234 tokens per abstract's sentences vs 200 for the abstract | ~5 min | ~1.4 h |
| ... pessimistic, by rows (725 rows/s) | | ~29 min | 7.3 h |
| Sentence embeddings, float32 | | 1.9 GB | 29.4 GB |
| Sentence text | 166 chars each | 0.2 GB | 3.2 GB (about twice that as Python strings) |

The embedding time by tokens takes the 725 abstracts/s measured for the corpus (manifest.json) as a token rate (about 145k tokens/s); sentences are short, so per-row speed would be higher. Both ends are given. Not measured: the GPU embedding, and spaCy with several processes (`n_process` > 1 should divide segmentation roughly by the process count, at some RAM per process).

**Recommendation: run LSA on the 200k subsample.** At full scale the sentence corpus does not fit in RAM (`SentenceCorpus.build` encodes all sentences in one call and keeps the float32 matrix; the cache then writes an `.npz` and a JSON of all sentences), segmentation alone is most of a working day on one core, and it would need new chunked/float16-memmap code. The 200k subsample costs under an hour of preprocessing and 2 GB, and is the same subsample the pilot uses, so LSA, the pilot and the baselines on the subsample are directly comparable. Say so in the report: LSA is run at 200k, not at 3.0M.

## 4. Cost of the summary methods

### 4.1 What a run costs

`mini_batch_summary_kmeans` runs a whole summary schedule on every batch. The summarizer is called once per non-empty cluster per summary step (a memo-cache hit or an empty cluster is free, so the call counts below are upper bounds):

    calls <= ceil(ceil(N / batch) / summary_every) x k x summary steps

For arXiv: N = 3,035,644 and k = 171. A call sends 10 abstracts (`force_context_length: 10`).

**Input tokens per call.** Measured on the sample with the Qwen3 tokenizer and the exact `HFSummarizer` prompt: 53 tokens of template, system prompt, task prompt and the `Topic:` prefill, plus 210 tokens per abstract and one newline each, so 53 + 10 x 211 = 2,163 tokens. Rendering 300 random 10-abstract prompts gave a mean of 2,134 (max 3,091, none near the 4,096 limit and none truncated). The CLINC150 prompts averaged about 1,570 tokens.

**Rate per input token.** From the CLINC150 table (`clinc150_summarizers.md` section 3): summarizer seconds divided by input tokens, at each size's logged batch size (8 / 4 / 1): 0.805, 1.124 and 1.656 ms per token. Those runs shared the GPU and were about 1.7x slow, so the estimator divides by 1.7 (`--gpu-slowdown`) to get an idle-card figure: **0.473, 0.661 and 0.974 ms per input token** for 0.6B / 1.7B / 4B. This is a single all-in rate (prefill, decoding of the roughly 15-25 output tokens, padding, tokenization), not a split into prefill and decode, because the data to separate them do not exist yet; the pilot will provide it.

**Check against the smoke run.** Qwen3-0.6B, 4 batches x 149 clusters x 5 steps = at most 2,980 calls at 2,163 tokens: 3,049 s predicted, 2,938 s measured (+4%). The smoke call count is an upper bound, and its GPU load is unknown, so the agreement is a sanity check on the order of magnitude, not a validation. Two independent anchors (CLINC150 deflated, and the smoke run) both give about 0.47 ms per token for the 0.6B.

### 4.2 Options

New code makes (c) possible: `mini_batch_summary_kmeans(summary_every=m)` runs the summary schedule only on batches 0, m, 2m, ... (the first batch is always summarized). The other batches run plain k-means from the carried centroids with the schedule's whole `max_iter` budget in one fit and make no calls. The default `summary_every=1` is unchanged and byte-identical. (b) and (d) need no code (`n_summaries`, `max_batch_size`).

GPU-hours on an idle card (bold = within 4 h); `scripts/estimate_stream_cost.py`:

| Option | batches | summarized | calls (max) | input tok/call | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) current: 10k batches, 5 steps, every batch | 304 | 304 | 259,920 | 2,163 | 73.9 | 103.3 | 152.2 |
| (b) one summary step per batch: 10k batches | 304 | 304 | 51,984 | 2,163 | 14.8 | 20.7 | 30.4 |
| (c1) 10k batches, 5 steps, every 20th batch | 304 | 16 | 13,680 | 2,163 | **3.9** | 5.4 | 8.0 |
| (c2) 10k batches, 1 step, every 4th batch | 304 | 76 | 12,996 | 2,163 | **3.7** | 5.2 | 7.6 |
| (d1) 100k batches, 5 steps, every batch | 31 | 31 | 26,505 | 2,163 | 7.5 | 10.5 | 15.5 |
| (d2) 100k batches, 2 steps, every batch | 31 | 31 | 10,602 | 2,163 | **3.0** | 4.2 | 6.2 |
| (d3) 100k batches, 1 step, every batch | 31 | 31 | 5,301 | 2,163 | **1.5** | **2.1** | **3.1** |
| (d4) 150k batches, 2 steps, every batch | 21 | 21 | 7,182 | 2,163 | **2.0** | **2.9** | 4.2 |
| (d5) 250k batches, 2 steps, every batch | 13 | 13 | 4,446 | 2,163 | **1.3** | **1.8** | **2.6** |

If the card is as slow as during the CLINC150 runs (`--gpu-slowdown 1`), multiply by 1.7: (d4) becomes 3.5 / 4.9 / 7.1 h, (d3) 2.6 / 3.6 / 5.3 h. The CPU k-means time is separate: at most 0.44 h in total for any option (every batch running all 120 Lloyd iterations at an assumed 30 GFLOP/s, independent of the batching; warm-started batches stop earlier). It is an assumption, not a measurement.

### 4.3 Recommendation

**Qwen3-1.7B, `max_batch_size: 150000` (21 batches), `schedule: {type: fixed, n_summaries: 2, max_iter: 120}`, `summary_every: 1`, `force_context_length: 10`, summarizer batch size 4: at most 7,182 calls, about 2.9 GPU-hours idle-equivalent (4.9 as logged), one seed.**

- **Why 1.7B:** it is the size the issue calls best on CLINC150, and it costs 1.4x the 0.6B and 0.68x the 4B per token. The 4B needs 4.2 h for the same schedule on an idle card (7.1 as logged).
- **Why large batches rather than every m-th batch:** cost depends on the number of summarized batches x steps, not on the rows. Big batches give each cluster hundreds of members per batch (about 880 at 150k) and every batch is summarized; (c) keeps small noisy batches and uses summaries on a few of them, so most of the stream never sees a summary. (c) also gives 4 to 20 plain batches in a row.
- **Why two steps, not five:** each step is 3,591 calls (1.4 GPU-hours). Two steps keep the summarize-then-refit loop that distinguishes the method; five steps (d1) need 10.5 h.

**Trade-off.** Fewer, larger batches mean fewer centroid updates from the stream (21 instead of 304), and fewer summary steps per batch than the 5 of the paper. Both reduce how often a summary can move a centroid, so the method gets less of a chance to differ from k-means than in the paper's setup. The summary-distance columns and the replacement diagnostic (#33) in #49 will show whether the summaries are doing anything. This is a budget choice, not evidence that 2 steps are enough. If the pilot's measured seconds per call is above about 1.7 s (idle-equivalent estimate: 1.43 s), drop to one summary step (about 1.4 h).

The seeds multiply the budget: seeds 0-2 would be about 9 GPU-hours idle-equivalent.

## 5. Pilot

`configs/arxiv_sub200k_pilot.yaml`: the 200k subsample (168 clusters), Qwen3-1.7B, two summary steps, `max_batch_size: 20000` (10 batches), `force_context_length: 10`, seed 0, the three baselines alongside, `gpu_check: {require_idle: fail}` (#35: it must run on an idle card).

- **Calls:** at most 10 x 168 x 2 = 3,360, 2,163 tokens each.
- **GPU time:** about 1.3 h idle-equivalent, 2.3 h as logged; the baselines on 200k add a few minutes of CPU and the model load about a minute. Plan for 1.5-2.5 hours wall.
- **Resources:** an RTX 3060 with enough free VRAM for the 1.7B at batch size 4 (the CLINC150 run peaked at 6.5 GB; the longer arXiv prompts raise the KV cache somewhat, so expect 7 GB or a little more), about 3 GB RAM. Not an LSA run.
- **What it settles:** the real seconds per call and per input token at arXiv prompt lengths for the 1.7B (the one number the estimate rests on), the real call count against the upper bound (memo hits, empty clusters), the cost fields of #30, and a first look at whether summaries move the centroids at 200k.
- **What it does not match:** the pilot has 10 batches of 20k rows; the full run has 21 batches of 150k. The per-call cost carries over, the batch dynamics do not.

## 6. What is left

- **Approvals:** the baselines run, the pilot run (commands in the pull request description).
- **#49:** the full summary run needs about 8 GB for the vectors and texts together (4.7 GB + about 3.3 GB of Python strings) before sklearn's buffers. The final evaluation step `assign_to_centroids(..., "lloyd1")` fits `KMeans(max_iter=1)` on the full X, which makes a second 4.7 GB copy; at full scale use `eval_assign: nearest` (chunked, no copy) or add `copy_x=False` there. Not changed here because `lloyd1` reproduces upstream's evaluation.
- **#49:** `aggregate_runs` expects each run directory to contain a summary method (`method_label` reads `cfg["summarizer"]`), so a baseline-only run directory cannot be passed to it yet; the paired comparison with the full-scale baselines needs that to be handled.
- **#49:** the baselines here use 10k batches while the recommended summary run uses 150k batches. `seqminibatchkmeans` with 10k batches is the stronger baseline; a 150k-batch run (`--set max_batch_size=150000`, minutes) would be the like-for-like comparison.
- **LSA on the 200k subsample:** a config and a run (about an hour of preprocessing, cached), not done.

## Reproducing the numbers

```
HF_HUB_OFFLINE=1 python scripts/estimate_stream_cost.py --measure --lsa --save-stats docs/results/arxiv_token_stats.json
python scripts/estimate_stream_cost.py                                           # the table in 4.2
python scripts/estimate_stream_cost.py --gpu-slowdown 1                          # as logged
python scripts/estimate_stream_cost.py --n-rows 20000 --k 149 --option 5000:5    # the smoke-run check
python scripts/estimate_stream_cost.py --n-rows 200000 --k 168 --option 20000:2  # the pilot
```

The measured sample statistics are in `docs/results/arxiv_token_stats.json`.
