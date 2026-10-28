# Summarizer comparison on CLINC150: k-means, LSA, Qwen3-0.6B / 1.7B / 4B

Issue #31. This is the Bank77 preliminary comparison (#20, [`prelim_bank77.md`](prelim_bank77.md)) repeated on CLINC150, with the same models and protocol, for the October 28 checkpoint.

## TL;DR

- **Size does not order the results.** On CLINC150 (22,500 in-scope queries, 150 intents, seeds 0–4):
  - **Qwen3-1.7B** has the best means (ACC 0.570, NMI 0.760, ARI 0.461). It is above k-means with paired 95% intervals that exclude zero on all three metrics, but its lead over LSA (4 of 5 seeds) cannot be separated from zero.
  - **Qwen3-0.6B** is level with or below k-means.
  - **Qwen3-4B** is the worst method on every metric (ACC 0.511), below both k-means and LSA in all 5 seeds.
- **The 4B's drop coincides with a prompt-following failure.**
  - The CLINC150 prompt asks for "a question that summarizes the following cluster". 34% of the 4B's summaries mention the cluster, and many are a question *about* it, most often "What is the main topic of the following cluster of questions?".
  - 16–21 of the 150 final clusters per seed end with exactly the same summary as another cluster, so they share a centroid at that summary step.
  - The 0.6B and 1.7B do this in 8% of summaries, with almost no duplicates. On BANKING77, whose prompt is worded differently, at most 9 of 1,925 summaries per size mention the cluster, and none are duplicated.
  - This is an association, not yet a tested cause (#33).
- **LSA is still the baseline to beat.** It beats k-means on every metric on both datasets. No SLM size beats LSA with an interval that excludes zero on either dataset.
- **The rankings flip between datasets.** On BANKING77 the 4B had the best SLM means on ACC and ARI and the 1.7B the weakest. With one model family, two datasets and 5 seeds, this is not a size trend.
- **Cost:** about 700 summarizer calls and 1.1 million input tokens per seed for every size; 15, 21 and 31 minutes per seed for the 0.6B, 1.7B and 4B, against 24 s for LSA. The times mix model size with batch size and were inflated by roughly 1.7× during the run, most likely by a background GPU load; the code is not the cause (Section 3).

## 1. Protocol

All settings were fixed before the run (`configs/clinc150_summarizers.yaml`, `scripts/run_clinc150.sh`).

- **Data:** `clinc150_intent`, the full CLINC150 in-scope set: 22,500 queries (train + validation + test), 150 intents, k = 150.
- **Out-of-scope queries:** excluded. `scripts/export_clinc150.py` drops all 1,200 `oos` queries of the original `data_full.json` (100 train, 100 validation, 1,000 test). OOS is not one intent, so it has no gold cluster to recover, and clustering it would need a 151st "cluster" with no coherent content.
- **Encoder:** `distilbert-base-nli-stsb-mean-tokens`, computed once, cached and shared by every method.
- **Initialization:** the k-means baseline and the first stage of every summary method use `KMeans(init="k-means++", random_state=seed)`. This is now checked directly: every metrics row carries `step0_sha1`, a hash of the step-0 labels (#30), and it is identical across all methods for each of seeds 0–4.
- **Methods:** k-means (sklearn defaults, `n_clusters=150`); k-NLPmeans with the LSA extractive summarizer (`top_k` 5); k-SLMmeans with `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`. No new model families.
- **Language models (identical for the three sizes):**
  - the CLINC prompt "The following is cluster of questions. Write a question that summarizes the following cluster:" with each member on a `Question:` line;
  - greedy decoding with thinking off;
  - `max_new_tokens` 64 and `max_input_tokens` 4096.
- **Member selection (identical for the three sizes):** all members of a cluster go into the prompt. If the rendered prompt exceeds 4,096 tokens, the cluster keeps the largest subset that fits, drawn from one fixed seeded permutation of its members and rendered in their original order (#28). Clusters that fit are passed unchanged.
- **Schedule (all summary methods):** fixed, 5 summary steps, 120 Lloyd iterations in 6 blocks of 20, `initial_iter` and `final_iter` on, `tol` 1e-4.
- **Seeds:** 0–4, paired across methods.
- **Hardware:** one RTX 3060 12 GB under WSL2, bf16 weights, Hugging Face `generate`.
- **Batch sizes:** 8 for 0.6B, 4 for 1.7B, and 1 for 4B.
  - In a probe on two prompts capped at about 3,700 tokens, the 4B at batch 2 overflowed the 12 GB card, and under WSL2 an overflow hangs the job instead of raising an error.
  - In the same probe, the 4B at batch 1 peaked at 9.18 GB reserved and the 1.7B at batch 4 at 6.78 GB. In the full runs the peaks were 9.42 and 6.68 GB reserved (Section 3).
  - Batch sizes differ between models, so per-call times mix model size with batching (Section 3).

## 2. Results

### 2.1 Means over seeds 0–4

Mean ± standard deviation over the 5 seeds. `s/seed` is wall-clock time per seed; see Section 3 before reading it.

| Method | ACC | NMI | ARI | s/seed | seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.551 ± 0.007 | 0.748 ± 0.002 | 0.439 ± 0.005 | 5.3 | 5 |
| k-NLPmeans (LSA) | 0.560 ± 0.008 | 0.752 ± 0.004 | 0.449 ± 0.007 | 23.7 | 5 |
| k-SLMmeans (Qwen3-0.6B) | 0.541 ± 0.013 | 0.747 ± 0.003 | 0.437 ± 0.008 | 903.9 | 5 |
| k-SLMmeans (Qwen3-1.7B) | 0.570 ± 0.011 | 0.760 ± 0.006 | 0.461 ± 0.012 | 1247.5 | 5 |
| k-SLMmeans (Qwen3-4B) | 0.511 ± 0.015 | 0.740 ± 0.006 | 0.421 ± 0.010 | 1861.7 | 5 |

### 2.2 Paired differences on CLINC150

Every method is compared with LSA and with k-means on the same seed, so each pair starts from the same step-0 clustering. `mean diff` is method minus reference averaged over the 5 seeds, the 95% CI is a Student-t interval of that mean (4 degrees of freedom), and `wins/n` counts the seeds where the method scores strictly higher. The intervals are per comparison and not corrected for the number of comparisons in these tables; with 5 seeds, an interval that only just excludes zero is weak evidence.

Paired difference vs k-NLPmeans (LSA) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.009 | [-0.017, -0.001] | 1/5 |
| k-means | NMI | -0.004 | [-0.008, -0.001] | 0/5 |
| k-means | ARI | -0.010 | [-0.018, -0.002] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.019 | [-0.032, -0.007] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.005 | [-0.012, +0.001] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.012 | [-0.023, -0.001] | 1/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.010 | [-0.011, +0.030] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.008 | [-0.002, +0.018] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.012 | [-0.007, +0.031] | 4/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.049 | [-0.072, -0.026] | 0/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.013 | [-0.023, -0.002] | 0/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.028 | [-0.043, -0.014] | 0/5 |

Paired difference vs k-means (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-NLPmeans (LSA) | ACC | +0.009 | [+0.001, +0.017] | 4/5 |
| k-NLPmeans (LSA) | NMI | +0.004 | [+0.001, +0.008] | 5/5 |
| k-NLPmeans (LSA) | ARI | +0.010 | [+0.002, +0.018] | 5/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.010 | [-0.018, -0.002] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.001 | [-0.004, +0.003] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.001 | [-0.007, +0.004] | 2/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.018 | [+0.004, +0.033] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.012 | [+0.005, +0.019] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.022 | [+0.011, +0.034] | 5/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.040 | [-0.058, -0.022] | 0/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.009 | [-0.017, -0.000] | 0/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.018 | [-0.027, -0.009] | 0/5 |

### 2.3 BANKING77 and CLINC150 side by side

The same paired differences for both datasets. The BANKING77 columns are recomputed with the same aggregator from the #20 run directories and agree with [`prelim_bank77.md`](prelim_bank77.md). Three protocol differences apply to the BANKING77 runs:

- **Prompt:** BANKING77 has its own prompt: "The following is a cluster of online banking questions. Write a single question that represents the following cluster concisely:", with each member on a `Sentence:` line.
- **Truncation:** they ran before #28, so an overflowing prompt kept a prefix of the cluster's members. The only truncation warning in that sweep is in the 4B log (seed 4, 1 of 128 queries dropped). The warning fires once per summarizer instance, so further overflows are not ruled out.
- **Batch size:** the 4B ran at batch 2 there and at batch 1 here.

Paired difference vs k-means (method minus reference): mean [95% CI] wins/n

| Method | Metric | BANKING77 | CLINC150 |
| --- | --- | ---: | ---: |
| k-NLPmeans (LSA) | ACC | +0.014 [+0.006, +0.022] 5/5 | +0.009 [+0.001, +0.017] 4/5 |
| k-NLPmeans (LSA) | NMI | +0.008 [+0.003, +0.013] 5/5 | +0.004 [+0.001, +0.008] 5/5 |
| k-NLPmeans (LSA) | ARI | +0.010 [+0.003, +0.018] 5/5 | +0.010 [+0.002, +0.018] 5/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.005 [-0.011, +0.021] 3/5 | -0.010 [-0.018, -0.002] 0/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.002 [-0.002, +0.005] 4/5 | -0.001 [-0.004, +0.003] 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.002 [-0.007, +0.003] 2/5 | -0.001 [-0.007, +0.004] 2/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.002 [-0.016, +0.020] 3/5 | +0.018 [+0.004, +0.033] 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.005 [+0.001, +0.009] 5/5 | +0.012 [+0.005, +0.019] 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | -0.003 [-0.015, +0.010] 1/5 | +0.022 [+0.011, +0.034] 5/5 |
| k-SLMmeans (Qwen3-4B) | ACC | +0.009 [-0.011, +0.029] 4/5 | -0.040 [-0.058, -0.022] 0/5 |
| k-SLMmeans (Qwen3-4B) | NMI | +0.005 [-0.008, +0.019] 4/5 | -0.009 [-0.017, -0.000] 0/5 |
| k-SLMmeans (Qwen3-4B) | ARI | +0.005 [-0.018, +0.028] 3/5 | -0.018 [-0.027, -0.009] 0/5 |

Paired difference vs k-NLPmeans (LSA) (method minus reference): mean [95% CI] wins/n

| Method | Metric | BANKING77 | CLINC150 |
| --- | --- | ---: | ---: |
| k-means | ACC | -0.014 [-0.022, -0.006] 0/5 | -0.009 [-0.017, -0.001] 1/5 |
| k-means | NMI | -0.008 [-0.013, -0.003] 0/5 | -0.004 [-0.008, -0.001] 0/5 |
| k-means | ARI | -0.010 [-0.018, -0.003] 0/5 | -0.010 [-0.018, -0.002] 0/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.009 [-0.025, +0.007] 2/5 | -0.019 [-0.032, -0.007] 0/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.007 [-0.011, -0.003] 0/5 | -0.005 [-0.012, +0.001] 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.012 [-0.018, -0.006] 0/5 | -0.012 [-0.023, -0.001] 1/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | -0.012 [-0.033, +0.008] 1/5 | +0.010 [-0.011, +0.030] 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | -0.004 [-0.009, +0.002] 1/5 | +0.008 [-0.002, +0.018] 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | -0.013 [-0.023, -0.004] 0/5 | +0.012 [-0.007, +0.031] 4/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.005 [-0.033, +0.022] 2/5 | -0.049 [-0.072, -0.026] 0/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.003 [-0.015, +0.010] 2/5 | -0.013 [-0.023, -0.002] 0/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.005 [-0.029, +0.018] 3/5 | -0.028 [-0.043, -0.014] 0/5 |

### 2.4 Summary surface checks

These are simple counts over `summaries.jsonl`, not the summary audit planned in #34.

- "All steps" means the 150 clusters × 5 summary steps × 5 seeds = 3,750 summaries per model, and "final" means the 750 summaries of the last step.
- "Mentions cluster/summar…" is a keyword match for summaries that talk about the cluster instead of standing in for it.
- "Clusters sharing an identical summary" counts, per seed, the final-step clusters whose summary text is exactly the same as another cluster's.

| Model | summaries (all steps) | mean words | ends with "?" | mentions cluster/summar… | final: mentions cluster/summar… | final: clusters sharing an identical summary (per seed) | final: largest identical group | starts with quote | `<think>` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 3,750 | 11.2 | 3,301 (88%) | 314 (8%) | 69/750 | 0–5 | 3 | 2 | 0 |
| Qwen3-1.7B | 3,750 | 21.0 | 3,623 (97%) | 316 (8%) | 68/750 | 0–4 | 2 | 0 | 0 |
| Qwen3-4B | 3,750 | 17.7 | 3,205 (85%) | 1,291 (34%) | 272/750 | 16–21 | 13 | 25 | 0 |

- **The 4B often writes a question about the cluster instead of a question from it.**
  - About a third of its summaries mention the cluster (1,291 of 3,750), against 8% for the other two sizes.
  - Most are the same few sentences. The most common final summary is "What is the main topic of the following cluster of questions?", and others include "What are the main topics covered in the cluster of questions?" and "What is the main theme of the cluster?".
  - Some go on to answer their own question, for example "What is the main topic of the following cluster of questions? The main topic of the cluster is **checking and confirming restaurant reservations**…".
  - As a result, 16–21 of the 150 final clusters per seed share their summary text exactly with at least one other cluster, in groups of up to 13. For the 0.6B and 1.7B it is at most 5.
- **Other sizes:**
  - Some 0.6B outputs repeat the instruction and then copy members, for example "Please summarize the following cluster of questions into a single question: **Can you reserve a table for 2 at Maxwell's on the 14th?** …".
  - The 1.7B's cluster-mentioning summaries mostly still name the topic, for example "Summarize the cluster of questions about scheduling, calendar management, reservations, meetings, and reminders.", so they rarely coincide.
- **BANKING77 for comparison** (same counts on the #20 runs): at most 9 of 1,925 summaries per model mention the cluster (0.6B 0, 1.7B 9, 4B 0).
- **Length:** mean words per summary are 11.2 (0.6B), 21.0 (1.7B) and 17.7 (4B).
- **No thinking text:** `<think>` and `</think>` appear in none of the summaries of any model.
- **Stray quotes:** 2 (0.6B) and 25 (4B) of 3,750 summaries start with a double quote. On BANKING77, 60 (0.6B) and 3 (4B) of the 77 final summaries of seed 0 did.

## 3. Cost

Per-seed means ± standard deviation, from the per-seed cost fields added in #30.

- `calls` counts clusters actually sent to the summarizer. Empty clusters, and clusters whose member set is unchanged since an earlier step (memo-cache hits), are not calls, so the maximum is 150 × 5 = 750.
- Token counts come from the model's tokenizer: input tokens are the rendered prompts, and output tokens are counted up to and including the first end-of-sequence token.
- `peak GPU GB` and `reserved GPU GB` are PyTorch's peak allocated and reserved memory for the run.

| Method | calls | input tokens | output tokens | summarizer s | total s | empty summaries | truncated clusters | dropped texts | peak GPU GB | reserved GPU GB | batch size | dtype | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a | n/a | 5.3 ± 1.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| k-NLPmeans (LSA) | 636 ± 8 | 0 ± 0 | 0 ± 0 | 14.8 ± 1.8 | 23.7 ± 2.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.37 ± 0.00 | 0.53 ± 0.08 | n/a | n/a | NVIDIA GeForce RTX 3060 |
| k-SLMmeans (Qwen3-0.6B) | 707 ± 7 | 1,113,037 ± 8,773 | 10,227 ± 211 | 895.8 ± 29.7 | 903.9 ± 29.6 | 0.0 ± 0.0 | 6.6 ± 3.5 | 173.0 ± 147.1 | 6.51 ± 0.00 | 6.70 ± 0.00 | 8 | torch.bfloat16 | NVIDIA GeForce RTX 3060 |
| k-SLMmeans (Qwen3-1.7B) | 702 ± 6 | 1,101,755 ± 11,449 | 18,288 ± 387 | 1238.6 ± 102.5 | 1247.5 ± 103.4 | 0.0 ± 0.0 | 6.8 ± 2.4 | 304.0 ± 42.8 | 6.54 ± 0.00 | 6.68 ± 0.00 | 4 | torch.bfloat16 | NVIDIA GeForce RTX 3060 |
| k-SLMmeans (Qwen3-4B) | 714 ± 6 | 1,120,190 ± 7,604 | 15,620 ± 463 | 1855.0 ± 188.6 | 1861.7 ± 189.8 | 0.0 ± 0.0 | 10.2 ± 3.4 | 502.4 ± 264.2 | 9.25 ± 0.00 | 9.42 ± 0.00 | 1 | torch.bfloat16 | NVIDIA GeForce RTX 3060 |

**What the numbers support:**

- **Tokens** do not depend on the hardware.
  - Every model size sends about 1.1 million input tokens per seed. The three Qwen3 sizes share one tokenizer (identical `tokenizer.json`), so the small differences come from the clusterings, not the models.
  - Output tokens per summary: about 14.5 (0.6B), 26 (1.7B) and 22 (4B).
- **Truncation is rare.**
  - On average 6.6 (0.6B), 6.8 (1.7B) and 10.2 (4B) prompts per seed overflowed 4,096 tokens and were trimmed to a seeded subset (per-seed range 3–14, about 1% of roughly 700 calls). On average that dropped 173, 304 and 502 queries per seed.
  - The truncated prompts belong to oversized clusters. The first warning in the 1.7B log reports a 396-member cluster losing 60 queries, and the 4B log a 367-member cluster losing 22. With 150 queries per intent, a cluster that size has merged several intents. The 4B truncates more on average, but the per-seed ranges overlap.
- **LSA** makes 636 calls per seed and spends about 15 s in the summarizer, 24 s per seed in total. That excludes sentence segmentation and sentence embeddings, which are computed once and cached.
- **Relative to LSA**, the language models take about 38× (0.6B), 53× (1.7B) and 79× (4B) the wall-clock time per seed, with the caveats below.

**What the timings do not support:**

- **Batching differs by model** (8 / 4 / 1), so per-call times mix model size with batching. #35 has the batching control.
- **The GPU ran slower during this run than in earlier runs and after it, and the code is not the cause.**
  - With the same 0.6B settings and batch size, the RQ2 fixed-n5 run (October 15–16) took 500–600 s per seed on seeds 0–4, against 904 s here. Even the first two 0.6B seeds of the aborted first attempt (below) took 847 and 900 s.
  - **Code check:** after the run (October 28, 00:49–00:57), BANKING77 seed 0 with the 0.6B at batch 8 was run four times, alternating the code from before #28 and #30 (`784e022`) with `8222515`. The old code took 114.8 and 113.5 s and the new code took 108.1 and 111.1 s, with identical ACC. The new member selection and logging add no measurable time.
  - **Time-of-day check:** the same new-code BANKING77 run took 185 s at 17:32 on October 27, minutes before the CLINC150 run started, against 108–111 s after it and 105 s on October 14 (#20). The slowdown follows the time, not the code.
  - **Likely cause:** a Windows-side process using the same card, which WSL's `nvidia-smi` does not list. Just before the recorded run started (19:09), with no WSL process on the GPU, the card showed 1.5 GB in use, 43% utilization and 85 W. During the after-run check it showed 2 GB and 3–4%. The load was not measured during the recorded runs.
  - **First attempt:** it was stopped at 18:56, when that load grew to about 6.7 GB and one 0.6B summary step took almost 10 minutes, against 2–3.5 minutes before. Its logs are kept in `results/clinc150_logs/attempt1/`, and none of its results are used.
  - **How to read the seconds:** treat them as inflated, by roughly 1.7× judging from the BANKING77 check (185 s against 105–115 s). Seed-to-seed differences may also reflect the load; for example, 1.7B seed 4 took 1,081 s against 1,260–1,367 s for seeds 0–3.
- **Memory:** the run used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which changes how PyTorch reserves memory, not what it computes. With it, the 0.6B reserved 6.70 GB, against 8.16 GB in the first attempt without it. Peak memory also depends on the batch size, so it is not comparable across models either.

## 4. Interpretation (preliminary)

**The three sizes land on different sides of the baselines.** On CLINC150:

- **1.7B** is above k-means on all three metrics, and each interval excludes zero:
  - ACC +0.018 [+0.004, +0.033], 4 of 5 seeds;
  - NMI +0.012 [+0.005, +0.019], 5 of 5;
  - ARI +0.022 [+0.011, +0.034], 5 of 5.

  It is also higher than LSA in 4 of 5 seeds on every metric (mean +0.010 ACC, +0.008 NMI, +0.012 ARI), but every one of those intervals includes zero. We cannot say it beats LSA.
- **0.6B** is below k-means on ACC (−0.010 [−0.018, −0.002], 0 of 5 seeds), level with it on NMI and ARI, and below LSA on ACC and ARI.
- **4B** is the worst method on every metric. It loses to k-means (ACC −0.040 [−0.058, −0.022]) and to LSA (ACC −0.049 [−0.072, −0.026]) in 5 of 5 seeds on all three metrics.

**The rankings flip between datasets.** On BANKING77, the 4B had the best SLM means on ACC and ARI and the 1.7B the weakest. All three sizes were close to k-means there (only the 1.7B's NMI interval excluded zero, at +0.005), and none beat LSA.

**The 0.6B mean is stable.** The RQ2 fixed-n5 run of the same model (#15) used the same settings, apart from #28's member selection and the code before #30. Its seeds 0–4 averaged ACC 0.544, against 0.541 here. Individual seeds moved by up to 0.026, though: seed 0 went from 0.559 to 0.534, from above k-means to below it. #28 is the most likely source of those seed-level changes: when no prompt overflows, the old and new code give identical results (the code check in Section 3). So the change in which members are dropped leaves the 5-seed mean where it was, but it is not neutral seed by seed.

**Most likely cause of the 4B's drop: it reads the prompt literally.**

- The CLINC150 prompt asks for "a question that summarizes the following cluster". About a third of the 4B's summaries mention the cluster, and many are the same question about it: "What is the main topic of the following cluster of questions?" (Section 2.4).
- Clusters with identical summary text get identical centroids at that summary step. At the final summary step of each seed, 16–21 clusters share a centroid with at least one other cluster, in groups of up to 13.
- On average more of the 4B's prompts overflow (10.2 per seed, against 6.6 and 6.8), though the per-seed ranges overlap. The overflowing clusters in the logs have 367 and 396 members, and RQ2 reported 231–398, all well above the 150 queries of one intent.
- The BANKING77 prompt asks for "a single question that represents the following cluster concisely", and there the 4B never did this.
- This is an association we have not tested. #33's replacement diagnostic measures it directly: where each summary embedding lands relative to its members, and which points it moves.

**What the results so far teach:**

- **A fluent summary is not automatically a useful representative.** "What is the main topic of the following cluster of questions?" is a well-formed question, and as a centroid it is useless: it carries none of the cluster's content and is the same for many clusters. The 0.6B's short questions are also below plain k-means on ACC. This is the "when does a fluent summary also make a useful representative" question in its sharpest form.
- **The prompt is part of the method.** The same instruction is followed differently by different sizes. This comparison is therefore of model size *under one prompt per dataset*, and a single prompt failure can dominate a size comparison. Prompt sensitivity has to be checked before any size claim (#34 summary audit, #36 size trend).
- **The 1.7B's gain is real but narrow.** It is the only SLM setting in the runs reported here whose intervals exclude zero against k-means on all three metrics, but it is one model on one dataset and cannot be separated from LSA.
- **LSA remains the baseline to beat.** It is above k-means on every metric on both datasets, with intervals that exclude zero, in about 24 s per seed. No SLM size beats it with an interval that excludes zero on either dataset.

## 5. Limitations and next steps

- **Preliminary scope:** one model family at three sizes, two datasets, 5 seeds, one prompt per dataset, greedy decoding and the fixed 5-step schedule. The intervals are uncorrected, and several sit close to zero.
- **Untested explanation:** the 4B explanation in Section 4 is an association from surface counts. No prompt variant was run, so we do not know how the 4B (or the other sizes) would do on CLINC150 with a prompt that asks for a representative question.
- **No cost-matched comparison.** The timings come from a shared GPU with different batch sizes per model (Section 3).
- **BANKING77 was not re-run** after #28 and #30. Its logs have a single truncation warning (a lower bound), so #28 probably matters less there than on CLINC150, but this is not measured. Its runs also have no cost or `step0_sha1` logging.
- **Next, from the professor's feedback (tracker #39):**
  - #32: access to the code and runs.
  - November 4:
    - #33 replacement diagnostic, which can test the collapsed-centroid explanation directly;
    - #34 summary audit, including summaries that describe the cluster;
    - #35 cost with a batching control;
    - #36 size trend, which now needs a prompt-sensitivity check.
  - November 18: #37 adaptive schedule, on one model and one dataset.
  - Backlog: #38 large-model reference, needed before any claim that a small model keeps a large model's gain.

## 6. Reproducibility

- Code: commit `8222515` (recorded with a clean tree in every run's `environment.json`). The tables use the aggregator from `28675b8`.
- Run: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHON=/path/to/venv/bin/python scripts/run_clinc150.sh` (steps `lsa`, `0.6b`, `1.7b`, `4b`, then the table).
- Run directories (under `results/`, gitignored):
  - `clinc150_lsa/20261027-174631` (k-means and LSA)
  - `clinc150_qwen3-0.6b/20261027-190908`
  - `clinc150_qwen3-1.7b/20261027-202532`
  - `clinc150_qwen3-4b/20261027-221042`
- Tables:

  ```
  python scripts/aggregate_results.py <the four run directories> \
      --out-md table.md --out-csv clinc150_summarizers_table.csv \
      --per-seed-csv clinc150_summarizers_per_seed.csv \
      --paired-md paired.md --paired-csv clinc150_summarizers_paired.csv --cost-md cost.md
  ```

  The BANKING77 columns come from the same command on the four `results/prelim_bank77_*` run directories of #20.
- Files in this directory:
  - `clinc150_summarizers_table.csv` (means and standard deviations)
  - `clinc150_summarizers_per_seed.csv` (`method, seed, acc, nmi, ari, runtime_s`)
  - `clinc150_summarizers_paired.csv` (paired differences, intervals and wins against LSA and k-means)
- Section 2.4 counts come from each run's `summaries.jsonl` (keys `summaries` and `summaries_evolution`).
