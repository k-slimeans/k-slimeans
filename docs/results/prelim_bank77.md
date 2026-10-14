# Preliminary results: k-SLMmeans with Qwen3-0.6B / 1.7B / 4B on BANKING77

## TL;DR

The professor's scoping for the first experiment: "Try to keep the first experiment manageable. One dataset, such as BANKING77, and two or three model sizes should be enough to establish an initial result."

We ran k-means, k-NLPmeans (LSA) and k-SLMmeans with three Qwen3 sizes (0.6B, 1.7B, 4B) on the BANKING77 test split, 5 seeds each. **In this setup no SLM size reliably beats the LSA extractive summarizer.** LSA has the highest mean ACC, NMI and ARI of the five methods, and 0.6B and 1.7B lose to it on ARI in all 5 seeds. The 4B model is the closest: it is within about one standard deviation of LSA and wins on NMI in 2 of 5 seeds. Against plain k-means, SLM NMI is higher on average (+0.002 to +0.005) but the gap is of the order of the seed-to-seed spread, so we regard it as suggestive at most. SLM runs cost 60 to 240 times more wall-clock time per seed than LSA. With 5 seeds, one dataset, one prompt and greedy decoding, all of this is preliminary.

## 1. Setup

- **Dataset:** BANKING77, test split, 3,080 queries, 77 intents. Embeddings: `distilbert-base-nli-stsb-mean-tokens` (`embedder: distilbert`), cached and shared by every method.
- **Methods**
  - k-means: sklearn `KMeans`, defaults except `n_clusters=77` and `random_state=seed`.
  - k-NLPmeans: LSA extractive summarizer, `top_k=5`.
  - k-SLMmeans: Qwen3-0.6B / 1.7B / 4B (`Qwen/Qwen3-*`), thinking disabled (the `HFSummarizer` default), greedy decoding, `max_new_tokens=64`, the upstream BANKING77 prompt, unchanged ("The following is a cluster of online banking questions. Write a single question that represents the following cluster concisely: ...").
- **Schedule (all summary methods):** fixed, 5 summary steps, `max_iter=120`, `initial_iter` and `final_iter` on, `tol=1e-4`. The 120 Lloyd iterations are split into 6 blocks of 20. All cluster members go into the prompt (`force_context_length=0`; the prompt is capped at `max_input_tokens=4096`). Seeds 0 to 4.
- **Hardware:** one RTX 3060 12 GB under WSL2. Per-model `batch_size` 8 / 4 / 2 for 0.6B / 1.7B / 4B, chosen to fit VRAM (rationale and measured peaks in the header of `scripts/run_prelim.sh`).
- **Commands**
  ```bash
  PYTHON=/path/to/venv/bin/python scripts/run_prelim.sh          # lsa, 0.6b, 1.7b, 4b, then the table
  PYTHON=/path/to/venv/bin/python scripts/run_prelim.sh table    # rebuild the table from existing runs
  ```
  Configs: `configs/offline_knlpmeans.yaml` (LSA, with `seeds=[0,1,2,3,4]`) and `configs/prelim_bank77.yaml` (SLMs, with the model and batch size set by the script).

## 2. Results

Mean ± sample std (ddof=1) over 5 seeds. `s/seed` is mean wall-clock seconds per seed (see Section 5 for caveats). Source: `docs/results/prelim_bank77_table.csv`.

| Method | ACC | NMI | ARI | s/seed | seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.435 ± 0.005 | 0.643 ± 0.003 | 0.295 ± 0.006 | 0.3 | 5 |
| k-NLPmeans (LSA) | 0.449 ± 0.009 | 0.651 ± 0.002 | 0.305 ± 0.005 | 1.6 | 5 |
| k-SLMmeans (Qwen3-0.6B) | 0.440 ± 0.010 | 0.645 ± 0.002 | 0.293 ± 0.006 | 102.7 | 5 |
| k-SLMmeans (Qwen3-1.7B) | 0.436 ± 0.010 | 0.648 ± 0.004 | 0.292 ± 0.008 | 229.0 | 5 |
| k-SLMmeans (Qwen3-4B) | 0.444 ± 0.016 | 0.648 ± 0.009 | 0.300 ± 0.014 | 381.9 | 5 |

## 3. Per-seed and paired view

**What is shared across methods for a given seed.** All methods run on the same cached embeddings. The k-means baseline and the first stage of every summary method both call `KMeans(n_clusters=77, init="k-means++", random_state=seed)` (`experiment.py::_run_baseline`, `core.py::summary_kmeans`) with sklearn defaults otherwise, so by code inspection they start from the same k-means++ initial centers. That is *all* that is shared:

- The baseline then runs up to sklearn's default `max_iter=300`, while the summary methods' first stage is capped at 20 iterations (`max_iter` 120 / 6 blocks).
- All later stages start from the summary embeddings (`init=` the summaries, no random state), so they diverge from the baseline after the first summary.
- LSA and the three SLM runs share the same first stage with each other, and differ only from the first summary onward.

We did not log first-stage labels, so "same initial centers" rests on the code and sklearn's determinism, not on a direct check. The pairing is therefore a variance-reduction device, not a controlled ablation.

NMI per seed:

| seed | k-means | k-NLPmeans (LSA) | k-SLMmeans (0.6B) | k-SLMmeans (1.7B) | k-SLMmeans (4B) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.642 | 0.652 | 0.643 | 0.643 | 0.651 |
| 1 | 0.645 | 0.648 | 0.647 | 0.648 | 0.656 |
| 2 | 0.639 | 0.651 | 0.643 | 0.647 | 0.654 |
| 3 | 0.646 | 0.652 | 0.644 | 0.654 | 0.634 |
| 4 | 0.641 | 0.652 | 0.647 | 0.645 | 0.647 |

Paired comparison: number of seeds (of 5) in which the SLM scores strictly higher than the reference, and the mean paired difference (SLM minus reference), from `docs/results/prelim_bank77_per_seed.csv`.

| SLM | Metric | Beats LSA | Mean diff vs LSA | Beats k-means | Mean diff vs k-means |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | ACC | 2 | -0.009 | 3 | +0.005 |
| Qwen3-0.6B | NMI | 0 | -0.007 | 4 | +0.002 |
| Qwen3-0.6B | ARI | 0 | -0.012 | 2 | -0.002 |
| Qwen3-1.7B | ACC | 1 | -0.012 | 3 | +0.002 |
| Qwen3-1.7B | NMI | 1 | -0.004 | 5 | +0.005 |
| Qwen3-1.7B | ARI | 0 | -0.013 | 1 | -0.003 |
| Qwen3-4B | ACC | 2 | -0.005 | 4 | +0.009 |
| Qwen3-4B | NMI | 2 | -0.003 | 4 | +0.005 |
| Qwen3-4B | ARI | 3 | -0.005 | 3 | +0.005 |

For reference, LSA beats k-means in 5 of 5 seeds on all three metrics (mean diff +0.014 ACC, +0.008 NMI, +0.010 ARI).

With n=5 we do not report p-values as evidence. As an indicative-only aside, a paired t-test on NMI gives an uncorrected p of about 0.01 for 0.6B vs LSA and about 0.03 for 1.7B vs k-means; neither survives any multiple-comparison correction across the 18 comparisons above, and they should not be read as conclusive.

## 4. Comparison with Issue #2

Issue #2 reports an independent CPU run of the same protocol by Samiksha (figures copied from the issue comment of 2026-10-14):

| Method | Issue #2 (CPU) ACC / NMI / ARI | This run (GPU) ACC / NMI / ARI |
| --- | --- | --- |
| k-means | 0.435 ± 0.005 / 0.643 ± 0.003 / 0.295 ± 0.006 | 0.435 ± 0.005 / 0.643 ± 0.003 / 0.295 ± 0.006 |
| k-NLPmeans (LSA) | 0.449 ± 0.009 / 0.651 ± 0.002 / 0.305 ± 0.005 | 0.449 ± 0.009 / 0.651 ± 0.002 / 0.305 ± 0.005 |

All 18 numbers agree to 3 decimals (checked against `prelim_bank77_table.csv`). k-means and LSA themselves run on the CPU (the GPU is only used by the SLMs, and possibly when the cached embeddings were first computed), so this shows that the two environments agree on the embeddings and clustering for these seeds. It says nothing about the SLM rows, which have no independent replication.

## 5. Interpretation (preliminary)

**SLM vs LSA.** No SLM size beat LSA on mean ACC, NMI or ARI. The 0.6B and 1.7B models lost to LSA on ARI in 5 of 5 seeds and the 0.6B on NMI in 5 of 5. The 4B model wins the most seeds against LSA (NMI on seeds 1 and 2, ARI on 3 of 5; 1.7B wins NMI only on seed 3), but its mean paired difference is still negative and its seed-to-seed spread (paired-difference std about 0.010 NMI) is larger than the gap. We cannot distinguish 4B from LSA with this data.

**SLM vs k-means.** Mean NMI is higher than k-means for all three sizes (+0.002, +0.005, +0.005) and 1.7B wins in 5 of 5 seeds. ARI is mixed (-0.002, -0.003, +0.005). These differences are small (a few thousandths of NMI) and we would not claim that SLM summaries improve on k-means from this experiment alone.

**Size trend.** There is no clean monotone trend. NMI rises from 0.645 (0.6B) to 0.648 (1.7B, 4B), ACC and ARI are non-monotone (1.7B is lowest on both), and the 4B has the highest mean ACC and ARI of the SLMs. The 4B is also the least stable: its std is 0.009 NMI (vs 0.002 / 0.004), 0.016 ACC and 0.014 ARI, driven by seed 3 (NMI 0.634, ACC 0.418, ARI 0.278), where it is the worst of all five methods; on seeds 1 and 2 it is the best. The only truncation warning in the 4B log appears during seed 4 (Section 7), so seed 3's drop is not explained by prompt truncation. We have no explanation for it beyond seed-level variation, and with 5 seeds one outlier moves the mean and std considerably.

**Cost.** Mean wall-clock per seed: LSA 1.6 s, 0.6B 102.7 s, 1.7B 229.0 s, 4B 381.9 s (k-means 0.3 s). The SLM methods are roughly 64x, 143x and 239x slower than LSA. Caveats: (i) LSA's time excludes spaCy sentence segmentation, which is cached; (ii) SLM time also depends on the batch size, which differed per model (8 / 4 / 2), so the ratios between model sizes mix model size and batching; (iii) these are single-run wall-clock times on one consumer GPU under WSL2, not a controlled cost measurement (see Issue #5).

All conclusions above are preliminary: one dataset, 5 seeds, one prompt, greedy decoding.

## 6. Qualitative examples (seed 0)

**Selection rule (deterministic, not hand-picked).** For each of the 4 summary methods, take the final summaries and assignments at seed 0. For each gold intent, the best-matching cluster is the one holding the most of its members (the `find_examples` max-count rule), and its score is overlap / gold class size. An intent's score is the minimum of this across the 4 methods, i.e. how well *every* method recovers it. We show the top 2 intents (`--top-recovered`; ties broken by class id) plus `card_arrival` as a harder case chosen by us. The top 5 were: verify_source_of_funds 0.975, terminate_account 0.925, visa_or_mastercard 0.900, card_about_to_expire 0.850, exchange_rate 0.825. Each entry is `(overlap/gold size, cluster size)`.

**verify_source_of_funds** (40 gold queries)

- LSA: "How can I check the source for my funds? How do I do this? My source of funds need verified. Where do my funds come from? Do I get charged if i am getting money?" (40/40, 59)
- 0.6B: "\"How can I verify the source of my funds?\"" (40/40, 58)
- 1.7B: "What is the source of my funds and how can I verify it?" (40/40, 78)
- 4B: "What are the methods and processes for sending, receiving, and verifying the source of funds in online banking?" (39/40, 40)

**terminate_account** (40 gold queries)

- LSA: "Can you delete my account? I am sick of this damn company and want to close out my account. This company isn't good, I would like to delete my account. Please delete my account. Can you delete my account?" (39/40, 51)
- 0.6B: "\"Please delete my account.\"" (37/40, 42)
- 1.7B: "How do I delete my account and get help in case my card or phone is lost?" (39/40, 44)
- 4B: "How do I freeze or delete my account?" (39/40, 52)

**card_arrival** (40 gold queries; harder, no method recovers it well)

- LSA: "Where are cards delivered to? Where can I use my card? How do I top up my card? When will my card arrive? Where will I find my card?" (10/40, 113)
- 0.6B: "What is the tracking number for my card that was mailed?" (11/40, 100)
- 1.7B: "I am still waiting for my card to arrive, and my transaction is pending. What can I do?" (14/40, 54)
- 4B: "\"I live in the US but want to get a credit card—will my existing card be accepted, and how can I apply for a new one?\"" (12/40, 24)

Observations: on easy intents the SLM summaries are a single fluent question, whereas LSA concatenates five extracted sentences (which can include off-topic ones, e.g. "How do I top up my card?" in card_arrival). In the hard case the SLM summaries drift toward one sub-topic of the cluster (tracking number, pending transaction, applying for a card). Max-count matching favors large clusters: LSA's best match for card_arrival is a 113-query cluster that holds only 10 of the 40 gold queries, so the displayed overlap (10/40) says little about how well its summary represents the intent. These are three hand-read illustrations at one seed, not evidence of a systematic difference.

## 7. Observations and quirks

- **Stray double quotes in summaries (seed 0, final summaries, 77 per model).** Counts of summaries that start with `"` (in every case they also end with `"`): 0.6B **60/77**, 1.7B **0/77**, 4B **3/77**. A previous smoke run of 0.6B saw 59/77. The quotes are embedded as part of the summary text, so they may shift the 0.6B centroids; we have not measured the effect (see Next steps).
- **No thinking text:** `<think>` / `</think>` appears in 0 of 1,925 summaries per model across the evolution of all 5 seeds (77 clusters x 5 steps x 5 seeds), and in none of the final summaries.
- **Prompt truncation (4B):** the log `results/prelim_logs/prelim_bank77_qwen3-4b.log` contains one warning: "a cluster's rendered prompt exceeded max_input_tokens=4096; dropped 1 of 128 text(s). (only warned once for this summarizer instance)". In the log it appears during seed 4's fifth summary step. Because it is warned once per summarizer instance, the log does not rule out further drops within that step; no warning appeared in the 0.6B or 1.7B logs, and none before seed 4 in the 4B log. The affected cluster had 128 members, so one summary of one cluster lost one query out of 128; the effect on the metrics is likely negligible but was not measured.
- **Qwen3-8B was not run** because it does not fit in 12 GB of VRAM.

## 8. Limitations and next steps

**Limitations**

- One dataset (BANKING77), 5 seeds; differences of a few thousandths are within seed noise.
- Greedy decoding and the upstream prompt, unchanged; no prompt or decoding sensitivity.
- Batch sizes differ per model (8 / 4 / 2), which confounds the timing comparison across sizes.
- No cost logging beyond wall-clock time (no tokens, FLOPs or energy).
- Only the Qwen3 family; no models of other families or above 4B.

**Next steps**

- #5: cost logging (tokens, per-step time).
- #6 / #11: fuller model ladder (Llama-3.2-3B, Phi-4-mini, 8B on a larger GPU).
- #4: CLINC150, to see whether the pattern holds on a second dataset.
- #7 / #8: genericity and label-alignment metrics, to measure what the summaries look like beyond ACC/NMI/ARI.
- Ablation: strip surrounding quotes from summaries before embedding (cheap; mainly affects 0.6B).
- More seeds (the paper protocol uses 10) before drawing any conclusion about differences of this size.

## 9. Reproducibility

- Git branch: `prelim-bank77-qwen3-sizes`; tracking issue: #20.
- Run directories (under `results/`, gitignored):
  - `prelim_bank77_lsa/20261014-105249`
  - `prelim_bank77_qwen3-0.6b/20261014-105306`
  - `prelim_bank77_qwen3-1.7b/20261014-110159`
  - `prelim_bank77_qwen3-4b/20261014-112130`
- Run and table: `PYTHON=... scripts/run_prelim.sh`, then `scripts/run_prelim.sh table`.
- Regenerate the files in `docs/results/` from the run directories (k-means rows come from the first run, LSA):
  ```bash
  RUNS="results/prelim_bank77_lsa/20261014-105249 results/prelim_bank77_qwen3-0.6b/20261014-105306 \
        results/prelim_bank77_qwen3-1.7b/20261014-110159 results/prelim_bank77_qwen3-4b/20261014-112130"
  python scripts/aggregate_results.py $RUNS --out-csv docs/results/prelim_bank77_table.csv \
      --per-seed-csv docs/results/prelim_bank77_per_seed.csv
  # Example selection (seed 0): top-2 intents and the hard case
  python scripts/aggregate_results.py $RUNS --top-recovered 5 --seed 0 \
      --examples verify_source_of_funds terminate_account card_arrival
  ```
- Files: `docs/results/prelim_bank77_table.csv` (aggregate), `docs/results/prelim_bank77_per_seed.csv` (columns `method, seed, acc, nmi, ari, runtime_s`). The win counts and mean paired differences in Section 3 are computed from the per-seed CSV (strictly-greater comparisons by seed). The quote and `<think>` counts come from `summaries.jsonl` in each run directory.
