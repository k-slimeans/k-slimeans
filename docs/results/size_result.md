# Why the larger model is not better here (#36)

Issue #36, written for the final feedback of #84 ("a short section built from the existing runs"; it feeds #50). **No new runs.** Every number comes from a committed file in `docs/results/`, named where it is used, or from [`size_result_table.csv`](size_result_table.csv), which `scripts/size_result.py` builds from those files (Section 3).

**Scope, in one place.** Qwen3-0.6B, 1.7B and 4B checkpoints, greedy decoding, thinking off, the prompts named below, one sentence encoder per dataset, one RTX 3060 12 GB, 5 seeds (0-4). This is a statement about these three checkpoints under these prompts. It is not a statement about parameter count in general, and it makes no size-threshold claim.

**Short answer.**

- On CLINC150 the October order (1.7B above 0.6B above 4B) came largely from how the 4B read that prompt. Under the clearer prompt (#46) no ACC order remains and the data do not show a size effect on ACC; a small NMI edge for the 4B (+0.006, 5 of 5 seeds over the 1.7B, uncorrected) is the one exception.
- BANKING77 and BBC show no separated size difference at all. Five seeds could not have separated differences of the size seen there.
- Size, prompt reading and batch size vary together in these runs, so none of them can be called *the* cause of the missing trend.

## 1. The result as measured

Paired by seed (same step-0 partition), larger size minus smaller size, mean difference [95% Student-t interval over 5 seeds, 4 df], seeds on which the larger size is higher. From `prompt_test_paired.csv` (CLINC150, both prompts), `bbc_tfidf_paired.csv` and `bbc_dense_paired.csv`, and for BANKING77 from `prelim_bank77_per_seed.csv` with `aggregate.paired_differences`. The intervals are per comparison and not corrected for the 45 below. A seed counts as a win only if the difference is above 0; BBC dense has tied seeds (the `ties` column of the CSV), so wins and losses need not sum to 5.

| Dataset | Pair | ACC | NMI | ARI |
| --- | --- | --- | --- | --- |
| CLINC150, October prompt | 1.7B minus 0.6B | +0.029 [+0.009, +0.048], 5/5 | +0.013 [+0.009, +0.017], 5/5 | +0.024 [+0.015, +0.033], 5/5 |
| CLINC150, October prompt | 4B minus 0.6B | -0.030 [-0.051, -0.008], 0/5 | -0.008 [-0.015, -0.001], 1/5 | -0.017 [-0.029, -0.004], 0/5 |
| CLINC150, October prompt | 4B minus 1.7B | -0.059 [-0.072, -0.045], 0/5 | -0.021 [-0.027, -0.015], 0/5 | -0.040 [-0.053, -0.028], 0/5 |
| CLINC150, clearer prompt | 1.7B minus 0.6B | +0.000 [-0.002, +0.003], 4/5 | -0.001 [-0.006, +0.004], 3/5 | +0.000 [-0.008, +0.009], 3/5 |
| CLINC150, clearer prompt | 4B minus 0.6B | +0.005 [-0.002, +0.012], 4/5 | +0.006 [+0.000, +0.011], 4/5 | +0.004 [-0.005, +0.013], 3/5 |
| CLINC150, clearer prompt | 4B minus 1.7B | +0.005 [-0.003, +0.012], 4/5 | +0.006 [+0.002, +0.011], 5/5 | +0.003 [-0.003, +0.010], 3/5 |
| BANKING77 (#20, upstream prompt) | 1.7B minus 0.6B | -0.003 [-0.012, +0.006], 3/5 | +0.003 [-0.003, +0.009], 4/5 | -0.001 [-0.012, +0.010], 3/5 |
| BANKING77 (#20, upstream prompt) | 4B minus 0.6B | +0.004 [-0.026, +0.034], 4/5 | +0.004 [-0.007, +0.015], 4/5 | +0.007 [-0.016, +0.029], 3/5 |
| BANKING77 (#20, upstream prompt) | 4B minus 1.7B | +0.007 [-0.021, +0.035], 4/5 | +0.001 [-0.014, +0.016], 4/5 | +0.008 [-0.012, +0.028], 4/5 |
| BBC TF-IDF (#18) | 1.7B minus 0.6B | +0.022 [-0.018, +0.062], 4/5 | +0.034 [-0.015, +0.083], 4/5 | +0.049 [-0.035, +0.134], 4/5 |
| BBC TF-IDF (#18) | 4B minus 0.6B | +0.003 [-0.037, +0.043], 2/5 | +0.005 [-0.038, +0.048], 3/5 | +0.002 [-0.087, +0.091], 2/5 |
| BBC TF-IDF (#18) | 4B minus 1.7B | -0.019 [-0.054, +0.017], 2/5 | -0.029 [-0.070, +0.012], 1/5 | -0.047 [-0.128, +0.034], 2/5 |
| BBC dense (#18) | 1.7B minus 0.6B | -0.057 [-0.215, +0.101], 1/5 | -0.034 [-0.129, +0.060], 2/5 | -0.059 [-0.221, +0.104], 2/5 |
| BBC dense (#18) | 4B minus 0.6B | -0.058 [-0.219, +0.102], 1/5 | -0.041 [-0.133, +0.052], 1/5 | -0.061 [-0.226, +0.104], 1/5 |
| BBC dense (#18) | 4B minus 1.7B | -0.002 [-0.007, +0.003], 2/5 | -0.006 [-0.019, +0.007], 2/5 | -0.002 [-0.012, +0.007], 2/5 |

Where the order holds, flips or vanishes:

- **CLINC150, October prompt: a clear, non-monotone order.** All nine intervals exclude 0: 1.7B above 0.6B above 4B (`prompt_test.md` section 8.7; ACC means 0.570 / 0.541 / 0.511). The 4B is the worst method of the five, below k-means and LSA in 5 of 5 seeds (`clinc150_summarizers.md` section 4).
- **CLINC150, clearer prompt: the ACC order vanishes.** ACC means are 0.563 / 0.563 / 0.568 (0.6B / 1.7B / 4B) and all six ACC and ARI intervals include 0. One thing remains: on NMI the 4B is ahead of the 1.7B by +0.006 [+0.002, +0.011] (5 of 5 seeds), and of the 0.6B by +0.006 whose lower bound is +0.0004. That is one metric of three, 5 seeds, among nine uncorrected intervals, and it does not separate the 0.6B from the 1.7B. `prompt_test.md` section 8.2 does not read it as an order, and nor does this note.
- **BANKING77: no separated order and no consistent direction.** The signs differ by metric: the 1.7B is below the 0.6B on ACC and ARI and above it on NMI. The 4B has the highest means (ACC 0.444, NMI 0.648, ARI 0.300, with the 0.6B 0.440 / 0.645 / 0.293 and the 1.7B 0.436 / 0.648 / 0.292, `prelim_bank77.md` section 2) and the widest spread, which the 4B's seed 3 drives (`prelim_bank77.md` section 5). This run has only the upstream prompt: BANKING77 under the clearer prompt was not run, so the October-versus-clearer contrast exists on CLINC150 only.
- **BBC: the order flips with the embedding.** TF-IDF has 1.7B above 4B above 0.6B in ACC; dense has the 0.6B first. Every one of the 18 intervals includes 0. The dense gaps against the 0.6B come from the two bad step-0 partitions of that run (`bbc.md` section 3), and `bbc.md` section 8 does not use the dense order as evidence. The narrow dense 4B-minus-1.7B interval is narrow because those two sizes end in nearly the same basin, not because the difference is well measured (`bbc.md` section 2.2).

So the order of the sizes changes between datasets (`clinc150_summarizers.md` section 4), between embeddings on one dataset (BBC), and between prompts on one dataset (CLINC150). A size order that moves with these has not been shown to be a property of size.

## 2. What explains it, and what kind of evidence each piece is

"Interventional" means the experimenter changed the input and re-ran or replayed. "Association" means a logged property goes with the outcome and nothing was changed.

**(a) The prompt. Interventional, CLINC150 only, several wording changes at once.** Re-running all three sizes with the clearer prompt on the same seeds and step-0 partitions (`prompt_test.md` sections 8.2 and 8.7) changed final ACC by +0.056 [+0.039, +0.074] for the 4B (5 of 5 seeds), +0.022 [+0.010, +0.033] for the 0.6B (5 of 5), and -0.007 [-0.023, +0.010] for the 1.7B (1 of 5, not separated from 0). The 4B moved from below k-means in 5 of 5 seeds to above it in 5 of 5. The prompt change altered the task wording, the "same style" request, "write only the question" and the banned words together, so it does not isolate any one failure (`prompt_test.md` section 8.6). It was chosen by the 4B's surface counts before any scoring (section 3 there), so the gains at the 0.6B and 4B and the null at the 1.7B are not a ranking of how the sizes like the prompt. Batch sizes were the same under both prompts (8 / 4 / 1), so batching cannot explain this contrast.

**(b) The 4B's failure modes under the October prompt.**

- *The summaries. Association.* Over the 3,750 summaries per run, 34% of the 4B's mention the cluster ("cluster" or "summar"), against 8% for the 1.7B and 0.6B; at the final step 16-21 clusters per seed share an identical summary (up to 13 in one group), against 0-4 and 0-5 (`prompt_test.md` section 8.3). Its "not fluent" share is 34%, against 28% for the 0.6B and 9% for the 1.7B, and its mean nearest-summary share (a coverage score) is lowest at 0.49 against 0.60 and 0.60 (`summary_quality.md` section 3). Under the clearer prompt mentions fall to 0 at every size and the 4B's duplicate groups to 0-2 clusters per seed. More of the 4B's prompts overflow the 4,096-token limit (10.2 clusters per seed against 6.6 and 6.8, ranges overlapping), also an association (`clinc150_summarizers.md` section 4).
- *Step-1 swaps. Interventional, at step 1 only (#47, `counterfactual_swaps.md` section 2).* Putting the 1.7B's summaries into the 4B's flagged clusters (43-56 of 150 per seed: duplicates and cluster mentions) raises the 4B's step-1 ACC by +0.026 [+0.019, +0.034] in 5 of 5 seeds, 99% of the 4B-to-1.7B gap, against -0.002 [-0.012, +0.007] for the same number of random unflagged clusters. Putting the 4B's flagged summaries into the 1.7B costs ACC against a matched control (same number of random unflagged clusters, r minus rd): -0.0248 [-0.0410, -0.0086]. That holds for the union flag only; the generic and duplicate-only intervals include 0 (`synthesis.md` section 10). Flagged clusters may also be harder clusters, and only step 1 was tested.
- *The guarded run. Interventional over five steps, and it does not recover the score (#47 stage 2, `summary_guard.md` section 1).* Starting flagged clusters from their k-means center at every step moves the 4B's final ACC by +0.009 [-0.006, +0.024] (4 of 5 seeds), and it stays 0.031 below k-means. An effect up to +0.024 is not excluded, and the guard replaces a summary with no summary, not with a better one.
- *Reading.* The flagged summaries differ from the 1.7B's in a way that matters at step 1. Whether that carries through to the 4B's final deficit is not shown by the swaps or the guard. The clearer prompt, which changes every summary and not only the flagged third, does remove the deficit (a). The best-supported statement is that the October 4B deficit goes with how the 4B read that prompt, not that cluster mentions alone caused it.

**(c) The 0.6B's quote wrapping and verbatim copies. Association, and the effect on scores is not measured.**

- On BANKING77, 60 of 77 seed-0 final summaries of the 0.6B start with a double quote, against 0 for the 1.7B and 3 for the 4B (`prelim_bank77.md` section 7); 83% of its last-step summaries trip the quote proxy (`summary_quality.md` section 3, an approximation on BANKING77). On CLINC150 quote wrapping is rare (2 of 3,750 for the 0.6B under the October prompt, 13 under the clearer one, `prompt_test.md` section 8.3). The quotes are embedded as part of the summary text. Stripping them before embedding (#36 question 3) was not run, so how much they move the 0.6B's BANKING77 scores is unknown.
- Verbatim copies of a member: 0.6B 15% (October) and 33% (clearer) of CLINC150 summaries, 4B 1% and 12%, 1.7B 1% and 1% (`summary_quality.md` section 3, clearer-prompt table; `synthesis.md` section 9). The two runs that gained most from the clearer prompt (0.6B, 4B) are the two with the most copies; the 1.7B has few and did not gain. That is three runs, and the link to ACC was not tested, so copying is not claimed as a cause or as harmless.

**(d) Batch size, which differs by model, so size and batching are confounded. Interventional for the summaries, not for the scores.** The runs used batch 8 / 4 / 1 for 0.6B / 1.7B / 4B on CLINC150 and BBC, and 8 / 4 / 2 on BANKING77 (`cost_justification.md` section 1; `prelim_bank77.md` section 1). Every size has one batch size per dataset, so batch size cannot be separated from size in the clustering results. Greedy decoding at a fixed batch size reproduces the logged summaries exactly (150 of 150 at every size), but at batch 1 instead of the logged batch 25 of 150 of the 0.6B's step-1 summaries (17%) and 42 of 150 of the 1.7B's (28%) differ (`cost_timing.md` section 5). So a model at two batch sizes is two slightly different summarizers. Whether the changed summaries cluster differently was not measured, and the 4B does not fit a larger batch on this card, so a size comparison at fixed batch size was not possible.

**(e) Failure modes read by hand, by size and prompt. Association, descriptive, one seed and one step, AI-assisted tags.**

The #36 review packet has 25 CLINC150 clusters at step 1 of seed 0, five from each of five purity bins. Every run starts step 1 from the same partition, so each cluster's six summaries (three sizes, two prompts) describe the same members. Each summary was tagged against the ten members shown in the packet (`size_failure_modes.md`, `size_failure_modes_tags_filled.csv`). The tags were made with AI assistance, as the annotation file's name says, and were not re-checked by a second reader.

Counts out of 25:

| tag | 0.6B old | 0.6B v2 | 1.7B old | 1.7B v2 | 4B old | 4B v2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| echoes the instruction | 1 | 0 | 1 | 0 | 8 | 0 |
| mentions the cluster or the questions | 1 | 0 | 2 | 0 | 9 | 0 |
| answers instead of asking | 0 | 0 | 0 | 0 | 2 | 0 |
| no "?" | 0 | 2 | 0 | 0 | 0 | 10 |
| covers only one sub-group | 13 | 17 | 3 | 12 | 1 | 10 |
| adds a detail no shown member has | 11 | 11 | 10 | 6 | 3 | 2 |

Quote marks (1, 1, 0, 0, 0, 2) and copies of a shown member (0 everywhere) are in the source table.

- **They agree with the surface counts of (b).** The old 4B mentions the cluster in 9 of 25 (34% over the full run), the smaller sizes in 1 and 2 of 25 (8%), and no v2 summary does.
- **They narrow (b).** Under the October prompt the 4B's failure is task-level talk about the cluster. The 0.6B fails differently: it covers only one sub-group in 13 of 25 and adds a detail absent from the shown members in 11 of 25, against 1 and 3 for the 4B. Part of the 4B's low sub-group count is that a task-level summary covers no sub-group at all, which the tag does not count. So the 4B is the size with the October prompt-reading failure, not the only size with failures.
- **The clearer prompt narrows coverage.** Under it, cluster talk is gone at every size, but covering only one sub-group goes up at every size: 0.6B 13 to 17, 1.7B 3 to 12, 4B 1 to 10. This fits the scope note of Section 4: the clearer prompt removed a surface failure and was not shown to make summaries more faithful. The 4B's v2 summaries lack a "?" in 10 of 25, which often follows the members' informal style rather than marking a failure of content (`size_failure_modes.md`).
- **Limits.** "Adds a detail" and "one sub-group" are judged from ten shown members, so an added detail is not a confirmed invention. These are 25 clusters at one step of one seed, tagged once.

## 3. What five seeds can and cannot separate

**Method.** The 45 rows of [`size_result_table.csv`](size_result_table.csv) are the three size pairs by three metrics for each of the five settings in Section 1. For each, `sd` is the sample standard deviation (ddof = 1) of the 5 per-seed paired differences, copied from `prompt_test_paired.csv`, `bbc_tfidf_paired.csv` and `bbc_dense_paired.csv`, or computed from `prelim_bank77_per_seed.csv` by `aggregate.paired_differences`. The 95% interval half-width is t(0.975, 4) * sd / sqrt(5) = 2.776 * sd / 2.236 = 1.242 * sd.

**Minimum detectable paired difference.** The central-t approximation for a two-sided paired t-test at alpha = 0.05 and power 0.8, n = 5:

    MDD = (t(0.975, 4) + t(0.80, 4)) * sd / sqrt(5) = (2.776 + 0.941) / 2.236 * sd = 1.662 * sd

(`kslimeans.size_result.min_detectable`). The same question solved with the noncentral t distribution (`min_detectable_exact`, power from the upper rejection region) gives 1.682 * sd, 1.2% above the approximation in every row (`min_detectable_exact` column of the CSV), so the approximation is used. Both treat the observed `sd` as the true one. With 4 degrees of freedom that is a loose estimate: the 95% range of the true sd is about 0.6 to 2.9 times the observed value, so each MDD below is indicative to a factor of about two, not a sharp threshold.

| Setting | Metric | SD of the three paired size differences | Minimum detectable | Observed gaps, absolute | Gaps below the minimum (of 3) | Intervals excluding 0 (of 3) |
| --- | --- | --- | --- | --- | --- | --- |
| CLINC150, October prompt | ACC | 0.011 to 0.017 | 0.018 to 0.029 | 0.029 to 0.059 | 0 | 3 |
| CLINC150, October prompt | NMI | 0.003 to 0.006 | 0.005 to 0.010 | 0.008 to 0.021 | 1 | 3 |
| CLINC150, October prompt | ARI | 0.007 to 0.010 | 0.012 to 0.017 | 0.017 to 0.040 | 1 | 3 |
| CLINC150, clearer prompt | ACC | 0.002 to 0.006 | 0.003 to 0.010 | 0.000 to 0.005 | 3 | 0 |
| CLINC150, clearer prompt | NMI | 0.004 to 0.004 | 0.006 to 0.007 | 0.001 to 0.006 | 2 | 2 |
| CLINC150, clearer prompt | ARI | 0.005 to 0.007 | 0.009 to 0.012 | 0.000 to 0.004 | 3 | 0 |
| BANKING77 | ACC | 0.007 to 0.024 | 0.012 to 0.040 | 0.003 to 0.007 | 3 | 0 |
| BANKING77 | NMI | 0.005 to 0.012 | 0.008 to 0.020 | 0.001 to 0.004 | 3 | 0 |
| BANKING77 | ARI | 0.009 to 0.018 | 0.015 to 0.030 | 0.001 to 0.008 | 3 | 0 |
| BBC TF-IDF | ACC | 0.029 to 0.032 | 0.048 to 0.054 | 0.003 to 0.022 | 3 | 0 |
| BBC TF-IDF | NMI | 0.033 to 0.039 | 0.055 to 0.065 | 0.005 to 0.034 | 3 | 0 |
| BBC TF-IDF | ARI | 0.065 to 0.072 | 0.108 to 0.119 | 0.002 to 0.049 | 3 | 0 |
| BBC dense | ACC | 0.004 to 0.129 | 0.007 to 0.215 | 0.002 to 0.058 | 3 | 0 |
| BBC dense | NMI | 0.010 to 0.076 | 0.017 to 0.127 | 0.006 to 0.041 | 3 | 0 |
| BBC dense | ARI | 0.008 to 0.133 | 0.013 to 0.221 | 0.002 to 0.061 | 3 | 0 |

What this says:

- **Separable: the October CLINC150 gaps.** Its ACC gaps (0.029 to 0.059) are above the MDD (0.018 to 0.029), and all nine intervals exclude 0. Two NMI and ARI gaps (4B against 0.6B, 0.008 and 0.017) are below it, although their intervals just exclude 0.
- **Not separable: everything else.** Of the 45 gaps, 37 are below their MDD. All 18 BBC and 9 BANKING77 intervals include 0 (none excludes it), and in the clearer-prompt CLINC150 runs only the two NMI intervals (4B above each smaller size) exclude 0, by +0.006 against an MDD of 0.006 to 0.007.
- **The detectable size is not small next to the gaps being discussed.** Under the clearer prompt, 5 seeds detect an ACC difference of about 0.003 to 0.010 between sizes. The observed ACC gaps (0.000 to 0.005) are inside that range: a true gap of 0.005 would be missed about as often as found (`prompt_test.md` section 8.6 says the same: "differences of about 0.005 ACC between the sizes cannot be ruled out"). On BANKING77 the MDD for ACC is 0.012 to 0.040 and the observed gaps are 0.003 to 0.007, and on BBC the MDD for TF-IDF ACC is about 0.05, so those datasets could not have shown a size gap of the size CLINC150 shows.
- **"Absent" is not "zero".** An interval that includes 0 is compatible with no difference and with a difference up to its limit (+0.012 ACC for the clearer-prompt 4B over the 1.7B).
- **What the seeds vary.** A seed changes the k-means start (the step-0 partition), and the same partition is given to every method, which is why the paired intervals are tighter than the unpaired spread. It does not vary the test set, the prompt, the checkpoint or the batch size. The intervals therefore describe variation over starting partitions for this dataset and these checkpoints, not variation over datasets, prompts or checkpoints. A fifth-seed result cannot say that a different dataset or another checkpoint of the same size would behave the same.
- **Multiplicity.** The 45 intervals are uncorrected. At 95%, chance alone would put about 2 of 45 away from 0. Eleven are: the 9 October CLINC150 ones, which are too many and too large to be chance, and the 2 clearer-prompt NMI ones, which alone are about the number chance would give. The 45 comparisons share seeds and partitions and are correlated, so the count of 11 is clustered rather than 11 independent results. That is another reason not to read the NMI edge as an order.
- **Seeds 5-9 were dropped** by the Nov 18 feedback (#84); ten seeds would cut the factor from 1.662 to about 1.0 (t(0.975, 9) + t(0.80, 9) over sqrt(10)). That is what was not bought, not a result.

## 4. Scope statement

- **Checkpoints and decoding.** Qwen3-0.6B, 1.7B and 4B, thinking off, greedy decoding, `max_new_tokens` 64. One model family: the sizes also differ in architecture and in how they were trained (Section 5), so "size" here means "these three checkpoints".
- **Prompts.** CLINC150: the dataset's own prompt (October) and one clearer prompt written before any run (#46); BANKING77, BBC: the upstream or dataset prompt only. No prompt search, and no per-size tuning.
- **Data and encoders.** CLINC150 (150 intents) and BANKING77 (77) with one sentence encoder; BBC (500 articles, k = 5) in TF-IDF and in a dense space; 5 seeds.
- **Hardware.** One 12 GB consumer GPU under WSL2, which fixes the batch sizes (the 4B only fits at batch 1 on CLINC150) and so ties batch size to model size.
- **Not claimed.** Nothing here says larger models are not better in general, that there is a size at which quality rises, or that the clearer prompt is a good prompt (it removed a surface failure; it was not shown to make summaries more faithful, `prompt_test.md` section 8.3).

## 5. How the three checkpoints differ in training (#36 question 6)

From the Qwen3 technical report (arXiv:2505.09388v1), summarized in `docs/qwen3_sizes_notes.md`. The numbers below were checked against the report on 2026-11-27.

- **Architecture (Table 1).** The 0.6B and 1.7B have 28 layers, 16 query and 8 key-value heads and a 32K context. The 4B has 36 layers, 32 query and 8 key-value heads and a 128K context. All three tie their embeddings. Our runs cap the input at 4,096 tokens, so the context difference does not apply.
- **Pre-training (Sections 3.1 to 3.2).** One recipe for the family: about 36 trillion tokens in 119 languages and dialects, in three stages (general, reasoning, then long context at 32,768 tokens), with learning rate and batch size set per model from scaling laws. The report gives no per-checkpoint data mixture or token count.
- **Post-training (Section 4.5).** All three are "lightweight" models trained by strong-to-weak distillation from a larger teacher (Qwen3-32B or Qwen3-235B-A22B): off-policy first, then on-policy, matching the teacher's logits by KL divergence. The report does not say which teacher, data or budget each student got.
- **Instruction following, non-thinking mode (Tables 18 and 20).** For 0.6B / 1.7B / 4B:
  - IFEval strict prompt: 54.5 / 68.2 / 81.2;
  - MMLU-Redux: 44.6 / 64.4 / 77.3;
  - WritingBench: 5.13 / 6.54 / 6.85.

**What this adds:**
- The checkpoints differ in more than parameter count: depth, heads, and a distillation whose details the report leaves out. So "size" here stays "these three checkpoints".
- The 4B scores highest on a general instruction-following benchmark, yet it read our October prompt worst (Section 2(b) and (e)). Following instructions in general is not the same property as producing a summary that works as a centroid.
- With one family and no controlled test, this describes these checkpoints. It does not explain the size result (`qwen3_sizes_notes.md` section 6).

## 6. Reproduce

    PYTHONPATH=src python scripts/size_result.py        # writes docs/results/size_result_table.csv
    PYTHONPATH=src python -m pytest tests/test_size_result.py

Reads only committed files in `docs/results/`; no GPU, no model, no run directory.
