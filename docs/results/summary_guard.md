# Do the 4B's flagged summaries cause its lower final scores? The guarded run (CLINC150, #47 stage 2)

Issue #47, stage 2. Preliminary: five seeds, one dataset, one encoder, one model. The guard is a diagnostic variant, not a method (README, "Summary guard"). It is written up here, not in `counterfactual_swaps.md`, because it is a different object: a new GPU run of five summary steps, with its own scores and cost, where stage 1 is an offline edit of one logged step. Stage 1 is cited where it explains a number.

**Setup.** `configs/clinc150_guard.yaml` is the #31 protocol for the Qwen3-4B (seeds 0-4, batch size 1, five summary steps, 150 clusters, 22,500 points) with one change. At every summary step a cluster is flagged when its summary is an exact duplicate of another cluster's, or mentions the cluster ("cluster" or "summar", any case). A flagged cluster starts that step's Lloyd block from its previous k-means center instead of its summary embedding. The summaries are still generated and logged, and `ref`, the summary embeddings and the `tol` shift stay on the summaries. The flag rules were fixed before the run. Comparison runs are the #31 4B, the #31 1.7B, and the k-means and LSA rows of the #31 LSA run, all with the same `step0_sha1` per seed (checked: shared for seeds 0-4). Everything below is computed from what the runs logged (`trace.jsonl`, `summaries.jsonl`, `metrics.jsonl`, `gpu.json`); nothing was re-embedded and no model was called.

## 1. Final scores, paired by seed

Means over 5 seeds (`summary.csv`; the first four rows are the table `run_clinc150_guard.sh` builds, plus the 1.7B):

| Method | ACC | NMI | ARI |
| --- | ---: | ---: | ---: |
| k-means | 0.551 | 0.748 | 0.439 |
| LSA | 0.560 | 0.752 | 0.449 |
| 4B (#31) | 0.511 | 0.740 | 0.421 |
| 4B guarded | 0.521 | 0.741 | 0.414 |
| 1.7B (#31) | 0.570 | 0.760 | 0.461 |

Paired differences, guarded 4B minus the reference, mean over 5 seeds [95% t-interval, n = 5, t = 2.776], seeds won by the guarded run (`summary_guard_paired.csv`, from `aggregate.paired_differences`):

| reference | ΔACC | ΔNMI | ΔARI |
| --- | --- | --- | --- |
| 4B (#31) | +0.0093 [-0.0059, +0.0244] 4/5 | +0.0010 [-0.0069, +0.0089] 2/5 | -0.0069 [-0.0193, +0.0055] 1/5 |
| k-means | -0.0310 [-0.0429, -0.0190] 0/5 | -0.0076 [-0.0146, -0.0006] 1/5 | -0.0249 [-0.0334, -0.0165] 0/5 |
| LSA | -0.0399 [-0.0509, -0.0288] 0/5 | -0.0119 [-0.0186, -0.0052] 0/5 | -0.0353 [-0.0430, -0.0277] 0/5 |
| 1.7B (#31) | -0.0494 [-0.0685, -0.0303] 0/5 | -0.0199 [-0.0303, -0.0096] 0/5 | -0.0474 [-0.0655, -0.0292] 0/5 |

- **The guard does not change the final scores detectably.** All three intervals against the unguarded 4B include zero. ACC is higher in 4 of 5 seeds (seed 1 is lower, -0.0058; the others +0.0044 to +0.0269), ARI is lower in 4 of 5.
- **Share of the gap closed** (mean gain over mean gap, `summary_guard_gap_closed.csv`): to k-means 23% of the ACC gap (gap 0.0402), 11% of NMI, and a negative share for ARI (-38%); to LSA 19% (ACC); to the 1.7B 16% (ACC). These are ratios of means. The gain's interval spans zero, so for ACC to k-means the share is anywhere from about -15% to +61% if the gap is taken as fixed (the gain interval over 0.0402). "About a quarter" is the point estimate, not a measured quantity.
- **The remaining gaps are large and clear:** the guarded 4B is still below k-means by 0.031 ACC (0/5 seeds), below LSA by 0.040 and below the 1.7B by 0.049.

## 2. What was guarded

Trace rows give `n_guarded`, `n_guarded_duplicate` and `n_guarded_mentions` per summary step. Every trace count equals a recount of the logged summaries by the surface rules (`summary_guard_check.txt`). Step `t` below is the Lloyd block that follows summary step `t - 1`, so step 1 is stage 1's step 1. Means over seeds [min-max], `summary_guard_counts.csv`:

| step | guarded (union) | share of 150 | duplicate | mentions | both | duplicate only | mentions only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 50.2 [43-56] | 33.5% | 18.8 [12-24] | 50.2 | 18.8 | 0 | 31.4 |
| 2 | 49.0 [45-53] | 32.7% | 16.8 [7-24] | 49.0 | 16.8 | 0 | 32.2 |
| 3 | 50.6 [46-55] | 33.7% | 17.8 [11-27] | 50.6 | 17.8 | 0 | 32.8 |
| 4 | 52.4 [49-57] | 34.9% | 17.8 [15-25] | 52.4 | 17.8 | 0 | 34.6 |
| 5 | 47.8 [45-52] | 31.9% | 12.6 [9-15] | 47.8 | 12.6 | 0 | 35.2 |

- **Why:** in all 25 seed-steps every duplicate summary also mentions the cluster, so the union equals the "mentions" set and "duplicate only" is empty (as in stage 1). A flagged cluster is a "mentions" cluster; about a third of those are also duplicates (419 duplicate clusters of 1,250 guarded cluster-steps in total).
- **Trend:** the guarded share is flat, 32% to 35% at every step, with no decay across the five steps; 1,250 of 3,750 cluster-steps were guarded. Duplicates fall at the last step (12.6 against 17-19) while mentions-only clusters rise (31 to 35); five seeds do not say whether that is more than noise.
- **Against stage 1:** step 1 is the same count: 49, 43, 48, 55, 56 guarded, with 18, 12, 18, 22, 24 duplicates, per seed (`counterfactual_swaps_flags.csv`, meta and dup). The step-1 summaries of the two runs are identical text in every seed, as they must be: same step-0 partition, greedy decoding. Counted on the #31 4B's own summaries (not applied), the same flags give 50.2, 49.0, 53.4, 51.2 and 54.4 flagged clusters at steps 1 to 5: the guard changes which partitions the summarizer sees, not visibly how often it writes a flagged summary.
- **Where it matters:** the guard acts on a third of the clusters, and the rest (about 100) still start from 4B summaries at every step. That is a limit on what the diagnostic can show (Section 6).

## 3. Where the guarded run diverges from the #31 4B

Step 0 is shared (identical partitions in every seed). **They differ from step 1**, the first block after the first summaries: in every seed 10,009 to 11,281 of 22,500 points have a different cluster id at step 1 (44% to 50%). ARI between the two partitions of the same fit, mean [min-max] over seeds (`summary_guard_steps.csv`):

| step | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARI, guarded vs #31 4B | 0.655 [0.616-0.688] | 0.615 [0.599-0.648] | 0.616 [0.593-0.647] | 0.609 [0.581-0.661] | 0.587 [0.564-0.616] |
| ARI, guarded 4B vs 1.7B | 0.689 | 0.645 | 0.645 | 0.614 | 0.617 |
| ARI, #31 4B vs 1.7B | 0.668 | 0.632 | 0.647 | 0.630 | 0.625 |

- The runs split at step 1 and never come back together; the distance changes little after step 1 (ARI 0.66 falling to 0.59). For scale, the guarded 4B is about as far from the #31 4B as the #31 4B is from the 1.7B at each step. Partitions of these runs are sensitive (stage 1, Section 3 notes that 1e-7 changes in start centers move the 4B's blocks), so a large divergence says "different trajectory", not "a different quality of result".
- **Step 1 reproduces stage 1.** The guarded run's step-1 ACC, NMI and ARI equal stage 1's variant b (meta, 20-iteration block) in every seed (max absolute difference 0 for ACC and NMI, 6e-17 for ARI). Stage 1 computed it offline from logged state, stage 2 got it from a fresh run, which is a check on both.
- **The guarded run moves fewer points per fit and calls the summarizer less.** The logged share of points that changed cluster in each fit is 0.24, 0.17, 0.14, 0.13, 0.14 for the guarded run against 0.45, 0.40, 0.42, 0.39, 0.41 for the #31 4B (means over seeds, steps 1-5). The guarded run made 676.6 summarizer calls per seed against 714.0 (666 to 684 against 706 to 721; fewer in all 5 seeds, 37.4 fewer on average, 5.2%), and 1.092M input tokens against 1.120M. The step-by-step calls are 150, 143.8, 135.2, 124.2, 123.4 against 150, 144.6, 142.2, 140.2, 137.0. The memo cache is why: a call is made only for a non-empty cluster whose exact member set no earlier summary step of the run had, and recomputing that from `assignments_evolution` reproduces every step's logged `n_calls` in both runs and all seeds (`summary_guard_check.txt`). So fewer calls come from more stable partitions, not from a cheaper step; the numbers are equal at step 2 and diverge later as the guarded partitions settle. This is a different number of calls because of different partitions, so it is not a measure of the guard's cost (it makes no extra calls).

## 4. Trajectory of the scores per step

Mean ACC / NMI over seeds after each fit, and the paired difference guarded minus #31 4B with its 95% interval (`summary_guard_steps_paired.csv`; step 5 is the final score of Section 1):

| step | ACC guarded | ACC #31 4B | ΔACC | ΔNMI | ΔARI |
| --- | ---: | ---: | --- | --- | --- |
| 0 | 0.5511 | 0.5511 | 0 | 0 | 0 |
| 1 | 0.5420 | 0.5356 | +0.0064 [-0.0058, +0.0185] | -0.0053 [-0.0092, -0.0013] | -0.0093 [-0.0180, -0.0007] |
| 2 | 0.5325 | 0.5260 | +0.0065 [-0.0147, +0.0276] | -0.0010 [-0.0119, +0.0099] | -0.0115 [-0.0305, +0.0075] |
| 3 | 0.5287 | 0.5171 | +0.0117 [-0.0082, +0.0315] | +0.0017 [-0.0135, +0.0170] | -0.0074 [-0.0325, +0.0178] |
| 4 | 0.5181 | 0.5161 | +0.0020 [-0.0039, +0.0080] | -0.0036 [-0.0077, +0.0005] | -0.0155 [-0.0198, -0.0112] |
| 5 | 0.5205 | 0.5112 | +0.0093 [-0.0059, +0.0244] | +0.0010 [-0.0069, +0.0089] | -0.0069 [-0.0193, +0.0055] |

- **Both runs lose ACC at every step from the shared k-means state (0.5511):** by step 5 the guarded run has lost 0.031 and the #31 4B 0.040. The guard slows the fall but does not stop it.
- **The guard's ACC advantage neither fades nor accumulates in these data.** It is +0.006 at step 1 and +0.009 at step 5, between +0.002 and +0.012 in between, with no trend and every interval including zero. It was not large at step 1, so there is nothing to fade from, and five seeds could not show a slow accumulation of about 0.01.
- **NMI and ARI do not follow ACC.** At step 1 the guard is lower in NMI (excluding zero) and ARI (excluding zero); ARI is lower at every step (intervals exclude zero at steps 1 and 4). So at best the guard recovers a small amount of ACC while not improving the other two measures.

## 5. Reading against #47

**What stage 2 adds.** Stage 1 tested one step: with the flagged summaries kept out of the step-1 centers (b), the block's ACC rises by +0.0064 (interval includes zero) and NMI falls. Stage 2 asks whether doing the same at every step gets the 4B's final scores back. It does not, within five seeds: final ACC +0.0093 [-0.0059, +0.0244], NMI and ARI unchanged, still 0.031 ACC below k-means and 0.049 below the 1.7B. The step-1 result (b) does not grow over the next four steps (Section 4).

**Reconciling with stage 1.** Stage 1 found (b) recovers nothing measurable of the 4B's deficit and (c) recovers the 1.7B gap: putting the 1.7B's summaries into the flagged clusters closed about all of the step-1 gap to the 1.7B (+0.026 ACC, against a control that does nothing). Stage 2 is stage 1's (b) repeated at every step (and step 1 equals it exactly, Section 3), so it was expected to look like (b), and it does. The two results fit together if the flagged summaries are worse than the 1.7B's summaries of the same clusters, but not very different from the k-means center of those clusters: removing them gives a small gain, replacing them with a better summary gives a large one. Falling back to the k-means center is not "a good summary"; it is no summary. Stage 2 therefore does not test (c) across steps, and it cannot say whether better summaries in the flagged clusters would carry through to the final score. Stage 1 also showed (b) beating its control (b - d, meta, ACC +0.034 [+0.025, +0.044], 5/5 seeds), so a k-means fallback in the flagged clusters is far better than the same fallback in a random set of unflagged clusters (whose summaries are, by contrast, worth keeping). That control mixes in another effect (stage 1, Section 5), so it supports no more than "the flagged summaries are not better centroids than k-means centers here".

**Does removing the flagged summaries' influence close the gap?** Not to k-means or the 1.7B within what five seeds can show. The point estimate is a gain of 0.009 ACC, 23% of the gap to k-means, 16% of the gap to the 1.7B, but the gain's interval includes zero and its sign in NMI and ARI is zero or negative.

**What is shown, and what is not.**

- Shown (interventional, for this model and these seeds): starting the flagged clusters from their k-means centers at every step does not recover the 4B's final scores to a degree five seeds can detect, and leaves it 0.031 ACC below k-means. So the flagged summaries (as a set, by this rule) are not by themselves the main cause of the 4B's final ACC deficit to k-means. A gain of up to about 0.024 ACC (the upper end of the interval) is not excluded, which is the order of 60% of that gap.
- Shown (observational, from the logs): the guard acts on about a third of the clusters at every step with no trend, and it changes the trajectory (ARI 0.59 to 0.66 against the #31 4B) and the number of summarizer calls (5% fewer) because partitions settle more.
- Not shown: that the flagged summaries do no harm. The guard replaces them with a k-means center, not a better summary, so it removes the flagged summaries' influence, not the cluster's difficulty. Stage 1's (c) effect (better summaries in the same clusters help at step 1) is not tested here at later steps. It also cannot say whether the unflagged 4B summaries, which still drive about two thirds of the centers, explain the rest of the decline. The remaining fall (0.031 ACC from the k-means state) is consistent with that, and not tested.
- Inconclusive: the sign of the true final ACC effect. +0.009 is about three times the mean encoder noise floor of stage 1 (0.0030 ACC), but the seed-to-seed spread (sd 0.012) dominates and the interval includes zero. NMI and ARI show no gain.

**Limits.** One model (the 4B), one dataset and encoder, five seeds (t-intervals with 4 degrees of freedom; no multiplicity correction is made across the tables). The flags were fixed in advance and are only exact duplicates and the keyword rule; #34's genericity flag, which stage 1 found to flag 41% to 50% of clusters, was not used. The guard uses the k-means center, so it removes the summary without adding information; a better summary is a different intervention. Each run is a single trajectory per seed, and these trajectories are sensitive to small changes in the start centers, so one seed's final ACC moves by about 0.003 under a re-encode (stage 1, Section 3). The comparison runs (#31) and the guarded run share the card type and software, but ran at different times (below).

## 6. Timing and the GPU check

The seconds are not a cost comparison. The guarded run took 1495.2 ± 53.5 s per seed (`summary.csv`) against 1861.7 s for the #31 4B. From 16:07 to about 17:06 a low-priority (nice 19) CPU analysis job ran alongside the guarded run (it is not in `gpu.json`, which records the GPU only), and the #31 runs were on a shared GPU (about 1.7x). Cost is measured in #35 part 3.

`gpu.json` of the guarded run: the pre-run idle check passed ("idle: true, median sample within all thresholds"; four samples 1 s apart at 1,215 MiB, 0% utilization and about 43 W, against the limits 1,400 MiB, 15% and 65 W). The monitor took 1,484 samples at 5 s (no errors): mean utilization 62.6%, maximum 100%, maximum memory 10,335 MiB, and at most 1,355 MiB attributed to other processes (non-torch memory 140 MiB above the pre-run baseline). So the card was idle at the start and had light other use; 62.6% mean utilization is the 4B itself with batch size 1.

## 7. Reproduce

CPU only, a few seconds, no model, no encoding. `PYTHON` is the project venv's interpreter; the cache is the real `cache/` of the checkout the runs were made from.

```
PYTHON=/path/to/venv/bin/python
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src $PYTHON scripts/summary_guard_report.py \
    --guard results/clinc150_guard_qwen3-4b/<stamp> --unguarded results/clinc150_qwen3-4b/<stamp> \
    --small results/clinc150_qwen3-1.7b/<stamp> --lsa results/clinc150_lsa/<stamp> \
    --cache-dir /path/to/main/checkout/cache --out-dir docs/results
```

This writes `summary_guard_{paired,gap_closed,counts,steps,steps_paired}.csv` and `summary_guard_check.txt` in `docs/results/`. The paired table can also be had from the existing tool:

```
$PYTHON scripts/aggregate_results.py results/clinc150_lsa/<stamp> results/clinc150_qwen3-4b/<stamp> \
    results/clinc150_qwen3-1.7b/<stamp> results/clinc150_guard_qwen3-4b/<stamp> \
    --references "k-SLMmeans-guard (Qwen3-4B)" --paired-md paired.md --cost-md cost.md
```

(there the guarded run is the reference, so the signs are the other way round). The guarded run itself is `scripts/run_clinc150_guard.sh`; `scripts/surface_counts.py` on the run directory gives the pooled surface checks of its summaries.
