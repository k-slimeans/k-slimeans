# Does dataset-order truncation (#28) invalidate the #15 numbers? Step 1 of #37

Issue #37, step 1. CPU only, from the logged `rq2_*` runs (60 run directories, 120 (method, seed) records). No model was run and nothing under `results/` was changed. The question: the #15 runs trimmed an overflowing cluster prompt to a leading prefix of its members in dataset order, and CLINC150 is stored grouped by intent (#28). Do the headline conclusions (trigger vs fixed-n5, fixed-n1, fixed-n2 and the call-matched random control) survive?

## TL;DR

- **Recommendation: keep the #15 numbers, with a caveat. Do not spend GPU time on a re-run to answer #28.** Truncation touched about 1% of the summarizer calls of a run and about 0.2% of the members offered to the summarizer. It cannot explain a 2.8-point loss to fixed-n1, and it hits the schedules being compared in the same or in a proportional way (Section 2.4). This is a judgement resting on the measurements below; Section 3 says which is which.
- **Measured.** Step 1 is exact: in 72 of 120 records exactly 1 or 2 of the 150 step-1 prompts overflowed (0.91 per record on average; 203-390 members, 2-62 members dropped, at most 16% of the cluster). Over all five steps, fixed-n5 had about 7 overflowing prompts per run out of about 700 calls, the trigger and its control about 1.7 out of about 235. The #31 run of the same schedule, where the true partitions exist, has 3-11 per run (Section 2.5).
- **Not a differential exposure.** Step-1 overflow is identical for the trigger, fixed-n5 and the random control on every seed (same partition, same prompts). Step-1 identity holds by construction (same seed, same first block, hence the same replayed partition); the independent evidence is the inertia match in Section 1. Overflowing prompts per 100 calls (the one rate used throughout, rebuilt count over logged `n_calls`) are 0.72 for the trigger and 0.72 for its control, so the trigger does not pick the big clusters more than chance does.
- **No visible effect on ACC.** The same fixed-n5 schedule and seeds 0-4 scored 0.003-0.026 apart between #15 (prefix rule) and #31 (random-subset rule): mean -0.003, sd 0.014, the size of the seed-to-seed spread, with no relation to how many prompts were cut. Across seeds, the paired ACC gaps do not track the paired truncation exposure (|rho| at most 0.61, all p at least 0.15; observational, 7-10 seeds).
- **A re-run is prepared anyway** (`configs/rq2_rerun.yaml`, `scripts/run_rq2_rerun.sh`, phase `R` of `scripts/run_rq2.py`), because #37's done-condition says the RQ2 report should follow the #31 protocol. About 16,200 summarizer calls, 3.4-5.8 GPU-hours (Section 4). It has not been run.

## 1. What was measured, and how exact it is

The #15 stdout logs, where the 58 truncation warnings were seen, are not among the files in `results/` (only the run directories are), so the "58 warnings over 60 runs" figure was not re-read. The run directories log no per-prompt truncation counts either (`n_truncated_clusters` and `n_dropped_texts` were added after #15). The prompts were therefore rebuilt from what is logged (`scripts/rq2_truncation_check.py`): the cached CLINC150 texts, the Qwen3-0.6B tokenizer, the same render path as `HFSummarizer` (system prompt, CLINC prompt, `Question:` prefill, `max_input_tokens` 4096) and the #15 rule (the longest leading prefix of the members in ascending dataset index that fits).

| Quantity | How it is obtained | Exactness |
| --- | --- | --- |
| Partition of summary step 1 | k-means++ with `random_state` = seed, `max_iter` = the first block (20 for fixed-n5/adaptive/random, 60 for fixed-n1, 40 for fixed-n2), replayed with scikit-learn | **Exact.** The replayed inertia matches the logged step-0 inertia to 3e-6 relative in all 120 records. Every schedule summarizes all 150 clusters here. |
| Partition of steps 2-5 | the run's final partition (the per-step partitions were not logged before #30) | **Approximate** |
| Clusters summarized at steps 2-5 | those whose summary text changed since the previous step | **Approximate**: misses a re-summary that returned the same text, and counts a free memo-cache hit only if its text changed |

How far the approximation is off can be checked, because the #31 Qwen3-0.6B run has the true partitions and logged counts (`rq2_truncation_check31.csv`, `--check`):

| #31 0.6B, seed | 0 | 1 | 2 | 3 | 4 | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| truncated prompts, logged | 3 | 8 | 3 | 11 | 8 | 33 |
| rebuilt from true partitions and memo cache | 3 | 8 | 3 | 11 | 8 | 33 |
| rebuilt with the #15 procedure (step 1 exact, later steps approximate) | 0 | 5 | 3 | 5 | 8 | 21 |
| dropped members, logged (= rebuilt from true partitions) | 112 | 138 | 74 | 433 | 108 | 865 |
| dropped members, #15 procedure | 0 | 104 | 12 | 541 | 329 | 986 |

With the true partitions the rebuild reproduces the logged calls, truncated prompts and dropped members exactly, which validates the shared render and token-count path (it exercises `kept_random`, the current `fit_texts_to_budget`, not the #15 rule). The #15 rule `kept_prefix` is not covered by that check: it was compared by code with `git show fb6eb01:src/kslimeans/summarizers/hf.py` (n <= 1 guard, binary search over [1, n-1], `<= max_tokens`, members in ascending index) and is unit-tested against a stub of the old routine. Step 1 counts as exact because the replayed step-0 inertia matches each run's logged value to about 1e-6 relative. The #15 procedure is the weak part: on this run it finds 21 of 33 overflowing prompts and its per-seed errors are up to 6 prompts and 221 members. **Treat step-1 numbers as exact and anything about steps 2-5 as rough, in either direction.** For fixed-n5 on seeds 0-4 the #31 run is a better estimate of the truth than the rebuild (Section 2.5).

## 2. Numbers

### 2.1 Overall (120 records, 60 run directories; `rq2_truncation_per_run.csv`, `rq2_truncation_prompts.csv`)

- The rebuild finds at least one overflowing prompt in **56 of the 60 run directories** (38 at step 1 alone), against 58 logged warnings. The four without are single-seed random controls of 240-471 calls (`rq2_random_d0.3_r0.4_seed0`, `rq2_random_d0.6_r0.7_seed0`, `rq2_random_d0.6_r0.7_s2_seed0`, `rq2_random_d0.6_r0.7_s2_seed2`). Seed 0 has no step-1 overflow (Section 2.3), and the later-step approximation can miss one.
- 426 overflowing prompts over 40,469 reconstructed summaries (1.05%), dropping 13,318 of 5.92 million offered members (0.22%). Per logged call that is 426 / 46,946 = 0.91%. These overall rates are rough: the steps 2-5 stand-in undercounted on the #31 check (21 of 33), but its partition choice can also overstate sizes, so the error could go either way.
- 112 of 120 records have at least one overflowing prompt; 72 have one at step 1.

An overflowing prompt, step 1 (exact, 109 prompts) and steps 2-5 (approximate, 317):

| | step 1, exact | steps 2-5, approximate |
| --- | ---: | ---: |
| cluster size (members) | 203-390, median 367 | 202-480, median 351 |
| members dropped | 2-62, median 22 | 1-204, median 25 |
| share of the cluster dropped | mean 6.7%, max 16% | mean 10.7%, max 49% |
| distinct intents in the cluster / intents with no member left in the prompt | 15.0 / 0.5 (48% of prompts lose at least one whole intent) | 24.0 / 1.2 |

The step-1 ranges match what #28 and `rq2_clinc150.md` report from the warnings (231-398 members, 1-70 dropped, at most about 18%). The later-step tail (49%, 204 dropped) is larger than anything logged, so it is probably an artefact of using the final partition, whose largest clusters are bigger than the intermediate ones.

### 2.2 By schedule (means per (run, seed) record; seeds 0-9 for the five headline variants)

| Variant | records | calls | overflowing prompts (min-max) | at step 1 | per 100 calls | members dropped | share of offered members |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-n1 | 10 | 150 | 1.3 (1-2) | 1.3 | 0.87 | 24 | 0.11% |
| fixed-n2 | 10 | 293 | 2.5 (1-4) | 1.3 | 0.85 | 50 | 0.13% |
| fixed-n5 | 10 | 703 | 6.7 (3-12) | 1.1 | 0.95 | 181 | 0.22% |
| adaptive-d0.6-r0.7 | 10 | 237 | 1.7 (0-3) | 1.1 | 0.72 | 49 | 0.16% |
| random-d0.6-r0.7 | 10 | 236 | 1.7 (0-4) | 1.1 | 0.72 | 46 | 0.15% |

fixed-n1 is exact (it has one summary step). The other rows are exact at step 1 and approximate after it. All 28 variants, with fixed-n3 and fixed-n4 and the other adaptive and control configs, are in `rq2_truncation_per_run.csv` (one row per record; group by `method`). They lie in the same band: 0-12 overflowing prompts per record and at most 0.8% of offered members.

### 2.3 Where it concentrates

- **In the largest clusters.** Every overflowing prompt is a cluster of at least 202 members (median 351), against a mean of 150. At step 1 a partition has about 29 clusters with at least 200 members and 0.9 that overflow on average (0 to 2 per seed), because token length matters as well as member count.
- **In a few seeds and in fixed-n5.** The step-1 overflow count of a seed is shared by the trigger, fixed-n5 and the control: seeds 0 and 2 have none, seeds 1, 6 and 9 have two, the others one (`rq2_truncation_per_run.csv`, `n_trunc_step1`). Later steps add about 5.6 overflowing prompts per fixed-n5 run (the same few big clusters re-summarized at each of four more steps) against 0.6 for the trigger and 0.6 for the control.
- **Not in the triggered clusters.** Per 100 calls the trigger sits at 0.72, the control at 0.72 and fixed-n5 at 0.95. If the trigger preferred big, high-churn clusters, its rate would be above the control's. It is not.
- **Dataset order, not a sample.** Under the #15 rule the dropped members are the trailing indices, which are whole trailing intents of a mixed cluster. At step 1, 48% of overflowing prompts lose at least one intent entirely (0.5 of 15.0 on average). #31 keeps a seeded random subset, which loses a few members of each intent instead; the expected number of wholly lost intents is close to zero but was not measured.

### 2.4 Could it change the headline comparison? What the numbers say

1. **Trigger vs fixed-n5 and vs the random control.** Step-1 summaries come from the same partition and the same prompts for all three on every seed (the step-1 overflow count is equal for the trigger, fixed-n5 and the control on all 10 seeds), so the step-1 truncation is common to the three (an identity by construction, backed by the inertia match). That it therefore cancels in the paired differences is a judgement: the same step-1 summaries feed later steps differently under each schedule. After step 1 fixed-n5 is exposed to about 5.6 overflowing prompts per run and the trigger and its control to about 0.6 each. If truncation hurts, it hurts fixed-n5 more, which would make the trigger look better against fixed-n5 than it would against a clean reference. The control is exposed to the same number as the trigger.
2. **Trigger vs fixed-n1 and fixed-n2** (ΔACC -0.028, 95% CI [-0.039, -0.018] and -0.022, [-0.036, -0.007]). fixed-n1 had 1.3 overflowing prompts (24 members dropped of 22,500) per run. Replacing all 150 centroids by 0.6B summaries moves ACC by 0.003 to 0.010 relative to k-means (`clinc150_summarizers.md` and `rq2_clinc150.md`: fixed-n1 0.554 vs k-means 0.551 on seeds 0-4; the #31 0.6B run -0.010 [-0.018, -0.002]). One or two of 150 centroids built from a prompt missing a tenth of its members is a small part of that, if the effect is anywhere near proportional (a judgement; Lloyd can amplify a small change, as Section 2.5 shows).
3. **Scale.** Each affected summary is one centroid of 150 built from a prompt that is on average 93% complete at step 1 (84% at worst).

### 2.5 The best direct evidence: the same schedule under both rules

fixed-n5, seeds 0-4, Qwen3-0.6B, same prompt, embeddings and schedule, with the #15 prefix rule (`rq2_fixed_n5`) and with the #31 random-subset rule (`clinc150_qwen3-0.6b`, #28). The two runs also differ in the code version and in GPU noise (greedy decoding with bf16 batches is not bit-reproducible), which this comparison cannot separate from the rule.

| seed | ACC, #15 | ACC, #31 | #31 minus #15 | overflowing prompts, rebuilt for #15 / logged in #31 | summaries identical to #31 at steps 1-5 (of 150) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.5593 | 0.5336 | -0.0258 | 3 / 3 | 150, 150, 149, 80, 53 |
| 1 | 0.5506 | 0.5572 | +0.0066 | 12 / 8 | 142, 75, 47, 44, 36 |
| 2 | 0.5452 | 0.5528 | +0.0076 | 3 / 3 | 150, 148, 101, 62, 44 |
| 3 | 0.5311 | 0.5342 | +0.0031 | 8 / 11 | 148, 83, 48, 34, 42 |
| 4 | 0.5324 | 0.5276 | -0.0048 | 9 / 8 | 147, 68, 49, 36, 39 |

- ACC differences: mean -0.003, sd 0.014, 95% t-interval [-0.020, +0.015] (n = 5). No sign of a systematic shift, and it is the size of the seed-to-seed sd of either run (0.013 in #31).
- The largest difference (seed 0, -0.026) is on a seed with no step-1 overflow and two steps of identical summaries; seed 3, with the most overflow in #31 (11 prompts), differs by +0.003, and seeds 1 and 4 (8 each) by +0.007 and -0.005. Trajectories diverge on every seed (36-53 of 150 summaries identical at step 5), whether or not a prompt was cut; most of that divergence is the amplification of small differences by Lloyd, not the truncation.
- **Real counts for fixed-n5.** In #31 this schedule had 3-11 overflowing prompts per run (mean 6.6, over 696-714 calls, 0.9%) and 74-433 dropped members (mean 173). The #15 rebuild gives 3-12 (mean 7.0) and 69-347 (mean 160) on the same seeds. The two agree on the mean; the #31 counts are the better estimate for fixed-n5, being exact for that run, whose partitions differ from #15's after the first changed summary.

### 2.6 Paired ACC gaps against truncation exposure (observational)

Spearman correlation across seeds between the paired ACC gap (trigger minus reference) and the paired difference in overflowing prompts or dropped members of the same two runs. Step-1 exposure is identical for fixed-n5 and the control by construction, so only later steps vary. With 7-10 points and noisy later-step counts this can only exclude a strong relation.

| Seeds | Reference | exposure | n | rho | p |
| --- | --- | --- | ---: | ---: | ---: |
| 0-9 | fixed-n5 | overflowing prompts | 10 | -0.40 | 0.25 |
| 0-9 | fixed-n5 | members dropped | 10 | -0.03 | 0.93 |
| 0-9 | random control | overflowing prompts | 10 | +0.36 | 0.31 |
| 0-9 | random control | members dropped | 10 | +0.34 | 0.34 |
| 3-9 (held out) | fixed-n5 | overflowing prompts | 7 | -0.25 | 0.58 |
| 3-9 (held out) | fixed-n5 | members dropped | 7 | +0.61 | 0.15 |
| 3-9 (held out) | random control | overflowing prompts | 7 | +0.45 | 0.31 |
| 3-9 (held out) | random control | members dropped | 7 | +0.40 | 0.38 |

The remaining eight rows (references fixed-n1 and fixed-n2) have |rho| at most 0.47 and p at least 0.21. The signs are mixed, none of the 16 is significant, and none survives any multiplicity correction. The per-seed values behind them are in `rq2_truncation_per_run.csv`. The largest per-seed ACC gaps vs fixed-n5 (seed 0, -0.032; seed 5, -0.028) are on seeds where the trigger had 0 and 2 overflowing prompts against 3 and 6 for fixed-n5; they point, if anywhere, to fixed-n5 being the more exposed run.

## 3. Recommendation

**Keep the #15 numbers and add a caveat; do not re-run for #28.**

Measured:
- Step-1 overflow is 0-2 prompts per run, equal across the trigger, fixed-n5 and the control on each seed.
- Overflow is about 1% of calls and 0.2% of members across the 120 records (rough, error either way), concentrated in the few largest clusters; the trigger's rate per call equals the control's and is below fixed-n5's.
- The same schedule under the #15 and #31 rules differs by -0.003 ACC on average, with an sd of 0.014.
- No paired-gap vs exposure correlation reaches p below 0.15.

Judgement:
- That a rate of about 1% truncated calls, each missing about a tenth of its cluster, cannot reverse a negative result of this size (fixed-n1 and fixed-n2 beat the trigger by 0.022-0.028 with intervals that exclude zero) or turn the control's -0.013 [-0.032, +0.007] into a clear trigger win or loss.
- That the direction of any bias favours the trigger against fixed-n5 (fixed-n5 is the more exposed run), so the "close to fixed-n5 on the mean" reading is, if anything, flattering to the trigger and not to its opponent.
- That the #15 and #31 comparison, with 5 seeds and GPU noise in the way, is consistent with no rule effect and cannot show that there is none; an effect up to about 0.015 ACC in either direction is inside its interval.

Suggested caveat for `rq2_clinc150.md` (one sentence, appended to the Input truncation limitation): "The prompts truncated by dataset order are about 1% of the summarizer calls (about 7 of 700 calls for fixed-n5, about 2 of 235 for the trigger), exact 0-2 per run at step 1 and identical across the trigger, fixed-n5 and the control; see `rq2_truncation_check.md`. A re-run on the #31 protocol was not needed to answer the question and was not done."

What would change this recommendation: a downstream decision that needs the RQ2 numbers on exactly the #31 protocol (the done-condition of #37 reads that way), or a second dataset whose order is label-correlated and whose clusters overflow far more often.

## 4. The prepared re-run (optional, not run)

Prepared so that it is one command if the done-condition is read literally. It re-runs only the headline comparison on the #31 protocol, with no threshold grid:

- `configs/rq2_rerun.yaml`: the #31 0.6B setup (`clinc150_summarizers.yaml`), fixed-n5 schedule, seeds 0-9, `max_input_tokens` spelled out as 4096; the k-means baseline rides on fixed-n5.
- Phase `R` of `scripts/run_rq2.py` (`rq2.plan_phase_r`): `rq2r_fixed_n5`, `rq2r_fixed_n1`, `rq2r_fixed_n2`, `rq2r_adaptive_d0.6_r0.7` (the config #15 chose, no re-selection) and, once the adaptive run is done, `rq2r_random_d0.6_r0.7_seed0-9` with each seed's per-step calls (`RandomSubsetSchedule`). Names start with `rq2r_` so they never mix with the #15 directories. The default phases (A-D) and the report are unchanged.
- `scripts/run_rq2_rerun.sh`: runs phase R, then builds a mean table and a per-seed CSV with `aggregate_results.py`. Resumable.

`tests/test_rq2_rerun.py` checks that the config parses and matches the #31 protocol (identity keys other than seeds and method label equal `clinc150_summarizers.yaml`, `max_input_tokens` 4096) and differs from the #15 base only in seeds and the explicit token cap, and that the plan has the right runs, seeds, baselines and call-matched controls.

**Command** (from a checkout that has `cache/` and where the GPU is free; the paths in the config are relative to the repository root, so run it from the main checkout after this branch is merged):

```bash
PYTHON=/path/to/venv/bin/python scripts/run_rq2_rerun.sh
# first: PYTHON=... scripts/run_rq2_rerun.sh plan      # 14 runs, 4 todo, 10 blocked until the adaptive run is done
```

**GPU-hours.** Calls per seed from the #15 means over seeds 0-9: fixed-n1 150, fixed-n2 293, fixed-n5 703, adaptive 237, control 236 = 1,619; times 10 seeds = **about 16,200 summarizer calls**. The adaptive calls will differ a little on the new run, and the controls follow them.

| Per-call cost | Basis | Hours |
| --- | --- | ---: |
| 1.28 s (wall clock per call, whole run) | #31 Qwen3-0.6B, 903.9 s per seed over about 707 calls (planning figure; that run was inflated about 1.7x by a background GPU load) | **5.8** |
| 0.92 s | #15 overall: about 12 GPU-hours over 46,946 calls | 4.1 |
| 0.75 s | #15 fixed-n5 runs, 0.70-0.85 s per call | 3.4 |

Budget **6 hours** on a free card; the k-means baseline and model loads add minutes.

Seeds 0-2 chose the adaptive config in #15 and so are no longer held out, but they are a fresh sample of it; report seeds 0-2 and 3-9 separately. The `rq2_report.py` report is tied to the A-D plan and would need a phase-R mode to produce the paired intervals; that is not part of this preparation.

## 5. Limitations

- Steps 2-5 are reconstructed with the final partition. Against the true partitions of the #31 run the procedure finds 21 of 33 overflowing prompts, with per-seed errors up to 6 prompts and 221 members (Section 1), and it overstates the tail (49% dropped against at most about 18% in the logs). Statements about "later steps" are rough; those about step 1, fixed-n1 and the #31 twin are exact or measured.
- The 58 warnings were not re-read, because the #15 stdout logs are not in `results/`. The rebuild's 56 of 60 directories is consistent with them.
- I did not audit every code change between `fb6eb01` (the #15 code) and this branch. `fit_texts_to_budget` is the change that matters for this question; that the rest of the run is unchanged rests on the #31 protocol note and the tests, not on my reading of the diff.
- "No visible effect" is the absence of evidence at n = 5 to 10, with seed noise of 0.014, not evidence of no effect.

## 6. Reproducibility

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/rq2_truncation_check.py \
    --results-dir /path/to/k-slimeans/results --cache-dir /path/to/k-slimeans/cache --out-dir docs/results
```

About 12 minutes on 2 CPU threads with the Qwen3-0.6B tokenizer in the Hugging Face cache (tokenizer only; no weights are loaded). `--check` prints only the #31 comparison. Outputs: `rq2_truncation_per_run.csv` (one row per (run, method, seed): calls, ACC, summarized clusters, overflowing prompts, dropped members, step-1 values), `rq2_truncation_prompts.csv` (one row per overflowing prompt: step, cluster size, members kept and dropped, intents lost, whether the step is exact), `rq2_truncation_check31.csv` (Section 1).
