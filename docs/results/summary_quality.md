# Summary quality: coverage, genericity and unsupported content, scored apart from fluency (issue #34)

Status: **automatic scores done for all runs, including the clearer-prompt (v2) runs at all three sizes. Hand check done by two human raters
(section 5, Nov 27). The earlier AI-rater stand-in (two Claude Sonnet 5.5 instances, not a human rating) stays in section 5 as a labelled comparison.**
Everything below is descriptive and preliminary. It says what the scores measure on the logged runs; it does not say why the clusterings
differ (that is #47) or how the scores relate to clustering outcome (that is #50).

## 1. What was scored

| Dataset | Runs | Seeds | Steps scored | Partition the members come from |
| --- | --- | --- | --- | --- |
| CLINC150 (22,500 queries, 150 intents) | `clinc150_lsa/20261027-174631`, `clinc150_qwen3-0.6b/20261027-190908`, `clinc150_qwen3-1.7b/20261027-202532`, `clinc150_qwen3-4b/20261027-221042`, and the clearer-prompt runs of #46 `clinc150_v2_qwen3-0.6b/20261112-200926`, `clinc150_v2_qwen3-1.7b/20261112-204127` and `clinc150_v2_qwen3-4b/20261112-181256` (method labels `k-SLMmeans-v2 (Qwen3-0.6B)`, `k-SLMmeans-v2 (Qwen3-1.7B)`, `k-SLMmeans-v2 (Qwen3-4B)`) | 0-4 | all 5 summary steps | the partition each summary was written from (`source_partition(j)`) |
| BANKING77 (3,080 queries, 77 intents) | `prelim_bank77_{lsa,qwen3-0.6b,qwen3-1.7b,qwen3-4b}` | 0-4 | last step only | the FINAL assignments (**approximation**) |

**BANKING77 caveat, applies to every BANKING77 number here.** These runs predate #30 and logged no per-step
assignments. The final summaries were written from the partition before the last Lloyd block, which is not logged, so
scoring them against the final partition mixes two partitions. No kept/lost/gained numbers exist for BANKING77. A
re-run under #33 would remove the approximation.

27,790 summaries were scored (26,250 CLINC150, 1,540 BANKING77). The v2 runs have no BANKING77 counterpart. The logged `summary_inertia` was reproduced from the
re-embedded summaries to a relative difference of at most 1.72e-7 on every CLINC150 step (the largest is the v2 0.6B, seed 4 step 3; 1.5e-7 or less on the October runs, 1.2e-7 or less on the v2 1.7B and 4B), so the CPU re-embedding matches the
embeddings the runs used. No cluster was empty at a scored step.

**Drift on the old rows when v2 was added.** Re-scoring the October runs reproduced every committed number to 1e-6 or less (`summary_inertia_replayed` moved by 0.1 on two rows), with one exception: near-tie Hungarian matches
flipped on 3 old 4B per-step rows (seed 0 step 3, seed 2 step 1, seed 4 step 2), each adding 0.0067 to the label-alignment accuracy (one cluster of 150). The old 4B all-steps label accuracy therefore moved from 0.322 to 0.323. A repeat run of the old 4B alone gave the new values exactly,
so the script is deterministic and the committed tables came from slightly different floating-point behaviour.

**Drift on the old rows when the v2 0.6B and 1.7B were added (16 Nov).** None. Re-scoring all runs with the `origin/main` code reproduced every number of the committed `summary_quality_table.csv` (14 rows) and `summary_quality_per_step.csv` (145 rows) exactly (largest absolute difference 0 on every numeric column, so inside the 1e-6 tolerance used for the earlier check), and every old row is byte-identical in the regenerated files. The committed files are the regenerated ones: the 4 new table rows and 50 new per-step rows (the v2 0.6B and 1.7B) were added, and some old rows moved down because the script writes them in method order.

## 2. The scores (definitions in `src/kslimeans/summary_quality.py`)

Four families, never merged into one number. Thresholds are constants in the module and were set before any hand rating.

- **Coverage.** *Nearest-summary share*: the share of the cluster's members whose nearest of the k summary embeddings
  (Euclidean, the metric of the assignment step) is this cluster's summary. Clusters with an identical summary text tie; the first copy
  is their representative for both coverage scores and later copies score 0 on both (the assignment step also gives tied points to the lowest index). *Intent coverage*: among gold intents with at least
  10% of the cluster's members, the share whose members are on average strictly closer to this summary than to any other.
  Flag `low coverage`: nearest-summary share below 0.5. #8's label alignment is included as an extra (Hungarian match of
  summaries to embedded label names per step: mean matched cosine / share of clusters matched to their majority intent).
- **Genericity.** *Specificity margin*: cosine to the cluster's own members' mean minus the highest cosine to another
  cluster's members' mean. *Max cos to another summary*. Flags: exact duplicate of another cluster's summary at the same step;
  near-duplicate (cosine at least 0.95 to another summary); mentions the cluster (`mentions_cluster`, the section 2.4 rule);
  margin at most 0. `generic` is their union. #7's method-level genericity is the mean off-diagonal cosine among the k
  summary embeddings, averaged over steps and seeds.
- **Unsupported information.** Share of the summary's content-word lemmas (spaCy `en_core_web_sm` NOUN/VERB/ADJ/PROPN, alphabetic,
  not stop words) that appear in no member, plus numbers (tokens with a digit) and named entities (PERSON, ORG, GPE, LOC, FAC, NORP,
  PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE) found in no member. Flag: lemma share at least 0.5, or any unsupported
  number or entity. This is lexical: a paraphrase counts as unsupported, so it overstates invented content.
- **Fluency proxies.** No "?" where a question was asked for (model runs only; the LSA summaries are extracted sentences), echoes the
  prompt (shares a word 3-gram with it that no member contains), is a verbatim copy of a member, stray markup or quotes,
  letters outside the Latin script, empty. `not fluent` is the union. These say nothing about content.

Members are lemmatized one text at a time and summaries one line at a time with the same spaCy pipeline (NER is on for summaries only), so the
extractive LSA summaries, which are member sentences, are analysed exactly as their sources. They act as a control: on CLINC150 they score 0 unsupported
lemmas in all but 1 of 3,750 summaries (one summary, not examined further) and 0% generic. On BANKING77, 29 of 385 LSA summaries have a non-zero share (4 flagged on any count). The likely cause is the final-partition approximation (an extracted
sentence whose source member is no longer in the cluster it is scored against); this was not checked per case.

## 3. Score tables

Pooled over clusters, seeds and steps ("all steps") or over the last summary step. Medians appear next to means where
the distribution is skewed. `n` is the number of summaries. Fluency proxies and the other three families are reported
separately on purpose.

### clinc150_intent

**Scores** (mean / median over clusters; `n` = summaries)

| method (scope) | n | nearest-summary share | intent coverage | specificity margin | max cos to another summary | unsupported lemma share | genericity (#7) | label align sim / acc (#8) | nearest-summary share without exact duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 3,750 | 0.791 / 0.849 | 0.959 | 0.247 / 0.248 | 0.567 | 0.000 / 0.000 | 0.204 | 0.390 / 0.523 | 0.791 |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 3,750 | 0.599 / 0.672 | 0.753 | 0.136 / 0.164 | 0.581 | 0.128 / 0.000 | 0.193 | 0.477 / 0.390 | 0.607 |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 3,750 | 0.604 / 0.675 | 0.764 | 0.109 / 0.134 | 0.638 | 0.298 / 0.267 | 0.274 | 0.419 / 0.451 | 0.608 |
| k-SLMmeans (Qwen3-4B) (all steps) | 3,750 | 0.493 / 0.548 | 0.595 | 0.003 / 0.084 | 0.702 | 0.438 / 0.409 | 0.290 | 0.409 / 0.323 | 0.555 |
| k-SLMmeans-v2 (Qwen3-4B) (all steps) | 3,750 | 0.664 / 0.750 | 0.811 | 0.148 / 0.172 | 0.627 | 0.131 / 0.000 | 0.224 | 0.510 / 0.485 | 0.667 |
| k-SLMmeans-v2 (Qwen3-0.6B) (all steps) | 3,750 | 0.682 / 0.737 | 0.864 | 0.202 / 0.203 | 0.561 | 0.015 / 0.000 | 0.182 | 0.498 / 0.498 | 0.682 |
| k-SLMmeans-v2 (Qwen3-1.7B) (all steps) | 3,750 | 0.590 / 0.650 | 0.751 | 0.119 / 0.133 | 0.595 | 0.288 / 0.250 | 0.224 | 0.434 / 0.423 | 0.590 |
| k-NLPmeans (last step) | 750 | 0.796 / 0.855 | 0.962 | 0.250 / 0.252 | 0.566 | 0.000 / 0.000 | 0.205 | 0.388 / 0.521 | 0.796 |
| k-SLMmeans (Qwen3-0.6B) (last step) | 750 | 0.607 / 0.684 | 0.764 | 0.146 / 0.171 | 0.570 | 0.128 / 0.000 | 0.193 | 0.472 / 0.345 | 0.613 |
| k-SLMmeans (Qwen3-1.7B) (last step) | 750 | 0.614 / 0.700 | 0.768 | 0.112 / 0.136 | 0.635 | 0.304 / 0.267 | 0.275 | 0.413 / 0.443 | 0.619 |
| k-SLMmeans (Qwen3-4B) (last step) | 750 | 0.494 / 0.569 | 0.590 | 0.003 / 0.086 | 0.702 | 0.454 / 0.429 | 0.289 | 0.402 / 0.304 | 0.564 |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 750 | 0.672 / 0.759 | 0.819 | 0.153 / 0.177 | 0.626 | 0.140 / 0.000 | 0.224 | 0.510 / 0.496 | 0.674 |
| k-SLMmeans-v2 (Qwen3-0.6B) (last step) | 750 | 0.689 / 0.740 | 0.869 | 0.204 / 0.212 | 0.565 | 0.013 / 0.000 | 0.186 | 0.496 / 0.509 | 0.689 |
| k-SLMmeans-v2 (Qwen3-1.7B) (last step) | 750 | 0.603 / 0.681 | 0.768 | 0.122 / 0.133 | 0.598 | 0.296 / 0.250 | 0.228 | 0.434 / 0.433 | 0.603 |

**Flags** (share of summaries)

| method (scope) | not fluent | low coverage | low coverage without exact duplicates | generic | duplicate | near-duplicate | mentions | margin <= 0 | unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 1% | 11% | 11% | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 28% | 35% | 34% | 19% | 1% | 2% | 8% | 16% | 12% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 9% | 34% | 34% | 22% | 1% | 1% | 8% | 19% | 24% |
| k-SLMmeans (Qwen3-4B) (all steps) | 34% | 48% | 41% | 46% | 11% | 14% | 34% | 39% | 46% |
| k-SLMmeans-v2 (Qwen3-4B) (all steps) | 43% | 26% | 25% | 17% | 1% | 1% | 0% | 17% | 10% |
| k-SLMmeans-v2 (Qwen3-0.6B) (all steps) | 38% | 24% | 24% | 5% | 0% | 0% | 0% | 5% | 1% |
| k-SLMmeans-v2 (Qwen3-1.7B) (all steps) | 2% | 37% | 37% | 20% | 0% | 0% | 0% | 20% | 26% |
| k-NLPmeans (last step) | 1% | 9% | 9% | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 29% | 35% | 34% | 17% | 1% | 2% | 9% | 14% | 12% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 9% | 32% | 32% | 21% | 1% | 1% | 9% | 19% | 25% |
| k-SLMmeans (Qwen3-4B) (last step) | 35% | 47% | 40% | 47% | 13% | 15% | 36% | 39% | 47% |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 43% | 24% | 24% | 17% | 1% | 1% | 0% | 16% | 12% |
| k-SLMmeans-v2 (Qwen3-0.6B) (last step) | 39% | 22% | 22% | 6% | 0% | 0% | 0% | 6% | 1% |
| k-SLMmeans-v2 (Qwen3-1.7B) (last step) | 2% | 36% | 36% | 19% | 0% | 0% | 0% | 19% | 29% |

**Fluency checks** (share of summaries failing each)

| method (scope) | no "?" | echoes prompt | copies a member | markup/quotes | non-Latin letters | empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 0% | 0% | 0% | 1% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 12% | 7% | 15% | 3% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 3% | 7% | 1% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (all steps) | 14% | 30% | 1% | 13% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-4B) (all steps) | 39% | 0% | 12% | 0% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-0.6B) (all steps) | 8% | 0% | 33% | 0% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-1.7B) (all steps) | 1% | 0% | 1% | 0% | 0% | 0% |
| k-NLPmeans (last step) | 0% | 0% | 0% | 1% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 12% | 8% | 15% | 3% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 3% | 7% | 1% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (last step) | 15% | 31% | 1% | 13% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 38% | 0% | 12% | 0% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-0.6B) (last step) | 7% | 0% | 34% | 0% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-1.7B) (last step) | 1% | 0% | 1% | 0% | 0% | 0% |

**Members at the step** (pooled over clusters, as a share of the members the summaries were written from, counted on the partition after the step's Lloyd block. A member is *uncovered* when its gold intent is one the summary does not cover; `lost if uncovered` / `lost if covered` are the shares of uncovered / covered members that left their cluster)

| method (scope) | kept | lost | gained | members uncovered | lost if uncovered | lost if covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 95% | 5% | 5% | 9% | 19% | 3% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 72% | 28% | 28% | 30% | 66% | 11% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 76% | 24% | 24% | 31% | 57% | 10% |
| k-SLMmeans (Qwen3-4B) (all steps) | 59% | 41% | 41% | 45% | 76% | 13% |
| k-SLMmeans-v2 (Qwen3-4B) (all steps) | 78% | 22% | 22% | 26% | 60% | 8% |
| k-SLMmeans-v2 (Qwen3-0.6B) (all steps) | 86% | 14% | 14% | 19% | 44% | 7% |
| k-SLMmeans-v2 (Qwen3-1.7B) (all steps) | 76% | 24% | 24% | 32% | 53% | 10% |
| k-NLPmeans (last step) | 98% | 2% | 2% | 9% | 9% | 1% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 76% | 24% | 24% | 28% | 61% | 10% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 75% | 25% | 25% | 31% | 58% | 10% |
| k-SLMmeans (Qwen3-4B) (last step) | 59% | 41% | 41% | 45% | 75% | 14% |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 80% | 20% | 20% | 25% | 57% | 7% |
| k-SLMmeans-v2 (Qwen3-0.6B) (last step) | 88% | 12% | 12% | 19% | 37% | 6% |
| k-SLMmeans-v2 (Qwen3-1.7B) (last step) | 79% | 21% | 21% | 30% | 48% | 9% |

### bank77

Last summary step scored against the FINAL assignments (approximation: the summaries were written from the partition before the last Lloyd block, which is not logged).

**Scores** (mean / median over clusters; `n` = summaries)

| method (scope) | n | nearest-summary share | intent coverage | specificity margin | max cos to another summary | unsupported lemma share | genericity (#7) | label align sim / acc (#8) | nearest-summary share without exact duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (final (approx.)) | 385 | 0.801 / 0.842 | 0.943 | 0.179 / 0.165 | 0.662 | 0.017 / 0.000 | 0.382 | 0.461 / 0.473 | 0.801 |
| k-SLMmeans (Qwen3-0.6B) (final (approx.)) | 385 | 0.736 / 0.804 | 0.877 | 0.172 / 0.165 | 0.641 | 0.040 / 0.000 | 0.288 | 0.575 / 0.468 | 0.736 |
| k-SLMmeans (Qwen3-1.7B) (final (approx.)) | 385 | 0.680 / 0.745 | 0.806 | 0.121 / 0.122 | 0.670 | 0.200 / 0.143 | 0.370 | 0.501 / 0.429 | 0.680 |
| k-SLMmeans (Qwen3-4B) (final (approx.)) | 385 | 0.680 / 0.765 | 0.809 | 0.119 / 0.122 | 0.686 | 0.177 / 0.000 | 0.375 | 0.541 / 0.418 | 0.680 |

**Flags** (share of summaries)

| method (scope) | not fluent | low coverage | low coverage without exact duplicates | generic | duplicate | near-duplicate | mentions | margin <= 0 | unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (final (approx.)) | 0% | 8% | 8% | 1% | 0% | 0% | 0% | 1% | 1% |
| k-SLMmeans (Qwen3-0.6B) (final (approx.)) | 89% | 16% | 16% | 5% | 0% | 0% | 0% | 5% | 3% |
| k-SLMmeans (Qwen3-1.7B) (final (approx.)) | 9% | 25% | 25% | 12% | 0% | 1% | 1% | 12% | 14% |
| k-SLMmeans (Qwen3-4B) (final (approx.)) | 7% | 25% | 25% | 17% | 0% | 0% | 0% | 17% | 11% |

**Fluency checks** (share of summaries failing each)

| method (scope) | no "?" | echoes prompt | copies a member | markup/quotes | non-Latin letters | empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (final (approx.)) | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (final (approx.)) | 26% | 0% | 30% | 83% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (final (approx.)) | 6% | 1% | 3% | 2% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (final (approx.)) | 3% | 0% | 2% | 3% | 0% | 0% |

Files: `summary_quality_table.md` / `.csv` (this table), `summary_quality_per_step.csv` (method-level numbers per step).

### What the tables show

- On CLINC150 the 4B is worst on coverage and specificity. Last step, mean nearest-summary share: LSA 0.80, 0.6B 0.61, 1.7B 0.61, 4B 0.49;
  share with a non-positive specificity margin: 0%, 14%, 19%, 39%. The 0.6B and 1.7B are about level on coverage (0.607 vs 0.614 nearest-summary
  share, 0.764 vs 0.768 intent coverage) and the 1.7B has a lower margin. The unsupported share rises across the three sizes on CLINC150 (flagged: 12%, 25%, 47%).
  The 4B has the most exact duplicates (13% of its last-step summaries) and the most that mention the cluster (36%). Model size is confounded with how each
  model reads the prompt (#46), and this analysis does not separate the two.
- The 4B's low coverage is partly the duplicate rule. Exact-duplicate rows (12.5% of the 4B's last-step summaries) share one embedding, only the first copy
  wins members, and the later copies score 0.001 on average. Without them the 4B's mean nearest-summary share is 0.564 instead of 0.494 and its low-coverage
  share 40% instead of 47%. The other methods barely change (0.6B 0.613, 1.7B 0.619). So the gap to the 0.6B and 1.7B is smaller than the headline 0.49, though it remains.
- Fluency proxies are not aligned with the other three families. The 1.7B fails the proxies least among the Qwen3 sizes (9%) but has
  more unsupported lemmas than the 0.6B (25% vs 12% flagged). The 0.6B and 4B fail the proxies in 28% and 34% of all summaries
  (for the 0.6B mostly copying a member verbatim and a missing "?"; for the 4B echoing the prompt, a missing "?" and quotes).
- Among CLINC150 summaries that pass every fluency proxy, a substantial share still fails another family
  (low coverage, generic or unsupported): 11% for LSA, 34% for the 0.6B, 42% for the 1.7B, 44% for the 4B.
- BANKING77 (approximate, last step): nearest-summary share is 0.80 / 0.74 / 0.68 / 0.68 (LSA, 0.6B, 1.7B, 4B). The unsupported share
  is 0.017 / 0.040 / 0.200 / 0.177, so it is not monotone in size there (1.7B above the 4B). The 4B has no duplicates and does not mention
  the cluster, consistent with `prelim_bank77.md`. The 0.6B's 89% "not fluent" is almost entirely the proxy for answers wrapped in double quotes (83%).
- Members at the step (CLINC150, "Members at the step" table, counted after the full Lloyd block): the share of members lost is larger for the
  Qwen3 summaries and largest for the 4B (last step: LSA 2%, 0.6B 24%, 1.7B 25%, 4B 41%). A member is much more likely to leave when its gold intent is one the
  summary does not cover than when it is covered: lost if uncovered vs lost if covered, last step, LSA 9% vs 1%, 0.6B 61% vs 10%, 1.7B 58% vs 10%, 4B 75% vs 14%.
  This association is partly built in: a member whose intent the summary does not cover is nearer to another summary, so it is the kind of member the next assignment moves. It
  is not evidence that the summary caused the move (#47).

### The clearer prompt at three sizes (#46)

Runs `clinc150_v2_qwen3-0.6b/20261112-200926`, `clinc150_v2_qwen3-1.7b/20261112-204127` and `clinc150_v2_qwen3-4b/20261112-181256`: Qwen3-0.6B, 1.7B and 4B, seeds 0-4, the same step-0 partitions as the October runs, a prompt that asks for one new question
in the same style and says not to mention the questions, a group, a cluster or a topic (the wording is in each run's `config.yaml`; the effect of the 4B version on clustering is in `prompt_test.md`
section 8). CLINC150 only. The "echoes the prompt" check compares each summary with the prompt the run was given: `scripts/summary_quality.py` reads
`summarizer.prompt` from the run's config and falls back to the dataset prompt (none of the October runs sets one, so their scores are unchanged). The four families stay apart.
Numbers are CLINC150, all steps (last step in brackets where the printed value differs; label alignment, the method-level genericity of #7 and the entity and number rows are all steps only). "0.6B", "1.7B" and "old 4B" are the October runs, "v2" the clearer-prompt run of the same size, "LSA" the extractive control.

| | LSA | 0.6B | v2 0.6B | 1.7B | v2 1.7B | old 4B | v2 4B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Coverage** | | | | | | | |
| nearest-summary share (mean) | 0.791 (0.796) | 0.599 (0.607) | 0.682 (0.689) | 0.604 (0.614) | 0.590 (0.603) | 0.493 (0.494) | 0.664 (0.672) |
| intent coverage (mean) | 0.959 (0.962) | 0.753 (0.764) | 0.864 (0.869) | 0.764 (0.768) | 0.751 (0.768) | 0.595 (0.590) | 0.811 (0.819) |
| low-coverage flag | 11% (9%) | 35% | 24% (22%) | 34% (32%) | 37% (36%) | 48% (47%) | 26% (24%) |
| label alignment sim / acc (#8) | 0.390 / 0.523 | 0.477 / 0.390 | 0.498 / 0.498 | 0.419 / 0.451 | 0.434 / 0.423 | 0.409 / 0.323 | 0.510 / 0.485 |
| **Genericity** | | | | | | | |
| specificity margin (mean) | 0.247 (0.250) | 0.136 (0.146) | 0.202 (0.204) | 0.109 (0.112) | 0.119 (0.122) | 0.003 | 0.148 (0.153) |
| max cos to another summary (mean) | 0.567 (0.566) | 0.581 (0.570) | 0.561 (0.565) | 0.638 (0.635) | 0.595 (0.598) | 0.702 | 0.627 (0.626) |
| duplicate / near-duplicate | 0% / 0% | 1% / 2% | 0% / 0% | 1% / 1% | 0% / 0% | 11% (13%) / 14% (15%) | 1% / 1% |
| mentions the cluster | 0% | 8% (9%) | 0% | 8% (9%) | 0% | 34% (36%) | 0% |
| margin <= 0 | 0% | 16% (14%) | 5% (6%) | 19% | 20% (19%) | 39% | 17% (16%) |
| generic (union) | 0% | 19% (17%) | 5% (6%) | 22% (21%) | 20% (19%) | 46% (47%) | 17% |
| method-level genericity (#7) | 0.204 | 0.193 | 0.182 | 0.274 | 0.224 | 0.290 | 0.224 |
| **Unsupported information** | | | | | | | |
| unsupported lemma share (mean / median) | 0.000 / 0.000 | 0.128 / 0.000 | 0.015 / 0.000 | 0.298 / 0.267 | 0.288 / 0.250 | 0.438 / 0.409 | 0.131 / 0.000 |
| summaries with an unsupported entity | 0.0% | 1.3% | 0.3% | 1.5% | 0.5% | 1.1% | 0.1% |
| summaries with an unsupported number | 0.0% | 0.4% | 0.1% | 0.0% | 0.5% | 0.0% | 0.0% |
| unsupported (flag) | 0% | 12% | 1% | 24% (25%) | 26% (29%) | 46% (47%) | 10% (12%) |
| **Fluency proxies** | | | | | | | |
| no "?" | 0% | 12% | 8% (7%) | 3% | 1% | 14% (15%) | 39% (38%) |
| echoes the prompt | 0% | 7% (8%) | 0% | 7% | 0% | 30% (31%) | 0% |
| copies a member verbatim | 0% | 15% | 33% (34%) | 1% | 1% | 1% | 12% |
| markup/quotes | 1% | 3% | 0% | 0% | 0% | 13% | 0% |
| non-Latin letters / empty | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| not fluent (union) | 1% | 28% (29%) | 38% (39%) | 9% | 2% | 34% (35%) | 43% |

Also measured on the same summaries. Exact-duplicate rows at the last step, per seed (seeds 0-4): 0.6B 0, 0, 0, 2, 5; 1.7B 0, 0, 0, 2, 4; old 4B 19, 16, 19, 19, 21; every v2 run 0, 0, 0, 0, 0 except the v2 4B (0, 0, 2, 2, 0). The mean summary is 8.7 words for the v2 0.6B (11.2 old), 13.5 for the v2 1.7B (21.0 old), 10.9 for the v2 4B (17.7 old) and 38.6 for LSA.
Members at the step (last step; kept / lost if uncovered / lost if covered): LSA 98% / 9% / 1%; 0.6B 76% / 61% / 10%, v2 88% / 37% / 6%; 1.7B 75% / 58% / 10%, v2 79% / 48% / 9%; old 4B 59% / 75% / 14%, v2 80% / 57% / 7%.
Among summaries that pass every fluency proxy, the share that still fails coverage, genericity or unsupported content is 11% (LSA), 34% / 32% (0.6B / v2), 42% / 53% (1.7B / v2) and 44% / 45% (old 4B / v2).

What the scores show, descriptively, and what changes across sizes:

- **The effect of the prompt is not the same at the three sizes.** At the 4B and the 0.6B the clearer prompt improves coverage, genericity and unsupported content together. At the 1.7B it removes the echoes, the cluster mentions and the duplicates and lifts the fluency proxies, but coverage and unsupported content do not improve. Under the October prompt the 0.6B and 1.7B were level and the 4B was last on coverage; under the clearer prompt the order is 0.6B, then 4B, then 1.7B. The three sizes are not ordered by size under either prompt, and this analysis does not separate model size from how each model reads the prompt (#46).
- **Coverage.** Nearest-summary share (all steps), October to v2: 0.6B 0.599 to 0.682, 1.7B 0.604 to 0.590, 4B 0.493 to 0.664. Intent coverage: 0.753 to 0.864, 0.764 to 0.751, 0.595 to 0.811. Low-coverage flag: 35% to 24%, 34% to 37%, 48% to 26%. All three v2 runs stay below LSA (0.791, 0.959, 11%); the v2 0.6B is the closest (0.682, 0.864, 24%), and it has the lowest low-coverage share of the model runs.
  Label alignment (#8) similarity / accuracy: 0.6B 0.477 / 0.390 to 0.498 / 0.498, 1.7B 0.419 / 0.451 to 0.434 / 0.423, 4B 0.409 / 0.323 to 0.510 / 0.485 (LSA 0.390 / 0.523). Exact duplicates explain little of the v2 numbers: without them the v2 means are 0.682, 0.590 and 0.667.
- **Genericity.** Specificity margin: 0.6B 0.136 to 0.202, 1.7B 0.109 to 0.119, 4B 0.003 to 0.148 (LSA 0.247). Mentions of the cluster fall to 0% at every size (8%, 8%, 34% before), duplicates to 1% or less, and the generic union goes 19% to 5% (0.6B), 22% to 20% (1.7B) and 46% to 17% (4B). The 1.7B's margin <= 0 share is unchanged (19% to 20%).
  The method-level genericity (#7) is 0.193 to 0.182, 0.274 to 0.224 and 0.290 to 0.224, against LSA's 0.204; the v2 0.6B (0.182, max cos 0.561) and the October 0.6B (0.193) are below LSA's on this number, while the 0.6B's own generic union is 5% against 0% for LSA, so the method-level number and the per-summary flags answer different questions.
- **Unsupported information.** Lemma share: 0.6B 0.128 to 0.015, 1.7B 0.298 to 0.288, 4B 0.438 to 0.131; flag 12% to 1%, 24% to 26%, 46% to 10%. Unsupported entities and numbers stay rare (1.5% or less of summaries in every run). This check is lexical and the v2 summaries are short, so a few lemmas decide the share; a median of 0.000
  (0.6B and 4B) means most of their summaries use only words that occur in some member, and it does not say they are supported in meaning (the hand check is meant to say that). The v2 1.7B is the exception: median 0.250.
- **Fluency proxies.** "Echoes the prompt" goes to 0% at every size (7%, 7%, 30% before), quotes and markup to 0% (3%, 0%, 13% before). The other proxies move in different directions at each size. The v2 4B's missing "?" rises from 14% to 39% (see below); the 1.7B's falls from 3% to 1%; the 0.6B's from 12% to 8%.
  "Copies a member verbatim" rises from 15% to 33% for the 0.6B (1,245 of 3,750 summaries) and from 1% to 12% for the 4B, and stays at 1% for the 1.7B. The union "not fluent" is 28% to 38% (0.6B), 9% to 2% (1.7B) and 34% to 43% (4B); about 30 points of the v2 0.6B's are summaries whose only failing proxy is the copy, about 31 points of the v2 4B's are summaries whose only failing proxy is the missing "?".
  It is a count of proxy failures, not a measure of how fluent the text is, and here it does not track the other families: the v2 1.7B fails the proxies least (2%) yet has the weakest coverage of the three v2 runs and the highest share of proxy-passing summaries that fail another family (53%).
- **The missing "?" is not the same thing at each size.** For the v2 4B, of the 1,450 summaries without a "?", 90% start with a question or auxiliary word (what, how, why, when, where, which, who, whom, whose, can, could, do, does, did, is, are, am, was, were, will, would, shall, should, have, has, had, may, might, must; matched on the lowercased first word; the share depends on this list)
  and only 4% start with a capital letter. The old 4B's missing-"?" summaries (521) are 94% capitalised and 80% start with one of those words, so for the old 4B the missing mark mostly goes with a capitalised sentence and for the v2 4B with a lowercase one. This supports reading the v2 4B cases as lowercase questions with the mark dropped
  (for example "what is the status of my flight"), which fits the pilot's finding that most lowercase summaries were still questions in form. For the 0.6B the picture differs: of its 296 v2 summaries without a "?", 16% start with such a word and 56% with a capital letter (October: 448, 13% and 96%), so most are not lowercase questions with the mark dropped, and the proxy may be right more often there.
  For the 1.7B (44 summaries, 89% start with such a word, 93% capitalised) the count is too small to say more. The check was written for the October outputs and counts these as failures; the hand check, not the proxy, should say whether they are.
- **Verbatim copies.** 451 of the 3,750 v2 4B summaries (12%) and 1,245 of the v2 0.6B's (33%) are word-for-word copies of a cluster member; the v2 1.7B has 31 (0.8%). Copying is a fluency proxy, not evidence that these summaries are worse. Three things follow from it.
  A copy has no unsupported lemmas by construction, so copies account for part of the drop in unsupported share: excluding them the v2 4B mean is 0.149 (0.131 with them, 0.438 for the old 4B), the v2 0.6B 0.022 (0.015 with them), the v2 1.7B 0.291 (0.288). For the 0.6B the low unsupported share therefore does not come from the copies alone. Copies also score better on coverage: the low-coverage flag is 6.7% for the 4B's copies against 28% for its other v2 summaries
  (mean nearest-summary share 0.83 against 0.64), and 14% against 29% for the 0.6B (0.74 against 0.65); for the 1.7B's 31 copies it is 35.5% against 37% (0.56 against 0.59). And a copied member makes that cluster's summary embedding a member embedding, which matters for reading the ACC gain in #46; that is left to #50, and no causal claim is made here.

By these scores the clearer prompt moves the 4B from the weakest model on coverage, genericity and unsupported content to one at or better than the 1.7B on all three, and gives the 0.6B the best coverage, margin and unsupported scores of the three v2 sizes, still below LSA on coverage and margin, with a third of its summaries copied from a member and more of its summaries failing the proxies than before. The 1.7B is the one size where the prompt removes the failure modes of the October summaries without raising coverage or lowering unsupported content. These are scores of the logged summaries. They do not say why the prompt helps at one size and not another (#47) or how this relates to
the clustering outcome (#50). Hand check: the v2 summaries are not in the hand-check sheets (section 5), and adding them is a choice for the person rating.

## 4. Worked examples: fluent summaries that do not represent their cluster

Selection rule (`scripts/summary_quality_examples.py`; applied once to the final scores. A first pass picked a 1.7B summary containing a Chinese word, which led to adding the non-Latin-letters fluency check; the rule itself did not change): per model, seed 0, last summary step (step 4), among summaries that pass every
fluency proxy, have a nearest-summary share below 0.3 and at least 50 members, the one that lost the most members. This picks the most
extreme case that fits the rule for each model. It is not a typical cluster: the tables above give the rates. The filter itself (a summary nearest to under 30% of its
members, then the most members lost) nearly guarantees a large loss, so these cases show what the loss looks like, not that it follows from the scores.
All kept/lost/gained counts are after the full Lloyd block (`partition_after`), not at the replacement itself.

**Qwen3-0.6B, cluster 69.** Summary: "What are the key transfers and actions you need to perform to move money between accounts and handle
account-related tasks?" A well-formed single question. 206 members, whose most common gold intents are order_checks (50), transfer (39),
freeze_account (23) and balance (15). Nearest-summary share 0.00, intent coverage 0.00 (none of its intents), margin -0.059. Flagged generic
(margin) and unsupported. After the Lloyd block all 206 members had left (134 of them to cluster 110) and 118 others had joined.

**Qwen3-1.7B, cluster 121.** Summary: "What is the impact of opening a new account on a credit score and how can it be managed?" A well-formed
question. 250 members: pin_change (99), account_blocked (51), freeze_account (33), report_fraud (16). Nearest-summary share 0.08, intent coverage 0.00,
margin -0.037. Flagged generic; **not** flagged unsupported, although none of its members' leading gold intents is about credit scores or opening accounts. Fewer than half of its content
lemmas are absent from the members, so the lexical unsupported rule does not fire; this shows the rule is loose. No member stayed; 109 went to cluster 103 and 65 to cluster 51.

**Qwen3-4B, cluster 81.** Summary: "What is the main theme of the cluster?" Well-formed, and about the task, not the content. 398 members:
restaurant_suggestion (113), meal_suggestion (95), restaurant_reviews (35), how_busy (31). Nearest-summary share 0.00, margin -0.309. Flagged generic (mentions
the cluster, negative margin) and unsupported. No member stayed; 313 went to cluster 123 and 37 to cluster 17, while 50 others joined.

In all three the cluster's members had left by the end of the step's Lloyd block and the cluster that carries the same id afterwards is made mostly of other queries (the "gained" counts). These are single seed-0 cases; they illustrate the kept/lost link and do not establish how often this happens (the tables do).

## 5. Hand check: two human raters

The October request was an audit of representative summaries against the automatic scores. The final feedback (#84) asked for two team members to rate the 40 summaries of each dataset independently, blind to the method. It also asked for their agreement, a comparison with the automatic measures, and whether the unsupported-content measure wrongly flags reasonable paraphrases.

### Results (Samiksha `is-sami`, Nabin `naween321`; 40 summaries per dataset)

**Design.**
- Two team members each rated both blank sheets by `docs/summary_audit_rubric.md`. Each summary was read against 10 shown members, with the method hidden.
- Ratings: the number of shown members the summary fits (`members_fit`), and generic, unsupported and fluent as yes or no. The hand labels are as in the rubric: low coverage means fewer than half of the shown members fit, and not fluent is fluent = 0.
- The numbers come from `scripts/human_eval_report.py` (commands below). Its checks all pass: the raters changed only the rating columns, every value is in range, and every cluster's member count matches.
- One BANKING77 `fluent` cell is empty (row `bank77-05-D`, `naween321`); that row is left out of that property.
- Nabin's first submission was made with an AI tool. It was discarded, and he re-rated both sheets by hand (Nov 27). Only the manual pass is used (limits below).

**Agreement between the two raters** (Cohen's kappa, raw agreement in brackets; n = 40, BANKING77 not fluent n = 39):

| property | CLINC150 | BANKING77 |
| --- | ---: | ---: |
| low coverage | 0.70 (0.85) | 0.36 (0.78) |
| generic | 0.07 (0.50) | -0.09 (0.62) |
| unsupported | 0.26 (0.65) | -0.10 (0.80) |
| not fluent | 0.06 (0.75) | 0.25 (0.67) |
| `members_fit` count: identical / Spearman / mean absolute difference | 20 of 40 / 0.70 / 1.52 | 8 of 40 / 0.53 / 2.42 |

- **Coverage is where the raters agree.** On CLINC150 they give the same low-coverage label to 34 of 40 summaries.
- **Generic and not fluent split them,** mostly over LSA's five-sentence extracts:
  - Samiksha marks LSA's extracts generic in 7 of 10 (CLINC150) and 10 of 10 (BANKING77), and not fluent in 9 of 10 and 8 of 10. Nabin marks none of them generic.
  - On the 30 Qwen3 summaries of CLINC150 the raters agree on not fluent for 29 of 30 (Samiksha 0, Nabin 1).
  - Generic still splits there: Samiksha marks 15 of 30, Nabin 4.
- **Unsupported agrees weakly on CLINC150 and not at all on BANKING77,** where the raters mark only 3 and 5 summaries.

**Human against the automatic flags** (kappa, raw agreement in brackets). Columns: Samiksha / Nabin / consensus. The consensus is the rows on which both raters agree on that property; its n is given.

| property | CLINC150 | BANKING77 |
| --- | --- | --- |
| low coverage | 0.36 (0.68) / 0.46 (0.72) / 0.48 (0.74; n 34) | 0.26 (0.75) / -0.12 (0.78) / -0.05 (0.84; n 31) |
| generic | 0.23 (0.60) / 0.26 (0.75) / 0.48 (0.85; n 20) | -0.09 (0.62) / 0.47 (0.95) / 0.00 (0.96; n 25) |
| unsupported | 0.16 (0.62) / -0.26 (0.47) / -0.20 (0.58; n 26) | 0.48 (0.95) / -0.04 (0.85) / no positives (1.00; n 32) |
| not fluent | -0.29 (0.55) / 0.06 (0.75) / -0.06 (0.70; n 30) | 0.04 (0.68) / 0.28 (0.69) / 0.15 (0.77; n 26) |

Counts on CLINC150, automatic flag / Samiksha / Nabin:
- low coverage 23 / 18 / 18;
- generic 12 / 22 / 4;
- unsupported 9 / 16 / 14;
- not fluent 9 / 9 / 3.

The not-fluent flags fall on Qwen3 summaries (9 of 30), which both raters read as fluent, while Samiksha's not-fluent marks fall on LSA's extracts. On BANKING77 every automatic flag is rare (1 to 7 of 40), so those kappas rest on a handful of summaries.

**Does the unsupported flag wrongly flag reasonable paraphrases? Partly: on CLINC150, 4 of the 9 flagged summaries are reasonable paraphrases or generalisations, and none is invented content.**

For each flagged summary, the words that set the flag off were sorted into a category. The categories are those of #34, plus `cluster_talk`, added because the five did not fit summaries that describe the cluster or the task. One reader did the sorting, without double coding. The tables are in `results/human_eval_filled/` (untracked, because they carry the key).

| | paraphrase | generalisation | cluster talk | unshown members | unsupported |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLINC150, flagged (9) | 2 | 2 | 5 | 0 | 0 |
| CLINC150, rater says unsupported, flag silent (16) | 0 | 2 | 2 | 11 | 1 |
| BANKING77, flagged (1) | 0 | 0 | 0 | 0 | 1 |
| BANKING77, rater says unsupported, flag silent (7) | 0 | 0 | 0 | 6 | 1 |

**Flagged on CLINC150 (9).**
- Two paraphrases: "express my gratitude and appreciation" for thank-you requests, and "request someone to repeat" for the repeat intent.
- Two generalisations: "travel planning" and "travel time estimation" for flight and airport questions.
- Both raters judged 3 of these 4 supported; the fourth split them.
- The other 5 are cluster talk, four of them the 4B's "What is the main topic of the following cluster of questions?" and its variants. The flag fires on them only because "cluster", "topic" and "main" occur in no member. Samiksha marks all 5 unsupported and Nabin none, so the raters split on whether talk about the cluster is "unsupported information".
- Both raters judged supported 3 of the 9 flagged summaries (33%), all among the paraphrases and generalisations.

**Flagged on BANKING77 (1).** A 1.7B summary that adds "PIN changes" and "balance discrepancies", which no member mentions. It is sorted as unsupported; the raters split.

**Rater says unsupported, flag silent.** Mostly a property of the sheets, not a miss by the flag. In 11 of 16 (CLINC150) and 6 of 7 (BANKING77) cases, the words the rater objected to occur only in members the rater was not shown, because the sheets show 10. Nine of those 17 are LSA extracts, which are member sentences by construction. The genuine misses are one summary per dataset that recombines details of two members into a claim no member makes: "How can I use contactless pay to receive my salary?" (BANKING77) and a Dallas-to-Los Angeles flight "for under $300" (CLINC150). This is the failure the lexical measure cannot see (section 4).

**What the hand check changes about how #50 uses the automatic scores.**
- **Low coverage:**
  - It is the one property with a stable human reading: the raters agree with each other (kappa 0.70 on CLINC150) and moderately with the flag (consensus 0.48).
  - The flag marks more summaries than either rater (23 against 18 and 18).
  - So the coverage half of the "readable but a poor representative" share (`synthesis.md` section 12) is roughly what a reader would see, with the flag on the strict side.
- **Generic:** the raters disagree with each other (kappa 0.07 and -0.09), so there is no human reference to check it against. Read the automatic genericity as its own definition (a duplicate, a mention of the cluster, or a non-positive specificity margin), not as a human judgment.
- **Unsupported:**
  - It does not track human judgment on CLINC150 (consensus kappa -0.20). In this sample it flags paraphrases, generalisations and cluster talk, not invented content.
  - It misses the recombinations that a reader catches.
  - #50 should therefore read the unsupported share as "words outside the members' vocabulary". On CLINC150 that mostly signals cluster talk, not hallucination.
- **Not fluent:**
  - The proxy is stricter than both raters on the Qwen3 summaries (9 of 30 flagged; the raters mark 0 and 1).
  - "Fluent" in `synthesis.md` section 12 is the proxy's definition, and a reader would call more of those summaries fluent.

**Limits.**
- Two raters, 40 summaries per dataset (10 per method), and few positives on BANKING77, so every kappa is rough.
- Nabin's first submission was AI-generated. He re-rated by hand after seeing it, so anchoring cannot be excluded.
  - His manual sheets differ from that version on 32 of 40 (BANKING77) and 35 of 40 (CLINC150) summaries.
  - They agree with the AI-rater stand-in about as much as Samiksha does: the same `members_fit` count on 13 to 17 of 40, against 24 to 27 for the discarded version.
- Samiksha's sheets were in a public pull request before Nabin's manual pass. His agreement with her is moderate (the same count on 8 and 20 of 40), with no sign of copying.
- Blinding is imperfect (section 6): LSA's extracts are recognisable as member sentences.
- One reader sorted the paraphrase categories, without double coding. The v2 summaries are not in the sheets.

### Materials

- Rubric, written before any summary was read for this audit: `docs/summary_audit_rubric.md`.
- Sample (`scripts/make_audit_sheet.py`, seed 0): 10 gold intents per dataset, 4 well-, 3 mixed-, 3 poorly-recovered, then for each intent the
  matched cluster of each of LSA / 0.6B / 1.7B / 4B at the last summary step (the October runs only; the v2 runs are not in the sheets), so 40 summaries per dataset, method names hidden and letters shuffled.
  Recovery uses the max-count rule of `prelim_bank77.md` section 6 to match an intent to a cluster, graded by the Jaccard overlap of the
  matched cluster with the intent (well at least 0.5, poor below 0.25, minimum over the four methods). Each row shows the cluster size and 10 random members.
- Blank sheets for the human rater: `summary_audit_sheet_clinc150_intent.csv` and `summary_audit_sheet_bank77.csv` (plus `.md` copies for reading). The keys
  (method, cluster and automatic flags) are deliberately **not** in the repo. They regenerate deterministically with
  `make_audit_sheet.py ... --write-key results/audit_keys` (command in section 7). The keys were regenerated locally (untracked) to unblind the AI ratings; the regenerated sheets are byte-identical to the committed blank ones, so nothing in the sheets changed. The human rater should rate from the sheets alone and not open the keys. The BANKING77 sheet uses the
  approximate final partition.
- After the human rating, `python scripts/audit_agreement.py SHEET results/audit_keys/summary_audit_key_<dataset>.csv` prints, per property (low coverage, generic, unsupported, not fluent), the 2 x 2 table, raw agreement and
  Cohen's kappa for one sheet; `scripts/human_eval_report.py` (below) does every rater and the comparisons between them.

### Scoring the human ratings

The two raters (`is-sami`, `naween321`) save their filled copies of the blank sheets as `docs/results/summary_audit_human_<login>_{clinc150_intent,bank77}.csv`. Once both sets are in, one command gives every number of the final-feedback request (no results yet). Keys stay out of the repo (`results/` is gitignored). First regenerate the keys and prove the regenerated sheets are the committed ones, so the keys belong to the sheets that were rated:

```bash
export HF_HUB_OFFLINE=1
R=results   # the logged runs (run from the repository root)
python scripts/make_audit_sheet.py $R/clinc150_lsa/20261027-174631 $R/clinc150_qwen3-0.6b/20261027-190908 \
    $R/clinc150_qwen3-1.7b/20261027-202532 $R/clinc150_qwen3-4b/20261027-221042 $R/prelim_bank77_*/* \
    --scores results/summary_quality/summary_quality_per_cluster.csv \
    --cache-dir cache --out-dir results/audit_regen --write-key results/audit_keys
for d in clinc150_intent bank77; do cmp results/audit_regen/summary_audit_sheet_$d.csv docs/results/summary_audit_sheet_$d.csv && echo "$d identical"; done
python scripts/human_eval_report.py --raters is-sami naween321 --sheets-dir docs/results \
    --key-dir results/audit_keys --runs-dir $R --cache-dir cache --out-dir results/human_eval
```

`--scores` is the per-cluster CSV that the first command of section 7 writes. Keys built from the 11 Nov and the 16 Nov versions of that file are identical (every column of all 80 audit rows, the automatic flags and lemma shares included), and the sheets do not depend on the v2 rows. If `cmp` prints a difference, stop: the keys would not match the rated sheets. The report (`results/human_eval/human_eval_report.md`, with CSVs beside it) has, per dataset: the agreement between the raters per property (2 x 2, raw agreement, kappa; for `members_fit` also the Spearman correlation and mean absolute difference of the counts) and the disagreeing rows, listed and not resolved; each rater and the consensus (rows where all raters agree on that property) against the automatic flags; per-method rates for each rater; and the paraphrase check. The script recomputes the unsupported flag from each cluster's full member set and stops if it does not reproduce the key's flag on every row. `paraphrase_flagged_<dataset>.csv` has every flagged summary with the summary's content lemmas that occur in no member (`unsupported_lemmas`), its unsupported numbers and entities, `trigger_path` (which rule set the flag off: `lemma_share` for an unsupported share of at least 0.5, `number`, `entity`; the flag fired on the lemmas only if `lemma_share` is there) and each rater's value; fill its `category` column by hand with one of `paraphrase`, `generalisation`, `inflection_tokenisation`, `unshown_members`, `unsupported`. `paraphrase_reverse_<dataset>.csv` holds the summaries a rater marks unsupported where the flag does not fire, with each summary's words split into those in the shown members, only in members not shown, and in no member. Because the flag reads every member, a flagged summary's triggers are absent from all members, shown or not, so `unshown_members` can only apply in the reverse CSV. The script also checks that the raters did not change `summary`, `cluster_size`, `members_shown` or `members` (against the committed blank sheet), that every 0/1 rating is 0, 1 or empty and `members_fit` is between 0 and `members_shown`, and that each cluster's loaded member count equals its `cluster_size`; any failure stops the run with the rater, row and column named.

The report rewrites its own CSVs on every run, so do not fill them in place. Copy them, fill the copies, and pass the copies back (the script refuses a `--categories` path that it writes itself):

```bash
mkdir -p results/human_eval_filled && cp results/human_eval/paraphrase_*.csv results/human_eval_filled/
# fill the `category` column of the copies, then rerun the report command above with
#   --categories results/human_eval_filled/paraphrase_flagged_*.csv results/human_eval_filled/paraphrase_reverse_*.csv
```

The category counts are given per file, so the flagged and the reverse summaries are counted separately. `--sheet-pattern` and `--kind` (for example `--kind "AI ratings"`) let the same script run on other raters' sheets, for example the AI raters.

### AI-rater stand-in (not a human rating; kept as a comparison)

**What was done.** Two independent Claude Sonnet 5.5 instances each rated both blank sheets (40 summaries per dataset) by `docs/summary_audit_rubric.md`. Each saw only the rubric and the sheets: blind to the method, the key, the automatic scores and every other file, and blind to the other rater. Their filled sheets are
`summary_audit_ai_rater_1_<dataset>.csv` and `summary_audit_ai_rater_2_<dataset>.csv` (same rows and columns as the blank sheets; checked row by row against the blank sheets, with identical `row_id`, summary, cluster size and members, and no empty rating cell). Rater 1 matched its ratings to rows by position, which is safe because the row order is the same in the `.md` and `.csv` sheets, as checked.
The agreement tables and the per-method counts come from `scripts/audit_ai_report.py` and are in `summary_audit_ai_raters.md` (all tables, both raters, both datasets). They use the same hand labels as the rubric (low coverage: fewer than half of the shown members fit; generic; unsupported; not fluent).

**Agreement between the two raters** (n = 40 per dataset; kappa from the 2 x 2 table; raw agreement in brackets):

| property | CLINC150 kappa (raw) | BANKING77 kappa (raw) |
| --- | ---: | ---: |
| low coverage | 1.00 (1.00) | 0.91 (0.97) |
| generic | 1.00 (1.00) | 0.00 (0.97) |
| unsupported | 0.66 (0.97) | 1.00 (1.00) |
| not fluent | 0.94 (0.97) | 1.00 (1.00) |

The two raters differ on 4 of the 320 labels (CLINC150: 1 unsupported, 1 not fluent; BANKING77: 1 low coverage, 1 generic). The BANKING77 generic kappa of 0.00 comes from one label: rater 1 marks one summary generic and rater 2 marks none, so there is no variation in one labelling. Such high agreement between two copies of the same model reading the same rubric is what one would expect from shared biases as much as from a clear rubric; it is not independent evidence (see the limits).

**Agreement with the automatic flags** (kappa, raw agreement in brackets; rater 1 / rater 2; n = 40; full 2 x 2 counts in `summary_audit_ai_raters.md`):

| property | CLINC150 | BANKING77 |
| --- | ---: | ---: |
| low coverage | 0.30 (0.62) / 0.30 (0.62) | 0.55 (0.90) / 0.48 (0.88) |
| generic | 0.44 (0.80) / 0.44 (0.80) | -0.03 (0.93) / 0.00 (0.95) |
| unsupported | -0.05 (0.75) / -0.09 (0.72) | 0.66 (0.97) / 0.66 (0.97) |
| not fluent | 0.87 (0.95) / 0.79 (0.93) | 0.72 (0.93) / 0.72 (0.93) |

Reading these as counts. On CLINC150 the automatic low-coverage flag marks 23 of the 40 summaries and the AI raters 10 (both: 9), and the automatic unsupported flag marks 9 while the raters mark 1 and 2 (both: 0 for each rater). The automatic coverage flag asks whether the summary is the nearest one to at least half of the cluster's members, the AI label whether at least half of the 10 shown members are about what the summary says; these are different questions, and the flag is the stricter one on this sample. The unsupported flag is lexical and counts paraphrase as unsupported (section 2), and the raters accepted paraphrase, which fits the zero overlap but does not show that the flag is wrong. Not fluent agrees best (kappa 0.72 to 0.87), where both sides test the same visible property.
On BANKING77 the flags are rare (1 to 7 of 40) so the kappas rest on a handful of summaries (for example generic: automatic 2, raters 1 and 0, both 0).

**Per-method rates** (CLINC150 and BANKING77, 10 summaries per method per dataset, so these are counts of 10 and rough; the key unblinds the methods; rater 1 / rater 2, with the automatic flag count of the same 10 summaries after the bar):

| dataset | method | low coverage | generic | unsupported | not fluent |
| --- | --- | ---: | ---: | ---: | ---: |
| CLINC150 | k-NLPmeans (LSA) | 1 / 1 \| 4 | 0 / 0 \| 0 | 0 / 0 \| 0 | 1 / 1 \| 0 |
| CLINC150 | k-SLMmeans (Qwen3-0.6B) | 4 / 4 \| 6 | 1 / 1 \| 2 | 0 / 0 \| 0 | 2 / 1 \| 2 |
| CLINC150 | k-SLMmeans (Qwen3-1.7B) | 1 / 1 \| 6 | 1 / 1 \| 4 | 1 / 2 \| 3 | 2 / 2 \| 2 |
| CLINC150 | k-SLMmeans (Qwen3-4B) | 4 / 4 \| 7 | 4 / 4 \| 6 | 0 / 0 \| 6 | 6 / 6 \| 5 |
| BANKING77 | k-NLPmeans (LSA) | 0 / 0 \| 0 | 0 / 0 \| 0 | 0 / 0 \| 0 | 0 / 0 \| 0 |
| BANKING77 | k-SLMmeans (Qwen3-0.6B) | 3 / 4 \| 2 | 0 / 0 \| 1 | 1 / 1 \| 0 | 5 / 5 \| 6 |
| BANKING77 | k-SLMmeans (Qwen3-1.7B) | 1 / 1 \| 2 | 1 / 0 \| 0 | 1 / 1 \| 1 | 1 / 1 \| 0 |
| BANKING77 | k-SLMmeans (Qwen3-4B) | 2 / 2 \| 0 | 0 / 0 \| 1 | 0 / 0 \| 0 | 0 / 0 \| 1 |

What they show, on this sample only: on CLINC150 the 4B has the most AI-marked generic (4 of 10, both raters) and not-fluent summaries (6 of 10), and the 0.6B and 4B the most low-coverage ones (4 of 10); LSA and the 1.7B have 1 of 10 low-coverage. The ordering of the methods on generic and not fluent is the same as in the automatic flags (the 4B first), but on CLINC150 the AI raters mark fewer summaries than the flags on low coverage, generic and unsupported. On BANKING77 the AI-marked problems are concentrated in the 0.6B (3 or 4 low coverage, 5 not fluent; the automatic proxy attributes the 0.6B's not-fluent flags to double quotes, section 3, and the AI notes were not checked against that), with the 4B marked low coverage twice and little else.
The sample is not random over clusters (it is 10 gold intents chosen by recovery class, section 5 above and section 6), so these counts are not rates over all summaries; the tables of section 3 are.

**Limits, stated plainly.**
- The two raters are two copies of one model family reading one rubric. They can share biases with each other and with LLM-written text (for example in how they judge a fluent, plausible question), so their agreement with each other is not independent evidence, and their agreement with the automatic flags is not the agreement of a human with them.
- n is 40 summaries per dataset (10 per method); every kappa and count above is rough, and several cells have 0 to 3 positives.
- The rubric was written for a human rater; the AI raters applied it as written and may have read parts of it differently from a person.
- Blinding is imperfect for the same reasons as in section 6 (LSA rows look like member sentences, the 0.6B quotes its answers, the 4B's "main theme of the cluster"); a model rater can notice these too. The v2 runs are not in the sheets.

**What this supports and does not.** It supports a first look at where the automatic flags and a careful reading of the same summaries agree (not fluent on both datasets; low coverage and generic agree on one dataset and not the other, with the flags marking more summaries on CLINC150) and where they measure different things (the unsupported flag against a reader's judgement on CLINC150, where they disagree; on BANKING77 its agreement rests on 1 to 2 positives; and the coverage flag's strictness). It also shows the sheets and rubric can be applied end to end and the agreement code runs on real ratings. It does **not** validate the automatic scores, does not stand in for the human hand check requested in October, and should not be quoted as a hand-check result or as an inter-rater study (#9). The two human raters' results at the top of this section are the hand-check result. These AI ratings stay as a comparison only.

## 6. Limitations

- Blinding is imperfect: the LSA rows are recognisably member sentences joined by line breaks, the 0.6B often wraps its answer in quotes, and the 4B's "main theme of the cluster" is recognisable. A rater can often tell the method.
- "Well recovered" requires every one of the four methods to reach Jaccard 0.5 with the intent, which tilts the sample toward easy intents and under-samples the 4B's failures.
- Lexical unsupported rule (paraphrase counts as unsupported; shared common words hide invented topics, see the 1.7B example).
- Nearest-summary share depends on the other k-1 summaries, so it is a property of the step's whole set, not of one summary alone. Identical summaries tie and only the first keeps its members.
- The specificity margin uses the members' mean as the cluster's representative; a cluster that mixes intents has a mean that fits nothing well.
- The fluency proxies were chosen to be cheap; no grammar checker was used.
- Thresholds (0.5 low coverage, 0.95 near-duplicate, 0.5 unsupported share, 10% major intent) are conventions fixed before rating, not tuned.
- Seeds are pooled; no intervals are given here.

## 7. Reproduce

```bash
export HF_HUB_OFFLINE=1
R=results   # the logged runs (run from the repository root)
python scripts/summary_quality.py $R/clinc150_lsa/20261027-174631 $R/clinc150_qwen3-0.6b/20261027-190908 \
    $R/clinc150_qwen3-1.7b/20261027-202532 $R/clinc150_qwen3-4b/20261027-221042 \
    $R/clinc150_v2_qwen3-0.6b/20261112-200926 $R/clinc150_v2_qwen3-1.7b/20261112-204127 \
    $R/clinc150_v2_qwen3-4b/20261112-181256 $R/prelim_bank77_*/* \
    --cache-dir cache --out-dir results/summary_quality
python scripts/make_audit_sheet.py <the October run dirs above, without the three v2 runs> \
    --scores results/summary_quality/summary_quality_per_cluster.csv \
    --cache-dir cache --out-dir docs/results --write-key results/audit_keys
python scripts/audit_ai_report.py --sheets-dir docs/results --key-dir results/audit_keys \
    --out docs/results/summary_audit_ai_raters.md
```

CPU only (the script hides the GPU); 27,790 summaries took about 270 s (4.5 minutes) at low priority, with the spaCy parse and the re-encoding of the summaries (the 20,290 summaries of the earlier run took about 12 minutes on 12 cores shared with other jobs, so the two times are not comparable).
`make_audit_sheet.py` reproduces the committed blank sheets byte for byte (checked on 16 Nov against the fresh scores, so the sample does not depend on the v2 rows). `audit_ai_report.py` needs the keys, which are untracked.
The per-cluster CSV (`summary_quality_per_cluster.csv`, about 11 MB, keyed by dataset, method, run, seed, step, cluster, with the kept/lost/gained counts and the cluster's
purity after the step, `purity_after`, for #33 and #50) is not committed; `results/` is untracked, and it regenerates with the command above. Join note: `method` is the label ("k-NLPmeans", "k-SLMmeans (Qwen3-4B)"), because the Qwen runs share the
method name `k-SLMmeans`; `run` (`<config name>/<timestamp>`) disambiguates.
