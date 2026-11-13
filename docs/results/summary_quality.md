# Summary quality: coverage, genericity and unsupported content, scored apart from fluency (issue #34)

Status: **automatic scores done; hand check pending** (sheets generated, no ratings made yet). Everything below is
descriptive and preliminary. It says what the scores measure on the logged runs; it does not say why the clusterings
differ (that is #47) or how the scores relate to clustering outcome (that is #50).

## 1. What was scored

| Dataset | Runs | Seeds | Steps scored | Partition the members come from |
| --- | --- | --- | --- | --- |
| CLINC150 (22,500 queries, 150 intents) | `clinc150_lsa/20261027-174631`, `clinc150_qwen3-0.6b/20261027-190908`, `clinc150_qwen3-1.7b/20261027-202532`, `clinc150_qwen3-4b/20261027-221042`, and the clearer-prompt 4B run `clinc150_v2_qwen3-4b/20261112-181256` (#46, method label `k-SLMmeans-v2 (Qwen3-4B)`) | 0-4 | all 5 summary steps | the partition each summary was written from (`source_partition(j)`) |
| BANKING77 (3,080 queries, 77 intents) | `prelim_bank77_{lsa,qwen3-0.6b,qwen3-1.7b,qwen3-4b}` | 0-4 | last step only | the FINAL assignments (**approximation**) |

**BANKING77 caveat, applies to every BANKING77 number here.** These runs predate #30 and logged no per-step
assignments. The final summaries were written from the partition before the last Lloyd block, which is not logged, so
scoring them against the final partition mixes two partitions. No kept/lost/gained numbers exist for BANKING77. A
re-run under #33 would remove the approximation.

20,290 summaries were scored (18,750 CLINC150, 1,540 BANKING77). The v2 run has no BANKING77 counterpart. The logged `summary_inertia` was reproduced from the
re-embedded summaries to a relative difference of at most 1.5e-7 on every CLINC150 step (1.1e-7 for the v2 run), so the CPU re-embedding matches the
embeddings the runs used. No cluster was empty at a scored step.

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
| k-NLPmeans (last step) | 750 | 0.796 / 0.855 | 0.962 | 0.250 / 0.252 | 0.566 | 0.000 / 0.000 | 0.205 | 0.388 / 0.521 | 0.796 |
| k-SLMmeans (Qwen3-0.6B) (last step) | 750 | 0.607 / 0.684 | 0.764 | 0.146 / 0.171 | 0.570 | 0.128 / 0.000 | 0.193 | 0.472 / 0.345 | 0.613 |
| k-SLMmeans (Qwen3-1.7B) (last step) | 750 | 0.614 / 0.700 | 0.768 | 0.112 / 0.136 | 0.635 | 0.304 / 0.267 | 0.275 | 0.413 / 0.443 | 0.619 |
| k-SLMmeans (Qwen3-4B) (last step) | 750 | 0.494 / 0.569 | 0.590 | 0.003 / 0.086 | 0.702 | 0.454 / 0.429 | 0.289 | 0.402 / 0.304 | 0.564 |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 750 | 0.672 / 0.759 | 0.819 | 0.153 / 0.177 | 0.626 | 0.140 / 0.000 | 0.224 | 0.510 / 0.496 | 0.674 |

**Flags** (share of summaries)

| method (scope) | not fluent | low coverage | low coverage without exact duplicates | generic | duplicate | near-duplicate | mentions | margin <= 0 | unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 1% | 11% | 11% | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 28% | 35% | 34% | 19% | 1% | 2% | 8% | 16% | 12% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 9% | 34% | 34% | 22% | 1% | 1% | 8% | 19% | 24% |
| k-SLMmeans (Qwen3-4B) (all steps) | 34% | 48% | 41% | 46% | 11% | 14% | 34% | 39% | 46% |
| k-SLMmeans-v2 (Qwen3-4B) (all steps) | 43% | 26% | 25% | 17% | 1% | 1% | 0% | 17% | 10% |
| k-NLPmeans (last step) | 1% | 9% | 9% | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 29% | 35% | 34% | 17% | 1% | 2% | 9% | 14% | 12% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 9% | 32% | 32% | 21% | 1% | 1% | 9% | 19% | 25% |
| k-SLMmeans (Qwen3-4B) (last step) | 35% | 47% | 40% | 47% | 13% | 15% | 36% | 39% | 47% |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 43% | 24% | 24% | 17% | 1% | 1% | 0% | 16% | 12% |

**Fluency checks** (share of summaries failing each)

| method (scope) | no "?" | echoes prompt | copies a member | markup/quotes | non-Latin letters | empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 0% | 0% | 0% | 1% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 12% | 7% | 15% | 3% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 3% | 7% | 1% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (all steps) | 14% | 30% | 1% | 13% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-4B) (all steps) | 39% | 0% | 12% | 0% | 0% | 0% |
| k-NLPmeans (last step) | 0% | 0% | 0% | 1% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 12% | 8% | 15% | 3% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 3% | 7% | 1% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (last step) | 15% | 31% | 1% | 13% | 0% | 0% |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 38% | 0% | 12% | 0% | 0% | 0% |

**Members at the step** (pooled over clusters, as a share of the members the summaries were written from, counted on the partition after the step's Lloyd block. A member is *uncovered* when its gold intent is one the summary does not cover; `lost if uncovered` / `lost if covered` are the shares of uncovered / covered members that left their cluster)

| method (scope) | kept | lost | gained | members uncovered | lost if uncovered | lost if covered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 95% | 5% | 5% | 9% | 19% | 3% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 72% | 28% | 28% | 30% | 66% | 11% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 76% | 24% | 24% | 31% | 57% | 10% |
| k-SLMmeans (Qwen3-4B) (all steps) | 59% | 41% | 41% | 45% | 76% | 13% |
| k-SLMmeans-v2 (Qwen3-4B) (all steps) | 78% | 22% | 22% | 26% | 60% | 8% |
| k-NLPmeans (last step) | 98% | 2% | 2% | 9% | 9% | 1% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 76% | 24% | 24% | 28% | 61% | 10% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 75% | 25% | 25% | 31% | 58% | 10% |
| k-SLMmeans (Qwen3-4B) (last step) | 59% | 41% | 41% | 45% | 75% | 14% |
| k-SLMmeans-v2 (Qwen3-4B) (last step) | 80% | 20% | 20% | 25% | 57% | 7% |

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

### The 4B under the clearer prompt (#46)

Run `clinc150_v2_qwen3-4b/20261112-181256`: Qwen3-4B, seeds 0-4, the same step-0 partitions as the October runs, a prompt that asks for one new question
in the same style and says not to mention the questions, a group, a cluster or a topic (the wording is in the run's `config.yaml`; the effect on clustering is in `prompt_test.md`
section 8). CLINC150 only. The "echoes the prompt" check compares each summary with the prompt the run was given: `scripts/summary_quality.py` reads
`summarizer.prompt` from the run's config and falls back to the dataset prompt (none of the October runs sets one, so their scores are unchanged). The four families stay apart.
Numbers are CLINC150, all steps (last step in brackets where it differs by a point or more); "old 4B" and "1.7B" are the October runs, "LSA" is the extractive control.

| | LSA | 1.7B | old 4B | v2 4B |
| --- | ---: | ---: | ---: | ---: |
| **Coverage** | | | | |
| nearest-summary share (mean) | 0.791 (0.796) | 0.604 (0.614) | 0.493 (0.494) | 0.664 (0.672) |
| intent coverage (mean) | 0.959 (0.962) | 0.764 (0.768) | 0.595 (0.590) | 0.811 (0.819) |
| low-coverage flag | 11% (9%) | 34% (32%) | 48% (47%) | 26% (24%) |
| label alignment sim / acc (#8) | 0.390 / 0.523 | 0.419 / 0.451 | 0.409 / 0.323 | 0.510 / 0.485 |
| **Genericity** | | | | |
| specificity margin (mean) | 0.247 (0.250) | 0.109 (0.112) | 0.003 | 0.148 (0.153) |
| max cos to another summary (mean) | 0.567 | 0.638 (0.635) | 0.702 | 0.627 (0.626) |
| duplicate / near-duplicate | 0% / 0% | 1% / 1% | 11% (13%) / 14% (15%) | 1% / 1% |
| mentions the cluster | 0% | 8% (9%) | 34% (36%) | 0% |
| margin <= 0 | 0% | 19% | 39% | 17% (16%) |
| generic (union) | 0% | 22% (21%) | 46% (47%) | 17% |
| method-level genericity (#7) | 0.204 | 0.274 | 0.290 | 0.224 |
| **Unsupported information** | | | | |
| unsupported lemma share (mean / median) | 0.000 / 0.000 | 0.298 / 0.267 | 0.438 / 0.409 | 0.131 / 0.000 |
| summaries with an unsupported entity | 0% | 1.5% | 1.1% | 0.1% |
| summaries with an unsupported number | 0% | 0% | 0% | 0% |
| unsupported (flag) | 0% | 24% (25%) | 46% (47%) | 10% (12%) |
| **Fluency proxies** | | | | |
| no "?" | 0% | 3% | 14% (15%) | 39% (38%) |
| echoes the prompt | 0% | 7% | 30% (31%) | 0% |
| copies a member verbatim | 0% | 1% | 1% | 12% |
| markup/quotes | 1% | 0% | 13% | 0% |
| non-Latin letters / empty | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| not fluent (union) | 1% | 9% | 34% (35%) | 43% |

Also measured on the same summaries. Exact-duplicate rows at the last step, per seed (seeds 0-4): old 4B 19, 16, 19, 19, 21; v2 4B 0, 0, 2, 2, 0. The mean summary is 10.9 words for the v2 4B, 17.7 for the old 4B,
21.0 for the 1.7B and 38.6 for LSA. Members at the step (last step; kept / lost if uncovered / lost if covered): LSA 98% / 9% / 1%, 1.7B 75% / 58% / 10%, old 4B 59% / 75% / 14%, v2 4B 80% / 57% / 7%.
Among summaries that pass every fluency proxy, the share that still fails coverage, genericity or unsupported content is 11% (LSA), 42% (1.7B), 44% (old 4B) and 45% (v2 4B).

What the scores show, descriptively:

- **Coverage.** The clearer prompt raises the 4B's coverage on every measure above and puts it above the 1.7B: nearest-summary share 0.664 vs 0.493 (old 4B) and 0.604 (1.7B), intent coverage 0.811 vs 0.595 and 0.764, low-coverage flag 26% vs 48% and 34%. It stays below LSA (0.791, 0.959, 11%). The v2 4B has the highest label alignment of the four (0.510 sim, 0.485 acc).
  The duplicate rule explains little of the gain now: without exact duplicates the v2 mean is 0.667 against 0.555 for the old 4B.
- **Genericity.** Every flag is at or below the 1.7B's level, and the specificity margin is 0.148 (1.7B 0.109, old 4B 0.003, LSA 0.247). Mentions of the cluster go from 34% to 0%, duplicates from 11% to 1%,
  the generic union from 46% to 17% (1.7B 22%); the margin <= 0 share is 17% (1.7B 19%). It is not zero as for LSA, and the method-level genericity (0.224) is above LSA's (0.204) but below the 1.7B's (0.274) and old 4B's (0.290).
- **Unsupported information.** The unsupported lemma share falls from 0.438 to 0.131 (1.7B 0.298) and the flag from 46% to 10% (1.7B 24%). Unsupported entities and numbers are rare in every model run. This check is lexical and the v2 summaries are short (10.9 words), so a
  few lemmas decide the share; a median of 0.000 means most v2 summaries use only words that occur in some member. Whether a summary is supported in meaning is not measured here (the hand check is meant to).
- **Fluency proxies.** "Echoes the prompt" drops from 30% to 0% and quotes from 13% to 0%. "No '?'" rises, from 14% to 39% (1.7B 3%), and so does "copies a member" (1% to 12%, i.e. the model wrote one of the cluster's own queries). By these proxies this is the one family that gets worse, and it is why "not fluent" is 43% against 34%.
  The "?" proxy probably overstates disfluency here. Of the 1,450 v2 summaries without a "?", 90% start with a question or auxiliary word (what, how, can, do, ...) and only 4% start with a capital letter; they read as lowercase questions with the mark dropped
  (for example "what is the status of my flight"), which fits the pilot's finding that most lowercase summaries were still questions in form. For the old 4B the same share is 80%. The check was written for the October outputs and counts these as failures; the hand check, not the proxy, should say whether they are.

By these scores the clearer prompt moves the 4B from the weakest model on coverage, genericity and unsupported content to one at or better than the 1.7B on all three, still below LSA on coverage and margin, with more of its summaries failing the "?" proxy. These are scores of the logged summaries. They do not say why the prompt helps (#47) or how this relates to
the clustering outcome (#50). The 0.6B and 1.7B under the clearer prompt are queued and will be added to this subsection the same way; nothing is assumed about them here. Hand check: the v2 summaries are not in the hand-check sheets (section 5), and adding them is a choice for the person rating.

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

## 5. Hand check: pending

The October request was an audit of representative summaries against the automatic scores. The materials exist; **no ratings have been made**, so
there is no agreement number yet.

- Rubric, written before any summary was read for this audit: `docs/summary_audit_rubric.md`.
- Sample (`scripts/make_audit_sheet.py`, seed 0): 10 gold intents per dataset, 4 well-, 3 mixed-, 3 poorly-recovered, then for each intent the
  matched cluster of each of LSA / 0.6B / 1.7B / 4B at the last summary step, so 40 summaries per dataset, method names hidden and letters shuffled.
  Recovery uses the max-count rule of `prelim_bank77.md` section 6 to match an intent to a cluster, graded by the Jaccard overlap of the
  matched cluster with the intent (well at least 0.5, poor below 0.25, minimum over the four methods). Each row shows the cluster size and 10 random members.
- Sheets: `summary_audit_sheet_clinc150_intent.csv` and `summary_audit_sheet_bank77.csv` (plus `.md` copies for reading); the keys
  (method, cluster and automatic flags) are deliberately **not** in the repo, so the rater cannot see them. They regenerate deterministically with
  `make_audit_sheet.py ... --write-key results/audit_keys` (command in section 7); do that after rating. The BANKING77 sheet uses the
  approximate final partition.
- After rating, `python scripts/audit_agreement.py SHEET results/audit_keys/summary_audit_key_<dataset>.csv` prints, per property (low coverage, generic, unsupported, not fluent), the 2 x 2 table, raw agreement and
  Cohen's kappa. One rater; with 40 summaries per dataset the numbers will be rough.

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
R=../k-slimeans/results   # the logged runs
python scripts/summary_quality.py $R/clinc150_lsa/20261027-174631 $R/clinc150_qwen3-0.6b/20261027-190908 \
    $R/clinc150_qwen3-1.7b/20261027-202532 $R/clinc150_qwen3-4b/20261027-221042 \
    $R/clinc150_v2_qwen3-4b/20261112-181256 $R/prelim_bank77_*/* \
    --cache-dir ../k-slimeans/cache --out-dir results/summary_quality
python scripts/make_audit_sheet.py <the same run dirs, without the v2 run> --scores results/summary_quality/summary_quality_per_cluster.csv \
    --cache-dir ../k-slimeans/cache --out-dir docs/results --write-key results/audit_keys
```

CPU only (the script hides the GPU); 20,290 summaries took about 12 minutes at low priority (12 cores shared with other jobs, spaCy plus re-encoding).
The per-cluster CSV (`summary_quality_per_cluster.csv`, 6.5 MB, keyed by dataset, method, run, seed, step, cluster, with the kept/lost/gained counts and the cluster's
purity after the step, `purity_after`, for #33 and #50). Join note: `method` is the label ("k-NLPmeans", "k-SLMmeans (Qwen3-4B)"), because the Qwen runs share the
method name `k-SLMmeans`; `run` (`<config name>/<timestamp>`) disambiguates is not committed; `results/` is untracked, and it regenerates with the command above.
