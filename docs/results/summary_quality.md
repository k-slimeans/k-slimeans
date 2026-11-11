# Summary quality: coverage, genericity and unsupported content, scored apart from fluency (issue #34)

Status: **automatic scores done; hand check pending** (sheets generated, no ratings made yet). Everything below is
descriptive and preliminary. It says what the scores measure on the logged runs; it does not say why the clusterings
differ (that is #47) or how the scores relate to clustering outcome (that is #50).

## 1. What was scored

| Dataset | Runs | Seeds | Steps scored | Partition the members come from |
| --- | --- | --- | --- | --- |
| CLINC150 (22,500 queries, 150 intents) | `clinc150_lsa/20261027-174631`, `clinc150_qwen3-0.6b/20261027-190908`, `clinc150_qwen3-1.7b/20261027-202532`, `clinc150_qwen3-4b/20261027-221042` | 0-4 | all 5 summary steps | the partition each summary was written from (`source_partition(j)`) |
| BANKING77 (3,080 queries, 77 intents) | `prelim_bank77_{lsa,qwen3-0.6b,qwen3-1.7b,qwen3-4b}` | 0-4 | last step only | the FINAL assignments (**approximation**) |

**BANKING77 caveat, applies to every BANKING77 number here.** These runs predate #30 and logged no per-step
assignments. The final summaries were written from the partition before the last Lloyd block, which is not logged, so
scoring them against the final partition mixes two partitions. No kept/lost/gained numbers exist for BANKING77. A
re-run under #33 would remove the approximation.

16,540 summaries were scored (15,000 CLINC150, 1,540 BANKING77). The logged `summary_inertia` was reproduced from the
re-embedded summaries to a relative difference of at most 1.4e-7 on every CLINC150 step, so the CPU re-embedding matches the
embeddings the runs used. No cluster was empty at a scored step.

## 2. The scores (definitions in `src/kslimeans/summary_quality.py`)

Four families, never merged into one number. Thresholds are constants in the module and were set before any hand rating.

- **Coverage.** *Nearest-summary share*: the share of the cluster's members whose nearest of the k summary embeddings
  (Euclidean, the metric of the assignment step) is this cluster's summary. *Intent coverage*: among gold intents with at least
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

The extractive LSA summaries act as a control: they are member sentences, so unsupported content should be near zero and
they are not expected to be generic. They are (0.1% unsupported lemmas, 0% generic on CLINC150).

## 3. Score tables

Pooled over clusters, seeds and steps ("all steps") or over the last summary step. Medians appear next to means where
the distribution is skewed. `n` is the number of summaries. Fluency proxies and the other three families are reported
separately on purpose.

### clinc150_intent

**Scores** (mean / median over clusters; `n` = summaries)

| method (scope) | n | nearest-summary share | intent coverage | specificity margin | max cos to another summary | unsupported lemma share | genericity (#7) | label align sim / acc (#8) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 3,750 | 0.791 / 0.849 | 0.959 | 0.247 / 0.248 | 0.567 | 0.001 / 0.000 | 0.204 | 0.390 / 0.523 |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 3,750 | 0.599 / 0.672 | 0.752 | 0.136 / 0.164 | 0.581 | 0.128 / 0.000 | 0.193 | 0.477 / 0.390 |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 3,750 | 0.604 / 0.675 | 0.764 | 0.109 / 0.134 | 0.638 | 0.298 / 0.267 | 0.274 | 0.419 / 0.451 |
| k-SLMmeans (Qwen3-4B) (all steps) | 3,750 | 0.493 / 0.548 | 0.595 | 0.003 / 0.084 | 0.702 | 0.438 / 0.409 | 0.290 | 0.409 / 0.322 |
| k-NLPmeans (last step) | 750 | 0.796 / 0.855 | 0.962 | 0.250 / 0.252 | 0.566 | 0.001 / 0.000 | 0.205 | 0.388 / 0.521 |
| k-SLMmeans (Qwen3-0.6B) (last step) | 750 | 0.607 / 0.684 | 0.764 | 0.146 / 0.171 | 0.570 | 0.128 / 0.000 | 0.193 | 0.472 / 0.345 |
| k-SLMmeans (Qwen3-1.7B) (last step) | 750 | 0.614 / 0.700 | 0.768 | 0.112 / 0.136 | 0.635 | 0.304 / 0.267 | 0.275 | 0.413 / 0.443 |
| k-SLMmeans (Qwen3-4B) (last step) | 750 | 0.494 / 0.569 | 0.590 | 0.003 / 0.086 | 0.702 | 0.454 / 0.429 | 0.289 | 0.402 / 0.304 |

**Flags** (share of summaries)

| method (scope) | not fluent | low coverage | generic | duplicate | near-duplicate | mentions | margin <= 0 | unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 1% | 11% | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 28% | 35% | 19% | 1% | 2% | 8% | 16% | 12% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 9% | 34% | 22% | 1% | 1% | 8% | 19% | 24% |
| k-SLMmeans (Qwen3-4B) (all steps) | 34% | 48% | 46% | 11% | 14% | 34% | 39% | 46% |
| k-NLPmeans (last step) | 1% | 9% | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 29% | 35% | 17% | 1% | 2% | 9% | 14% | 12% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 9% | 32% | 21% | 1% | 1% | 9% | 19% | 25% |
| k-SLMmeans (Qwen3-4B) (last step) | 35% | 47% | 47% | 13% | 15% | 36% | 39% | 47% |

**Fluency checks** (share of summaries failing each)

| method (scope) | no "?" | echoes prompt | copies a member | markup/quotes | non-Latin letters | empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 0% | 0% | 0% | 1% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 12% | 7% | 15% | 3% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 3% | 7% | 1% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (all steps) | 14% | 30% | 1% | 13% | 0% | 0% |
| k-NLPmeans (last step) | 0% | 0% | 0% | 1% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 12% | 8% | 15% | 3% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 3% | 7% | 1% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (last step) | 15% | 31% | 1% | 13% | 0% | 0% |

**Members at the step** (pooled over clusters, as a share of the members the summaries were written from; `lost in uncovered intents` is the share of lost members whose gold intent the summary does not cover, to compare with `members in uncovered intents`)

| method (scope) | kept | lost | gained | lost in uncovered intents | members in uncovered intents |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (all steps) | 95% | 5% | 5% | 40% | 9% |
| k-SLMmeans (Qwen3-0.6B) (all steps) | 72% | 28% | 28% | 72% | 31% |
| k-SLMmeans (Qwen3-1.7B) (all steps) | 76% | 24% | 24% | 72% | 31% |
| k-SLMmeans (Qwen3-4B) (all steps) | 59% | 41% | 41% | 83% | 45% |
| k-NLPmeans (last step) | 98% | 2% | 2% | 36% | 9% |
| k-SLMmeans (Qwen3-0.6B) (last step) | 76% | 24% | 24% | 72% | 28% |
| k-SLMmeans (Qwen3-1.7B) (last step) | 75% | 25% | 25% | 73% | 31% |
| k-SLMmeans (Qwen3-4B) (last step) | 59% | 41% | 41% | 82% | 45% |

### bank77

Last summary step scored against the FINAL assignments (approximation: the summaries were written from the partition before the last Lloyd block, which is not logged).

**Scores** (mean / median over clusters; `n` = summaries)

| method (scope) | n | nearest-summary share | intent coverage | specificity margin | max cos to another summary | unsupported lemma share | genericity (#7) | label align sim / acc (#8) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (final (approx.)) | 385 | 0.801 / 0.842 | 0.943 | 0.179 / 0.165 | 0.662 | 0.017 / 0.000 | 0.382 | 0.461 / 0.473 |
| k-SLMmeans (Qwen3-0.6B) (final (approx.)) | 385 | 0.736 / 0.804 | 0.877 | 0.172 / 0.165 | 0.641 | 0.040 / 0.000 | 0.288 | 0.575 / 0.468 |
| k-SLMmeans (Qwen3-1.7B) (final (approx.)) | 385 | 0.680 / 0.745 | 0.806 | 0.121 / 0.122 | 0.670 | 0.200 / 0.143 | 0.370 | 0.501 / 0.429 |
| k-SLMmeans (Qwen3-4B) (final (approx.)) | 385 | 0.680 / 0.765 | 0.809 | 0.119 / 0.122 | 0.686 | 0.177 / 0.000 | 0.375 | 0.541 / 0.418 |

**Flags** (share of summaries)

| method (scope) | not fluent | low coverage | generic | duplicate | near-duplicate | mentions | margin <= 0 | unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (final (approx.)) | 0% | 8% | 1% | 0% | 0% | 0% | 1% | 1% |
| k-SLMmeans (Qwen3-0.6B) (final (approx.)) | 89% | 16% | 5% | 0% | 0% | 0% | 5% | 3% |
| k-SLMmeans (Qwen3-1.7B) (final (approx.)) | 9% | 25% | 12% | 0% | 1% | 1% | 12% | 14% |
| k-SLMmeans (Qwen3-4B) (final (approx.)) | 7% | 25% | 17% | 0% | 0% | 0% | 17% | 11% |

**Fluency checks** (share of summaries failing each)

| method (scope) | no "?" | echoes prompt | copies a member | markup/quotes | non-Latin letters | empty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (final (approx.)) | 0% | 0% | 0% | 0% | 0% | 0% |
| k-SLMmeans (Qwen3-0.6B) (final (approx.)) | 26% | 0% | 30% | 83% | 0% | 0% |
| k-SLMmeans (Qwen3-1.7B) (final (approx.)) | 6% | 1% | 3% | 2% | 0% | 0% |
| k-SLMmeans (Qwen3-4B) (final (approx.)) | 3% | 0% | 2% | 3% | 0% | 0% |

Files: `summary_quality_table.md` / `.csv` (this table), `summary_quality_per_step.csv` (method-level numbers per step).

### What the tables show

- On CLINC150 the Qwen3 summaries get worse on coverage, specificity and support with size, while fluency does not follow that order.
  Last step, mean nearest-summary share: LSA 0.80, 0.6B 0.61, 1.7B 0.61, 4B 0.49; share with a non-positive specificity margin:
  0%, 14%, 19%, 39%; share flagged unsupported: 0%, 12%, 25%, 47%. The 4B has the most exact duplicates (13% of its
  last-step summaries) and the most that mention the cluster (36%).
- Fluency proxies are not aligned with the other three families. The 1.7B fails the proxies least among the Qwen3 sizes (9%) but has
  more unsupported lemmas than the 0.6B (25% vs 12% flagged). The 0.6B and 4B fail the proxies in 28% and 34% of all summaries
  (for the 0.6B mostly copying a member verbatim and a missing "?"; for the 4B echoing the prompt, a missing "?" and quotes).
- Among CLINC150 summaries that pass every fluency proxy, a substantial share still fails another family
  (low coverage, generic or unsupported): 11% for LSA, 34% for the 0.6B, 42% for the 1.7B, 44% for the 4B.
- BANKING77 (approximate, last step): the same ordering by size appears for coverage and for the unsupported share but is weaker
  (nearest-summary share 0.80 / 0.74 / 0.68 / 0.68, in the order LSA, 0.6B, 1.7B, 4B), and the 4B has no duplicates and does not mention
  the cluster, consistent with `prelim_bank77.md`. The 0.6B's 89% "not fluent" is almost entirely the proxy for answers wrapped in double quotes (83%).
- Members at the step (CLINC150, "Members at the step" table): the share of members lost at a replacement step is larger for the
  Qwen3 summaries and largest for the 4B (last step: LSA 2%, 0.6B 24%, 1.7B 25%, 4B 41%). Among lost members, 72-83% belong to a gold intent the
  summary does not cover, against 28-45% of all members in such intents. This association is partly built in:
  a member whose intent the summary does not cover is a member that is nearer to another summary, so it is the kind of member that moves. It
  is not evidence that the summary caused the move (#47).

## 4. Worked examples: fluent summaries that do not represent their cluster

Selection rule (`scripts/summary_quality_examples.py`; applied once to the final scores. A first pass picked a 1.7B summary containing a Chinese word, which led to adding the non-Latin-letters fluency check; the rule itself did not change): per model, seed 0, last summary step (step 4), among summaries that pass every
fluency proxy, have a nearest-summary share below 0.3 and at least 50 members, the one that lost the most members. This picks the most
extreme case that fits the rule for each model. It is not a typical cluster: the tables above give the rates.

**Qwen3-0.6B, cluster 69.** Summary: "What are the key transfers and actions you need to perform to move money between accounts and handle
account-related tasks?" A well-formed single question. 206 members, whose most common gold intents are order_checks (50), transfer (39),
freeze_account (23) and balance (15). Nearest-summary share 0.00, intent coverage 0.00 (none of its intents), margin -0.059. Flagged generic
(margin) and unsupported. At the replacement all 206 members left (134 of them to cluster 110) and 118 others joined.

**Qwen3-1.7B, cluster 121.** Summary: "What is the impact of opening a new account on a credit score and how can it be managed?" A well-formed
question. 250 members: pin_change (99), account_blocked (51), freeze_account (33), report_fraud (16). Nearest-summary share 0.08, intent coverage 0.00,
margin -0.037. Flagged generic; **not** flagged unsupported, although none of its members' leading gold intents is about credit scores or opening accounts. Fewer than half of its content
lemmas are absent from the members, so the lexical unsupported rule does not fire; this shows the rule is loose. No member stayed; 109 went to cluster 103 and 65 to cluster 51.

**Qwen3-4B, cluster 81.** Summary: "What is the main theme of the cluster?" Well-formed, and about the task, not the content. 398 members:
restaurant_suggestion (113), meal_suggestion (95), restaurant_reviews (35), how_busy (31). Nearest-summary share 0.00, margin -0.309. Flagged generic (mentions
the cluster, negative margin) and unsupported. No member stayed; 313 went to cluster 123 and 37 to cluster 17, while 50 others joined.

In all three the cluster's members left at the replacement step and the cluster that carries the same id afterwards is made mostly of other queries (the "gained" counts). These are single seed-0 cases; they illustrate the kept/lost link and do not establish how often this happens (the tables do).

## 5. Hand check: pending

The October request was an audit of representative summaries against the automatic scores. The materials exist; **no ratings have been made**, so
there is no agreement number yet.

- Rubric, written before any summary was read for this audit: `docs/summary_audit_rubric.md`.
- Sample (`scripts/make_audit_sheet.py`, seed 0): 10 gold intents per dataset, 4 well-, 3 mixed-, 3 poorly-recovered, then for each intent the
  matched cluster of each of LSA / 0.6B / 1.7B / 4B at the last summary step, so 40 summaries per dataset, method names hidden and letters shuffled.
  Recovery uses the max-count rule of `prelim_bank77.md` section 6 to match an intent to a cluster, graded by the Jaccard overlap of the
  matched cluster with the intent (well at least 0.5, poor below 0.25, minimum over the four methods). Each row shows the cluster size and 10 random members.
- Sheets: `summary_audit_sheet_clinc150_intent.csv` and `summary_audit_sheet_bank77.csv` (plus `.md` copies for reading); the keys
  (`summary_audit_key_*.csv`) hold the method and the automatic flags and should stay closed while rating. The BANKING77 sheet uses the
  approximate final partition.
- After rating, `python scripts/audit_agreement.py SHEET KEY` prints, per property (low coverage, generic, unsupported, not fluent), the 2 x 2 table, raw agreement and
  Cohen's kappa. One rater; with 40 summaries per dataset the numbers will be rough.

## 6. Limitations

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
    $R/clinc150_qwen3-1.7b/20261027-202532 $R/clinc150_qwen3-4b/20261027-221042 $R/prelim_bank77_*/* \
    --cache-dir ../k-slimeans/cache --out-dir results/summary_quality
python scripts/make_audit_sheet.py <the same run dirs> --scores results/summary_quality/summary_quality_per_cluster.csv \
    --cache-dir ../k-slimeans/cache --out-dir docs/results
```

CPU only (the script hides the GPU); 16,540 summaries took about 2.5 minutes (12 cores, spaCy plus re-encoding).
The per-cluster CSV (`summary_quality_per_cluster.csv`, 6.5 MB, keyed by dataset, method, run, seed, step, cluster, with the kept/lost/gained counts
for #33 and #50) is not committed; `results/` is untracked, and it regenerates with the command above.
