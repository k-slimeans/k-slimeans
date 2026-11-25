# Dec 2 talk: deck skeleton and Cooper's slides (#83)

30-minute talk plus about 10 minutes of questions, all three of us presenting (#83, #84). Slides are submitted with the
revised manuscript on Nov 30. Draft slides from each speaker are due Nov 28; the rehearsal on Dec 1 aims for 28 minutes.

Contents:

1. **Deck skeleton**: every part of the talk, its speaker, its time, its slides and where its numbers come from. Samiksha and Nabin
   fill in their own parts; the skeleton only fixes titles, order and sources.
2. **Cooper's slides** (parts 1, 3 and 8): on-slide text, figure, speaker notes and a source footer for every slide.
3. Backup slides, 4. likely questions for Cooper's parts, 5. open items.

**Rules for every slide** (from #83 and `synthesis.md` §10):

- Every number comes from `docs/results/`, cited by file and section in the slide footer.
- Every claim carries its scope: step 1 or final; which model; 5 seeds; CLINC150 unless the slide says otherwise.
- Every piece of evidence is labelled **interventional** (we changed the input and re-ran or replayed), **association** (logged
  properties go together), **measurement** (a logged run, nothing compared causally) or **identity** (true by algebra).
- No step-1 result is used to explain a final score.
- The human evaluation (#34) has no numbers yet: its slots are `TODO(#34)` placeholders. The AI-rater stand-in, if it is shown at all, is labelled
  "not a human rating" on the slide itself.
- No BBC article text on any slide (the repo is public); ids only.

**Footer convention**, so the three of us look the same: `dataset · step 1 / final · models · 5 seeds · evidence kind — file §section`.
Example: `CLINC150 · step 1 · old 4B, 1.7B · 5 seeds · interventional — counterfactual_swaps.md §2, §4`.

"Old" means the October prompt (#31); "v2" means the clearer prompt (#46). "Step 1" is the first summary step and the Lloyd block after it;
"final" is the score after all five summary steps.

---

## Part 1. Deck skeleton

| # | Part | Speaker | Min | Slides | Sources | Open inputs |
| --- | --- | --- | ---: | --- | --- | --- |
| 1 | The question; k-means with summaries as centroids; the setup | Cooper | 4 | 1.1 to 1.4 (below) | `clinc150_summarizers.md` §1, §2, §2.4; README "How the algorithm maps to the code" | none |
| 2 | Closest work and what we add | Samiksha, Nabin | 3 | 2.1 closest work; 2.2 what we add | #80 | `TODO(#80, Samiksha and Nabin)`: related work |
| 3 | What a replacement does; how far the causal evidence goes (step 1 vs final) | Cooper | 8 | 3.1 to 3.9 (below) | `replacement_diagnostic.md` §3 to §8; `counterfactual_swaps.md`; `summary_guard.md`; `prompt_test.md` §8; `synthesis.md` §5, §10, §11 | none |
| 4 | Is the summary a good representative? Automatic scores and the human evaluation | Samiksha | 5 | 4.1 the four score families; 4.2 readable but a poor representative; 4.3 to 4.5 human evaluation | `summary_quality.md` §2 (definitions) and its results sections; `synthesis.md` §12 | `TODO(#34)`: inter-rater agreement, human against automatic, the paraphrase check |
| 5 | Clusters a summary helped or hurt | Samiksha | 2 | 5.1, 5.2 worked clusters | `synthesis.md` §12 (worked examples); `synthesis_case_studies.md` | `TODO(#50 case studies, Samiksha)` |
| 6 | Why the larger model isn't better here | Nabin | 4 | 6.1 the size result under both prompts; 6.2 what explains it, by kind of evidence; 6.3 what 5 seeds can detect; 6.4 scope | `size_result.md` §1 to §4; `prompt_test.md` §8.7 | `TODO(#36, Nabin)`: failure-mode tags (`size_failure_modes.md`), Qwen3 training notes (`docs/qwen3_sizes_notes.md`) |
| 7 | What it costs, and when it's worth it: an SLM or an extractive summarizer | Nabin | 3 | 7.1 cost table; 7.2 cost per ACC point; 7.3 SLM or LSA | `cost_justification.md` §2, §3; `cost_justification_pareto.png` | `CONFIRM(#10, Nabin)`: dollar prices (`docs/cost_assumptions.*`) |
| 8 | The answer to the main question; limits | Cooper | 1 | 8.1 (below) | `synthesis.md` §13; `cost_justification.md` §3 | `TODO(#34)`, `CONFIRM(#10)` |
| | **Total** | | **30** | | | |

Coordination notes:

- **Parts 3, 4 and 5 share material.** Slide 3.4 (the sign of the purity change goes with purity before) is the step from "assignments change" to "cluster
  quality". Part 4 should not repeat it; part 5's worked clusters can point back to it ("this is the mixed-cluster case of slide 3.4").
- **Parts 3 and 6 share the prompt result.** Slide 3.8 gives the clearer prompt as an intervention on the 4B's final score. The size order
  under both prompts is Nabin's (part 6), so slide 3.8 does not show it.
- **Parts 7 and 8 share the cost result.** Slide 8.1 states it in one line and points to part 7.
- **Timing.** The parts add up to 30 minutes; the rehearsal target is 28 (#83). Cooper's three parts are written for 3.5, 7.5 and 1 minutes.

---

## Part 2. Cooper's slides

### Part 1: The question and the setup (4 min in the plan, written for 3.5; 4 slides)

#### Slide 1.1 Title and the question (0.5 min)

On the slide:

- **k-SLiMeans: small-language-model summaries as k-means centroids**
- `TODO(names)`: full names of the three presenters, course, date
- The question (Nov 18 feedback, #84): *"When does a generated summary provide a useful centroid, and when is its additional computational
  cost justified?"*

Speaker notes: This talk answers one question, asked of us by the course after our November report. Everything we show comes
from runs we already had; following the Nov 18 feedback we added no new models, datasets or GPU runs (#84).

Footer: `#84`

#### Slide 1.2 The method: a summary replaces each centroid (1 min)

On the slide (diagram, left to right):

1. Embed the texts; run k-means (k-means++ start) for one block: **step 0**.
2. Repeat 5 times (**summary steps 1 to 5**): for each cluster, write a summary of its members → embed the summary → the summary
   embedding becomes the cluster's start center → run up to 20 Lloyd iterations.
3. The summarizer is the only thing that changes:
   - **k-NLPmeans**: an extractive LSA summary (top 5 sentences), the upstream method (arXiv:2502.09667);
   - **k-SLMmeans**: a summary written by Qwen3-0.6B, 1.7B or 4B.

Speaker notes: This is the upstream loop, re-implemented; our version gives the same assignments as the upstream code in every check
we ran (BANKING77 with the extractive summarizers, synthetic fixtures, mini-batch; README, "Reproduction status"). The schedule is fixed: 5 summary steps, 120 Lloyd iterations in 6 blocks of 20.
One design point matters for everything later: for a given seed, every method starts from the same step-0 partition, so at summary step 1
cluster *c* has the same members whichever summarizer wrote its summary. Differences at step 1 therefore come from the summarizers alone, and
all comparisons are paired by seed.

Figure: redraw the loop from the README diagram ("How the algorithm maps to the code") as a simple flow.

Footer: `README "How the algorithm maps to the code", "Reproduction status" — clinc150_summarizers.md §1 — replacement_diagnostic.md §1`

#### Slide 1.3 Setup (1 min)

On the slide:

| | |
| --- | --- |
| Main dataset | CLINC150: 22,500 queries, 150 intents, k = 150 |
| Other datasets | BANKING77 (3,080 queries, 77 intents); BBC News (500 articles, 5 topics; TF-IDF and a dense encoder); arXiv (200k-abstract pilot, 1 seed) |
| Encoder (CLINC150) | `distilbert-base-nli-stsb-mean-tokens`, shared by every method |
| Summarizers | LSA (extractive); Qwen3-0.6B, 1.7B, 4B: greedy decoding, thinking off, 64 new tokens, 4,096 input tokens; batch 8 / 4 / 1 on CLINC150 and BBC (BANKING77 8 / 4 / 2) |
| Prompts on CLINC150 | October: *"The following is cluster of questions. Write a question that summarizes the following cluster:"*; clearer prompt (#46), fixed before any run |
| Seeds | 0 to 4; all methods share the step-0 partition per seed |
| Scores | ACC (Hungarian-matched), NMI, ARI; paired difference per seed, 95% Student-t interval over 5 seeds (4 df), not corrected for multiple comparisons |
| Hardware | one RTX 3060, 12 GB |

Speaker notes: CLINC150 is where we have the per-cluster replay, so it carries the mechanism results; the other datasets are
method-level only. Five seeds give wide intervals: an interval that only just excludes zero is weak evidence, and "not detectable" means
not detectable with five seeds, not "no effect". BANKING77 and BBC used their own dataset prompts; only CLINC150 was run under two prompts.

Footer: `clinc150_summarizers.md §1, §2.2 — summary_quality.md §1 — bbc.md §1 — arxiv_status.md §5.1 — prompt_test.md §1 — cost_justification.md §1`

#### Slide 1.4 Where we started: the largest model was the worst (1 min)

On the slide:

| CLINC150, final, October prompt | ACC | vs k-means [95%], seeds above |
| --- | ---: | --- |
| k-means | 0.551 | |
| LSA (k-NLPmeans) | 0.560 | +0.009 [+0.001, +0.017], 4/5 |
| Qwen3-0.6B | 0.541 | −0.010 [−0.018, −0.002], 0/5 |
| Qwen3-1.7B | 0.570 | +0.018 [+0.004, +0.033], 4/5 |
| Qwen3-4B | 0.511 | −0.040 [−0.058, −0.022], 0/5 |

- The 4B is below k-means and LSA in 5 of 5 seeds.
- Over all five summary steps, 34% of its summaries mention "cluster" or "summar" (0.6B and 1.7B: 8%). The most common one: *"What is the main topic of the following cluster of
  questions?"*
- That is readable text, and a poor stand-in for the cluster.

Speaker notes: This was our October result. A larger model doing worse is the puzzle the rest of the talk takes apart. The summaries
show one obvious failure: the 4B often writes a question *about* the cluster instead of a question *from* it, and at the final step 16 to 21
of 150 clusters per seed share an identical summary. At this point that is an association: the failure and the low score go together.
Part 3 tests it. Part 4 asks the general version: a summary can read well and still be a poor representative.

Footer: `CLINC150 · final (summary counts: all steps) · k-means, LSA, Qwen3 0.6B / 1.7B / 4B, October prompt · 5 seeds · measurement; the link to the summaries is an association — clinc150_summarizers.md §2.1, §2.2, §2.4`

Transition to part 2: "Before we open up the replacement step, Samiksha and Nabin place this in the closest work."

---

### Part 3: What a replacement does, and how far the causal evidence goes (8 min in the plan, written for 7.5; 9 slides)

#### Slide 3.1 Replacing the centroids raises the objective, by exactly the movement (0.75 min)

On the slide:

- Objective with the assignment C held fixed: J(C, Z) = Σᵢ ‖xᵢ − z_{cᵢ}‖²
- **Identity:** J(C, Z) = J(C, X̄) + Σₖ nₖ ‖zₖ − x̄ₖ‖², so for fixed C the cluster means minimise J.
- Replacing the centers M by the summary embeddings S, assignments kept, changes J by exactly
  Σₖ nₖ (‖sₖ − x̄ₖ‖² − ‖mₖ − x̄ₖ‖²) ≈ Σₖ nₖ ‖sₖ − x̄ₖ‖²: the size-weighted squared movement of the centers.
- Checked on all 25 logged step-1 replacements (LSA, old 0.6B / 1.7B / 4B, v2 4B; 5 seeds): largest relative residual 2.9e-16.
- Step 1: J rises by 8,675 (LSA) to 14,939 (old 4B), from 10,721.

Speaker notes: This is the corrected version of the objective argument in our November report (Section 4.1), which the
feedback asked us to fix. The statement that holds is about **fixed assignments**: the mean is the best center for a fixed set of members, so
putting anything else there raises the objective, and by exactly how far the new center is from the mean, weighted by cluster size. We checked the
identity on every logged step-1 replacement; the two sides agree to floating-point precision.

Footer: `CLINC150 · step 1 · 5 methods · 5 seeds · identity, checked numerically — replacement_diagnostic.md §3; replacement_objective_identity.csv; synthesis.md §10; #84 (the Section 4.1 request)`

#### Slide 3.2 After reassignment: what follows, and what does not (1 min)

On the slide:

- **Follows:** moving each point to its nearest summary center cannot raise J, and no later Lloyd step can. From the raised value J can only
  stay level or fall.
- **Measured:** at step 1, after reassignment J is still 70% (v2 4B) to 98% (old 4B) above where it started. Averaged over all steps, Lloyd
  removes 88% to 97% of the rise by iteration 1 (October runs).
- **Does not follow:** where the block ends. The run from the summary centers and the run without replacement are two Lloyd runs from different
  starts; either can end lower, and a lower J does not mean a higher ACC.
- Example: the 1.7B and the old 4B raise J about equally (+95%, +98%) and both end above the no-replacement run in J (+0.18%, +1.4%), yet
  in ACC they end on opposite sides of it (+0.0106, −0.0159).

Figure: `replacement_diagnostic_objective.png`, **step-1 column only** (seed 0; top: objective over the objective before, log scale; bottom: percent above
it from iteration 2). The full figure has five step columns and is too dense for a slide; crop it.

Speaker notes: The November report drew a conclusion about the objective after reassignment that the identity does not support.
Once points move, the identity no longer applies: the replacement run and the plain-Lloyd run are different trajectories, and nothing about
the final objective, or about a score, follows from the identity. The 1.7B and the old 4B are the clean example: the same jump, the same direction in
J, opposite outcomes in ACC. The size of the jump does not rank the outcomes.

Footer: `CLINC150 · step 1 · 5 methods · 5 seeds · identity (monotone part); measurement of logged runs; the J-versus-ACC contrast is descriptive — replacement_diagnostic.md §3, §6; synthesis.md §10`

#### Slide 3.3 Summaries move the centers about one radius, and about half the points change cluster (0.75 min)

On the slide:

| step 1, mean over seeds | LSA | 0.6B | 1.7B | old 4B | v2 4B |
| --- | ---: | ---: | ---: | ---: | ---: |
| distance, summary to replaced center (median over clusters, RMS radii) | 0.89 | 0.97 | 1.05 | 1.08 | 0.88 |
| share of all points that change cluster at the replacement | 25% | 47% | 47% | 56% | 40% |
| share of all points still in a different cluster after the block | 9% | 33% | 29% | 45% | 26.5% |

- Lloyd does not undo the replacement: the block-end centers return close to the old ones (0.04 to 0.33 radii), but the partition does not.
- The moves are not random: moves toward a point's gold intent are 24 to 31 times what shuffled destinations give, but few (723 of the old
  4B's 12,620 movers).

Figure: `replacement_diagnostic_movement.png` (caption: **pooled over all clusters, steps and seeds**; the table is step 1). The x-axis labels overlap
in the committed file; relabel on the slide.

Speaker notes: This is what the replacement does geometrically. A summary embedding sits about one cluster radius from the mean it replaces,
for the extractive LSA summary as for the model summaries. The model summaries move about half of all points at once, LSA a quarter. After 20 Lloyd
iterations the centers come back near where they were, but 29% to 45% of the points stay in a different cluster (LSA 9%). For scale,
re-encoding noise of 1e-6 changes about 4.5% of the old 4B's labels after a block. These are measurements from an exact replay of the logged
runs: 120 of 120 fits of the October runs and 30 of 30 of the v2 4B reproduce the logged partitions with no label mismatches.

Footer: `CLINC150 · step 1 · LSA, old 0.6B / 1.7B / 4B, v2 4B · 5 seeds · measurement (exact replay) — replacement_diagnostic.md §2, §4, §8`

#### Slide 3.4 Whether a move helps depends on the cluster's starting purity, not on the summary's quality score (1 min)

On the slide:

| step 1: change in members' purity minus plain Lloyd [95%] | 0.6B | 1.7B | old 4B | v2 4B |
| --- | ---: | ---: | ---: | ---: |
| mixed clusters, purity ≤ 0.4 (187 of 750) | +0.050 [+0.024, +0.076] | +0.057 [+0.039, +0.074] | +0.042 [+0.018, +0.067] | +0.034 [+0.006, +0.062] |
| pure clusters, purity > 0.9 (169 of 750) | −0.060 [−0.103, −0.018] | −0.036 [−0.056, −0.016] | −0.090 [−0.137, −0.042] | −0.026 [−0.071, +0.019] |

- LSA shows the same signs, smaller (+0.012, −0.017): it moves fewer members.
- Against random moves of the same size, the summaries' moves do better on average in every purity stratum (+0.03 to +0.26, old Qwen3 models).
- The quality scores do not tell which case a cluster is in: partial rank correlation with the purity change −0.06 to +0.14 over five
  methods including LSA (4 of 23 intervals exclude 0): weak and inconsistent.

Figure: `synthesis_clinc150_intent_help_reference.png`, **top-left panel only** (purity change minus the counterfactual, by purity before). Note: this figure
colours the old 4B pink and the v2 4B yellow; the objective figure (slide 3.2) colours the old 4B dark orange and the v2 4B pink. Label the series on each slide.

Speaker notes: Moving members is not the same as harm. The same movement raises the purity of a mixed cluster and lowers it for a pure one.
Part of that is mechanical: a pure cluster can only lose purity, a mixed one has room to gain. That is why we compare with two references,
plain Lloyd from the same start, and random moves of the same size. Against both, the pattern holds. What does *not* separate the two cases is any
summary-quality score: once purity and cluster size are controlled, the scores are weak and inconsistent. The members a cluster loses do track
coverage (rank correlation about −0.7 to −0.8), but that is largely built into how coverage is defined. All of this is association; the
replacement was not randomised per cluster.

Footer: `CLINC150 · step 1 · old 0.6B / 1.7B / 4B, v2 4B (LSA as reference; the partial correlations include LSA) · 5 seeds · association against two references — synthesis.md §3, §5, §10, §11 links 3 and 3b`

Transition: "So the quality scores don't predict harm cluster by cluster. But the old 4B is clearly worse as a whole. Is it its odd summaries?"

#### Slide 3.5 The old 4B at step 1 (0.5 min)

On the slide:

| step 1, per seed (mean) | 1.7B | old 4B | v2 4B |
| --- | ---: | ---: | ---: |
| summaries that mention the cluster | 13.6 | 50.2 | 0 |
| clusters sharing an identical summary | 0.8 | 18.8 | 1.2 |
| clusters left empty by the replacement | 0.6 | 10.8 | 0.4 |
| step-1 ACC minus plain Lloyd from the same start | +0.0106 (0/5 seeds below) | −0.0159 (5/5 below) | +0.0024 (2/5 below) |

Speaker notes: Clusters with an identical summary get an identical center, so all their points go to the first copy and the rest are
emptied. The old 4B's step-1 partition is below plain Lloyd in ACC in all five seeds, and the deficit survives running both sides to convergence
(−0.0138, 5 of 5). Not in NMI, though: there the difference is about zero. These are measurements, and the link between the odd summaries and
the deficit is still an association. The next slide tests it.

Footer: `CLINC150 · step 1 · 1.7B, old 4B, v2 4B · 5 seeds · measurement — replacement_diagnostic.md §5, §6, §8`

#### Slide 3.6 Swap test at step 1: the flagged summaries carry the gap (1.25 min)

On the slide:

- **Flag** (fixed before any outcome): the 4B's summary duplicates another cluster's or mentions "cluster" / "summar". 43 to 56 of 150 clusters
  per seed.
- Replay the step-1 block with edited start centers (ACC change, 95% interval, 5 seeds):

| variant | ΔACC |
| --- | --- |
| the 1.7B's summaries in the 4B's flagged clusters | +0.026 [+0.019, +0.034], 5/5 seeds: 99% of the 4B-to-1.7B gap |
| control: the 1.7B's summaries in as many random unflagged clusters | −0.002 [−0.012, +0.007] |
| **difference (swap minus control)** | **+0.0285 [+0.0149, +0.0420]** |
| reverse: the 4B's flagged summaries put into the 1.7B, minus its control | −0.0248 [−0.0410, −0.0086] |
| flagged clusters keep their k-means center instead | +0.0064 [−0.0058, +0.0185]: not shown |

- At step 1, the 4B-to-1.7B difference travels with the text of the flagged summaries.

Speaker notes: This is the one experiment where we changed the summaries themselves. Because every method starts step 1 from the same
partition, we can take the 4B's start centers and swap individual rows. Putting the 1.7B's summaries into the flagged clusters closes all of
the step-1 gap between the two models; putting them into the same number of random unflagged clusters does nothing; and the reverse swap opens
a gap of about the same size in the 1.7B. Both directions agree. The effect is about nine times the mean re-encoding noise floor, and the converged
blocks give the same answer. What it does not show: falling back to the k-means center in those clusters does not recover a measurable amount,
so the flagged summaries are worse than the 1.7B's, not clearly worse than no summary. And two limits: flagged clusters may also be harder
clusters, and this is step 1 only, one model pair.

Footer: `CLINC150 · step 1 only · old 4B and 1.7B · 5 seeds · interventional — counterfactual_swaps.md §1, §2, §3, §4, §5; synthesis.md §10`

#### Slide 3.7 Over all five steps, removing the flagged summaries does not recover the final score (1 min)

On the slide:

- **Guarded run** (a new run, #47 stage 2): at every summary step, the flagged 4B clusters start from their previous k-means center instead of their summary. About a third of the clusters (32% to 35%) are guarded at every step.
- Final, guarded minus unguarded 4B: ACC +0.0093 [−0.0059, +0.0244] (4/5 seeds); NMI +0.0010 [−0.0069, +0.0089]; ARI −0.0069 [−0.0193, +0.0055].
- It stays 0.031 ACC below k-means (−0.0310 [−0.0429, −0.0190], 0/5 seeds).
- **Not shown** that the flagged summaries explain the 4B's final deficit. **Not shown** that they don't: a gain up to +0.024 is not excluded.

Speaker notes: This is where the step-1 result stops. The guard removes the flagged summaries' influence at every step, and the final score
does not come back by an amount five seeds can detect. This does not contradict the swap test: the guard replaces a bad summary with *no*
summary, not with a better one, and at step 1 that variant also showed nothing. The swap itself was never run over five steps, because later steps need new
summaries of new partitions, which means new GPU runs, and the Nov 18 feedback ruled out new GPU runs (#84). Across methods, step-1 and final
scores do go together (rank correlation +0.60 over 25 method-seed pairs), but that rests on the old 4B (+0.24 without it), and within a method
nothing is detectable (−0.11). So we do not use the step-1 result to explain any final score.

Footer: `CLINC150 · final · old 4B only · 5 seeds · interventional (the guard); association (step 1 against final, across 5 methods) — summary_guard.md §1, §2, §5; counterfactual_swaps.md §5; synthesis.md §10, §11 link 4; #84`

#### Slide 3.8 What did change the old 4B's final score: a clearer prompt (0.5 min)

On the slide:

- Clearer prompt (#46), written and committed before any run: *"…Write one new question that a user could have asked, in the same style, that covers
  what these questions have in common. Write only the question. Do not mention the questions, a group, a cluster or a topic."*
- Final ACC, clearer minus October, same seeds and step-0 partitions:
  - 4B **+0.056 [+0.039, +0.074]**, 5/5 seeds (0.511 → 0.568)
  - 0.6B +0.022 [+0.010, +0.033], 5/5
  - 1.7B −0.007 [−0.023, +0.010], 1/5 (not separated from 0)
- The v2 4B is above k-means (+0.016 [+0.009, +0.023]) and not separated from LSA (+0.007 [−0.001, +0.016]).
- At step 1 its surface failures are gone (0 cluster mentions, 1.2 duplicate clusters per seed, 0.4 empty clusters), but its step-1 ACC against
  plain Lloyd, +0.0024 (2/5 seeds below), is not distinguishable from noise.

Speaker notes: The prompt is the other intervention: we changed one input and re-ran all five steps on the same seeds and start partitions.
It removes the 4B's final deficit and its cluster-talk at once. But it changed several things in the wording together, and we chose it by the 4B's
surface counts, so it does not tell us which change mattered, and it does not isolate the cluster-mentioning failure. Nabin will show what it
does to the comparison between model sizes.

Footer: `CLINC150 · final (and step 1 for the last line) · Qwen3 0.6B / 1.7B / 4B, both prompts · 5 seeds · interventional, two runs, several changes at once — prompt_test.md §1, §3, §8.2, §8.6; replacement_diagnostic.md §8`

#### Slide 3.9 How far the evidence goes (0.75 min)

On the slide:

| Claim | Kind | Scope | Status |
| --- | --- | --- | --- |
| Replacing the centers raises J at fixed assignments, by the size-weighted squared movement | identity | step 1, 25 replays | shown |
| The objective after the block does not order the outcomes (1.7B and old 4B: same rise, opposite ACC) | association | step 1, two models | shown (descriptive, not a test) |
| Summaries move centers about one radius; 47% to 56% of points change cluster (LSA 25%, v2 4B 40%), and Lloyd does not undo it | measurement | step 1; LSA, old 0.6B / 1.7B / 4B, v2 4B | shown |
| Quality scores predict whether a cluster's purity rises or falls | association | step 1 | not shown |
| Mixed clusters gain purity, pure ones lose | association, two references | step 1 | shown as a pattern; no mechanism |
| The 1.7B's summaries in the 4B's flagged clusters close its gap | interventional | step 1, 4B and 1.7B | shown |
| The flagged summaries explain the 4B's final gap | interventional | final, 4B | not shown (up to +0.024 not excluded) |
| The step-1 effect carries to the final score | not tested | step 1 → final | not shown |
| The clearer prompt removes the 4B's final drop | interventional, several changes at once | final | shown for the final score; which step carries it and which change in the wording: not shown |

Bottom line on the slide: **No step-1 result is used to explain a final score.**

Speaker notes: This is the summary of part 3; we propose the same table for the manuscript. Read it by the last column. The step-1 swap is the
strongest causal result we have, and it is about step 1. The final-score interventions are the guard, which is inconclusive, and the prompt,
which works but changed several things at once.

Footer: `CLINC150 unless noted · 5 seeds — synthesis.md §10`

Transition to part 4: "Samiksha now asks the direct question: are these summaries good representatives, by our scores and by human raters?"

---

### Part 8: The answer (1 min, 1 slide)

#### Slide 8.1 When is a summary a useful centroid, and when is it worth the cost?

On the slide:

**Useful centroid** (CLINC150, Qwen3, 5 seeds)

- A summary keeps its cluster when most members are nearest to it. That protects; it does not improve (association, largely by construction).
- At step 1 a summary raised members' purity only in mixed clusters (purity ≤ 0.4: +0.04 to +0.06 against plain Lloyd, old Qwen3; v2 4B +0.034).
  The quality scores do not tell which clusters those are: weak and inconsistent once purity and size are controlled (association).
- Readable is not representative: 36% to 44% of the fluent Qwen3 summaries fail coverage or specificity (automatic proxies, step 1).
  Human check: `TODO(#34)`.
- The old 4B's failed stand-ins (duplicates, questions about "the cluster") are worse than the 1.7B's summaries of the same clusters: swapping
  those in closes the old 4B's step-1 gap to the 1.7B (interventional, step 1). That they explain its final scores is not shown.
- A clearer prompt removed the old 4B's final deficit (+0.056 ACC; interventional, two runs, several wording changes at once, so it does not
  isolate the cluster-mentioning failure).

**Worth the cost?**

- CLINC150: LSA gains +0.009 ACC over k-means for 14.8 s of summarizer time per seed. No SLM is shown above LSA on ACC; the SLM runs that
  beat k-means (October 1.7B, the three v2 sizes) took 25 to 84 times LSA's summarizer seconds (as logged).
- BANKING77, BBC TF-IDF: no measured SLM benefit over LSA on ACC. BBC dense: 1.7B and 4B +0.005 to +0.006 ACC over LSA, not read as a method difference.
- Dollars: `CONFIRM(#10)`.
- **Wherever no SLM benefit over LSA was measured (everything above except BBC dense), the extractive summarizer is the better-value choice;
  whether an SLM adds a small gain cannot be told from five seeds.**

Limits (small type): one model family; per-cluster analysis on one dataset; 5 seeds, uncorrected intervals; step-1 interventions on one model pair;
arXiv is a one-seed pilot with no comparison.

Speaker notes (about 150 words): Our answer has two halves, both mostly from CLINC150 and Qwen3. A generated summary keeps its cluster when it
actually stands in for its members: then it moves little. When it does move members, at step 1 that helped mixed clusters and hurt pure ones, and
our quality scores did not reliably tell the two apart in advance. The old 4B's failed stand-ins were worse than the 1.7B's summaries of the same
clusters at step 1, which we showed by swapping them; that they explain its final scores, we could not show. On cost: on ACC, no small model was
shown to beat the extractive summarizer, apart from a half-point gain on BBC with a dense encoder that we do not read as a method difference, and
the extractive summarizer takes a small fraction of the time. So the extra cost of an SLM is not shown to be justified here.

Footer: `synthesis.md §10, §12, §13; cost_justification.md §2, §3; bbc.md §2.2; prompt_test.md §8.2`

**One-line answer** (for the abstract and the closing sentence of the manuscript):

> On CLINC150 with Qwen3 (5 seeds), a generated summary kept its cluster when the cluster's members were nearest to it, which protects the
> cluster rather than improving it; at the first summary step it raised members' purity only in mixed clusters, and the quality scores did not
> reliably identify those clusters. On ACC, no small language model was shown to beat the extractive summarizer, apart from a gain of half an
> ACC point on BBC with a dense encoder that we do not read as a method difference. On CLINC150 the extractive summarizer took 1/25 to 1/84 of
> the summarizer time of the SLM runs that beat k-means, so the added cost is not shown to be justified.

---

## Part 3. Backup slides (for questions)

| # | Title | Content | Source |
| --- | --- | --- | --- |
| B1 | The replay is exact; the noise floor | 120 of 120 fits reproduce the logged partitions (0 label mismatches) with GPU-encoded summaries; a CPU re-encode (about 1e-6 per row) changed up to 9% of the old 4B's labels. Tied-noise floor for the old 4B: 0.0030 ACC mean, 0.0084 at most. | `replacement_diagnostic.md` §2 |
| B2 | The old 4B's step-1 deficit against plain Lloyd is not an iteration-budget artifact | Both sides run to convergence: −0.0138 ACC, 5/5 seeds below (20-iteration block: −0.0159). | `replacement_diagnostic.md` §6 |
| B3 | Why control (d) is not used | Putting random unflagged clusters back on their k-means centers lowers ACC by 0.019 to 0.032 and leaves 28 to 39 empty clusters, so (b) minus (d) mostly measures (d)'s damage; (b) against (a) is the comparison used. | `counterfactual_swaps.md` §2 |
| B4 | The chain from summary to final score | Links 1 to 5 with strength and kind: quality → movement (built in), movement → members lost (partly built in), members lost → purity (sign set by purity before), step-1 partition → final (across methods only, rests on the old 4B), cost. | `synthesis.md` §11 |
| B5 | The objective at the end of the step-1 block | End vs no-replacement run: LSA −0.17%, 0.6B +0.35%, 1.7B +0.18%, old 4B +1.4%, v2 4B +0.09%; only LSA ends below the objective before the replacement. | `replacement_diagnostic.md` §3 |

## Part 4. Likely questions for Cooper's parts

1. **"If replacing the centroid raises the objective, how can it ever help?"** The identity is about fixed assignments. After points move,
   the replacement run and plain Lloyd are different trajectories: the 1.7B ends 0.18% above plain Lloyd in J and +0.0106 above it in ACC at step 1.
   A lower objective does not mean a better partition against the gold labels. (`replacement_diagnostic.md` §3)
2. **"Does the swap experiment explain the 4B's final scores?"** No. It is step 1 only. Over five steps the guard gives +0.009 [−0.006, +0.024]
   and the 4B stays 0.031 below k-means; the swap itself was never run over five steps. (`synthesis.md` §10)
3. **"Why not run the swap over all five steps?"** Later steps need new summaries of the new partitions, which cannot be produced offline;
   that is a new GPU run, which the Nov 18 feedback asked us not to add. (`counterfactual_swaps.md` §5; #84)
4. **"Aren't the flagged clusters just harder clusters?"** They may be; that is a stated limit. The swap changes the text in the same clusters,
   and the matched control (the same number of random unflagged clusters) does nothing, so "1.7B summaries are better everywhere" is ruled out; "harder clusters" is not.
   (`counterfactual_swaps.md` §4, §5)
5. **"Is the prompt's gain from copying members verbatim?"** Verbatim copies rose (v2 4B 12% of summaries, old 1.4%), but copying accounts for
   about 0.02 of the +0.16 coverage gain, and its link to ACC was not tested. (`synthesis.md` §2, §9)
6. **"Five seeds and no multiplicity correction: how much should we trust this?"** Each interval is per comparison. The results we lean on are
   far from the noise: the step-1 swap is about 9 times the mean re-encoding noise floor (3.1 times its maximum) and holds in 5 of 5 seeds. "Not detectable" means not detectable
   with five seeds. Seeds 5 to 9 were dropped after the Nov 18 feedback. (`counterfactual_swaps.md` §3; `size_result.md` §3)

## Part 5. Open items

- `TODO(names)`: slide 1.1.
- `TODO(#80)`, `TODO(#34)`, `TODO(#50 case studies)`, `TODO(#36)`, `CONFIRM(#10)`: teammates' inputs, as in the skeleton.
- Slide tool and shared deck (#83): not chosen yet. This outline does not depend on the tool.
- Figures: crop `replacement_diagnostic_objective.png` to the step-1 column and `synthesis_clinc150_intent_help_reference.png` to the top-left
  panel; relabel the x axis of `replacement_diagnostic_movement.png`. The colours differ between figures (old 4B dark orange in one, pink in the other):
  label each series on the slide, or redraw with one colour map (a plotting run from the committed CSVs, not done here).
