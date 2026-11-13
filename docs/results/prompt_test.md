# CLINC150 prompt test (#46): does a clearer prompt change the Qwen3 results?

Issue #46 asks whether part of the Qwen3-4B's lower CLINC150 scores comes from how it reads the prompt (background in
[`clinc150_summarizers.md`](clinc150_summarizers.md) §2.4 and §4). This file holds the pilot, which needs no clustering
scores. Section 8 holds the main runs: the 4B under the clearer prompt is in; the 0.6B and 1.7B are still to run.

Everything below comes from outputs that already exist. The pilot's run directories are under `results/` and not
committed; [`prompt_test_pilot.csv`](prompt_test_pilot.csv) and [`prompt_test_pilot_examples.md`](prompt_test_pilot_examples.md)
are written from them by `scripts/prompt_test_pilot.py` (see "Reproduce").

## 1. Setup

- **Fixed workload.** The inputs of seed 0's first summary step in the logged #31 run of each size: 150 clusters, from the
  step-0 partition that every method shares for a seed (`step0_sha1` 2b3858a965ae721d5e2bb27201fbc3728c31e716).
  The workload is rebuilt from the logged partition and the cached embeddings, so the members and their order are the
  ones the #31 run sent. The input tokens of the old-prompt rows are identical across sizes (234,011), as expected for one workload.
- **Batch sizes.** The logged ones: 4B 1, 1.7B 4, 0.6B 8. A larger batch for the 4B overflows the 12 GB card.
- **What changes.** Only the instruction. The system prompt, the `Question:` member lines, greedy decoding, thinking off,
  `max_new_tokens` 64, `max_input_tokens` 4096 and the #28 member selection are the #31 ones. The old prompt is the CLINC150
  dataset prompt, "The following is cluster of questions. Write a question that summarizes the following cluster:".
  The clearer prompt is `summarizer.prompt` in [`configs/clinc150_prompt_v2.yaml`](../../configs/clinc150_prompt_v2.yaml):
  "The following questions were asked by users. Write one new question that a user could have asked, in the same style, that
  covers what these questions have in common. Write only the question. Do not mention the questions, a group, a cluster or a topic."
- **One prompt, fixed in advance.** It was written and committed before any run (commit fe257e2, 2026-11-11; the pilot ran on 2026-11-12). There was no prompt search.
- **Counts.** The surface rules are those of §2.4 (`kslimeans.surface`): a summary "mentions the cluster" if it contains
  "cluster" or "summar" in any case; a duplicate group is a set of clusters with exactly the same summary.

## 2. Pilot table

Seed 0, first summary step, 150 summaries per row. The same numbers are in `prompt_test_pilot.csv`.

| Size | Prompt | Batch | Mention the cluster | Duplicate groups (clusters in them) | Mean words | Ends with "?" | Input tokens | Output tokens | Equal to logged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | old | 1 | 49 (33%) | 4 (18) | 17.2 | 130 (87%) | 234,011 | 3,160 | 100% |
| 4B | clearer | 1 | 0 (0%) | 1 (2) | 11.1 | 98 (65%) | 239,561 | 1,975 | 2% |
| 1.7B | old | 4 | 14 (9%) | 1 (2) | 20.4 | 146 (97%) | 234,011 | 3,786 | 100% |
| 1.7B | clearer | 4 | 0 (0%) | 0 (0) | 13.4 | 148 (99%) | 239,561 | 2,410 | 0% |
| 0.6B | old | 8 | 13 (9%) | 2 (4) | 11.3 | 133 (89%) | 234,011 | 2,157 | 100% |
| 0.6B | clearer | 8 | 0 (0%) | 0 (0) | 8.8 | 135 (90%) | 239,561 | 1,725 | 5% |

"Equal to logged" is the share of the 150 summaries identical to the summary the #31 run wrote for the same cluster.
For the clearer prompt it is not expected to be high (3, 0 and 7 summaries). The 4B's largest old-prompt group has 7 clusters
("What is the main topic of the following cluster of questions?"). Its one remaining clearer-prompt group has 2 clusters
("How can I transfer money between different bank accounts?"). Input tokens are 2.4% higher with the clearer prompt (its
instruction is longer), the same at every size.

## 3. Decision

The rule, fixed before the pilot: choose the prompt with the fewest cluster-mentioning and duplicate 4B summaries. The
prompt is never chosen by clustering scores.

Outcome: the clearer prompt wins on both counts for the 4B (cluster-mentioning 49 to 0; clusters in duplicate groups 18 to 2),
so **the clearer prompt is adopted** for the main runs. The system-prompt variant (also dropping "clusters" from the system
prompt) was reserved for the case where the 4B still writes about the cluster under the clearer prompt. No size has any
cluster-mentioning summary under it (0 of 150 at each size), so **the variant is not needed** and is not run.

## 4. Determinism

With the old prompt, every size regenerated its logged summaries exactly (150 of 150, 100%) at the logged batch size.
So greedy decoding at that batch size, with this code and environment, reproduces the #31 run on this workload. The clearer-prompt rows
are therefore comparable to the old ones: the difference between them is the prompt. (Each pilot ran once, so the pilot
alone does not test repeatability; the 100% match to the run of 27 October is the evidence.)

## 5. Side-by-side examples

Eight clusters of the 150 (all three sizes, same clusters), each with three member questions. Drawn by a seeded rule that
looks only at the old prompt's summaries: `random.Random(46)` draws 3 clusters at random from those where the old prompt
gave, at any size, a summary that mentions the cluster or an exact duplicate (`old_prompt_flags`), then 5 more at random from
the other clusters, and the result is sorted by index. Members are 3 random members of the cluster's selected members
(same generator, seed 46). Long summaries are cut at 200 characters and marked. Nothing was picked by hand; the heading
of a flagged cluster says why it was drawn. A different draw would show different clusters. The examples are not a
sample for estimating rates; the table above is.

#### Cluster 8

Members (sampled):
- how old is al
- tell me your age please
- how many years do you have

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | What is your age? | how old are you |
| 1.7b | How old are you? | What is the exact date you were born? |
| 0.6b | How old are you? | how old are you again? |

#### Cluster 16 (old prompt: 4b mentions the cluster, 0.6b mentions the cluster)

Members (sampled):
- how many points do i have with my american express
- how expensive is my internet
- what's the status of the order i placed online last week with target

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | What is the summary of the following cluster of questions? The cluster of questions revolves around various financial and account-related inquiries, including transferring funds between accounts, chec [...] | Can you check the status of my recent online purchases and provide information about my account balances and transaction fees? |
| 1.7b | What are the key financial and transaction-related concerns and requests related to multiple accounts, including checking, savings, credit cards, and online purchases, and how can they be addressed? | What are the common concerns users have about their financial transactions and card fees? |
| 0.6b | What are the key points you're asking about in this cluster of questions? | What are the international transaction fees associated with my card? |

#### Cluster 48

Members (sampled):
- when will the replaecment card arrive
- how soon will you mail me a new card
- how can i find a new credit card to apply to

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | When can I expect to receive my new or replacement credit card and what is the typical delivery time? | When will my new credit card arrive? |
| 1.7b | What is the expected timeline for receiving a replacement card, including delivery time, shipping, and when the new card will arrive? | When will my new credit card arrive in the mail? |
| 0.6b | How soon will you send me a replacement card? | when will my new card arrive? |

#### Cluster 63 (old prompt: 4b mentions the cluster, 4b duplicate)

Members (sampled):
- how do you spell "exactly
- will you help me with a math problem
- please help me apply for a silver card

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | What is the main topic that the cluster of questions revolves around? | How do you spell various words and translate terms into different languages? |
| 1.7b | What are some common spelling challenges and how can I learn to spell words like "potato," "water," "university," and others, especially when I'm unsure of the correct spelling? | How do you spell "emolument" for me? |
| 0.6b | What is the correct way to spell "emolument"? | can you spell "emolument" for me? |

#### Cluster 106

Members (sampled):
- how do they say no in germany
- do i need a socket converter if i'm going to be in germany
- how would i say i love you if i were german

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | What are the common German language phrases, travel tips, and cultural considerations for visiting Germany? | How do you say common phrases and ask about travel information in german? |
| 1.7b | What are the most common ways to communicate in German, including greetings, farewells, and everyday phrases, as well as travel-related questions and recommendations? | How do you express feelings in German? |
| 0.6b | How do you say "goodnight" in German? | How do I know if I need a visa to travel to Germany? |

#### Cluster 124

Members (sampled):
- how long will a new card take to reach me
- how long will it take to get a new card
- whats the wait time for a replacement card

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | How long does it take to replace a credit card and what is the expiration date of a credit card? | How long does it take to replace a credit card and what is the expiration date of my credit card? |
| 1.7b | What is the typical time frame for receiving a replacement credit card after losing or having a stolen credit card, including the time it takes for the new card to be mailed and delivered? | How long does it take for a credit card company to process and send a replacement card after a loss or theft? |
| 0.6b | How long will it take for your credit card to be replaced after it gets stolen? | How long will it take for my credit card to be replaced when I lose it? |

#### Cluster 133

Members (sampled):
- change back to default settings
- i want my original settings back
- i need you to switch to my original settings

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | **How can I reset the settings to their original or default state?** | Can you revert to the original settings please? |
| 1.7b | How can I revert my settings to their original state? | Can you reset all settings to their original state? |
| 0.6b | Please go back to the original settings. | What is the main factor that influences the success of a project? |

#### Cluster 146 (old prompt: 4b mentions the cluster, 0.6b mentions the cluster)

Members (sampled):
- text christopher and tell him i will stop by
- text steve and tell him i am running late
- make sure my to do list is completely clear please

| Size | Old prompt | Clearer prompt |
| --- | --- | --- |
| 4b | The cluster of questions summarizes a variety of requests related to scheduling, reminders, task management, travel alerts, account information, and communication tasks. | How can I manage my schedule, set reminders, confirm reservations, and handle notifications? |
| 1.7b | How can I manage my calendar, reminders, and appointments, including setting reminders, confirming reservations, adding events, and ensuring my schedule is clear and up-to-date? | How can I ensure my calendar is clear and organized for upcoming events and tasks? |
| 0.6b | Please summarize the key points of the cluster of questions. | Please schedule a meeting with Sam at 1:00 PM on Tuesday. |

## 6. Observations

These describe the 150 pilot summaries per row. They are not tested for significance.

- **The share ending in "?" drops for the 4B, 87% to 65%.** 52 of the 4B's 150 clearer-prompt summaries do not end in "?" (old: 20,
  11 of them multi-line "What is ... cluster? ... The cluster ..." answers). All 52 are fully lowercase, as are 2 more
  that end in "?", and the old prompt gave no lowercase summary. The members are lowercase questions without "?", so this
  looks like copying the users' style, "in the same style" in the prompt. Of the 52, 46 open with a question word or an
  auxiliary verb ("how do i change the oil in my car", "are you a bot or a human"): they are questions in form with the mark
  left off. The other 6 are requests ("order everything on my shopping list", "play the next song on the playlist",
  "set a timer or alarm for me", "cancel my dinner reservation please", "help me find my lost phone and disconnect it from your system",
  "please remove an item from my list"). The question-word check is a heuristic on the first word; it was read through by hand for the 4B. The 1.7B and 0.6B
  shares are 99% and 90% (old: 97% and 89%).
- **Summaries get shorter.** Mean words go from 17.2 to 11.1 (4B), 20.4 to 13.4 (1.7B) and 11.3 to 8.8 (0.6B).
- **Output tokens fall:** 3,160 to 1,975 (-37.5%, 4B), 3,786 to 2,410 (-36%, 1.7B) and 2,157 to 1,725 (-20%, 0.6B). This is for
  one step of 150 summaries. It says the generated text is shorter, not what a full run costs.
- **Cluster-mentioning summaries and duplicates vanish at the 1.7B and 0.6B too** (14 to 0 and 13 to 0 mentions). Those sizes
  had few of them to begin with (9%).
- **Clearer is not the same as faithful.** The examples include clearer-prompt summaries that look unrelated to their
  members (0.6B, clusters 133 and 146). The pilot counts surface form; it does not
  measure whether a summary represents its cluster. That is what the main runs' clustering scores and #33/#34 address.

## 7. Timing caveat

The pilot's seconds are not used for cost. A low-priority CPU job (the `nice -n 19` replays of #33, #47 and #50 from the GPU summary embeddings, recorded in the run log, not in these directories) ran at the same time, and the pilot has no warm-up pass (repeat 0
includes the one-off CUDA start-up). Cost timing is #35 part 3. The idle check each directory's `gpu.json` recorded before the
model loaded says idle for all six ("median sample within all thresholds"; median 1,215 MiB used, 0-1% utilization, 43-45 W):

| Directory | Idle at start |
| --- | --- |
| `pilot_old_4b/20261112-154657` | yes |
| `pilot_v2_4b/20261112-155229` | yes |
| `pilot_old_1.7b/20261112-155653` | yes |
| `pilot_v2_1.7b/20261112-160011` | yes |
| `pilot_old_0.6b/20261112-160249` | yes |
| `pilot_v2_0.6b/20261112-160456` | yes |

The idle check says nothing about the CPU job, which the check does not look at.

## 8. Main runs

Seeds 0-4 on the #31 protocol (batch size 1 for the 4B). Only the Qwen3-4B has run under the clearer prompt so far.
The #31 rows are the old-prompt runs; "k-SLMmeans-v2" marks the clearer prompt.

### 8.1 Results

| Method | ACC | NMI | ARI | s/seed | seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.551 ± 0.007 | 0.748 ± 0.002 | 0.439 ± 0.005 | 5.3 | 5 |
| k-NLPmeans (LSA) | 0.560 ± 0.008 | 0.752 ± 0.004 | 0.449 ± 0.007 | 23.7 | 5 |
| k-SLMmeans (Qwen3-0.6B) | 0.541 ± 0.013 | 0.747 ± 0.003 | 0.437 ± 0.008 | 903.9 | 5 |
| k-SLMmeans (Qwen3-1.7B) | 0.570 ± 0.011 | 0.760 ± 0.006 | 0.461 ± 0.012 | 1247.5 | 5 |
| k-SLMmeans (Qwen3-4B) | 0.511 ± 0.015 | 0.740 ± 0.006 | 0.421 ± 0.010 | 1861.7 | 5 |
| k-SLMmeans-v2 (Qwen3-4B) | 0.568 ± 0.010 | 0.760 ± 0.005 | 0.457 ± 0.012 | 1098.8 | 5 |

Mean ± standard deviation over seeds. The s/seed column is not comparable across the old and clearer prompt (Section 8.4).
The k-means and LSA rows are from the #31 LSA run; neither depends on the prompt.

**Pending: Qwen3-0.6B and Qwen3-1.7B under the clearer prompt.** They are queued and have not run. No v2 row exists for them,
and nothing in this file says how the sizes order under the clearer prompt.

### 8.2 Paired differences

Same seed, same step-0 partition, difference per seed, 95% t-interval over the 5 seeds (df = 4). "Wins" is the number of seeds
with a difference above 0. The full tables (every method against the old 4B, k-means, LSA and the old 1.7B) are in
[`prompt_test_paired.md`](prompt_test_paired.md) and [`prompt_test_paired.csv`](prompt_test_paired.csv). They will be
regenerated with all sizes once the 0.6B and 1.7B run under the clearer prompt. The rows below are copied from them.

| Comparison (first minus second) | ACC | NMI | ARI |
| --- | --- | --- | --- |
| v2 4B minus old 4B | +0.056 [+0.039, +0.074], 5/5 | +0.021 [+0.013, +0.029], 5/5 | +0.036 [+0.019, +0.053], 5/5 |
| v2 4B minus k-means | +0.016 [+0.009, +0.023], 5/5 | +0.012 [+0.007, +0.017], 5/5 | +0.018 [+0.007, +0.029], 5/5 |
| v2 4B minus LSA | +0.007 [-0.001, +0.016], 5/5 | +0.008 [+0.002, +0.014], 5/5 | +0.008 [-0.002, +0.017], 4/5 |
| v2 4B minus old 1.7B | -0.002 [-0.021, +0.017], 2/5 | -0.000 [-0.007, +0.007], 2/5 | -0.005 [-0.023, +0.014], 2/5 |
| old 4B minus k-means (from #31) | -0.040 [-0.058, -0.022], 0/5 | -0.009 [-0.017, -0.000], 0/5 | -0.018 [-0.027, -0.009], 0/5 |
| old 1.7B minus k-means (from #31) | +0.018 [+0.004, +0.033], 4/5 | +0.012 [+0.005, +0.019], 5/5 | +0.022 [+0.011, +0.034], 5/5 |

- Under the clearer prompt the 4B is above k-means in ACC on 5 of 5 seeds, by 0.016. Under the old prompt it was below
  k-means on 5 of 5 seeds, by 0.040.
- The 4B against LSA: the ACC and ARI intervals include 0 (NMI does not). With 5 seeds this does not separate the two.
- The 4B against the old-prompt 1.7B: the interval is wide, [-0.021, +0.017] in ACC, and the sign changes by seed.
  The data do not tell the two apart. That is a comparison with an old-prompt run, not a v2 size comparison.

**What was checked before pairing.** The pairing needs the v2 4B and the #31 runs to differ only in the prompt and to start
from the same step-0 partition.

- Config: `diff` of the `config.yaml` saved in the two 4B run directories shows four differences. `name` and `method_name`
  (labels), the added `summarizer.prompt`, and the added `gpu_check` policy (the monitor, no effect on results). The old
  config also lists `baselines: kmeans` (the k-means row, not rerun here). Model, batch size (1), `max_new_tokens`,
  `max_input_tokens`, schedule, `tol`, seeds, embedder and dataset are identical.
- Step-0 partition: `step0_sha1` (a hash of the step-0 labels) is identical per seed in the v2 4B run and in every #31 run
  (seeds 0-4: 2b3858a9, 41db4cd8, 4f942755, 142872d5, f74393e0, as the paired script's own check also says). The step-0
  `inertia` in `trace.jsonl` agrees between the v2 4B and the old 4B for all five seeds to float32 rounding (seeds 1-3 differ
  by 0.001 in about 10,700). The k-means row is the same in all #31 runs (ACC 0.5473, 0.5600, 0.5569, 0.5495, 0.5437).
- Not checked: the step-1 summaries' inputs. They are the same by construction (same partition and member selection), and
  the pilot (Section 4) showed the same inputs reproduce the logged summaries exactly. After step 1 the partitions
  differ between prompts, as they should.

### 8.3 Surface counts on the full runs

From `scripts/surface_counts.py`, pooled over the 5 seeds and 5 summary steps (750 summaries per step set, 3,750 per run), with the
rules of Section 1 and of `clinc150_summarizers.md` §2.4. The table is in [`prompt_test_surface.md`](prompt_test_surface.md).

| Run | Summaries | Mean words | End with "?" | Mention the cluster | Final step: mention | Final step: clusters in an identical-summary group, per seed | Largest group |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 4B | 3,750 | 10.9 | 2,300 (61%) | 0 (0%) | 0/750 | 0-2 | 2 |
| old 4B | 3,750 | 17.7 | 3,205 (85%) | 1,291 (34%) | 272/750 | 16-21 | 13 |
| old 1.7B | 3,750 | 21.0 | 3,623 (97%) | 316 (8%) | 68/750 | 0-4 | 2 |
| old 0.6B | 3,750 | 11.2 | 3,301 (88%) | 314 (8%) | 69/750 | 0-5 | 3 |

- The pilot's result holds over the whole run: no summary of the v2 4B mentions the cluster, against 34% under the old
  prompt, and the final-step identical-summary groups go from 16-21 clusters per seed to 0-2.
- The 4B's share ending in "?" is 61% (old 85%), below the pilot's 65% at step 1. Summaries are shorter: 10.9 words against
  17.7. The pilot (Section 6) found that the "?"-less ones are lowercase copies of the users' style; the full run was not
  read for this.
- The old 1.7B and 0.6B counts are the #31 ones. The v2 counts for those sizes are pending.
- Starting with a quote mark: 0 for the v2 4B, 25 for the old 4B. No `<think>` text in any run.

### 8.4 Cost

From `--cost-md` ([`prompt_test_cost.md`](prompt_test_cost.md), the two 4B rows here; the file has all rows); mean ± std over seeds.

| Run | Calls | Input tokens | Output tokens | Summarizer s | Total s | Truncated clusters | Dropped texts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old 4B | 714 ± 6 | 1,120,190 ± 7,604 | 15,620 ± 463 | 1855.0 ± 188.6 | 1861.7 ± 189.8 | 10.2 ± 3.4 | 502.4 ± 264.2 |
| v2 4B | 698 ± 10 | 1,131,772 ± 11,868 | 9,068 ± 185 | 1093.7 ± 20.7 | 1098.8 ± 20.6 | 3.8 ± 2.2 | 86.2 ± 69.0 |

- **Do not read the drop from 1862 to 1099 s/seed as an effect of the prompt.** The #31 timings were inflated about 1.7 times by another process on
  the card (see the header of `scripts/run_clinc150.sh` and #35). The v2 run was idle-checked
  before start (`gpu.json`: idle, median sample within all thresholds; one low-priority CPU job is not covered by that
  check, Section 7). The two times differ by contention and by prompt, and the data cannot split them.
  The clean comparison is the #35 re-time of the old-prompt 4B at seed 0 under idle checks. It is queued.
- **Tokens are not affected by contention.** Output tokens fall from 15,620 to 9,068 per seed (-42%), in line with the
  pilot's -37.5% at step 1. Input tokens rise by 1% (1,120,190 to 1,131,772): the instruction is longer, and the
  members differ after step 1.
- The call count is 698 against 714. Calls are made for clusters whose summary has to be rewritten, so it depends on how the
  partitions move. The difference of 16 in about 700 is small and is not read as an effect.
- Peak GPU memory is the same (9.25 GB allocated, 9.41 GB reserved).
- Truncation: the log of the v2 run warns once, `a cluster's rendered prompt exceeded max_input_tokens=4096; dropped 2 of 192
  text(s)`. The warning is printed once per summarizer, so the counts come from the metrics: 1-7 truncated clusters per seed and 15-195
  dropped texts (v2) against 5-14 clusters and 183-821 texts (old), in about 700 calls. The figures are lower under the clearer prompt, but
  they depend on the partitions of later steps, so they are not a prompt effect. The truncation rate on
  the #15 runs is in [`rq2_truncation_check.md`](rq2_truncation_check.md) (#37).

### 8.5 Lloyd iterations

`trace.jsonl` logs the iterations of every k-means fit (`last_n_iter` against the cap `last_max_iter`). Under both prompts
every fit ran to its cap: 20 iterations in each of 6 fits per seed, 120 per seed, in all 5 seeds (`converged` is False in every record). The counts
are the same in the v2 and the old 4B, so they say nothing about the prompt. They only show that neither run converged by the cap.

### 8.6 Reading

What the run shows:

- With the clearer prompt, the 4B scores 0.568 ACC against 0.511 with the old one, +0.056 [+0.039, +0.074] on the same seeds
  and step-0 partitions, higher on all 5 seeds. NMI and ARI move the same way (+0.021, +0.036).
- Under the clearer prompt the 4B is above k-means (+0.016 ACC [+0.009, +0.023], 5/5) and not distinguishable from LSA
  (+0.007 [-0.001, +0.016]) with 5 seeds.
- The surface problem of the old prompt is gone in this run: no summary mentions the cluster, and at most 2 clusters per seed
  share a summary at the end.
- This agrees with #47. There, the 4B's flagged summaries (duplicates and cluster mentions) were worse than the 1.7B's at step 1,
  but guarding them with the k-means centroid did not bring the 4B back above k-means. A different prompt, which stops those
  summaries from being written in the first place, does.

What it does not show:

- It is one prompt change that altered several things in the wording at once (the task, the "same style", "write only the question",
  and the banned words). It does not say which of them matters. No test of single parts was run.
- It is one dataset (CLINC150) and 5 seeds. The intervals are narrow for the old-versus-new comparison and wide for the
  comparisons with LSA and the 1.7B.
- The size order under the clearer prompt is not known. The 0.6B and 1.7B have not run under it. The v2 4B being level with the
  old 1.7B (-0.002 [-0.021, +0.017]) is a comparison across prompts and is not that answer.
- It does not say the summaries are better or more faithful: the surface counts measure form. #33 and #34 look at that.
- It does not give a prompt effect on cost (Section 8.4).
- Truncation affected some prompts in both runs; the warning was seen once (2 of 192 texts dropped).

Per the reading rule of the issue, the 4B recovers to the level of k-means and above it, and close to the old 1.7B. So the
old size comparison is confounded by prompt reading for the 4B, and sizes have to be reported under both prompts once the
other two have run.

### 8.7 Run directories handed on

- v2 4B, for #33 (replacement diagnostic) and #34 (summary quality): `results/clinc150_v2_qwen3-4b/20261112-181256`
  (git 341e420, clean tree, GPU idle-checked; has `gpu.json` and `gpu_samples.csv`).
- The old-prompt runs it pairs with: `results/clinc150_lsa/20261027-174631`, `results/clinc150_qwen3-0.6b/20261027-190908`,
  `results/clinc150_qwen3-1.7b/20261027-202532`, `results/clinc150_qwen3-4b/20261027-221042`.
- The v2 0.6B and 1.7B directories will be added when they run.

## Reproduce

Run in the checkout that holds `results/` and `cache/` (CPU only, no model runs):

```bash
# Pilot (Sections 1-7)
python scripts/compare_workloads.py $(for n in old_4b v2_4b old_1.7b v2_1.7b old_0.6b v2_0.6b; do ls -d results/fixed_workload/pilot_$n/* | sort | tail -1; done)
python scripts/prompt_test_pilot.py --out-csv docs/results/prompt_test_pilot.csv --out-examples docs/results/prompt_test_pilot_examples.md
```

The pilot directories were made with `scripts/fixed_workload.py` (README, "Prompt test and fixed workloads (#46)").

Main runs (Section 8). The v2 4B run, then the 0.6B and 1.7B when their turn comes:

```bash
PYTHON=/path/to/venv/bin/python scripts/run_clinc150_prompt_v2.sh 4b
PYTHON=/path/to/venv/bin/python scripts/run_clinc150_prompt_v2.sh 0.6b 1.7b   # pending
```

Tables for Section 8 (the five runs that exist so far; add `results/clinc150_v2_qwen3-{0.6b,1.7b}/<stamp>` when they run):

```bash
OLD="results/clinc150_lsa/20261027-174631 results/clinc150_qwen3-0.6b/20261027-190908 results/clinc150_qwen3-1.7b/20261027-202532 results/clinc150_qwen3-4b/20261027-221042"
V2=results/clinc150_v2_qwen3-4b/20261112-181256

python scripts/aggregate_results.py $OLD $V2 \
    --paired-md docs/results/prompt_test_paired.md --paired-csv docs/results/prompt_test_paired.csv \
    --references "k-SLMmeans (Qwen3-4B)" "k-means" "k-NLPmeans (LSA)" "k-SLMmeans (Qwen3-1.7B)"
python scripts/aggregate_results.py $OLD $V2 --out-md /dev/null --cost-md docs/results/prompt_test_cost.md
python scripts/surface_counts.py $V2 results/clinc150_qwen3-4b/20261027-221042 \
    results/clinc150_qwen3-1.7b/20261027-202532 results/clinc150_qwen3-0.6b/20261027-190908 > docs/results/prompt_test_surface.md
```

The first command without `--paired-*` (or `scripts/run_clinc150_prompt_v2.sh table`) prints the Section 8.1 table. The
Lloyd iteration counts (Section 8.5) were read from each run's `trace.jsonl` (`last_n_iter`, `last_max_iter`).
