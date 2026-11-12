# CLINC150 prompt test (#46): does a clearer prompt change the Qwen3 results?

Issue #46 asks whether part of the Qwen3-4B's lower CLINC150 scores comes from how it reads the prompt (background in
[`clinc150_summarizers.md`](clinc150_summarizers.md) §2.4 and §4). This file holds the pilot, which needs no clustering
scores. The main runs are still in progress and are listed at the end.

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
- **One prompt, fixed in advance.** It was written and committed before any run. There was no prompt search.
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
  which are mostly the multi-line "What is ... cluster? ... The cluster ..." answers). All 52 are fully lowercase, as are 2 more
  that end in "?", and the old prompt gave no lowercase summary. The members are lowercase questions without "?", so this
  looks like copying the users' style, "in the same style" in the prompt. Of the 52, 46 open with a question word or an
  auxiliary verb ("how do i change the oil in my car", "are you a bot or a human"): they are questions in form with the mark
  left off. The other 6 are requests ("order everything on my shopping list", "play the next song on the playlist",
  "set a timer or alarm for me", "cancel my dinner reservation please", "help me find my lost phone and disconnect it from your system",
  "please remove an item from my list"). The question-word check is a heuristic on the first word; it was read through by hand for the 4B. The 1.7B and 0.6B
  shares stay at 99% and 90%.
- **Summaries get shorter.** Mean words go from 17.2 to 11.1 (4B), 20.4 to 13.4 (1.7B) and 11.3 to 8.8 (0.6B).
- **Output tokens fall:** 3,160 to 1,975 (-37.5%, 4B), 3,786 to 2,410 (-36%, 1.7B) and 2,157 to 1,725 (-20%, 0.6B). This is for
  one step of 150 summaries. It says the generated text is shorter, not what a full run costs.
- **Cluster-mentioning summaries and duplicates vanish at the 1.7B and 0.6B too** (14 to 0 and 13 to 0 mentions). Those sizes
  had few of them to begin with (9%).
- **Clearer is not the same as faithful.** The examples include clearer-prompt summaries that look unrelated to their
  members (0.6B, clusters 133 and 146). The pilot counts surface form; it does not
  measure whether a summary represents its cluster. That is what the main runs' clustering scores and #33/#34 address.

## 7. Timing caveat

The pilot's seconds are not used for cost. A low-priority CPU job ran at the same time, and the pilot has no warm-up pass (repeat 0
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

## Main runs (pending)

To be added when the runs finish (`scripts/run_clinc150_prompt_v2.sh`, config `configs/clinc150_prompt_v2.yaml`, #31 protocol, batch sizes 4B 1, 1.7B 4, 0.6B 8):

- the 4B under the clearer prompt on seeds 0-4, then the 0.6B and the 1.7B;
- paired differences (same seed, same step-0 partition, 95% t-intervals as in #31): the 4B under the clearer prompt against the 4B
  under the old prompt, and each size against k-means and LSA under the clearer prompt;
- whether the order of the sizes changes;
- the §2.4 surface counts on the full runs' summaries, and the run directories handed to #33 and #34.

Commands:

```bash
PYTHON=/home/cooper/mst/clusteralgs/summaryCentroids/.venv/bin/python scripts/run_clinc150_prompt_v2.sh 4b 0.6b 1.7b

# Paired tables, all runs together: each v2 size and each old size against k-means, LSA and the old 4B.
python scripts/aggregate_results.py results/clinc150_lsa/20261027-174631 results/clinc150_qwen3-0.6b/20261027-190908 \
    results/clinc150_qwen3-1.7b/20261027-202532 results/clinc150_qwen3-4b/20261027-221042 \
    results/clinc150_v2_qwen3-{0.6b,1.7b,4b}/<stamp> \
    --paired-md docs/results/prompt_test_paired.md --paired-csv docs/results/prompt_test_paired.csv \
    --references "k-means" "k-NLPmeans (LSA)" "k-SLMmeans (Qwen3-4B)"

# Surface counts of the full runs.
python scripts/surface_counts.py results/clinc150_v2_qwen3-{4b,1.7b,0.6b}/<stamp>
```

## Reproduce

Run in the checkout that holds `results/` and `cache/` (CPU only, no model runs):

```bash
python scripts/compare_workloads.py $(for n in old_4b v2_4b old_1.7b v2_1.7b old_0.6b v2_0.6b; do ls -d results/fixed_workload/pilot_$n/* | sort | tail -1; done)
python scripts/prompt_test_pilot.py --out-csv docs/results/prompt_test_pilot.csv --out-examples docs/results/prompt_test_pilot_examples.md
```

The pilot directories were made with `scripts/fixed_workload.py` (README, "Prompt test and fixed workloads (#46)").
