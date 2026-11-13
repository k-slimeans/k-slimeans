# BBC News (#18): k-means, LSA and Qwen3 summaries in a sparse TF-IDF space and with a dense control

Issue #18, the course's high-dimensional dataset. 500 BBC News articles, 5 topics, k = 5, seeds 0-4, the #31 protocol, two
embeddings of the same articles: TF-IDF (500 x 784, the main variant) and all-MiniLM-L6-v2 (500 x 384, the control).
Five seeds, one dataset, one model family. Numbers are means over seeds unless stated; intervals are paired 95% t-intervals (n = 5, df = 4).

## TL;DR

- **TF-IDF.** The means order as k-means 0.788 < LSA 0.839 < Qwen3-0.6B 0.851 < Qwen3-4B 0.854 < Qwen3-1.7B 0.873 in ACC. The seed-to-seed
  spread (0.02-0.08) is as large as most of the gaps. Of the 30 paired intervals (10 pairs, 3 metrics), 2 exclude zero (1.7B against k-means in
  ARI, 1.7B against LSA in NMI), with no correction for the number of intervals. Size does not order the SLMs (1.7B above 4B above 0.6B).
- **Dense control.** Means are k-means 0.866 +- 0.129, LSA 0.897 +- 0.131, 0.6B 0.960 +- 0.001, 1.7B 0.904 +- 0.127, 4B 0.902 +- 0.130. The large
  spreads come from **two seeds (0 and 2)** whose shared step-0 partition is a bad local optimum in which business and tech are
  (mostly) in one cluster (step-0 ACC 0.702 and 0.750; the other seeds 0.954-0.962). Seed 2 is repaired by all four summary methods and
  not by k-means. Seed 0 is repaired only by the 0.6B; LSA, 1.7B and 4B end at 0.662, 0.676 and 0.670, below where they started. The 0.6B's low
  spread is therefore one seed's outcome, not a stable difference (Section 3).
- **TF-IDF logging.** 49.5-54.0% of the Qwen summary tokens are outside the 784-term vocabulary (71.1% for LSA's extracted sentences). The same
  share for the articles themselves is 53.8%. No summary has an all-zero vector (0 of 500 summaries in the four runs).
- **Surface.** No summary mentions the cluster, and none is duplicated. Every Qwen summary contains the words "news stories" from the prompt.
- **Replacement diagnostic (TF-IDF, CPU, exact).** All 120 fits replay the logged runs with 0 label mismatches. A summary sits 0.98-1.08 RMS radii
  from the centroid it replaces, about what CLINC150 showed (0.89-1.08), but the cosine is much lower (0.17-0.43 against 0.67-0.80).
- **Cost.** About 24 summarizer calls and 84,000-87,000 input tokens per seed in the TF-IDF variant; 27, 46 and 87 s per seed for 0.6B, 1.7B and 4B.

## 1. Setup

- **Data.** `bbc500`: 500 articles of the BBC News dataset (Greene and Cunningham, ICML 2006), 100 per topic (business,
  entertainment, politics, sport, tech), drawn after near-duplicate removal with seed 0 and shuffled. Only the article file ids are in the repository
  (`data/bbc500_ids.csv`, with the SHA-256 of the generated CSV); the text is rebuilt with `python scripts/prepare_bbc.py` and is not redistributed (`data/README.md`).
  The 500 articles average 401 words (minimum 120, maximum 3,295).
  Nothing below quotes article text; the only quoted text is a prompt and one short model-written summary.
- **TF-IDF embedder.** scikit-learn `TfidfVectorizer`, fit once on the 500 articles: English stop words removed, terms in fewer than 20 articles removed
  (`min_df: 20`), no `max_df` or `max_features`, raw term frequency, L2-normalized rows. The vocabulary has **784 terms** (the fit raises below 700).
  Summaries are only transformed with that fitted vectorizer. It is never refit. The settings and the scikit-learn version (1.9.1) are in the cache key
  (`tfidf-mindf20-maxdf1p-maxfall-sub0-en-sk1.9.1`).
- **Dense control.** all-MiniLM-L6-v2 (384 dimensions, 256-token window; articles longer than the window are cut by the encoder), chosen over the CLINC150
  encoder (distilbert, 128-token window) because it reads twice as much of each article (`configs/bbc_dense.yaml`).
- **Prompt.** The news prompt written with the #46 lesson (no "cluster", no "articles", no topic word). User turn: the instruction, a blank line, then one
  member per line:

  > Below are several news stories. Write one sentence that says what they all have in common:

  The model's reply is prefilled with `Sentence:`. The system prompt is the repository default (`DEFAULT_SYSTEM_PROMPT` in `summarizers/hf.py`), unchanged from the
  other datasets. Greedy decoding, thinking off, `max_new_tokens` 64, `max_input_tokens` 4096. The prompt was fixed before the runs; there was no prompt search.
- **Member selection.** Articles are about 400 words, so a cluster of about 100 does not fit in 4,096 tokens. Both configs use `summarizer.max_words_per_text: 30`:
  each member is cut to its first 30 words (the headline and the lead) before the token budget is applied, and what still overflows is cut by the seeded random subset of #28.
  This was chosen over the #28 subset alone, which would show the model a random handful of about 7 whole articles per cluster. The ablation (`0.6b-random` in `scripts/run_bbc.sh`) was not run.
  How often the fallback was needed is in Section 6.
- **Methods and protocol.** k-means, k-NLPmeans (LSA, top-5 sentences), k-SLMmeans with Qwen3-0.6B, 1.7B and 4B (batch 8, 4 and 1 as in #31), k = 5, seeds 0-4. The #31 schedule:
  k-means++ initialization, a step-0 fit, then 5 summary steps (a fixed schedule of 5 summaries), each followed by a block of at most 20 Lloyd iterations (120 in total), `initial_iter` and `final_iter` on,
  `tol` 1e-4. The "k-means" row is the step-0 partition: its ACC equals the step-0 ACC of every seed in both variants, and `step0_sha1` is identical across all methods of a seed.
- **Environment.** All eight runs (`environment.json`): git `341e420`, clean tree (`git_dirty` false), Python 3.14.7, torch 2.14.0+cu130, transformers 5.17.0, sentence-transformers 6.0.1, scikit-learn 1.9.1,
  RTX 3060, bfloat16. The BBC configs, `scripts/run_bbc.sh`, `data.py` and `embeddings.py` are unchanged since that commit. The GPU idle check (`gpu.json`) passed for all eight runs ("median sample
  within all thresholds"); the queue logs record the card at 1,216 MiB, 0% utilization and about 43 W before each variant started. The TF-IDF runs started 2026-11-12 19:45 and finished by 20:00; the dense runs ran 20:01 to 20:09.

## 2. Results

### 2.1 TF-IDF space

Mean +- standard deviation over seeds 0-4 ([`bbc_tfidf_table.md`](bbc_tfidf_table.md), [`.csv`](bbc_tfidf_table.csv); per seed in [`bbc_tfidf_per_seed.csv`](bbc_tfidf_per_seed.csv)).

| Method | ACC | NMI | ARI | s/seed | seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.788 ± 0.080 | 0.659 ± 0.062 | 0.592 ± 0.098 | 0.0 | 5 |
| k-NLPmeans (LSA) | 0.839 ± 0.033 | 0.679 ± 0.033 | 0.641 ± 0.075 | 2.9 | 5 |
| k-SLMmeans (Qwen3-0.6B) | 0.851 ± 0.022 | 0.697 ± 0.025 | 0.668 ± 0.048 | 26.6 | 5 |
| k-SLMmeans (Qwen3-1.7B) | 0.873 ± 0.021 | 0.732 ± 0.027 | 0.717 ± 0.043 | 46.2 | 5 |
| k-SLMmeans (Qwen3-4B) | 0.854 ± 0.022 | 0.702 ± 0.024 | 0.670 ± 0.052 | 87.3 | 5 |

Paired differences (same seed, same step-0 partition; first minus second; difference per seed, 95% t-interval over 5 seeds, then the number of seeds with a difference above 0).
All 10 pairs and 3 metrics: [`bbc_tfidf_paired.md`](bbc_tfidf_paired.md), [`.csv`](bbc_tfidf_paired.csv).

| First minus second | ACC | NMI | ARI |
| --- | --- | --- | --- |
| LSA minus k-means | +0.051 [-0.020, +0.121], 4/5 | +0.020 [-0.037, +0.077], 3/5 | +0.049 [-0.011, +0.109], 4/5 |
| 0.6B minus k-means | +0.063 [-0.053, +0.179], 3/5 | +0.038 [-0.060, +0.136], 3/5 | +0.076 [-0.092, +0.244], 3/5 |
| 1.7B minus k-means | +0.085 [-0.009, +0.178], 5/5 | +0.072 [-0.001, +0.146], 5/5 | +0.125 [+0.007, +0.243], 5/5 |
| 4B minus k-means | +0.066 [-0.027, +0.159], 4/5 | +0.043 [-0.023, +0.110], 4/5 | +0.078 [-0.036, +0.192], 4/5 |
| 0.6B minus LSA | +0.012 [-0.051, +0.075], 2/5 | +0.018 [-0.046, +0.082], 3/5 | +0.027 [-0.114, +0.167], 2/5 |
| 1.7B minus LSA | +0.034 [-0.014, +0.082], 4/5 | +0.052 [+0.001, +0.103], 4/5 | +0.076 [-0.030, +0.182], 4/5 |
| 4B minus LSA | +0.015 [-0.015, +0.045], 3/5 | +0.023 [-0.002, +0.049], 5/5 | +0.029 [-0.038, +0.095], 3/5 |
| 1.7B minus 0.6B | +0.022 [-0.018, +0.062], 4/5 | +0.034 [-0.015, +0.083], 4/5 | +0.049 [-0.035, +0.134], 4/5 |
| 4B minus 0.6B | +0.003 [-0.037, +0.043], 2/5 | +0.005 [-0.038, +0.048], 3/5 | +0.002 [-0.087, +0.091], 2/5 |
| 4B minus 1.7B | -0.019 [-0.054, +0.017], 2/5 | -0.029 [-0.070, +0.012], 1/5 | -0.047 [-0.128, +0.034], 2/5 |

- Every Qwen size and LSA have higher mean ACC than k-means (+0.051 to +0.085), but on 3 to 5 of 5 seeds, and 28 of the 30 intervals include 0.
  The two that do not are 1.7B against k-means in ARI (+0.007 to +0.243) and 1.7B against LSA in NMI (+0.001 to +0.103); both lower ends are close to 0.
- The 1.7B is the best mean on every metric and is above k-means on 5 of 5 seeds. Its lead over the 4B and the 0.6B has an interval that includes 0.
- The 4B is not above the 1.7B and not above the 0.6B in any way these seeds can separate.

### 2.2 Dense control (MiniLM)

Mean +- standard deviation over seeds 0-4 ([`bbc_dense_table.md`](bbc_dense_table.md), [`.csv`](bbc_dense_table.csv), [`bbc_dense_per_seed.csv`](bbc_dense_per_seed.csv)).
The s/seed column is not comparable with the TF-IDF one: the loop stopped before the fifth summary step in several seeds (Section 6).

| Method | ACC | NMI | ARI | s/seed | seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.866 ± 0.129 | 0.823 ± 0.086 | 0.798 ± 0.139 | 0.0 | 5 |
| k-NLPmeans (LSA) | 0.897 ± 0.131 | 0.845 ± 0.073 | 0.837 ± 0.128 | 1.0 | 5 |
| k-SLMmeans (Qwen3-0.6B) | 0.960 ± 0.001 | 0.888 ± 0.004 | 0.904 ± 0.002 | 17.5 | 5 |
| k-SLMmeans (Qwen3-1.7B) | 0.904 ± 0.127 | 0.854 ± 0.077 | 0.845 ± 0.132 | 24.5 | 5 |
| k-SLMmeans (Qwen3-4B) | 0.902 ± 0.130 | 0.848 ± 0.076 | 0.843 ± 0.134 | 38.9 | 5 |

Paired differences ([`bbc_dense_paired.md`](bbc_dense_paired.md), [`.csv`](bbc_dense_paired.csv)), same layout.

| First minus second | ACC | NMI | ARI |
| --- | --- | --- | --- |
| LSA minus k-means | +0.032 [-0.091, +0.154], 2/5 | +0.022 [-0.090, +0.133], 2/5 | +0.038 [-0.113, +0.190], 2/5 |
| 0.6B minus k-means | +0.095 [-0.064, +0.254], 4/5 | +0.065 [-0.039, +0.170], 4/5 | +0.105 [-0.066, +0.277], 4/5 |
| 1.7B minus k-means | +0.038 [-0.084, +0.160], 2/5 | +0.031 [-0.087, +0.149], 2/5 | +0.047 [-0.109, +0.202], 2/5 |
| 4B minus k-means | +0.036 [-0.083, +0.156], 3/5 | +0.025 [-0.082, +0.131], 2/5 | +0.045 [-0.104, +0.193], 3/5 |
| 0.6B minus LSA | +0.063 [-0.100, +0.226], 5/5 | +0.044 [-0.046, +0.133], 5/5 | +0.067 [-0.091, +0.225], 5/5 |
| 1.7B minus LSA | +0.006 [+0.001, +0.012], 5/5 | +0.009 [+0.003, +0.016], 5/5 | +0.008 [+0.003, +0.014], 5/5 |
| 4B minus LSA | +0.005 [+0.001, +0.009], 4/5 | +0.003 [-0.006, +0.012], 3/5 | +0.006 [-0.004, +0.016], 3/5 |
| 1.7B minus 0.6B | -0.057 [-0.215, +0.101], 1/5 | -0.034 [-0.129, +0.060], 2/5 | -0.059 [-0.221, +0.104], 2/5 |
| 4B minus 0.6B | -0.058 [-0.219, +0.102], 1/5 | -0.041 [-0.133, +0.052], 1/5 | -0.061 [-0.226, +0.104], 1/5 |
| 4B minus 1.7B | -0.002 [-0.007, +0.003], 2/5 | -0.006 [-0.019, +0.007], 2/5 | -0.002 [-0.012, +0.007], 2/5 |

- 4 of the 30 intervals exclude 0 (1.7B against LSA on all three metrics, 4B against LSA in ACC), all for differences of at most 0.009. In those seeds LSA, 1.7B and 4B end in the
  same basin and differ by a few articles; the intervals are narrow because the seed-to-seed differences are nearly constant. This is not read as a method difference.
- The large intervals elsewhere are the two bad seeds below.

## 3. Step-0 partitions and the bad seeds

The "k-means" row and the start of every summary method are the same step-0 partition per seed. Cluster-by-topic counts for every seed's step-0 partition and for every run's final
partition are in [`bbc_dense_partitions.md`](bbc_dense_partitions.md) / [`.csv`](bbc_dense_partitions.csv) and [`bbc_tfidf_partitions.md`](bbc_tfidf_partitions.md) / [`.csv`](bbc_tfidf_partitions.csv)
(topic counts only; made by `scripts/bbc_report.py` from the ids file and the logged assignments).

**Dense: seeds 0 and 2 start in a merged-topic local optimum; the other three seeds start at the topic structure.**

| Seed | Step-0 ACC | Step-0 clusters (topics with at least 10 members) |
| ---: | ---: | --- |
| 0 | 0.702 | tech 95 + business 93 ; politics 97 ; entertainment 96 ; sport 63 ; sport 37 |
| 1 | 0.962 | one topic per cluster |
| 2 | 0.750 | business 70 + tech 70 ; entertainment 90 ; sport 92 ; politics 93 ; business 30 + tech 26 (that cluster also has 7 entertainment, 6 politics and 7 sport) |
| 3 | 0.960 | one topic per cluster |
| 4 | 0.954 | one topic per cluster |

In both bad seeds business and tech share a cluster (seed 0: all of both topics but 12 articles; seed 2: 70 of each plus a mixed remainder), a business/tech merge in 2 of the 5 seeds (checked from the counts, not assumed).
In seed 0 the cluster that this frees up splits sport 63/37. In seed 2 a fifth cluster holds the remaining business and tech articles with a few from other topics.

Final ACC per seed, dense (from [`bbc_dense_per_seed.csv`](bbc_dense_per_seed.csv)):

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-means (= step 0) | 0.702 | 0.962 | 0.750 | 0.960 | 0.954 |
| LSA | 0.662 | 0.956 | 0.956 | 0.956 | 0.956 |
| Qwen3-0.6B | 0.960 | 0.960 | 0.960 | 0.962 | 0.960 |
| Qwen3-1.7B | 0.676 | 0.960 | 0.962 | 0.960 | 0.960 |
| Qwen3-4B | 0.670 | 0.962 | 0.956 | 0.962 | 0.960 |

- **Seed 2:** k-means stays at 0.750, and all four summary methods reach 0.956-0.962 with one cluster per topic.
- **Seed 0:** only the 0.6B leaves the bad optimum (0.960). LSA, 1.7B and 4B keep business and tech together and split the rest differently, and their final ACC (0.662, 0.676, 0.670) is below the step-0 value (0.702).
- On the three good seeds every method ends at 0.954-0.962, so the methods differ by at most 0.008 there.
- So the dense means are driven by two seeds, and the 0.6B's 0.960 +- 0.001 is the result of leaving the bad optimum in both of them. With one seed in which the 0.6B differs from the other three, this is not evidence that the 0.6B is better. The
  paired interval of 0.6B against LSA, [-0.100, +0.226] in ACC, says the same.
- An example of what the 0.6B wrote for seed 0's merged cluster at its first summary step: "All the news stories highlight the importance of technology and innovation in various sectors."
  The summary mentions technology only.

**TF-IDF: no two-mode pattern.** Step-0 ACC is 0.854, 0.774, 0.658, 0.806, 0.850 (seeds 0-4) and the finals sit between 0.784 and 0.904 for every method. The errors are spread over several clusters, mostly entertainment articles
spread into other clusters: in 24 of the 25 final partitions one cluster is mostly entertainment (58-63 articles), and usually one to three other clusters (up to three) each hold 10 or more entertainment articles. The exception is the 1.7B in seed 0, whose entertainment cluster has 86 entertainment and 16 sport articles and no other cluster has 10 entertainment articles.
Seed 2 is the worst start (0.658): one cluster holds 92 business, 80 tech, 31 entertainment and 14 politics articles. LSA's seed-2 final (0.784) has 34 tech articles in the business cluster and a separate tech cluster of 57, while the three Qwen sizes
end at 0.872, 0.874 and 0.834. The TF-IDF seeds are therefore all moderate rather than split into good and bad.

## 4. The TF-IDF vocabulary and the summaries (issue logging requirements)

The runs logged these counters themselves (`embed_tokens`, `embed_oov_tokens`, `embed_zero_vectors` in `metrics.jsonl`, `trace.jsonl` and `summary.csv`); nothing was computed offline for this table.
Tokens are the scikit-learn analyzer's tokens after stop-word removal, summed over the 5 seeds and all summary steps (5 clusters x 5 summary steps x 5 seeds = 125 summaries per method).
[`bbc_tfidf_oov.md`](bbc_tfidf_oov.md) / [`.csv`](bbc_tfidf_oov.csv), made by `scripts/bbc_report.py` from `metrics.jsonl` and `summaries.jsonl`.

| Method | summaries embedded | tokens | outside the vocabulary | share | all-zero vectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-NLPmeans (LSA) | 125 | 4,426 | 3,145 | 71.1% | 0 |
| k-SLMmeans (Qwen3-0.6B) | 125 | 1,626 | 878 | 54.0% | 0 |
| k-SLMmeans (Qwen3-1.7B) | 125 | 2,418 | 1,332 | 55.1% | 0 |
| k-SLMmeans (Qwen3-4B) | 125 | 2,052 | 1,016 | 49.5% | 0 |

- **No summary is an all-zero vector** (0 of 500), so no run starts a step from a degenerate TF-IDF centroid.
- About half the Qwen tokens are outside the vocabulary. For scale, computed offline from the cached fit (not logged by the runs): 53.8% of the stop-word-filtered tokens of the 500 articles are outside the 784-term vocabulary, and 54.9% of those in the first 30 words of each article.
  So the Qwen summaries lose about the same share of their words as the articles do, and LSA's sentences, taken from the articles, lose more (71.1%). Not interpreted further.
- The dense variant has no such counters.

## 5. Summary surface

[`bbc_tfidf_surface.md`](bbc_tfidf_surface.md) and [`bbc_dense_surface.md`](bbc_dense_surface.md), from `scripts/surface_counts.py` (rules of `clinc150_summarizers.md` §2.4), pooled over seeds and summary steps.

| Variant | Method | summaries | mean words | mention "cluster" or "summar…" | identical-summary clusters at the final step | ends with "?" |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | LSA | 125 | 67.0 | 0 | 0 | 1 |
| TF-IDF | 0.6B | 125 | 21.4 | 0 | 0 | 0 |
| TF-IDF | 1.7B | 125 | 29.1 | 0 | 0 | 0 |
| TF-IDF | 4B | 125 | 23.3 | 0 | 0 | 0 |
| dense | LSA | 85 | 99.4 | 0 | 0 | 0 |
| dense | 0.6B | 115 | 20.1 | 0 | 0 | 0 |
| dense | 1.7B | 110 | 25.6 | 0 | 0 | 0 |
| dense | 4B | 75 | 21.7 | 0 | 0 | 0 |

- **Mentions of the cluster:** 0 of 885 summaries in the eight runs, against 34% for the 4B on CLINC150 (#31, `clinc150_summarizers.md`; a different prompt and task). **Duplicates:** none at the final step in any seed.
- **Length:** the Qwen summaries average 20-29 words. LSA's are extracted sentences of 67 words (TF-IDF) and 99 words (dense).
- **A prompt echo.** Every Qwen summary in all six runs (75 to 125 per run) contains "news stories", words of the prompt. Almost all begin "All the news stories ..." or "The news stories ..." ("All the news stories": 122 of 125 for the 0.6B and 115 of 115 in the
  dense 0.6B run; the 4B mostly writes "The news stories cover/all ..."). The words are in every summary of every cluster; their effect on the clustering is not measured here. Distinct texts per run (of 125 or fewer): 82, 89, 81 in the TF-IDF runs and 29, 40, 24 in the dense runs;
  a cluster whose members did not change keeps its summary, so the same text recurs across steps.

## 6. Cost

[`bbc_tfidf_cost.md`](bbc_tfidf_cost.md) and [`bbc_dense_cost.md`](bbc_dense_cost.md) (`aggregate_results.py --cost-md`), per seed, mean +- std over 5 seeds.

| Variant | Method | calls | input tokens | output tokens | clusters truncated | texts dropped | s/seed | peak GPU GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | LSA | 21 ± 5 | 0 | 0 | 0 | 0 | 2.9 | 0.00 |
| TF-IDF | 0.6B | 23 ± 1 | 83,813 ± 3,752 | 588 ± 41 | 10.6 ± 0.9 | 302.6 ± 38.1 | 26.6 | 4.35 |
| TF-IDF | 1.7B | 24 ± 1 | 86,898 ± 1,916 | 868 ± 30 | 11.6 ± 1.9 | 266.8 ± 54.2 | 46.2 | 6.27 |
| TF-IDF | 4B | 24 ± 1 | 87,474 ± 4,390 | 732 ± 31 | 10.6 ± 0.9 | 270.2 ± 67.0 | 87.3 | 8.98 |
| dense | LSA | 13 ± 6 | 0 | 0 | 0 | 0 | 1.0 | 0.11 |
| dense | 0.6B | 15 ± 6 | 59,061 ± 21,200 | 351 ± 136 | 4.4 ± 1.1 | 51.2 ± 70.1 | 17.5 | 4.45 |
| dense | 1.7B | 13 ± 7 | 47,923 ± 21,493 | 425 ± 234 | 3.8 ± 1.1 | 114.0 ± 205.7 | 24.5 | 6.37 |
| dense | 4B | 11 ± 5 | 41,438 ± 15,096 | 311 ± 149 | 3.4 ± 0.9 | 104.6 ± 194.9 | 38.9 | 9.07 |

- **Calls.** At most 5 summary steps x 5 clusters = 25 per seed. The TF-IDF runs make 23-24 on average, and all log 6 fits per seed. The dense runs make 11-15 on average with a large spread: the number of logged fits per seed ranges from 2 to 6 (step 0 and up to 5 blocks),
  so several seeds stopped before the fifth summary step; all four methods ran the full 6 fits in seed 0. The dense cost columns therefore depend on how many steps each seed ran.
- **Truncation.** With the 30-word cut, 10.6-11.6 clusters per seed still overflowed the 4,096-token budget in the TF-IDF runs, and the seeded random subset dropped 267-303 member texts per seed, about 23-29 per truncated cluster. The other clusters (about half the calls)
  had no member dropped. Input per call is about 3,600 tokens. Dense: 3.4-4.4 clusters and 51-114 dropped texts, with a spread as large as the mean.
- **Time.** s/seed is wall time of the method for one seed, 27, 46 and 87 s (TF-IDF) for 0.6B, 1.7B and 4B, against 2.9 s for LSA. Caveat: during these runs, light low-priority CPU work for other issues ran at times on the same machine; the GPU was idle-checked before each
  variant and each run (`gpu.json`, all passed), so the GPU timings are not affected by other GPU jobs, but the CPU parts (tokenization, scikit-learn, LSA) may be slightly inflated. Times also mix model size with batch size (8, 4, 1).

## 7. Replacement diagnostic (issue step 5)

**Tool support.** `scripts/embed_logged_summaries.py` and `scripts/replacement_diagnostic.py` work on the TF-IDF runs without a change (`load_run_embedder` rebuilds the fitted vectorizer from the 500 articles; the cached 500 x 784 matrix is used). Two small gaps, neither fixed here:
the script's final figure step raises a `KeyError` for any run name that is not one of the four CLINC150 names (the CSVs and tables are written before it, and nothing needed for this section is lost), and its `examples.md` quotes article text, so all outputs were written to a scratch directory and only the numeric files were copied here (the examples file is **not committed**).

**Verification (TF-IDF, CPU).** TF-IDF transforms are deterministic, so the summaries were re-embedded on the CPU with the same fitted vectorizer
(`embed_logged_summaries.py --device cpu`, 9 s) and the logged runs replayed (`replacement_diagnostic.py`, 62 s wall time, nice 19). Result ([`bbc_tfidf_replacement_verification.md`](bbc_tfidf_replacement_verification.md), scikit-learn 1.9.1):

| run | fits | exact fits | label mismatches (max) | `n_iter` differs | max rel. error, start objective | max rel. error, end objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSA | 30 | 30 | 0 | 0 | 3.6e-10 | 1.7e-07 |
| Qwen3-0.6B | 30 | 30 | 0 | 0 | 7.2e-09 | 1.7e-07 |
| Qwen3-1.7B | 30 | 30 | 0 | 0 | 5.4e-09 | 2.2e-07 |
| Qwen3-4B | 30 | 30 | 0 | 0 | 5.2e-09 | 2.9e-07 |

All 120 fits (step 0 and 5 blocks, 5 seeds, 4 runs) reproduce the logged partition with 0 mismatching labels and the logged iteration count, and the real-`KMeans` mechanism check (first seed of every method) agrees (0 label mismatches, same `n_iter`, [`bbc_tfidf_replacement_mechanism.csv`](bbc_tfidf_replacement_mechanism.csv)).
Committed outputs: [`bbc_tfidf_replacement_tables.md`](bbc_tfidf_replacement_tables.md) (all tables), [`_steps.csv`](bbc_tfidf_replacement_steps.csv), [`_outcome.csv`](bbc_tfidf_replacement_outcome.csv), [`_verification.csv`](bbc_tfidf_replacement_verification.csv), [`_mechanism.csv`](bbc_tfidf_replacement_mechanism.csv).
No figures were made. Definitions are those of [`replacement_diagnostic.md`](replacement_diagnostic.md) §1; step 1 is the first replacement, from the step-0 partition every method shares.

**Centroid jump, step 1, mean over seeds, TF-IDF beside CLINC150** (CLINC150 values from `replacement_diagnostic.md` §3-4; its step 1 is the same step):

| Method | jump in J (rel.), TF-IDF / CLINC | summary to centroid, RMS radii, TF-IDF / CLINC | cosine, TF-IDF / CLINC | points moved at the replacement, TF-IDF / CLINC |
| --- | --- | --- | --- | --- |
| LSA | 0.92 / 0.72 | 0.975 / 0.891 | 0.43 / 0.80 | 26% / 25% |
| Qwen3-0.6B | 1.13 / 0.78 | 1.08 / 0.965 | 0.17 / 0.75 | 50% / 47% |
| Qwen3-1.7B | 1.10 / 0.95 | 1.06 / 1.05 | 0.19 / 0.71 | 46% / 47% |
| Qwen3-4B | 1.13 / 0.98 | 1.08 / 1.08 | 0.20 / 0.67 | 47% / 56% |

- **In RMS radii the jump is about the same as on CLINC150** (0.98-1.08 against 0.89-1.08): a summary sits about one cluster radius from the centroid it replaces. The issue expected a larger jump in TF-IDF space; in this unit it is not larger.
- **In cosine it is much farther**: 0.17-0.43 against 0.67-0.80, and the relative rise of the k-means objective at the replacement is larger (0.92-1.13 against 0.72-0.98).
- The share of points that change cluster at the replacement is close to CLINC150's (26%, 50%, 46%, 47% against 25%, 47%, 47%, 56%). Over the block the net share that stays moved is 11-16% (CLINC150: 9-45%), and the partition after the block is closer to the old one
  (ARI with the pre partition 0.71-0.75; CLINC150 0.65-0.85).
- Duplicates and collapsed centroids: none (0 duplicate summaries, 0 empty clusters at the replacement or at the end of any block, 0 reseeded).
- Outcome at step 1, logged partition minus plain Lloyd from the same step-0 partition (mean ACC difference, seeds below): LSA +0.020 (1/5), 0.6B +0.022 (1/5), 1.7B +0.057 (0/5), 4B +0.046 (1/5). On CLINC150 the same quantity was +0.005, +0.002, +0.011 and -0.016.
  In NMI and ARI the means are +0.029 / +0.059 (1.7B) and +0.016 / +0.044 (4B), but about zero or mixed for LSA (NMI -0.015, ARI +0.017) and the 0.6B (-0.002, +0.003). So the step-1 improvement over plain Lloyd shows in ACC for the 1.7B and 4B and in all three metrics only for those two, with 5 seeds and one start per seed. This is a description of 20 (method, seed) pairs, not an effect estimate.
- Share of the final summaries that no longer describe the final partition (Jaccard below 0.5, or nearer another cluster's final center): LSA 0 of 25, 0.6B 5, 1.7B 1, 4B 4.

**Dense variant (not run).** An exact replay needs the summaries re-encoded by MiniLM on the GPU the runs used (as for CLINC150, where a CPU encode differs by about 1e-6 per row and did not reproduce the 4B's blocks). This is one queued GPU step of about a minute, then CPU replays.
The commands, with `<stamp>` the run directories of `bbc_dense_{lsa,qwen3-0.6b,qwen3-1.7b,qwen3-4b}`, and a scratch output directory so that the CLINC150 files in `docs/results/` are not overwritten and no article text is committed:

```bash
python scripts/embed_logged_summaries.py results/bbc_dense_lsa/<stamp> results/bbc_dense_qwen3-0.6b/<stamp> \
    results/bbc_dense_qwen3-1.7b/<stamp> results/bbc_dense_qwen3-4b/<stamp> \
    --cache-dir cache --out results/replacement_diagnostic/bbc_dense_summary_embeddings_cuda.npz --device cuda
CUDA_VISIBLE_DEVICES="" python scripts/replacement_diagnostic.py results/bbc_dense_lsa/<stamp> ... \
    --cache-dir cache --summary-embeddings results/replacement_diagnostic/bbc_dense_summary_embeddings_cuda.npz \
    --out-dir results/replacement_diagnostic/bbc_dense --work-dir results/replacement_diagnostic/bbc_dense_work
```

## 8. Reading

Five seeds, one dataset with 5 topics, one model family, one prompt, one member-selection rule (first 30 words), no multiplicity correction. The claims are limited to these runs.

- **In the TF-IDF space, every summary method has a higher mean than k-means, and the 1.7B has the highest.** The gaps are 0.05-0.09 in ACC against seed spreads of 0.02-0.08, and almost all paired intervals include 0. Two of 30 do not, both with a lower end near 0.
  The data do not separate LSA from the Qwen sizes, or the Qwen sizes from each other.
- **Size does not order the Qwen results.** In TF-IDF the order 1.7B above 4B above 0.6B is within noise (every paired interval between sizes includes 0), as on CLINC150 and Bank77 (`clinc150_summarizers.md`). The dense ordering is not used as evidence: it is one seed's outcome (Section 3). With one model family this is not a trend either way.
- **The dense control mostly shows the effect of the starting partition.** Seeds 0 and 2 start in a business/tech merged optimum and decide the means and spreads. Whether a method leaves that optimum in a given seed is not explained by these runs.
  On the other three seeds all methods are within 0.008 of each other.
- **Compared with CLINC150 (#31), qualitatively:** the 4B is not below k-means here (on CLINC150 it was below k-means on all 5 seeds), there are no cluster-mentioning or duplicate summaries (CLINC150 had 34% mentions for the 4B; a different prompt, dataset and task, so not a controlled comparison), and the replacement moves
  a summary about as far from the centroid in RMS radii as on CLINC150 but much farther in cosine. Whether any of these differences are due to the data, the embedding, the prompt or the 30-word members is not tested here; no cause is claimed.
- **What is not shown:** that summaries beat numerical centroids on this dataset; that either embedding suits the method better; that the 0.6B is better in the dense space (one seed).

## 9. Limits and next steps

- The first-30-words cut and the news prompt were fixed in advance. The #28 random subset of whole articles (`0.6b-random`) was not run, so how much the member selection matters is open.
- The dense replacement diagnostic needs the queued GPU encode (Section 7).
- The TF-IDF vocabulary is 784 terms at `min_df` 20; other settings were not tried.
- Timings come from a shared machine (Section 6).
- The README's BBC section says articles average ~380 words; the 500-article sample averages 401 (whitespace-split words).

## 10. Reproduce

Run from the checkout that holds `results/` and `cache/`; `python` is an environment with the project installed. `<stamp>` is each run's timestamp directory.

```bash
python scripts/prepare_bbc.py                 # data/bbc500.csv from the ids file (not redistributed)
scripts/run_bbc.sh tfidf                      # k-means, LSA, Qwen3 0.6B/1.7B/4B, seeds 0-4; table in results/bbc_tfidf_table.md
scripts/run_bbc.sh dense

TFIDF_RUNS="results/bbc_tfidf_lsa/<stamp> results/bbc_tfidf_qwen3-0.6b/<stamp> results/bbc_tfidf_qwen3-1.7b/<stamp> results/bbc_tfidf_qwen3-4b/<stamp>"
for v in tfidf dense; do
  RUNS="results/bbc_${v}_lsa/<stamp> results/bbc_${v}_qwen3-0.6b/<stamp> results/bbc_${v}_qwen3-1.7b/<stamp> results/bbc_${v}_qwen3-4b/<stamp>"
  python scripts/aggregate_results.py $RUNS \
      --out-md docs/results/bbc_${v}_table.md --out-csv docs/results/bbc_${v}_table.csv \
      --per-seed-csv docs/results/bbc_${v}_per_seed.csv \
      --paired-md docs/results/bbc_${v}_paired.md --paired-csv docs/results/bbc_${v}_paired.csv \
      --references "k-means" "k-NLPmeans (LSA)" "k-SLMmeans (Qwen3-0.6B)" "k-SLMmeans (Qwen3-1.7B)" "k-SLMmeans (Qwen3-4B)" \
      --cost-md docs/results/bbc_${v}_cost.md
  python scripts/surface_counts.py $RUNS > docs/results/bbc_${v}_surface.md
  python scripts/bbc_report.py $RUNS --ids data/bbc500_ids.csv \
      --partitions-md docs/results/bbc_${v}_partitions.md --partitions-csv docs/results/bbc_${v}_partitions.csv
done
python scripts/bbc_report.py $TFIDF_RUNS --ids data/bbc500_ids.csv \
    --oov-md docs/results/bbc_tfidf_oov.md --oov-csv docs/results/bbc_tfidf_oov.csv   # TF-IDF runs only

# Replacement diagnostic, TF-IDF (CPU, about 70 s). Write to a scratch directory, not docs/results: the file names carry no
# dataset prefix and an examples file with article text is written; the numeric files are then copied under the committed names below.
CUDA_VISIBLE_DEVICES="" python scripts/embed_logged_summaries.py $TFIDF_RUNS --cache-dir cache \
    --out results/replacement_diagnostic/bbc_tfidf_summary_embeddings.npz --device cpu
CUDA_VISIBLE_DEVICES="" python scripts/replacement_diagnostic.py $TFIDF_RUNS --cache-dir cache \
    --summary-embeddings results/replacement_diagnostic/bbc_tfidf_summary_embeddings.npz \
    --out-dir results/replacement_diagnostic/bbc_tfidf --work-dir results/replacement_diagnostic/bbc_tfidf_work
# Commit only these (not examples.md, which quotes article text; not iterations.csv):
OUT=results/replacement_diagnostic/bbc_tfidf
for f in verification.csv verification.md mechanism.csv steps.csv outcome.csv tables.md; do
  cp $OUT/replacement_diagnostic_$f docs/results/bbc_tfidf_replacement_$f
done
```

The script ends with a `KeyError` in the figure step (Section 7) after writing all tables and CSVs. The dense replay is in Section 7.
Tests: `tests/test_bbc_report.py`.
