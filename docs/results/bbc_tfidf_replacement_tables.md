### Step 1 (shared step-0 partition; replayed from the summary embeddings): means over seeds

Objective (J_before; relative jump at replacement; relative change at block end vs before and vs the no-replacement counterfactual; iterations run)

| model | j_before | jump_rel | end_vs_before_rel | end_vs_cf_rel | n_iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 430 | 0.923 | -0.00154 | -0.00154 | 7.2 |
| bbc_tfidf_qwen3-0.6b | 430 | 1.13 | -0.0019 | -0.0019 | 10.8 |
| bbc_tfidf_qwen3-1.7b | 430 | 1.1 | -0.00194 | -0.00194 | 9.4 |
| bbc_tfidf_qwen3-4b | 430 | 1.13 | -0.00183 | -0.00183 | 9.6 |

Centroid movement (per-cluster medians in pre RMS radii, cosine, share whose nearest pre center is its own, distance from summary to block-end center in radii, distance from block-end center to the pre center in radii, fraction by which the block shortens the summary-to-center distance)

| model | drift_median | cosine_median | own_nearest_share | to_end_radii_median | end_vs_pre_radii_median | pulled_back_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0.975 | 0.426 | 1 | 0.977 | 0.0474 | 0.00248 |
| bbc_tfidf_qwen3-0.6b | 1.08 | 0.169 | 0.84 | 1.08 | 0.0639 | 0.000811 |
| bbc_tfidf_qwen3-1.7b | 1.06 | 0.189 | 0.84 | 1.06 | 0.0726 | 0.000448 |
| bbc_tfidf_qwen3-4b | 1.08 | 0.195 | 0.76 | 1.08 | 0.0531 | 0.000886 |

Partition change (fraction of points moved at the replacement and over the block; ARI and mean cluster Jaccard against the pre partition)

| model | frac_moved_replacement | frac_moved_end | ari_replacement_vs_pre | ari_end_vs_pre | jaccard_end_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0.262 | 0.114 | 0.436 | 0.754 | 0.813 |
| bbc_tfidf_qwen3-0.6b | 0.502 | 0.142 | 0.161 | 0.712 | 0.791 |
| bbc_tfidf_qwen3-1.7b | 0.459 | 0.156 | 0.21 | 0.714 | 0.774 |
| bbc_tfidf_qwen3-4b | 0.472 | 0.122 | 0.191 | 0.735 | 0.807 |

Movers at the replacement (total, counts toward/away from the gold majority and ending in their own gold majority, each beside the mean of a shuffled-destination baseline; share of all points in their own cluster's majority; move rates by margin and flag)

| model | n_moved_replacement | toward_replacement | toward_shuffled | away_replacement | away_shuffled | ends_in_own_majority_replacement | ends_in_own_majority_shuffled | share_in_own_majority_pre | move_rate_boundary | move_rate_core | move_rate_flagged | move_rate_unflagged | share_movers_from_flagged | share_points_in_flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 131 | 16.8 | 10.1 | 78.2 | 64.4 | 17.8 | 25 | 0.79 | 0.57 | 0.16 | n/a | 0.262 | 0 | 0 |
| bbc_tfidf_qwen3-0.6b | 251 | 27.2 | 17.8 | 171 | 146 | 29 | 44.2 | 0.79 | 0.693 | 0.439 | n/a | 0.502 | 0 | 0 |
| bbc_tfidf_qwen3-1.7b | 230 | 32 | 16 | 148 | 125 | 34.8 | 42 | 0.79 | 0.656 | 0.394 | n/a | 0.459 | 0 | 0 |
| bbc_tfidf_qwen3-4b | 236 | 32 | 15.6 | 159 | 130 | 32.6 | 45.2 | 0.79 | 0.666 | 0.407 | n/a | 0.472 | 0 | 0 |

Duplicates and collapsed centroids (clusters per step)

| model | n_dup_clusters | n_meta_clusters | n_empty_replacement | n_empty_end | n_reseeded |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0 | 0 | 0 | 0 | 0 |
| bbc_tfidf_qwen3-0.6b | 0 | 0 | 0 | 0 | 0 |
| bbc_tfidf_qwen3-1.7b | 0 | 0 | 0 | 0 | 0 |
| bbc_tfidf_qwen3-4b | 0 | 0 | 0 | 0 | 0 |

### All five steps pooled (replayed; from step 2 on the previous block is replayed too): means over seeds

Objective (J_before; relative jump at replacement; relative change at block end vs before and vs the no-replacement counterfactual; iterations run)

| model | j_before | jump_rel | end_vs_before_rel | end_vs_cf_rel | n_iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 429 | 0.929 | -0.000407 | -0.000407 | 8.96 |
| bbc_tfidf_qwen3-0.6b | 429 | 1.13 | -0.000434 | -0.000433 | 10.6 |
| bbc_tfidf_qwen3-1.7b | 429 | 1.11 | -0.000258 | -0.000258 | 9.68 |
| bbc_tfidf_qwen3-4b | 429 | 1.12 | -0.00045 | -0.00045 | 10.6 |

Centroid movement (per-cluster medians in pre RMS radii, cosine, share whose nearest pre center is its own, distance from summary to block-end center in radii, distance from block-end center to the pre center in radii, fraction by which the block shortens the summary-to-center distance)

| model | drift_median | cosine_median | own_nearest_share | to_end_radii_median | end_vs_pre_radii_median | pulled_back_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0.973 | 0.433 | 1 | 0.974 | 0.0355 | -0.000125 |
| bbc_tfidf_qwen3-0.6b | 1.08 | 0.177 | 0.856 | 1.08 | 0.0568 | 0.000496 |
| bbc_tfidf_qwen3-1.7b | 1.07 | 0.215 | 0.92 | 1.07 | 0.0544 | 0.00011 |
| bbc_tfidf_qwen3-4b | 1.07 | 0.179 | 0.752 | 1.07 | 0.0506 | 0.000555 |

Partition change (fraction of points moved at the replacement and over the block; ARI and mean cluster Jaccard against the pre partition)

| model | frac_moved_replacement | frac_moved_end | ari_replacement_vs_pre | ari_end_vs_pre | jaccard_end_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0.25 | 0.0566 | 0.459 | 0.862 | 0.905 |
| bbc_tfidf_qwen3-0.6b | 0.469 | 0.128 | 0.19 | 0.772 | 0.825 |
| bbc_tfidf_qwen3-1.7b | 0.427 | 0.0918 | 0.233 | 0.793 | 0.851 |
| bbc_tfidf_qwen3-4b | 0.464 | 0.0894 | 0.209 | 0.782 | 0.856 |

Movers at the replacement (total, counts toward/away from the gold majority and ending in their own gold majority, each beside the mean of a shuffled-destination baseline; share of all points in their own cluster's majority; move rates by margin and flag)

| model | n_moved_replacement | toward_replacement | toward_shuffled | away_replacement | away_shuffled | ends_in_own_majority_replacement | ends_in_own_majority_shuffled | share_in_own_majority_pre | move_rate_boundary | move_rate_core | move_rate_flagged | move_rate_unflagged | share_movers_from_flagged | share_points_in_flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 125 | 11.6 | 7.28 | 84.2 | 68.3 | 11.8 | 23.4 | 0.821 | 0.497 | 0.167 | n/a | 0.25 | 0 | 0 |
| bbc_tfidf_qwen3-0.6b | 234 | 24.9 | 13.7 | 171 | 144 | 25.2 | 41.8 | 0.825 | 0.636 | 0.413 | n/a | 0.469 | 0 | 0 |
| bbc_tfidf_qwen3-1.7b | 213 | 25.3 | 14.1 | 154 | 130 | 25.9 | 38.8 | 0.845 | 0.63 | 0.359 | n/a | 0.427 | 0 | 0 |
| bbc_tfidf_qwen3-4b | 232 | 28.6 | 13.2 | 171 | 141 | 28.7 | 42.9 | 0.839 | 0.63 | 0.409 | n/a | 0.464 | 0 | 0 |

Duplicates and collapsed centroids (clusters per step)

| model | n_dup_clusters | n_meta_clusters | n_empty_replacement | n_empty_end | n_reseeded |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0 | 0 | 0 | 0 | 0 |
| bbc_tfidf_qwen3-0.6b | 0 | 0 | 0 | 0 | 0 |
| bbc_tfidf_qwen3-1.7b | 0 | 0 | 0 | 0 | 0 |
| bbc_tfidf_qwen3-4b | 0 | 0 | 0 | 0 | 0 |

### Outcome

**Step 1 headline.** Step 0 replays exactly, so at step 1 the pre state and the plain-Lloyd counterfactual are exact and `assignments_evolution[1]` is the real post-block partition. Mean over seeds of (score of the logged partition after the block) minus (score after plain Lloyd from the step-0 partition, same 20 iterations); `seeds_below` counts seeds with a lower ACC than the counterfactual; `d_acc_replayed` is the same from the replay, as a check.

| model | d_acc | d_nmi | d_ari | seeds_below | d_acc_replayed | replay_vs_logged_points |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| bbc_tfidf_lsa | +0.0204 | -0.0147 | +0.0172 | 1/5 | +0.0204 | 0 |
| bbc_tfidf_qwen3-0.6b | +0.0220 | -0.0024 | +0.0026 | 1/5 | +0.0220 | 0 |
| bbc_tfidf_qwen3-1.7b | +0.0572 | +0.0294 | +0.0590 | 0/5 | +0.0572 | 0 |
| bbc_tfidf_qwen3-4b | +0.0464 | +0.0157 | +0.0439 | 1/5 | +0.0464 | 0 |

Per seed, step 1, logged partition minus counterfactual.

| model | seed | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0 | +0.0040 | -0.0015 | +0.0131 |
| bbc_tfidf_lsa | 1 | +0.0820 | +0.0199 | +0.0682 |
| bbc_tfidf_lsa | 2 | +0.0100 | -0.0291 | +0.0021 |
| bbc_tfidf_lsa | 3 | +0.0160 | -0.0319 | +0.0272 |
| bbc_tfidf_lsa | 4 | -0.0100 | -0.0308 | -0.0246 |
| bbc_tfidf_qwen3-0.6b | 0 | -0.0360 | -0.0382 | -0.0827 |
| bbc_tfidf_qwen3-0.6b | 1 | +0.1020 | +0.0685 | +0.1183 |
| bbc_tfidf_qwen3-0.6b | 2 | +0.0380 | -0.0175 | +0.0030 |
| bbc_tfidf_qwen3-0.6b | 3 | +0.0040 | -0.0217 | -0.0259 |
| bbc_tfidf_qwen3-0.6b | 4 | +0.0020 | -0.0029 | +0.0004 |
| bbc_tfidf_qwen3-1.7b | 0 | +0.0060 | +0.0022 | +0.0020 |
| bbc_tfidf_qwen3-1.7b | 1 | +0.0480 | -0.0222 | -0.0163 |
| bbc_tfidf_qwen3-1.7b | 2 | +0.1900 | +0.1465 | +0.2294 |
| bbc_tfidf_qwen3-1.7b | 3 | +0.0260 | -0.0058 | +0.0423 |
| bbc_tfidf_qwen3-1.7b | 4 | +0.0160 | +0.0262 | +0.0376 |
| bbc_tfidf_qwen3-4b | 0 | -0.0320 | -0.0368 | -0.0722 |
| bbc_tfidf_qwen3-4b | 1 | +0.0780 | +0.0350 | +0.0562 |
| bbc_tfidf_qwen3-4b | 2 | +0.1020 | +0.0159 | +0.0693 |
| bbc_tfidf_qwen3-4b | 3 | +0.0480 | +0.0175 | +0.0871 |
| bbc_tfidf_qwen3-4b | 4 | +0.0360 | +0.0469 | +0.0791 |

**Step 1, both sides run to sklearn's own stop** (cap 300 iterations; replayed from the summary embeddings). A fixed 20-iteration budget can leave the replacement block less converged than the counterfactual (which starts converged), so this separates a worse end state from less convergence. `n_iter` and `n_iter_cf` are mean iterations to the stop, `capped*` the runs that hit the cap, `j_end_rel` the mean relative objective of the replacement run over the counterfactual's, and `d_*` the score differences (replacement minus counterfactual).

| model | n_iter | n_iter_cf | capped | capped_cf | j_end_rel | d_acc | d_nmi | d_ari | seeds_below |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bbc_tfidf_lsa | 7.2 | 1 | 0 | 0 | -0.0015 | +0.0204 | -0.0147 | +0.0172 | 1/5 |
| bbc_tfidf_qwen3-0.6b | 10.8 | 1 | 0 | 0 | -0.0019 | +0.0220 | -0.0024 | +0.0026 | 1/5 |
| bbc_tfidf_qwen3-1.7b | 9.4 | 1 | 0 | 0 | -0.0019 | +0.0572 | +0.0294 | +0.0590 | 0/5 |
| bbc_tfidf_qwen3-4b | 9.6 | 1 | 0 | 0 | -0.0018 | +0.0464 | +0.0157 | +0.0439 | 1/5 |

**Per step, replayed.** From step 2 on every row starts from a replayed previous block.

Mean over seeds of (replayed score after the block) minus (score after plain Lloyd from the replayed pre-replacement state, same iterations).

| model | step | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 1 | +0.0204 | -0.0147 | +0.0172 |
| bbc_tfidf_lsa | 2 | +0.0208 | +0.0213 | +0.0243 |
| bbc_tfidf_lsa | 3 | +0.0120 | +0.0161 | +0.0153 |
| bbc_tfidf_lsa | 4 | -0.0064 | -0.0074 | -0.0164 |
| bbc_tfidf_lsa | 5 | +0.0040 | +0.0048 | +0.0088 |
| bbc_tfidf_qwen3-0.6b | 1 | +0.0220 | -0.0024 | +0.0026 |
| bbc_tfidf_qwen3-0.6b | 2 | +0.0156 | +0.0048 | +0.0122 |
| bbc_tfidf_qwen3-0.6b | 3 | +0.0224 | +0.0252 | +0.0535 |
| bbc_tfidf_qwen3-0.6b | 4 | -0.0008 | +0.0131 | +0.0007 |
| bbc_tfidf_qwen3-0.6b | 5 | +0.0036 | -0.0026 | +0.0068 |
| bbc_tfidf_qwen3-1.7b | 1 | +0.0572 | +0.0294 | +0.0590 |
| bbc_tfidf_qwen3-1.7b | 2 | +0.0164 | +0.0186 | +0.0408 |
| bbc_tfidf_qwen3-1.7b | 3 | +0.0012 | +0.0029 | +0.0021 |
| bbc_tfidf_qwen3-1.7b | 4 | -0.0008 | -0.0045 | -0.0053 |
| bbc_tfidf_qwen3-1.7b | 5 | +0.0108 | +0.0260 | +0.0285 |
| bbc_tfidf_qwen3-4b | 1 | +0.0464 | +0.0157 | +0.0439 |
| bbc_tfidf_qwen3-4b | 2 | +0.0260 | +0.0420 | +0.0487 |
| bbc_tfidf_qwen3-4b | 3 | +0.0004 | -0.0025 | +0.0047 |
| bbc_tfidf_qwen3-4b | 4 | -0.0132 | -0.0176 | -0.0272 |
| bbc_tfidf_qwen3-4b | 5 | +0.0064 | +0.0056 | +0.0077 |

Final scores against k-means from the same initialization (logged values; mean over seeds).

| model | acc | kmeans_acc | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0.839 | 0.788 | +0.0508 | +0.0200 | +0.0492 |
| bbc_tfidf_qwen3-0.6b | 0.851 | 0.788 | +0.0628 | +0.0381 | +0.0758 |
| bbc_tfidf_qwen3-1.7b | 0.873 | 0.788 | +0.0848 | +0.0724 | +0.1251 |
| bbc_tfidf_qwen3-4b | 0.854 | 0.788 | +0.0660 | +0.0432 | +0.0778 |

Spearman correlation across all (model, seed) pairs at step 1; a descriptive number, not a test. The pooled correlation mixes between-model and within-model variation.

| step-1 measure | outcome | spearman (n=20) |
| --- | --- | ---: |
| jump_rel | d_step_acc | +0.13 |
| jump_rel | d_final_acc | -0.00 |
| jump_rel | d_final_nmi | +0.06 |
| drift_median | d_step_acc | +0.07 |
| drift_median | d_final_acc | -0.06 |
| drift_median | d_final_nmi | +0.05 |
| n_empty_replacement | d_step_acc | n/a |
| n_empty_replacement | d_final_acc | n/a |
| n_empty_replacement | d_final_nmi | n/a |
| frac_moved_replacement | d_step_acc | +0.20 |
| frac_moved_replacement | d_final_acc | +0.20 |
| frac_moved_replacement | d_final_nmi | +0.32 |

### Do the displayed (final-step) summaries describe the final partition?

Share of final summaries that are out of date: Jaccard between the members each summary was written from and the final members below the threshold, or its embedding nearest to another cluster's final center; `stale@t` is either condition. All seeds pooled; 0.5 is the documented convention, 0.3 and 0.7 show sensitivity.

| model | summaries | mean_jaccard | jaccard<0.3 | jaccard<0.5 | jaccard<0.7 | nearest_not_own | stale@0.3 | stale@0.5 | stale@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 25 | 0.944 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bbc_tfidf_qwen3-0.6b | 25 | 0.886 | 0 | 0 | 0.04 | 0.2 | 0.2 | 0.2 | 0.2 |
| bbc_tfidf_qwen3-1.7b | 25 | 0.857 | 0 | 0 | 0.04 | 0.04 | 0.04 | 0.04 | 0.08 |
| bbc_tfidf_qwen3-4b | 25 | 0.918 | 0 | 0 | 0 | 0.16 | 0.16 | 0.16 | 0.16 |

Share stale@0.5 by summary step (summary of step j against the partition after its own block).

| model | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbc_tfidf_lsa | 0.120 | 0.000 | 0.000 | 0.000 | 0.000 |
| bbc_tfidf_qwen3-0.6b | 0.240 | 0.120 | 0.120 | 0.200 | 0.200 |
| bbc_tfidf_qwen3-1.7b | 0.320 | 0.040 | 0.000 | 0.120 | 0.040 |
| bbc_tfidf_qwen3-4b | 0.320 | 0.240 | 0.240 | 0.320 | 0.160 |
