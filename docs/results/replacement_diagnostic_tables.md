### Step 1 (shared step-0 partition; replayed from the summary embeddings): means over seeds

Objective (J_before; relative jump at replacement; relative change at block end vs before and vs the no-replacement counterfactual; iterations run)

| model | j_before | jump_rel | end_vs_before_rel | end_vs_cf_rel | n_iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 10,721 | 0.721 | -0.00252 | -0.00168 | 20 |
| qwen3-0.6b | 10,721 | 0.777 | 0.00261 | 0.00345 | 20 |
| qwen3-1.7b | 10,721 | 0.95 | 0.000989 | 0.00183 | 20 |
| qwen3-4b | 10,721 | 0.981 | 0.0132 | 0.014 | 20 |

Centroid movement (per-cluster medians in pre RMS radii, cosine, share whose nearest pre center is its own, distance from summary to block-end center in radii, distance from block-end center to the pre center in radii, fraction by which the block shortens the summary-to-center distance)

| model | drift_median | cosine_median | own_nearest_share | to_end_radii_median | end_vs_pre_radii_median | pulled_back_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.891 | 0.802 | 0.997 | 0.889 | 0.0409 | -3.64e-06 |
| qwen3-0.6b | 0.965 | 0.754 | 0.825 | 0.964 | 0.161 | 0.000184 |
| qwen3-1.7b | 1.05 | 0.708 | 0.817 | 1.06 | 0.114 | -0.000253 |
| qwen3-4b | 1.08 | 0.668 | 0.619 | 1.09 | 0.333 | -0.000238 |

Partition change (fraction of points moved at the replacement and over the block; ARI and mean cluster Jaccard against the pre partition)

| model | frac_moved_replacement | frac_moved_end | ari_replacement_vs_pre | ari_end_vs_pre | jaccard_end_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.253 | 0.0904 | 0.57 | 0.852 | 0.859 |
| qwen3-0.6b | 0.467 | 0.331 | 0.388 | 0.686 | 0.599 |
| qwen3-1.7b | 0.468 | 0.287 | 0.371 | 0.708 | 0.652 |
| qwen3-4b | 0.561 | 0.448 | 0.317 | 0.652 | 0.473 |

Movers at the replacement (total, counts toward/away from the gold majority and ending in their own gold majority, each beside the mean of a shuffled-destination baseline; share of all points in their own cluster's majority; move rates by margin and flag)

| model | n_moved_replacement | toward_replacement | toward_shuffled | away_replacement | away_shuffled | ends_in_own_majority_replacement | ends_in_own_majority_shuffled | share_in_own_majority_pre | move_rate_boundary | move_rate_core | move_rate_flagged | move_rate_unflagged | share_movers_from_flagged | share_points_in_flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 5,698 | 603 | 19.3 | 2,141 | 2,249 | 724 | 32.7 | 0.578 | 0.569 | 0.148 | n/a | 0.253 | 0 | 0 |
| qwen3-0.6b | 10,502 | 694 | 26.5 | 4,666 | 4,875 | 937 | 61.5 | 0.578 | 0.731 | 0.379 | 0.849 | 0.427 | 0.172 | 0.0953 |
| qwen3-1.7b | 10,520 | 731 | 28.2 | 4,487 | 4,711 | 985 | 57.5 | 0.578 | 0.747 | 0.374 | 0.725 | 0.441 | 0.145 | 0.0933 |
| qwen3-4b | 12,620 | 723 | 30.5 | 5,707 | 5,931 | 978 | 60.2 | 0.578 | 0.802 | 0.481 | 0.904 | 0.364 | 0.586 | 0.364 |

Duplicates and collapsed centroids (clusters per step)

| model | n_dup_clusters | n_meta_clusters | n_empty_replacement | n_empty_end | n_reseeded |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0 | 0 | 0 | 0 | 0 |
| qwen3-0.6b | 3 | 13.6 | 1.2 | 0 | 1.2 |
| qwen3-1.7b | 0.8 | 13.6 | 0.6 | 0 | 0.6 |
| qwen3-4b | 18.8 | 50.2 | 10.8 | 0 | 10.8 |

### All five steps pooled (replayed; from step 2 on the previous block is replayed too): means over seeds

Objective (J_before; relative jump at replacement; relative change at block end vs before and vs the no-replacement counterfactual; iterations run)

| model | j_before | jump_rel | end_vs_before_rel | end_vs_cf_rel | n_iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 10,685 | 0.727 | -0.00102 | -0.000778 | 19.4 |
| qwen3-0.6b | 10,725 | 0.78 | 0.00012 | 0.00133 | 20 |
| qwen3-1.7b | 10,713 | 0.964 | -0.000276 | 0.000776 | 20 |
| qwen3-4b | 10,842 | 0.979 | 0.00354 | 0.0057 | 20 |

Centroid movement (per-cluster medians in pre RMS radii, cosine, share whose nearest pre center is its own, distance from summary to block-end center in radii, distance from block-end center to the pre center in radii, fraction by which the block shortens the summary-to-center distance)

| model | drift_median | cosine_median | own_nearest_share | to_end_radii_median | end_vs_pre_radii_median | pulled_back_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.886 | 0.804 | 0.998 | 0.888 | 0.0228 | -7.29e-07 |
| qwen3-0.6b | 0.956 | 0.764 | 0.843 | 0.958 | 0.0987 | 6.25e-05 |
| qwen3-1.7b | 1.05 | 0.709 | 0.813 | 1.06 | 0.0758 | -0.000131 |
| qwen3-4b | 1.09 | 0.662 | 0.606 | 1.1 | 0.284 | -0.000282 |

Partition change (fraction of points moved at the replacement and over the block; ARI and mean cluster Jaccard against the pre partition)

| model | frac_moved_replacement | frac_moved_end | ari_replacement_vs_pre | ari_end_vs_pre | jaccard_end_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.236 | 0.0456 | 0.589 | 0.921 | 0.926 |
| qwen3-0.6b | 0.438 | 0.277 | 0.405 | 0.731 | 0.665 |
| qwen3-1.7b | 0.444 | 0.244 | 0.383 | 0.747 | 0.708 |
| qwen3-4b | 0.537 | 0.414 | 0.336 | 0.69 | 0.489 |

Movers at the replacement (total, counts toward/away from the gold majority and ending in their own gold majority, each beside the mean of a shuffled-destination baseline; share of all points in their own cluster's majority; move rates by margin and flag)

| model | n_moved_replacement | toward_replacement | toward_shuffled | away_replacement | away_shuffled | ends_in_own_majority_replacement | ends_in_own_majority_shuffled | share_in_own_majority_pre | move_rate_boundary | move_rate_core | move_rate_flagged | move_rate_unflagged | share_movers_from_flagged | share_points_in_flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 5,314 | 552 | 17.4 | 2,015 | 2,088 | 637 | 29.8 | 0.584 | 0.542 | 0.134 | n/a | 0.236 | 0 | 0 |
| qwen3-0.6b | 9,862 | 648 | 26.7 | 4,422 | 4,547 | 804 | 58.2 | 0.571 | 0.715 | 0.346 | 0.823 | 0.396 | 0.187 | 0.1 |
| qwen3-1.7b | 9,994 | 690 | 25.6 | 4,394 | 4,575 | 900 | 54.2 | 0.592 | 0.73 | 0.349 | 0.739 | 0.413 | 0.156 | 0.0927 |
| qwen3-4b | 12,088 | 660 | 29.5 | 5,265 | 5,470 | 896 | 59.6 | 0.556 | 0.785 | 0.454 | 0.886 | 0.333 | 0.608 | 0.369 |

Duplicates and collapsed centroids (clusters per step)

| model | n_dup_clusters | n_meta_clusters | n_empty_replacement | n_empty_end | n_reseeded |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0 | 0 | 0 | 0 | 0 |
| qwen3-0.6b | 2 | 12.6 | 0.84 | 0 | 0.84 |
| qwen3-1.7b | 0.92 | 12.6 | 0.48 | 0 | 0.48 |
| qwen3-4b | 16.9 | 51.6 | 10.5 | 0 | 10.5 |

### Outcome

**Step 1 headline.** Step 0 replays exactly, so at step 1 the pre state and the plain-Lloyd counterfactual are exact and `assignments_evolution[1]` is the real post-block partition. Mean over seeds of (score of the logged partition after the block) minus (score after plain Lloyd from the step-0 partition, same 20 iterations); `seeds_below` counts seeds with a lower ACC than the counterfactual; `d_acc_replayed` is the same from the replay, as a check.

| model | d_acc | d_nmi | d_ari | seeds_below | d_acc_replayed | replay_vs_logged_points |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| lsa | +0.0051 | +0.0019 | +0.0048 | 0/5 | +0.0051 | 0 |
| qwen3-0.6b | +0.0016 | +0.0022 | +0.0048 | 3/5 | +0.0016 | 0 |
| qwen3-1.7b | +0.0106 | +0.0076 | +0.0177 | 0/5 | +0.0106 | 0 |
| qwen3-4b | -0.0159 | +0.0011 | -0.0012 | 5/5 | -0.0159 | 0 |

Per seed, step 1, logged partition minus counterfactual.

| model | seed | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: |
| lsa | 0 | +0.0047 | +0.0032 | +0.0085 |
| lsa | 1 | +0.0040 | +0.0014 | +0.0038 |
| lsa | 2 | +0.0144 | +0.0045 | +0.0110 |
| lsa | 3 | +0.0016 | +0.0003 | +0.0004 |
| lsa | 4 | +0.0007 | +0.0000 | +0.0004 |
| qwen3-0.6b | 0 | +0.0162 | +0.0032 | +0.0099 |
| qwen3-0.6b | 1 | -0.0077 | +0.0011 | -0.0012 |
| qwen3-0.6b | 2 | -0.0055 | -0.0023 | -0.0005 |
| qwen3-0.6b | 3 | -0.0102 | -0.0022 | -0.0045 |
| qwen3-0.6b | 4 | +0.0149 | +0.0113 | +0.0201 |
| qwen3-1.7b | 0 | +0.0044 | +0.0066 | +0.0107 |
| qwen3-1.7b | 1 | +0.0117 | +0.0086 | +0.0191 |
| qwen3-1.7b | 2 | +0.0151 | +0.0092 | +0.0261 |
| qwen3-1.7b | 3 | +0.0140 | +0.0064 | +0.0187 |
| qwen3-1.7b | 4 | +0.0078 | +0.0070 | +0.0139 |
| qwen3-4b | 0 | -0.0104 | +0.0035 | +0.0001 |
| qwen3-4b | 1 | -0.0264 | +0.0005 | -0.0017 |
| qwen3-4b | 2 | -0.0128 | +0.0020 | -0.0000 |
| qwen3-4b | 3 | -0.0212 | -0.0041 | -0.0090 |
| qwen3-4b | 4 | -0.0087 | +0.0035 | +0.0048 |

**Step 1, both sides run to sklearn's own stop** (cap 300 iterations; replayed from the summary embeddings). A fixed 20-iteration budget can leave the replacement block less converged than the counterfactual (which starts converged), so this separates a worse end state from less convergence. `n_iter` and `n_iter_cf` are mean iterations to the stop, `capped*` the runs that hit the cap, `j_end_rel` the mean relative objective of the replacement run over the counterfactual's, and `d_*` the score differences (replacement minus counterfactual).

| model | n_iter | n_iter_cf | capped | capped_cf | j_end_rel | d_acc | d_nmi | d_ari | seeds_below |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lsa | 31.8 | 15.2 | 0 | 0 | -0.0019 | +0.0050 | +0.0019 | +0.0049 | 0/5 |
| qwen3-0.6b | 55.6 | 15.2 | 0 | 0 | +0.0019 | +0.0029 | +0.0031 | +0.0057 | 3/5 |
| qwen3-1.7b | 48.4 | 15.2 | 0 | 0 | +0.0008 | +0.0121 | +0.0088 | +0.0190 | 0/5 |
| qwen3-4b | 52.8 | 15.2 | 0 | 0 | +0.0110 | -0.0138 | +0.0026 | +0.0015 | 5/5 |

**Per step, replayed.** From step 2 on every row starts from a replayed previous block.

Mean over seeds of (replayed score after the block) minus (score after plain Lloyd from the replayed pre-replacement state, same iterations).

| model | step | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: |
| lsa | 1 | +0.0051 | +0.0019 | +0.0048 |
| lsa | 2 | +0.0041 | +0.0023 | +0.0056 |
| lsa | 3 | -0.0005 | -0.0004 | -0.0007 |
| lsa | 4 | -0.0012 | -0.0006 | -0.0009 |
| lsa | 5 | +0.0012 | +0.0008 | +0.0016 |
| qwen3-0.6b | 1 | +0.0016 | +0.0022 | +0.0048 |
| qwen3-0.6b | 2 | -0.0060 | -0.0014 | -0.0040 |
| qwen3-0.6b | 3 | -0.0080 | -0.0051 | -0.0057 |
| qwen3-0.6b | 4 | -0.0036 | -0.0017 | -0.0027 |
| qwen3-0.6b | 5 | -0.0030 | +0.0003 | -0.0009 |
| qwen3-1.7b | 1 | +0.0106 | +0.0076 | +0.0177 |
| qwen3-1.7b | 2 | +0.0067 | +0.0028 | +0.0036 |
| qwen3-1.7b | 3 | +0.0046 | +0.0017 | +0.0047 |
| qwen3-1.7b | 4 | -0.0082 | -0.0024 | -0.0069 |
| qwen3-1.7b | 5 | +0.0011 | -0.0007 | +0.0003 |
| qwen3-4b | 1 | -0.0159 | +0.0011 | -0.0012 |
| qwen3-4b | 2 | -0.0120 | -0.0076 | -0.0096 |
| qwen3-4b | 3 | -0.0123 | -0.0053 | -0.0088 |
| qwen3-4b | 4 | -0.0040 | -0.0017 | -0.0029 |
| qwen3-4b | 5 | -0.0074 | -0.0039 | -0.0070 |

Final scores against k-means from the same initialization (logged values; mean over seeds).

| model | acc | kmeans_acc | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.560 | 0.551 | +0.0089 | +0.0043 | +0.0104 |
| qwen3-0.6b | 0.541 | 0.551 | -0.0104 | -0.0009 | -0.0014 |
| qwen3-1.7b | 0.570 | 0.551 | +0.0184 | +0.0123 | +0.0224 |
| qwen3-4b | 0.511 | 0.551 | -0.0402 | -0.0086 | -0.0181 |

Spearman correlation across all (model, seed) pairs at step 1; a descriptive number, not a test. The pooled correlation mixes between-model and within-model variation.

| step-1 measure | outcome | spearman (n=20) |
| --- | --- | ---: |
| jump_rel | d_step_acc | -0.32 |
| jump_rel | d_final_acc | -0.44 |
| jump_rel | d_final_nmi | -0.36 |
| drift_median | d_step_acc | -0.44 |
| drift_median | d_final_acc | -0.35 |
| drift_median | d_final_nmi | -0.29 |
| n_empty_replacement | d_step_acc | -0.51 |
| n_empty_replacement | d_final_acc | -0.77 |
| n_empty_replacement | d_final_nmi | -0.68 |
| frac_moved_replacement | d_step_acc | -0.55 |
| frac_moved_replacement | d_final_acc | -0.63 |
| frac_moved_replacement | d_final_nmi | -0.52 |

### Do the displayed (final-step) summaries describe the final partition?

Share of final summaries that are out of date: Jaccard between the members each summary was written from and the final members below the threshold, or its embedding nearest to another cluster's final center; `stale@t` is either condition. All seeds pooled; 0.5 is the documented convention, 0.3 and 0.7 show sensitivity.

| model | summaries | mean_jaccard | jaccard<0.3 | jaccard<0.5 | jaccard<0.7 | nearest_not_own | stale@0.3 | stale@0.5 | stale@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 750 | 0.965 | 0.00133 | 0.00533 | 0.00933 | 0 | 0.00133 | 0.00533 | 0.00933 |
| qwen3-0.6b | 750 | 0.699 | 0.18 | 0.244 | 0.351 | 0.155 | 0.236 | 0.287 | 0.387 |
| qwen3-1.7b | 750 | 0.711 | 0.159 | 0.231 | 0.325 | 0.209 | 0.256 | 0.301 | 0.384 |
| qwen3-4b | 750 | 0.484 | 0.416 | 0.484 | 0.56 | 0.423 | 0.496 | 0.547 | 0.597 |

Share stale@0.5 by summary step (summary of step j against the partition after its own block).

| model | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.060 | 0.035 | 0.013 | 0.008 | 0.005 |
| qwen3-0.6b | 0.381 | 0.345 | 0.301 | 0.283 | 0.287 |
| qwen3-1.7b | 0.351 | 0.319 | 0.291 | 0.292 | 0.301 |
| qwen3-4b | 0.525 | 0.517 | 0.552 | 0.519 | 0.547 |
