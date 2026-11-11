### Step 1 (shared step-0 partition): means over seeds

Objective (J_before; relative jump at replacement; relative change at block end vs before and vs the no-replacement counterfactual; iterations run)

| model | j_before | jump_rel | end_vs_before_rel | end_vs_cf_rel | n_iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 10,721 | 0.721 | -0.00252 | -0.00168 | 20 |
| qwen3-0.6b | 10,721 | 0.777 | 0.00238 | 0.00322 | 20 |
| qwen3-1.7b | 10,721 | 0.95 | 0.000989 | 0.00183 | 20 |
| qwen3-4b | 10,721 | 0.981 | 0.0131 | 0.0139 | 20 |

Centroid movement (per-cluster medians in pre RMS radii, cosine, share whose nearest pre center is its own, distance from summary to block-end center in radii, distance from block-end center to the pre center in radii, fraction by which the block shortens the summary-to-center distance)

| model | drift_median | cosine_median | own_nearest_share | to_end_radii_median | end_vs_pre_radii_median | pulled_back_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.891 | 0.802 | 0.997 | 0.889 | 0.0409 | -3.64e-06 |
| qwen3-0.6b | 0.965 | 0.754 | 0.825 | 0.965 | 0.167 | 0.000198 |
| qwen3-1.7b | 1.05 | 0.708 | 0.817 | 1.06 | 0.114 | -0.000253 |
| qwen3-4b | 1.08 | 0.668 | 0.619 | 1.09 | 0.333 | -0.000303 |

Partition change (fraction of points moved at the replacement and over the block; ARI and mean cluster Jaccard against the pre partition)

| model | frac_moved_replacement | frac_moved_end | ari_replacement_vs_pre | ari_end_vs_pre | jaccard_end_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.253 | 0.0904 | 0.57 | 0.852 | 0.859 |
| qwen3-0.6b | 0.467 | 0.334 | 0.388 | 0.688 | 0.597 |
| qwen3-1.7b | 0.468 | 0.287 | 0.371 | 0.708 | 0.652 |
| qwen3-4b | 0.561 | 0.447 | 0.317 | 0.653 | 0.473 |

Movers at the replacement (counts toward/away from the gold majority, ends in own majority, total, move rates by margin and flag)

| model | toward_replacement | away_replacement | ends_in_own_majority_replacement | n_moved_replacement | move_rate_boundary | move_rate_core | move_rate_flagged | move_rate_unflagged | share_movers_from_flagged | share_points_in_flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 603 | 2,141 | 724 | 5,698 | 0.569 | 0.148 | n/a | 0.253 | 0 | 0 |
| qwen3-0.6b | 694 | 4,666 | 937 | 10,502 | 0.731 | 0.379 | 0.849 | 0.427 | 0.172 | 0.0953 |
| qwen3-1.7b | 731 | 4,487 | 985 | 10,520 | 0.747 | 0.374 | 0.725 | 0.441 | 0.145 | 0.0933 |
| qwen3-4b | 723 | 5,707 | 978 | 12,620 | 0.802 | 0.481 | 0.904 | 0.364 | 0.586 | 0.364 |

Duplicates and collapsed centroids (clusters per step)

| model | n_dup_clusters | n_meta_clusters | n_empty_replacement | n_empty_end | n_reseeded |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0 | 0 | 0 | 0 | 0 |
| qwen3-0.6b | 3 | 13.6 | 0.8 | 0 | 0.8 |
| qwen3-1.7b | 0.8 | 13.6 | 0.6 | 0 | 0.6 |
| qwen3-4b | 18.8 | 50.2 | 11.2 | 0 | 11.2 |

### All five steps pooled: means over seeds

Objective (J_before; relative jump at replacement; relative change at block end vs before and vs the no-replacement counterfactual; iterations run)

| model | j_before | jump_rel | end_vs_before_rel | end_vs_cf_rel | n_iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 10,685 | 0.727 | -0.00102 | -0.000778 | 19.4 |
| qwen3-0.6b | 10,724 | 0.78 | 0.00012 | 0.00132 | 20 |
| qwen3-1.7b | 10,713 | 0.964 | -0.000357 | 0.000672 | 20 |
| qwen3-4b | 10,839 | 0.979 | 0.00371 | 0.00573 | 20 |

Centroid movement (per-cluster medians in pre RMS radii, cosine, share whose nearest pre center is its own, distance from summary to block-end center in radii, distance from block-end center to the pre center in radii, fraction by which the block shortens the summary-to-center distance)

| model | drift_median | cosine_median | own_nearest_share | to_end_radii_median | end_vs_pre_radii_median | pulled_back_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.886 | 0.804 | 0.998 | 0.888 | 0.0228 | -7.29e-07 |
| qwen3-0.6b | 0.957 | 0.763 | 0.841 | 0.958 | 0.0997 | 0.000133 |
| qwen3-1.7b | 1.05 | 0.709 | 0.812 | 1.06 | 0.0757 | -0.000149 |
| qwen3-4b | 1.09 | 0.659 | 0.601 | 1.1 | 0.292 | -0.00023 |

Partition change (fraction of points moved at the replacement and over the block; ARI and mean cluster Jaccard against the pre partition)

| model | frac_moved_replacement | frac_moved_end | ari_replacement_vs_pre | ari_end_vs_pre | jaccard_end_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.236 | 0.0456 | 0.589 | 0.921 | 0.926 |
| qwen3-0.6b | 0.44 | 0.28 | 0.405 | 0.732 | 0.663 |
| qwen3-1.7b | 0.444 | 0.244 | 0.383 | 0.747 | 0.708 |
| qwen3-4b | 0.54 | 0.417 | 0.335 | 0.691 | 0.487 |

Movers at the replacement (counts toward/away from the gold majority, ends in own majority, total, move rates by margin and flag)

| model | toward_replacement | away_replacement | ends_in_own_majority_replacement | n_moved_replacement | move_rate_boundary | move_rate_core | move_rate_flagged | move_rate_unflagged | share_movers_from_flagged | share_points_in_flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 552 | 2,015 | 638 | 5,313 | 0.542 | 0.134 | n/a | 0.236 | 0 | 0 |
| qwen3-0.6b | 644 | 4,438 | 803 | 9,891 | 0.716 | 0.347 | 0.823 | 0.397 | 0.187 | 0.1 |
| qwen3-1.7b | 689 | 4,402 | 896 | 9,999 | 0.73 | 0.349 | 0.739 | 0.413 | 0.156 | 0.0927 |
| qwen3-4b | 653 | 5,320 | 880 | 12,149 | 0.788 | 0.457 | 0.887 | 0.337 | 0.604 | 0.368 |

Duplicates and collapsed centroids (clusters per step)

| model | n_dup_clusters | n_meta_clusters | n_empty_replacement | n_empty_end | n_reseeded |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0 | 0 | 0 | 0 | 0 |
| qwen3-0.6b | 2 | 12.6 | 0.68 | 0 | 0.68 |
| qwen3-1.7b | 0.92 | 12.6 | 0.48 | 0 | 0.48 |
| qwen3-4b | 16.9 | 51.6 | 10.5 | 0 | 10.5 |

### Outcome per step

Mean over seeds of (score after the block) minus (score after plain Lloyd from the pre-replacement state, same iterations).

| model | step | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: |
| lsa | 1 | +0.0051 | +0.0019 | +0.0048 |
| lsa | 2 | +0.0041 | +0.0023 | +0.0056 |
| lsa | 3 | -0.0005 | -0.0004 | -0.0008 |
| lsa | 4 | -0.0011 | -0.0006 | -0.0009 |
| lsa | 5 | +0.0012 | +0.0008 | +0.0016 |
| qwen3-0.6b | 1 | +0.0029 | +0.0022 | +0.0052 |
| qwen3-0.6b | 2 | -0.0067 | -0.0013 | -0.0042 |
| qwen3-0.6b | 3 | -0.0077 | -0.0049 | -0.0054 |
| qwen3-0.6b | 4 | -0.0037 | -0.0018 | -0.0028 |
| qwen3-0.6b | 5 | -0.0029 | +0.0001 | -0.0006 |
| qwen3-1.7b | 1 | +0.0106 | +0.0076 | +0.0177 |
| qwen3-1.7b | 2 | +0.0067 | +0.0028 | +0.0036 |
| qwen3-1.7b | 3 | +0.0045 | +0.0018 | +0.0046 |
| qwen3-1.7b | 4 | -0.0080 | -0.0024 | -0.0071 |
| qwen3-1.7b | 5 | +0.0023 | -0.0000 | +0.0017 |
| qwen3-4b | 1 | -0.0165 | +0.0009 | -0.0014 |
| qwen3-4b | 2 | -0.0125 | -0.0070 | -0.0098 |
| qwen3-4b | 3 | -0.0105 | -0.0040 | -0.0064 |
| qwen3-4b | 4 | -0.0059 | -0.0029 | -0.0054 |
| qwen3-4b | 5 | -0.0092 | -0.0052 | -0.0096 |

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
| drift_median | d_step_acc | -0.45 |
| drift_median | d_final_acc | -0.35 |
| drift_median | d_final_nmi | -0.29 |
| n_empty_replacement | d_step_acc | -0.58 |
| n_empty_replacement | d_final_acc | -0.73 |
| n_empty_replacement | d_final_nmi | -0.61 |
| frac_moved_replacement | d_step_acc | -0.52 |
| frac_moved_replacement | d_final_acc | -0.63 |
| frac_moved_replacement | d_final_nmi | -0.52 |

### Do the displayed (final-step) summaries describe the final partition?

Share of final summaries that are out of date: Jaccard between the members each summary was written from and the final members below the threshold, or its embedding nearest to another cluster's final center; `stale@t` is either condition. All seeds pooled; 0.5 is the documented convention, 0.3 and 0.7 show sensitivity.

| model | summaries | mean_jaccard | jaccard<0.3 | jaccard<0.5 | jaccard<0.7 | nearest_not_own | stale@0.3 | stale@0.5 | stale@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 750 | 0.965 | 0.00133 | 0.00533 | 0.00933 | 0 | 0.00133 | 0.00533 | 0.00933 |
| qwen3-0.6b | 750 | 0.699 | 0.18 | 0.244 | 0.351 | 0.155 | 0.236 | 0.287 | 0.387 |
| qwen3-1.7b | 750 | 0.711 | 0.161 | 0.231 | 0.325 | 0.212 | 0.259 | 0.303 | 0.384 |
| qwen3-4b | 750 | 0.484 | 0.417 | 0.484 | 0.559 | 0.427 | 0.497 | 0.545 | 0.596 |

Share stale@0.5 by summary step (summary of step j against the partition after its own block).

| model | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.060 | 0.035 | 0.013 | 0.008 | 0.005 |
| qwen3-0.6b | 0.388 | 0.345 | 0.300 | 0.280 | 0.287 |
| qwen3-1.7b | 0.351 | 0.319 | 0.289 | 0.292 | 0.303 |
| qwen3-4b | 0.531 | 0.515 | 0.549 | 0.517 | 0.545 |
