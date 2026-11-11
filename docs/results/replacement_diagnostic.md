# What happens when a summary replaces a centroid: the replacement diagnostic on CLINC150

Issue #33. Preliminary: five seeds, one dataset, one encoder. **Everything below except the logged-partition columns that say so was computed from summary embeddings re-computed on a CPU, not the original GPU ones (Section 2); the 4B's replayed blocks in particular do not reproduce the logged partitions.** Measurement and explanation are kept apart: Sections 3 to 7 report what was measured, Section 8 lists what is and is not concluded. No causal claim is made here; testing causes is #47.

## 1. What was measured and how

The logged CLINC150 runs (#31: LSA, Qwen3-0.6B, 1.7B and 4B, seeds 0-4, 5 summary steps, 20 Lloyd iterations per block, `initial_iter` and `final_iter` on) were replayed offline with `kslimeans.replay`. No language model was called. Each Lloyd block is deterministic given its start centers, so the logged summaries were embedded again (`scripts/embed_logged_summaries.py`, the same one-call-per-step encoding as core), and every block was re-run one iteration at a time with sklearn's own Lloyd step (empty-cluster relocation, the tol stop and the strict-convergence stop all as in a single fit). The replay also runs the counterfactual **no replacement**: plain Lloyd iterations from the pre-replacement state (the previous block's final centers, which the logs do not hold) for the same number of iterations.

The measures are in `kslimeans.diagnostic` (definitions in its docstring). The ones that need a convention:

- **Step numbering** is 1-based: step 1 is the first replacement. Every method starts step 1 from the same step-0 partition (`step0_sha1`), so cluster `c` has the same members for every summarizer at step 1 and differences there come from the summarizers alone.
- **Logged versus replayed.** Step 0 replays exactly for all 20 (method, seed) pairs, so at step 1 the pre-replacement state and the counterfactual are exact, and `assignments_evolution[1]` is the real post-block partition. Step 1 outcomes are therefore reported from the logged partition. **From step 2 on, every measure starts from a replayed previous block, which for the 4B is mostly not the real one** (the replayed pre-replacement partition differs from the logged source partition in 650 to 1,300 points on average at steps 2 to 5, `n_pre_vs_source` in `steps.csv`; for the 0.6B at steps 2, 4, 5 in 76 to 390; LSA in at most 1 point, the 1.7B in 0 to 92).
- **J** is the k-means objective (sum of squared distances to the nearest center). *Before* is the objective of the pre-replacement centers; *old assignment* uses the summary centers with every point kept in its old cluster; *replaced* re-assigns every point to the nearest summary center (the logged `summary_inertia`); *end* is after the block.
- **Centroid movement** is the distance from a cluster's pre-replacement center to its summary embedding in units of that cluster's RMS radius (the drift of #15), and their cosine.
- **Movers** are points whose label changes, at the replacement (versus the pre partition) and net over the block (end versus pre). The **gold direction** uses the majority gold label of each cluster *in the pre partition* (ties go to the smallest label): a mover from `a` to `b` moves *toward* its gold label if `majority[b]` is its label and `majority[a]` is not, *away* if the reverse, and *neutral* if both or neither majority is its label. Two baselines go with it: the share of all points whose label is the majority of their own pre cluster, and the same counts after permuting the movers' destinations among the movers (mean of 20 shuffles).
- **Boundary versus core** is a convention, not a derived quantity: a point is a boundary point if its pre-replacement margin (distance to the second-nearest pre center minus the nearest) is in the lowest quartile of that step's margins.
- **Flags** are the §2.4 surface rules (`kslimeans.surface`): *dup* (the summary is identical to another cluster's at that step), *meta* ("cluster" or "summar" appears), *flagged* is either. They are joined to the logged source partition. #34's genericity flag is not available yet.
- **Out-of-date summaries**: a displayed summary is out of date if the Jaccard similarity between the members it was written from and its cluster's members in the **logged** post-block partition is below a threshold (0.5 is the documented convention; 0.3 and 0.7 are reported), or its embedding is nearest to another cluster's final center (**this rule uses replayed centers**). Empty summaries are excluded.

Committed outputs (`docs/results/`, prefix `replacement_diagnostic_`): `verification.csv`/`.md`, `mechanism.csv`, `steps.csv` (one row per method, seed and step), `iterations.csv` (per iteration, replay and counterfactual), `outcome.csv`, `sensitivity.csv`, `sensitivity_summary.csv`/`.md`, `tables.md` (all tables, including the pooled-over-steps ones), `examples.md`, and the two figures.

**Not committed, regenerate:** the per-cluster file `replacement_diagnostic_clusters.csv` (one row per method, seed, step and cluster; key `run, method, seed, step, cluster`), `replacement_diagnostic_stale.csv` (same key) and the per-point mover files are written to the gitignored `results/replacement_diagnostic/`. #34 and #50 join on those keys after running the script (commands and runtime in Section 9); they are not in git.

## 2. Verification of the replay, and the noise floor

Every fit of every run was compared with the log: replayed labels against `assignments_evolution`, `n_iter` against `last_n_iter`, and the objective at the start and end of each block against `summary_inertia` and `inertia`. scikit-learn 1.9.1 throughout (recorded in `verification.csv`; the replay uses private sklearn functions and warns on any other version).

| run | method | fits | exact_fits | mismatched_points_median | mismatched_points_max | n_iter_differs | max_rel_err_summary_inertia | max_rel_err_inertia | sklearn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| clinc150_lsa | k-NLPmeans | 30 | 29 | 0 | 6 | 0 | 3.88e-08 | 3.11e-06 | 1.9.1 |
| clinc150_qwen3-0.6b | k-SLMmeans | 30 | 26 | 0 | 1180 | 0 | 2.84e-08 | 2.08e-03 | 1.9.1 |
| clinc150_qwen3-1.7b | k-SLMmeans | 30 | 28 | 0 | 630 | 0 | 4.48e-08 | 2.03e-03 | 1.9.1 |
| clinc150_qwen3-4b | k-SLMmeans | 30 | 8 | 778 | 2057 | 0 | 3.72e-08 | 2.14e-03 | 1.9.1 |

The summary embeddings used here were computed **on a CPU** (`results/replacement_diagnostic/summary_embeddings_cpu.npz`: torch 2.14.0, sentence-transformers 6.0.1, batch size 32); the original runs encoded on a GPU.

- **Step 0 is exact** for all 20 (method, seed) pairs (0 label mismatches, same `n_iter`), and the number of iterations matches the trace in all 120 fits.
- **The replay mechanism is exact on real data.** From the same start centers the replayed block gives 0 label mismatches and the same `n_iter` as a real `KMeans(init=start, max_iter=20)` fit, for the first seed of every method (`mechanism.csv`; centers differ by at most 3e-8, which is the run-to-run variation of KMeans itself). `tests/test_replay.py` checks it on synthetic data, including empty clusters, identical centers, both stops, a stop on tol, a final summary step with no block, and every iteration prefix.
- **Blocks do not always match the logs.** `exact_fits` counts fits including the five exact step-0 fits, so the 25 blocks per method are exact in 24 (LSA), 21 (0.6B), 23 (1.7B) and 3 (4B) cases. The worst mismatches are 6 points for LSA, 1,180 (0.6B), 630 (1.7B) and 2,057 (4B) of 22,500; the 4B's median over its 30 fits is 778 points. Where labels differ the end objective differs by up to 0.2%; the start objective agrees to 5e-8 in every fit.
- **Probable cause: GPU versus CPU encoder noise (unconfirmed).** A CPU re-encode of 300 dataset texts differs from the cached (GPU) embeddings by a median 1.0e-6 (L2 per row), and no row is bit-equal. The start objectives differ by 3e-8 to 4.5e-8, which a bit-identical encoder would not give, and noise of that size on the start centers reproduces mismatches of the observed size (next section). Untested alternatives: differences in thread count or BLAS between the original runs and this machine (KMeans itself is not bit-repeatable run to run), or library versions. Whether a GPU encode of the summaries makes the replay exact has not been tested. CONFIRM(#33).

### Sensitivity to noise in the start centers

For each method, seeds 0-2 and steps 1, 3 and 5, the block was replayed from the CPU embeddings (reference) and from copies perturbed by `start * (1 + rel * N(0, 1))`, 2 draws per size (fixed seeds). **float32 cannot represent relative changes below about 6e-8, so perturbations of 1e-9 change no element and 1e-8 changes only 0.04% of them** (about 40 of the 115,200 values; the `elements changed` column). Those rows are below float32 resolution and show what a handful of one-unit changes do, not a noise level. `iid` perturbs every row independently, so identical start centers stop being identical; `tied` gives identical rows the same noise, so ties stay exact. "Tied points" are those assigned at the replacement to a center that has an identical twin. Columns report the share of the 22,500 labels that differ from the reference after the block, the ARI between the two partitions, and the mean absolute change of ACC, NMI and ARI against gold.

| model | noise | rel. noise | elements changed | tied clusters | labels changed | of tied points | of other points | ARI vs reference | mean abs dACC | mean abs dNMI | mean abs dARI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | iid | 1e-09 | 0 | 0.0 | 0.000 | n/a | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| lsa | iid | 1e-08 | 0.00037 | 0.0 | 0.000 | n/a | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| lsa | iid | 1e-07 | 0.67 | 0.0 | 0.000 | n/a | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| lsa | iid | 1e-06 | 0.97 | 0.0 | 0.000 | n/a | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| lsa | tied | 1e-07 | 0.67 | 0.0 | 0.000 | n/a | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| lsa | tied | 1e-06 | 0.97 | 0.0 | 0.000 | n/a | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-0.6b | iid | 1e-09 | 0 | 1.1 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-0.6b | iid | 1e-08 | 0.00036 | 1.1 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-0.6b | iid | 1e-07 | 0.67 | 1.1 | 0.016 | 0.385 | 0.015 | 0.9776 | 0.0010 | 0.0005 | 0.0006 |
| qwen3-0.6b | iid | 1e-06 | 0.97 | 1.1 | 0.010 | 0.359 | 0.010 | 0.9850 | 0.0012 | 0.0005 | 0.0007 |
| qwen3-0.6b | tied | 1e-07 | 0.67 | 1.1 | 0.001 | 0.006 | 0.001 | 0.9988 | 0.0000 | 0.0000 | 0.0001 |
| qwen3-0.6b | tied | 1e-06 | 0.97 | 1.1 | 0.001 | 0.069 | 0.001 | 0.9978 | 0.0000 | 0.0001 | 0.0001 |
| qwen3-1.7b | iid | 1e-09 | 0 | 0.9 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-1.7b | iid | 1e-08 | 0.00037 | 0.9 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-1.7b | iid | 1e-07 | 0.67 | 0.9 | 0.010 | 0.263 | 0.009 | 0.9852 | 0.0006 | 0.0002 | 0.0006 |
| qwen3-1.7b | iid | 1e-06 | 0.97 | 0.9 | 0.012 | 0.391 | 0.011 | 0.9837 | 0.0009 | 0.0004 | 0.0005 |
| qwen3-1.7b | tied | 1e-07 | 0.67 | 0.9 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-1.7b | tied | 1e-06 | 0.97 | 0.9 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-4b | iid | 1e-09 | 0 | 17.4 | 0.000 | 0.000 | 0.000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| qwen3-4b | iid | 1e-08 | 0.00036 | 17.4 | 0.008 | 0.024 | 0.007 | 0.9888 | 0.0005 | 0.0002 | 0.0003 |
| qwen3-4b | iid | 1e-07 | 0.67 | 17.4 | 0.113 | 0.565 | 0.111 | 0.8969 | 0.0037 | 0.0022 | 0.0030 |
| qwen3-4b | iid | 1e-06 | 0.97 | 17.4 | 0.114 | 0.579 | 0.112 | 0.8972 | 0.0055 | 0.0023 | 0.0043 |
| qwen3-4b | tied | 1e-07 | 0.67 | 17.4 | 0.035 | 0.137 | 0.035 | 0.9559 | 0.0021 | 0.0019 | 0.0023 |
| qwen3-4b | tied | 1e-06 | 0.97 | 17.4 | 0.047 | 0.160 | 0.047 | 0.9441 | 0.0026 | 0.0017 | 0.0026 |

- **The tied rows are the realistic noise model**: identical texts encode identically, so real encoder noise does not break ties, and the `iid` rows overstate the noise floor for the 4B. From the tied 1e-6 rows (the size of the measured CPU/GPU difference; 9 cells per method): LSA and 1.7B change no label; 0.6B changes 0.14% on average (at most 1.5%); the **4B changes 4.7% of labels on average (sd 3.0% across cells, at most 9.5%), with a mean absolute change of 0.0026 ACC (at most 0.0072), 0.0017 NMI and 0.0026 ARI**. The `iid` rows give 11% and 0.0055 ACC for the 4B.
- **LSA is insensitive** to every noise size, matching its replay being exact in 29 of 30 fits.
- **0.6B and 1.7B** change about 1% of labels under `iid` noise at 1e-7 and 1e-6, almost entirely at the roughly one duplicate-summary cluster per step, and not at all under tied noise (0.6B 0.1%, 1.7B none).
- **The 4B** is the most sensitive even to the 40-element perturbation (0.8% of labels at 1e-8). Comparing `iid` with `tied` noise at 1e-7 and 1e-6, tie-preserving noise gives 3.5-4.7% against 11% of labels changed. So breaking the ties by noise raises the change, but the 4B stays sensitive to tie-preserving noise too. A cascade from the relocation of emptied clusters is a possible mechanism; it was not tested.
- **Reading the later tables:** the noise floor for partition-level quantities of the 4B (who moved where, the Jaccard with the pre partition) is of the order of 5% of labels (tied noise) to 11% (iid noise); for 0.6B and 1.7B at most about 1%; for LSA zero. ACC/NMI/ARI differences between methods below about 0.003 to 0.007 should not be interpreted. Objective values are much less affected (a few 1e-3 relative at most).

## 3. Objective around the replacement

CPU-embedding replay (exact for LSA and 1.7B at step 1; the 4B step-1 block differs from the logged one in about 680 points).

![Objective around each replacement](replacement_diagnostic_objective.png)

Top: objective divided by the objective before, log scale, seed 0 (solid: replacement, dashed: plain Lloyd). Bottom: the same without the spike, in percent above the pre-replacement objective, from iteration 2.

Step 1, means over seeds (`j_before` 10,721 for every method; the counterfactual after the same iterations is 10,712):

| model | j_before | jump_rel | end_vs_before_rel | end_vs_cf_rel | n_iter |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 10,721 | 0.721 | -0.00252 | -0.00168 | 20 |
| qwen3-0.6b | 10,721 | 0.777 | 0.00238 | 0.00322 | 20 |
| qwen3-1.7b | 10,721 | 0.95 | 0.000989 | 0.00183 | 20 |
| qwen3-4b | 10,721 | 0.981 | 0.0131 | 0.0139 | 20 |

Mean objectives at step 1, before / old assignments / re-assigned / after the block / counterfactual: LSA 10,721 / 19,396 / 18,446 / 10,694 / 10,712; 0.6B 10,721 / 22,157 / 19,052 / 10,746 / 10,712; 1.7B 10,721 / 23,352 / 20,907 / 10,731 / 10,712; 4B 10,721 / 25,660 / 21,238 / 10,861 / 10,712.

- Every replacement raises the objective by 72% (LSA) to 98% (4B) on average at step 1. Lloyd removes most of that rise quickly: averaged over all steps and seeds, 88% to 97% of it by iteration 1 and 94% to 99% by iteration 2 (4B lowest).
- After 20 iterations the 4B is the only method clearly above the pre-replacement objective: +1.3% on average at step 1, against +0.1% to +0.2% for 0.6B and 1.7B and -0.25% for LSA. In the seed-0 figure its curve lies above the others' for most of every block (at step 4 the 0.6B ends higher), 0.1% to 0.6% above the pre-replacement objective at iteration 20 depending on the step.
- The size of the jump does not rank the outcomes: 1.7B and 4B have similar jumps (0.95 and 0.98) and opposite outcomes (Section 6). LSA's blocks stop early in 8 of its 25 cases (10 end on a strict-convergence stop, 2 of them at exactly iteration 20); the other methods' blocks always run the full 20 iterations.

## 4. Centroid movement and movers

Replayed, CPU embeddings. Movement (summary versus pre-replacement center) depends only on the exact step-0 state at step 1 and on the embeddings.

![Centroid movement](replacement_diagnostic_movement.png)

Per-cluster distance from the pre-replacement center to the summary embedding in RMS radii (left, log scale) and their cosine (right), all clusters, steps and seeds.

Step 1, means over seeds of per-step medians:

| model | drift_median | cosine_median | own_nearest_share | to_end_radii_median | end_vs_pre_radii_median | pulled_back_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.891 | 0.802 | 0.997 | 0.889 | 0.0409 | -3.64e-06 |
| qwen3-0.6b | 0.965 | 0.754 | 0.825 | 0.965 | 0.167 | 0.000198 |
| qwen3-1.7b | 1.05 | 0.708 | 0.817 | 1.06 | 0.114 | -0.000253 |
| qwen3-4b | 1.08 | 0.668 | 0.619 | 1.09 | 0.333 | -0.000303 |

- Summary embeddings sit about one RMS radius (0.89 for LSA to 1.08 for the 4B) away from the center they replace. The cosine falls from 0.80 (LSA) to 0.67 (4B).
- The share of clusters whose summary is nearest to its own pre-replacement center is 0.997 for LSA, 0.82 for 0.6B and 1.7B and 0.62 for the 4B.
- The block ends close to where it started: the median block-end center is 0.04 (LSA), 0.17, 0.11 and 0.33 (4B) radii from the pre-replacement center, so it is about as far from the summary as the pre-replacement center was (`pulled_back_median` is about zero). Yet the partition does not return:

| model | frac_moved_replacement | frac_moved_end | ari_replacement_vs_pre | ari_end_vs_pre | jaccard_end_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.253 | 0.0904 | 0.57 | 0.852 | 0.859 |
| qwen3-0.6b | 0.467 | 0.334 | 0.388 | 0.688 | 0.597 |
| qwen3-1.7b | 0.468 | 0.287 | 0.371 | 0.708 | 0.652 |
| qwen3-4b | 0.561 | 0.447 | 0.317 | 0.653 | 0.473 |

- At the replacement 25% (LSA), 47%, 47% and 56% (4B) of the points change cluster. After the block, net, 9%, 33%, 29% and 45% differ from the pre partition (ARI 0.85, 0.69, 0.71, 0.65). Lloyd does not undo the replacement; it settles in a partition that differs from the old one, most for the 4B (mean per-cluster Jaccard with the pre partition 0.47). The noise floor above applies (4B about 5-11% of labels).

Movers at the replacement, step 1 (counts per step, means over seeds), with the gold-direction baselines:

| model | n_moved_replacement | toward_replacement | toward_shuffled | away_replacement | away_shuffled | ends_in_own_majority_replacement | ends_in_own_majority_shuffled | share_in_own_majority_pre | move_rate_boundary | move_rate_core | move_rate_flagged | move_rate_unflagged | share_movers_from_flagged | share_points_in_flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 5,698 | 603 | 19.3 | 2,141 | 2,249 | 724 | 32.7 | 0.578 | 0.569 | 0.148 | n/a | 0.253 | 0 | 0 |
| qwen3-0.6b | 10,502 | 694 | 26.4 | 4,666 | 4,875 | 937 | 61.5 | 0.578 | 0.731 | 0.379 | 0.849 | 0.427 | 0.172 | 0.0953 |
| qwen3-1.7b | 10,520 | 731 | 28.2 | 4,487 | 4,711 | 985 | 57.5 | 0.578 | 0.747 | 0.374 | 0.725 | 0.441 | 0.145 | 0.0933 |
| qwen3-4b | 12,620 | 723 | 30.5 | 5,707 | 5,931 | 978 | 60.2 | 0.578 | 0.802 | 0.481 | 0.904 | 0.364 | 0.586 | 0.364 |

- **Gold direction, with baselines.** 57.8% of all points sit in the cluster whose majority gold label is their own before the replacement. Against that, the movers: of the 4B's 12,620 movers per step, 723 move toward their gold label, 5,707 away, 978 end in a cluster whose majority is their own label, and the remaining 6,190 (49%) are neutral (both or neither cluster's majority is their label); LSA 603 toward, 2,141 away, 724 own-majority ends of 5,698. With the movers' destinations shuffled among the movers the same counts are about 30 toward, 5,931 away and 60 own-majority ends for the 4B (LSA 19, 2,249, 33). So the **toward** moves are 24 to 31 times and the own-majority ends 15 to 22 times what random destinations would give, for every method, while the **away** count is about what shuffling gives (about 4% to 5% below it for the 4B and LSA). Many "away" moves are expected by construction, because most points start in their majority cluster; the shuffled-destination baseline, not the raw away count, is the reference. Neither baseline says whether a post-block partition is better; the ACC/NMI/ARI columns do.
- **Boundary versus core.** Movers are concentrated at small pre margins for every method: the lowest-margin quartile of points moves at the replacement with rate 0.57 (LSA), 0.73, 0.75 and 0.80 (4B), the rest at 0.15, 0.38, 0.37 and 0.48.
- **Flagged clusters.** For 0.6B and 1.7B the flagged clusters (dup or meta; 9% of points) move at 0.85 and 0.73 against 0.43 and 0.44 for unflagged ones, and give 17% and 15% of the movers. For the 4B, flagged clusters hold 36% of the points, move at 0.90 against 0.36, and give 59% of the movers. LSA has no flagged cluster.

## 5. The 4B: duplicate summaries and collapsed centroids

Replayed, CPU embeddings (step 1 is exact up to the encoder noise; the counts of duplicates come from the logged summaries and are exact).

At step 1 the 4B has 18.8 clusters per seed that share their summary with another cluster (12 to 24 across seeds; 0.6B 3.0, 1.7B 0.8, LSA none), and 50 clusters whose summary mentions the cluster (0.6B and 1.7B 13.6). Clusters with an identical summary get identical centers, and every point goes to the first of them.

- **Empty clusters right after the replacement:** 11.2 per seed on average at step 1 (8 to 16 across seeds; 0.6B 0.8, 1.7B 0.6, LSA 0). Over all steps and seeds, 262 empty clusters for the 4B, 17 for the 0.6B, 12 for the 1.7B.
- **How sklearn re-seeds them:** every cluster that was empty at the replacement is empty in the first E-step and is relocated at iteration 1 (the `first_nonempty_t` column is 1 for all 291); no cluster is empty at the end of any block. sklearn's relocation puts an empty cluster's center on points that are far from their current centers. These re-seeded clusters hold on average 56 points at the end of the block, against 157 for clusters that were not empty.
- **The duplicate clusters themselves** hold 145 points on average before the replacement, 4 right after it, and 70 at the end of the block (over all steps and seeds, 4B), and their end members overlap almost not at all with their pre members (mean Jaccard 0.009; per-cluster values are in the regenerated `clusters.csv`). Their summary embedding is nearest to its own pre-replacement center in none of the 4B's and 1.7B's duplicate cases (4% for the 0.6B); identical summaries cannot all be nearest to their own center.
- **Examples** (4B, seed 0, step 1; `replacement_diagnostic_examples.md`): seven clusters share "What is the main topic of the following cluster of questions?". Points moved out of them include "send 50 dollars between bank of america and chase accounts", which lands in a cluster summarized "How can I transfer money between bank accounts and check my bank balances?", and also "how would you say can i borrow five dollars in portuguese", which lands in "How do I change my oil and when should I do it?".
- Mean J rises by 98% at the replacement and is 1.3% above the pre-replacement objective after the block, against the counterfactual 0.1% below (Section 3).

## 6. Relating it to the outcome

**Step 1, from the logged partition** (exact pairing; the replayed values are a check). Mean over seeds of the score of the real post-block partition minus the score after plain Lloyd from the step-0 partition, the same 20 iterations:

| model | d_acc | d_nmi | d_ari | seeds_below | d_acc_replayed | replay_vs_logged_points |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| lsa | +0.0051 | +0.0019 | +0.0048 | 0/5 | +0.0051 | 0 |
| qwen3-0.6b | +0.0016 | +0.0022 | +0.0048 | 3/5 | +0.0029 | 390 |
| qwen3-1.7b | +0.0106 | +0.0076 | +0.0177 | 0/5 | +0.0106 | 0 |
| qwen3-4b | -0.0159 | +0.0011 | -0.0012 | 5/5 | -0.0165 | 682 |

Per seed (ACC), logged minus counterfactual: LSA +0.0047, +0.0040, +0.0144, +0.0016, +0.0007 (5/5 above); 0.6B +0.0162, -0.0077, -0.0055, -0.0102, +0.0149 (3/5 below); 1.7B +0.0044, +0.0117, +0.0151, +0.0140, +0.0078 (5/5 above); 4B -0.0104, -0.0264, -0.0128, -0.0212, -0.0087 (5/5 below, mean -0.0159). The 4B's ACC deficit exceeds the noise floor of Section 2 (tied noise: 0.0026 mean, 0.0072 at most); its NMI and ARI differences are about zero (+0.0011, -0.0012), so the deficit is in ACC only. The 0.6B's mean (+0.0016) and sign pattern are within the noise.

**A confound, and a run to convergence.** The counterfactual starts from a converged state, so a fixed 20-iteration budget can end the replacement block less converged. All three SLM methods' blocks run the full 20 iterations at step 1, so "below the counterfactual" mixes a worse end state with being less converged. Running both sides to sklearn's own stop at step 1 (cap 300; replayed from the CPU embeddings, no run hit the cap):

| model | n_iter | n_iter_cf | capped | capped_cf | j_end_rel | d_acc | d_nmi | d_ari | seeds_below |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lsa | 31.8 | 15.2 | 0 | 0 | -0.0019 | +0.0050 | +0.0019 | +0.0049 | 0/5 |
| qwen3-0.6b | 53.8 | 15.2 | 0 | 0 | +0.0019 | +0.0033 | +0.0028 | +0.0059 | 3/5 |
| qwen3-1.7b | 48.4 | 15.2 | 0 | 0 | +0.0008 | +0.0121 | +0.0088 | +0.0190 | 0/5 |
| qwen3-4b | 53.6 | 15.2 | 0 | 0 | +0.0112 | -0.0131 | +0.0033 | +0.0023 | 5/5 |

The replacement runs need 48 to 54 iterations (LSA 32) against 15 for the counterfactual. With both converged the 4B is still below the counterfactual in ACC (-0.0131, 5/5 seeds) with an objective 1.1% higher, the 1.7B above (+0.0121), the 0.6B at +0.0033 (3/5 below, within noise) and LSA at +0.0050. So at step 1 the 4B's lower ACC is not an artifact of the iteration budget, although this converged comparison is replayed with CPU summary embeddings rather than logged, and the 4B's tied-noise floor reaches 0.0072 ACC (Section 2). This is still a comparison between two ways of continuing from the same step-0 partition, not a statement about why.

**Later steps (replayed; from step 2 on each row starts from a replayed previous block).** Mean over seeds of (replayed score after the block) minus (score after plain Lloyd from the replayed pre-replacement state, same iterations):

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

- In this replay the 4B's block-end ACC is below the counterfactual's at every step (mean -0.006 to -0.017); the 1.7B's is above it at steps 1-3 and 5 and below at step 4. For the 4B these step 2-5 values are replays of blocks that mostly differ from the real ones, so they carry the noise of Section 2 and are not a measurement of the logged runs.

Final scores against k-means from the same initialization (logged values; mean over seeds):

| model | acc | kmeans_acc | d_acc | d_nmi | d_ari |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.560 | 0.551 | +0.0089 | +0.0043 | +0.0104 |
| qwen3-0.6b | 0.541 | 0.551 | -0.0104 | -0.0009 | -0.0014 |
| qwen3-1.7b | 0.570 | 0.551 | +0.0184 | +0.0123 | +0.0224 |
| qwen3-4b | 0.511 | 0.551 | -0.0402 | -0.0086 | -0.0181 |

- Final scores against k-means: LSA +0.009 ACC, 0.6B -0.010, 1.7B +0.018, 4B -0.040.
- **Descriptively, across the 20 (model, seed) pairs at step 1** (Spearman, with the step outcome from the logged partition; not a test, and it pools between-model and within-model variation with 5 seeds per model): the final ACC difference to k-means goes with the number of clusters left empty at the replacement (-0.73), the fraction of points moved at the replacement (-0.63), the relative objective jump (-0.44) and the median drift (-0.35). These associations are not evidence for a mechanism.

| step-1 measure | outcome | spearman (n=20) |
| --- | --- | ---: |
| jump_rel | d_step_acc | -0.32 |
| jump_rel | d_final_acc | -0.44 |
| jump_rel | d_final_nmi | -0.36 |
| drift_median | d_step_acc | -0.44 |
| drift_median | d_final_acc | -0.35 |
| drift_median | d_final_nmi | -0.29 |
| n_empty_replacement | d_step_acc | -0.60 |
| n_empty_replacement | d_final_acc | -0.73 |
| n_empty_replacement | d_final_nmi | -0.61 |
| frac_moved_replacement | d_step_acc | -0.55 |
| frac_moved_replacement | d_final_acc | -0.63 |
| frac_moved_replacement | d_final_nmi | -0.52 |

- No large relative objective jump reliably goes with a worse result: the 1.7B and the 4B jump by about the same and end on opposite sides of the counterfactual.

## 7. Do the displayed summaries describe the final partition?

Jaccard against the **logged** final partition (`assignments_evolution[-1]`); only the nearest-final-centroid rule uses replayed centers (CPU embeddings, step 5 of the 4B replayed from a replayed chain). With `final_iter` on, a Lloyd block follows the last summary step. All seeds pooled, 750 final summaries per method:

| model | summaries | mean_jaccard | jaccard<0.3 | jaccard<0.5 | jaccard<0.7 | nearest_not_own | stale@0.3 | stale@0.5 | stale@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsa | 750 | 0.965 | 0.00133 | 0.00533 | 0.00933 | 0 | 0.00133 | 0.00533 | 0.00933 |
| qwen3-0.6b | 750 | 0.699 | 0.18 | 0.244 | 0.351 | 0.155 | 0.236 | 0.287 | 0.387 |
| qwen3-1.7b | 750 | 0.711 | 0.159 | 0.231 | 0.325 | 0.212 | 0.259 | 0.303 | 0.384 |
| qwen3-4b | 750 | 0.484 | 0.416 | 0.484 | 0.56 | 0.427 | 0.499 | 0.547 | 0.597 |

Share stale@0.5 by summary step (summary of step j against the partition after its own block).

| model | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lsa | 0.060 | 0.035 | 0.013 | 0.008 | 0.005 |
| qwen3-0.6b | 0.383 | 0.345 | 0.300 | 0.283 | 0.287 |
| qwen3-1.7b | 0.351 | 0.319 | 0.291 | 0.292 | 0.303 |
| qwen3-4b | 0.525 | 0.519 | 0.552 | 0.519 | 0.547 |

- By the 0.5 convention, 0.5% of LSA's final summaries are out of date, 29% of the 0.6B's, 30% of the 1.7B's and 55% of the 4B's. With a 0.3 threshold the shares are 0.1%, 24%, 26% and 50%; with 0.7, 0.9%, 39%, 38% and 60%.
- The Jaccard-only shares (0.005, 0.244, 0.231, 0.484 at 0.5) come from logged partitions alone and are the part of this table that does not depend on the replay. The 4B's final summary embedding is nearest to another cluster's final center for 43% of its clusters (LSA 0%, 0.6B 16%, 1.7B 21%; replay-dependent), and the mean Jaccard between the members a summary was written from and the final members is 0.48 (LSA 0.97, 0.6B 0.70, 1.7B 0.71). The share out of date at every step for the 4B is 0.52 to 0.55 (last table).
- The noise floor of Section 2 applies to the nearest-center rule (4B).

## 8. What can and cannot be said

Measured: replacing the centroids with these summaries raises the objective by 72% to 98% and Lloyd removes most of it within a couple of iterations; the partition however ends 9% to 45% different from the pre-replacement partition, most for the 4B; the 4B has an order of magnitude more duplicate summaries and emptied clusters than the other methods, its movers come disproportionately from flagged clusters, and at step 1 its real post-block partition has a lower ACC than the counterfactual in all five seeds (mean -0.016; -0.013 with both sides run to convergence); more than half of its displayed final summaries no longer match their cluster by the convention used here.

Not concluded:

- That the duplicates or the cluster-mentioning summaries *cause* the 4B's lower scores. These are associations between measures; #47 tests this with swapped start centers (the replay's `swap` hook).
- That the replay is exact against the original GPU run. Step 0 and the mechanism are exact; LSA, 0.6B and 1.7B blocks are mostly exact; 22 of the 25 4B blocks differ from the logs (median over fits 778 points, at most 2,057), probably through CPU versus GPU encoder noise, which the 4B's blocks amplify. A GPU encode of the summaries (`scripts/embed_logged_summaries.py --device cuda`) followed by a rerun would settle it. CONFIRM(#33). Anything from step 2 on starts from replayed previous blocks and, for the 4B, should be read as a replay.
- That "below the counterfactual" means a worse basin. The convergence comparison removes the fixed-budget confound at step 1, but the counterfactual (plain Lloyd from the step-0 partition) is one reference among others.
- Gold direction is measured against the pre-partition majority labels; it does not say whether the post-block partition is better, only the ACC/NMI/ARI columns do.
- Boundary versus core, the 0.5 Jaccard threshold and the flags are conventions fixed before looking at outcomes (the thresholds 0.3 and 0.7 are reported next to 0.5).
- BANKING77 is not covered. The #20 runs lack per-step assignments and would need a re-run with #30 logging (about one GPU hour, not yet approved).
- Five seeds, one dataset, one encoder.

## 9. Reproduce

```bash
CUDA_VISIBLE_DEVICES="" python scripts/embed_logged_summaries.py RUN_DIR... --cache-dir cache --out results/replacement_diagnostic/summary_embeddings_cpu.npz --device cpu
CUDA_VISIBLE_DEVICES="" python scripts/replacement_diagnostic.py RUN_DIR... --cache-dir cache --summary-embeddings results/replacement_diagnostic/summary_embeddings_cpu.npz --out-dir docs/results
CUDA_VISIBLE_DEVICES="" python scripts/replacement_sensitivity.py RUN_DIR... --cache-dir cache --summary-embeddings results/replacement_diagnostic/summary_embeddings_cpu.npz --out-dir docs/results
```

The second command also writes the gitignored `replacement_diagnostic_clusters.csv`, `replacement_diagnostic_stale.csv` and the mover files to `results/replacement_diagnostic/` (`--work-dir`); that is how #34 and #50 get their per-cluster table.

CPU time on the machine used (12 cores, WSL2): embedding the 100 logged summary steps 2.7 minutes; the full replay and all measures for 20 (method, seed) pairs with counterfactuals, real-KMeans checks and the step-1 convergence runs 8.7 minutes; the sensitivity probe 13.7 minutes. About 1.2 s per replayed 20-iteration block.
