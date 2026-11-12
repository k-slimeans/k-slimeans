# Do the 4B's flagged summaries cause its lower scores? Counterfactual replacements at step 1 (CLINC150)

Issue #47, stage 1. Preliminary: five seeds, one dataset, one encoder, **one summary step**. Everything was computed offline from the logged runs (#31) with the replay of #33 (`kslimeans.replay`), from summary embeddings re-computed on a CPU (#33 section 2), so the noise floor of #33 applies (Section 3). No language model was called. Stage 2 (the guarded 4B runs that test whether this carries through to final scores) has not been run.

## 1. What was done

At summary step 1 every method of a seed starts from the same step-0 partition (`step0_sha1`), and step 0 replays exactly (checked again here: the replayed step-0 labels equal `assignments_evolution[0]` of both runs, all seeds). Cluster `c` therefore has the same members whichever model summarized it, and the start centers of the step-1 Lloyd block can be edited row by row. Each variant below changes some rows of the 4B's step-1 start centers (the embeddings of its summaries) and replays the block that follows: the logged one (20 Lloyd iterations, sklearn's stops applied: "fixed") and, separately, one run to convergence as in #33 section 6 ("converged", cap 300 iterations).

**Flags, defined before any outcome was computed** (`kslimeans.counterfactual`, docstring). Per cluster, on the 4B's step-1 summaries, written from the step-0 partition:

- **dup** (i): the summary is identical to another cluster's (every copy is flagged);
- **meta** (ii): the summary contains "cluster" or "summar", any case (`surface.mentions_cluster`, the section 2.4 rule);
- **union** = (i) or (ii);
- **generic** (iii): #34's `flag_generic`: duplicate, near-duplicate (cosine of summary embeddings at least 0.95), mentions the cluster, or non-positive specificity margin. It contains the union. (`summary_quality.generic_flags` computes the flag without spaCy; a test checks it against `score_step`.)

In these data **dup is a subset of meta in every seed, so union and meta are the same set**; their rows are identical and are not separate evidence. Flagged clusters per seed (`counterfactual_swaps_flags.csv`; k = 150, 22,500 points):

| flag | clusters flagged (seeds 0-4) | points in them |
| --- | --- | --- |
| dup | 18, 12, 18, 22, 24 | 2,000 to 3,700 |
| meta = union | 49, 43, 48, 55, 56 | 7,200 to 9,400 (32% to 42%) |
| generic | 63, 62, 75, 69, 72 | 10,300 to 11,800 (46% to 53%) |

**Variants** (`own` = the 4B, `other` = the 1.7B of the same seed; all with the flagged set of one flag at a time):

| name | start centers of the step-1 block |
| --- | --- |
| a | the 4B's summaries as logged |
| b | flagged clusters keep their k-means center (`pre.centers[c]`), the others take the 4B summary |
| c | flagged clusters take the 1.7B's summary embedding of the same cluster |
| d | control: as many randomly drawn **unflagged** clusters as flagged ones keep their k-means center (20 draws per seed and flag set, fixed RNG; mean over draws, then over seeds) |
| m | the 1.7B's summaries as logged |
| r | reverse: the 1.7B's, with the 4B's flagged summaries put into the same clusters |
| rd | control for r: the 4B's summaries put into the same number of random unflagged clusters of the 1.7B (same draws as d) |
| ref | no replacement: plain Lloyd from the k-means state (#33's counterfactual) |

(An addition to the brief: `rd`, so that "the 4B's flagged summaries hurt the 1.7B" can be compared with "any 4B summaries hurt the 1.7B".)

Each variant is measured after the replacement (iteration 0: nearest-center assignment) and after the block: the objective, ACC (Hungarian), NMI and ARI of the partition, empty clusters, and movers (points whose cluster differs from the pre-replacement one). The tables below give the end-of-block ACC and NMI; ARI, the objective and the replacement-stage measures are in `counterfactual_swaps_tables.md` and `counterfactual_swaps_per_seed.csv`. Differences are paired by seed against (a) (reverse: against m), with Student-t 95% intervals over the 5 seeds (n = 5, t = 2.776), as in #31.

## 2. Results

Absolute levels, means over seeds (fixed block): ACC / NMI of (a) 0.5350 / 0.7490, of the 1.7B (m) 0.5621 / 0.7557, of no replacement (ref) 0.5515 / 0.7482. So at step 1 the 4B is below the 1.7B by 0.027 ACC and 0.007 NMI, and below k-means by 0.0165 ACC but **not below k-means in NMI** (ref minus a: -0.0009 [-0.0047, +0.0030]).

Paired differences, end of block, mean over 5 seeds [95% interval], fixed 20-iteration block. "Closes" is the mean difference over the mean gap (ACC): b over the gap to ref, c over the gap to m.

| variant (flag) | flagged | ΔACC | ΔNMI | closes |
| --- | ---: | --- | --- | ---: |
| ref - a (k-means, no replacement) | | +0.0165 [+0.0069, +0.0261] | -0.0009 [-0.0047, +0.0030] | |
| m - a (the 1.7B) | | +0.0271 [+0.0132, +0.0410] | +0.0067 [+0.0024, +0.0109] | |
| b (dup) | 12-24 | -0.0075 [-0.0311, +0.0161] | -0.0090 [-0.0177, -0.0003] | -0.46 |
| b (meta = union) | 43-56 | +0.0070 [-0.0065, +0.0205] | -0.0051 [-0.0097, -0.0004] | 0.42 |
| b (generic) | 62-75 | +0.0114 [-0.0064, +0.0292] | -0.0025 [-0.0089, +0.0039] | 0.69 |
| c (dup) | 12-24 | +0.0154 [+0.0004, +0.0303] | +0.0031 [-0.0017, +0.0079] | 0.57 |
| c (meta = union) | 43-56 | +0.0268 [+0.0213, +0.0322] | +0.0071 [+0.0034, +0.0109] | 0.99 |
| c (generic) | 62-75 | +0.0255 [+0.0179, +0.0331] | +0.0080 [+0.0036, +0.0123] | 0.94 |
| d (dup) | = dup | -0.0179 [-0.0322, -0.0036] | -0.0103 [-0.0157, -0.0049] | |
| d (meta = union) | = union | -0.0274 [-0.0394, -0.0155] | -0.0164 [-0.0228, -0.0100] | |
| d (generic) | = generic | -0.0308 [-0.0402, -0.0215] | -0.0159 [-0.0212, -0.0107] | |
| r - m (dup) | | -0.0110 [-0.0212, -0.0008] | -0.0032 [-0.0090, +0.0025] | |
| r - m (meta = union) | | -0.0258 [-0.0443, -0.0074] | -0.0095 [-0.0144, -0.0046] | |
| r - m (generic) | | -0.0210 [-0.0379, -0.0041] | -0.0051 [-0.0109, +0.0007] | |
| rd - m (dup) | | -0.0028 [-0.0102, +0.0045] | -0.0003 [-0.0023, +0.0017] | |
| rd - m (meta = union) | | -0.0023 [-0.0098, +0.0052] | -0.0001 [-0.0024, +0.0021] | |
| rd - m (generic) | | -0.0030 [-0.0149, +0.0088] | +0.0006 [-0.0031, +0.0043] | |

Same, **run to convergence** (the 4B's (a) runs 54 iterations on average, ref 15):

| variant (flag) | ΔACC | ΔNMI | closes |
| --- | --- | --- | ---: |
| ref - a | +0.0131 [+0.0048, +0.0214] | -0.0033 [-0.0070, +0.0005] | |
| m - a | +0.0252 [+0.0135, +0.0369] | +0.0055 [+0.0008, +0.0102] | |
| b (dup) | -0.0059 [-0.0274, +0.0155] | -0.0081 [-0.0160, -0.0002] | -0.45 |
| b (meta = union) | +0.0053 [-0.0132, +0.0238] | -0.0068 [-0.0131, -0.0005] | 0.40 |
| b (generic) | +0.0085 [-0.0101, +0.0271] | -0.0046 [-0.0116, +0.0023] | 0.65 |
| c (dup) | +0.0146 [+0.0007, +0.0286] | +0.0022 [-0.0042, +0.0087] | 0.58 |
| c (meta = union) | +0.0266 [+0.0195, +0.0337] | +0.0066 [+0.0023, +0.0110] | 1.06 |
| c (generic) | +0.0251 [+0.0207, +0.0294] | +0.0073 [+0.0034, +0.0113] | 1.00 |
| d (dup) | -0.0159 [-0.0304, -0.0014] | -0.0096 [-0.0151, -0.0040] | |
| d (meta = union) | -0.0244 [-0.0353, -0.0134] | -0.0150 [-0.0211, -0.0089] | |
| d (generic) | -0.0277 [-0.0408, -0.0146] | -0.0135 [-0.0211, -0.0059] | |
| r - m (dup) | -0.0126 [-0.0232, -0.0020] | -0.0046 [-0.0096, +0.0005] | |
| r - m (meta = union) | -0.0241 [-0.0402, -0.0079] | -0.0086 [-0.0140, -0.0032] | |
| r - m (generic) | -0.0202 [-0.0324, -0.0081] | -0.0045 [-0.0074, -0.0016] | |
| rd - m (dup) | -0.0024 [-0.0107, +0.0058] | -0.0004 [-0.0023, +0.0016] | |
| rd - m (meta = union) | -0.0022 [-0.0088, +0.0044] | -0.0003 [-0.0014, +0.0008] | |
| rd - m (generic) | -0.0031 [-0.0142, +0.0081] | +0.0004 [-0.0020, +0.0028] | |

Fixed and converged agree in sign and size everywhere (the convergence confound of #33 section 6 does not change any conclusion here). The full set, including ARI, the relative objective, (b) minus (d), (c) minus (d), (r) minus (rd) and the seeds above zero, is in `counterfactual_swaps_tables.md` / `counterfactual_swaps_paired.csv`.

Observations (measured):

- **(c) is the clear effect.** Giving the flagged clusters the 1.7B's summaries lifts the 4B by +0.027 ACC (union) in 5/5 seeds, which is the whole 4B-to-1.7B gap (99% fixed, 106% converged), and +0.007 NMI (the gap is +0.007). For the 18 duplicate clusters alone, +0.015 (57% of the gap). The objective after the block falls by 1.2% (union).
- **Reverse:** putting the 4B's union-flagged summaries into the 1.7B costs -0.026 ACC (0/5 seeds above zero), against -0.002 [-0.010, +0.005] for the same number of random unflagged clusters. The 4B's summaries elsewhere cost the 1.7B nothing measurable.
- **(b) is not shown.** Keeping the k-means center in the flagged clusters changes ACC by +0.007 (union; +0.011 generic; -0.008 dup), every interval includes zero, and NMI goes down (-0.005 union, interval excludes zero). It closes 42% (union) to 69% (generic) of an ACC gap to k-means that is itself only 0.0165.
- **The control (d) is not neutral, it hurts.** Reverting the k-means center of random unflagged clusters lowers ACC by 0.018 to 0.031 (0/5 seeds above zero) and NMI by 0.010 to 0.016. After the replacement it leaves 29 to 39 empty clusters, against 8 (b, union) and 14 (a): near-data k-means centers sitting beside far-away summary embeddings take points from them. The spread over the 20 draws within a seed is sd 0.007 to 0.011 ACC. So (b) minus (d) is large (+0.034 union [+0.025, +0.044]) mainly because (d) damages the partition, not because (b) repairs it; **(b) against (a) is the cleaner comparison and it is inconclusive.** Why the mixture hurts has not been tested.

## 3. Noise floor and what is not interpretable

- **(a) replayed against the logged step-1 partition** (`counterfactual_swaps_fidelity.csv`; the replay is from CPU embeddings, the logged run used GPU ones). 4B: 421, 293, 759, 1,937 and 0 of 22,500 points differ (seeds 0-4; mean 682), ACC differs by 0.0009, 0.0008, 0.0040, 0.0008 and 0 (mean 0.0013). 1.7B: 0 points in every seed. So the table's (a) is a replay, not the logged 4B; the logged 4B ACC (mean 0.5356) and the replay's (0.5350) are within 0.001.
- **Noise floor** from #33 section 2 for the 4B under tie-preserving noise of the size of the CPU/GPU encoder difference: mean absolute change 0.003 ACC (at most 0.007), 0.002 NMI. Differences below that are **uninterpretable**. Against it: (c) union (+0.027 ACC, +0.007 NMI), (r) union (-0.026), (d) (-0.018 to -0.031) and c-dup (+0.015) are several times larger. (b) union (+0.007 ACC) is at the maximum of the floor, and its interval includes zero anyway; (rd) and (r) dup (-0.011) are in the range where the interval is the guide.
- **Stability under perturbation** (`counterfactual_swaps_stability.csv`): the paired ACC and NMI differences of b, c and r were recomputed from three tie-preserving perturbations of both models' embeddings (`start * (1 + 1e-6 N(0, 1))`, identical texts get the same noise; the k-means centers are not perturbed, being exact). The mean ACC difference moved by at most 0.002 (for example c union fixed +0.0268 against +0.0259 to +0.0278; b union +0.0070 against +0.0062 to +0.0080) and kept its sign in all 3 reruns for all 12 comparisons. This probes the encoder noise only; it does not reproduce the GPU embeddings, which were not re-encoded here.
- The `iid` noise rows of #33 (which break ties) were not used.

## 4. Reading against #47's criteria

Criterion: flagged summaries cause the drop if (b) and (c) close most of the gap to k-means or the 1.7B, (d) does not, and (for the reverse) they hurt the 1.7B; stage 2 must recover the final scores.

- **(c) holds** (closes about all of the gap to the 1.7B, 5/5 seeds, at several times the noise floor). **The reverse holds** (the 4B's flagged summaries lower the 1.7B by 0.026 ACC, the same number of random 4B summaries by 0.002), which says the effect travels with the summary text and is not something else in the 4B run. **(d) closes nothing**, but because it harms the partition, so it does not play its intended role of showing that (b) works only through replacing fewer centroids.
- **(b) does not hold at step 1.** Falling back to the k-means center in the flagged clusters does not recover a measurable amount of ACC and lowers NMI. Together with the NMI of the unreplaced k-means state being no higher than the 4B's, **the 4B at step 1 is not clearly worse than "no replacement"; it is worse than the 1.7B.** The flagged 4B summaries are worse than the 1.7B's summaries of the same clusters; they are not shown to be worse than the cluster's own centroid.
- **So: partly.** At step 1, replacing the flagged summaries with the 1.7B's removes the 4B's deficit to the 1.7B, and the reverse transplant creates a comparable deficit, so the flagged summaries (as a set, by any of dup, meta/union or generic) carry the difference between the two models. What is not established is that they make the 4B worse than a k-means centroid would, or that the whole 4B deficit lies in them (the 4B's ACC deficit to k-means, 0.0165, is not explained by the flagged set in (b), and in the rd test random 4B summaries cost the 1.7B nothing measurable). The remaining question, whether the effect survives later steps, is stage 2.
- Flag by flag: dup alone (12 to 24 clusters) recovers about 57% of the 1.7B gap in (c) and costs the 1.7B -0.011 in (r); meta/union and generic recover about all of it. Because generic contains the union and the union is 64% to 80% of generic, the data cannot separate the flags' contributions beyond "more flagged clusters, more effect".

## 5. Limits

- **One step.** Later steps need new summaries of the new partitions, which cannot be produced offline. The statement supported is: at step 1, replacing the flagged summaries with X changes the block's ACC by Y. It does not carry to whole runs; the final logged scores (#31) are not tested here. Stage 2 (GPU) does that.
- **CPU-embedding noise.** The 4B's blocks are sensitive to 1e-7 changes in the start centers (Section 3). The (a) replay itself differs from the logged 4B partition in up to 1,937 points. The perturbation reruns show the differences are stable to that noise, not that they match the original GPU run.
- **Five seeds,** one dataset, one encoder, one model pair. Intervals are t-intervals with 4 degrees of freedom. About 100 comparisons are listed; no multiplicity correction. The dup, meta and generic rows share the same seeds and overlapping clusters, so they are not independent confirmations.
- **No matched control for (c).** There is no run that puts the 1.7B's summaries into random unflagged 4B clusters; (rd) is the reverse-direction analogue. If any 1.7B summary were better than the 4B's for any cluster, (c) would show a gain with no link to the flags. The rd result (4B summaries cost the 1.7B nothing in random clusters) argues against it but does not replace the control.
- **The flags describe clusters, which differ in other ways.** Flagged clusters may be harder clusters (mixed intents), and the replaced summary comes from a model that saw the same members. The swap is an intervention on the summary text of those clusters, but not on which clusters are hard.
- **The control (d) design** mixes near-data and far centers (Section 2); a control that replaced random unflagged clusters' summaries with something else of similar kind would be cleaner.

## 6. Reproduce

```
CUDA_VISIBLE_DEVICES="" python scripts/counterfactual_swaps.py \
    --own results/clinc150_qwen3-4b/20261027-221042 --other results/clinc150_qwen3-1.7b/20261027-202532 \
    --cache-dir cache --summary-embeddings results/replacement_diagnostic/summary_embeddings_cpu.npz --out-dir docs/results
```

CPU only. 14.1 minutes on 12 cores (WSL2), 5 seeds in parallel with 2 OpenMP threads each (the thread count affects the summation order of the Lloyd step, so it is fixed). Committed: `counterfactual_swaps_{flags,fidelity,per_seed,draws,paired,gap_closed,stability}.csv` and `counterfactual_swaps_tables.md`. Not committed: every row including each control draw, in the gitignored `results/counterfactual_swaps/`.
