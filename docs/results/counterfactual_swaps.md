# Do the 4B's flagged summaries cause its lower scores? Counterfactual replacements at step 1 (CLINC150)

Issue #47, stage 1. Preliminary: five seeds, one dataset, one encoder, **one summary step**. Everything was computed offline from the logged runs (#31) with the replay of #33 (`kslimeans.replay`), from the logged summaries re-embedded on the GPU (#33 section 2), which makes the replay exact: the replayed step-1 partition of the 4B equals the logged one in every seed (Section 3). The noise floor of #33 is therefore not a replay error; it says how much a result moves under re-encoding noise of 1e-6 (Section 3). An earlier version of this note used CPU-encoded summaries; every paired ACC difference moved by at most 0.0013 and no headline conclusion changed, but the dup-only result weakened: several dup intervals now include zero (Section 4 says which). No language model was called. Stage 2 (the guarded 4B runs that test whether this carries through to final scores) has not been run.

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
| cd | control for c: the 1.7B's summaries put into the same number of random **unflagged** 4B clusters (same 20 draws as d) |
| m | the 1.7B's summaries as logged |
| r | reverse: the 1.7B's, with the 4B's flagged summaries put into the same clusters |
| rd | control for r: the 4B's summaries put into the same number of random unflagged clusters of the 1.7B (same draws as d) |
| ref | no replacement: plain Lloyd from the k-means state (#33's counterfactual) |

(Additions to the brief: `rd`, so that "the 4B's flagged summaries hurt the 1.7B" can be compared with "any 4B summaries hurt the 1.7B", and `cd`, so that "the 1.7B's summaries in the flagged clusters help" can be compared with "any 1.7B summaries help".)

Each variant is measured after the replacement (iteration 0: nearest-center assignment) and after the block: the objective, ACC (Hungarian), NMI and ARI of the partition, empty clusters, and movers (points whose cluster differs from the pre-replacement one). The tables below give the end-of-block ACC and NMI; ARI, the objective and the replacement-stage measures are in `counterfactual_swaps_tables.md` and `counterfactual_swaps_per_seed.csv`. Differences are paired by seed against (a) (reverse: against m), with Student-t 95% intervals over the 5 seeds (n = 5, t = 2.776), as in #31.

## 2. Results

Absolute levels, means over seeds (fixed block): ACC / NMI of (a) 0.5356 / 0.7493, of the 1.7B (m) 0.5621 / 0.7557, of no replacement (ref) 0.5515 / 0.7482. So at step 1 the 4B is below the 1.7B by 0.026 ACC and 0.006 NMI, and below k-means by 0.0159 ACC but **not below k-means in NMI** (ref minus a: -0.0011 [-0.0050, +0.0028]).

Paired differences, end of block, mean over 5 seeds [95% interval], fixed 20-iteration block. ACC and NMI: positive = higher. "Closes" is the mean difference over the mean gap (ACC): b over the gap to ref, c, cd and c - cd over the gap to m. The relative objective (`end_j_rel`, in the other tables) is a raw relative difference where **lower is better**.

| variant (flag) | flagged | ΔACC | ΔNMI | closes |
| --- | ---: | --- | --- | ---: |
| ref - a (k-means, no replacement) | | +0.0159 [+0.0065, +0.0253] | -0.0011 [-0.0050, +0.0028] | |
| m - a (the 1.7B) | | +0.0265 [+0.0133, +0.0397] | +0.0065 [+0.0025, +0.0104] | |
| b (dup) | 12-24 | -0.0081 [-0.0313, +0.0151] | -0.0092 [-0.0177, -0.0007] | -0.51 |
| b (meta = union) | 43-56 | +0.0064 [-0.0058, +0.0185] | -0.0053 [-0.0092, -0.0013] | 0.40 |
| b (generic) | 62-75 | +0.0108 [-0.0052, +0.0267] | -0.0027 [-0.0083, +0.0029] | 0.68 |
| c (dup) | 12-24 | +0.0147 [-0.0006, +0.0301] | +0.0029 [-0.0018, +0.0075] | 0.56 |
| c (meta = union) | 43-56 | +0.0261 [+0.0185, +0.0337] | +0.0069 [+0.0029, +0.0109] | 0.99 |
| c (generic) | 62-75 | +0.0249 [+0.0162, +0.0336] | +0.0077 [+0.0037, +0.0118] | 0.94 |
| d (dup) | = dup | -0.0187 [-0.0313, -0.0060] | -0.0104 [-0.0153, -0.0056] | |
| d (meta = union) | = union | -0.0279 [-0.0384, -0.0174] | -0.0165 [-0.0226, -0.0103] | |
| d (generic) | = generic | -0.0315 [-0.0391, -0.0240] | -0.0160 [-0.0206, -0.0113] | |
| r - m (dup) | | -0.0114 [-0.0229, +0.0001] | -0.0033 [-0.0086, +0.0021] | |
| r - m (meta = union) | | -0.0270 [-0.0458, -0.0083] | -0.0096 [-0.0159, -0.0034] | |
| r - m (generic) | | -0.0213 [-0.0389, -0.0036] | -0.0050 [-0.0106, +0.0006] | |
| rd - m (dup) | | -0.0028 [-0.0102, +0.0046] | -0.0003 [-0.0023, +0.0017] | |
| rd - m (meta = union) | | -0.0023 [-0.0098, +0.0052] | -0.0001 [-0.0024, +0.0021] | |
| rd - m (generic) | | -0.0030 [-0.0149, +0.0088] | +0.0006 [-0.0031, +0.0043] | |
| cd (dup) | | -0.0040 [-0.0089, +0.0008] | -0.0029 [-0.0058, +0.0001] | -0.15 |
| cd (meta = union) | | -0.0023 [-0.0120, +0.0074] | -0.0030 [-0.0065, +0.0005] | -0.09 |
| cd (generic) | | +0.0005 [-0.0108, +0.0118] | -0.0009 [-0.0028, +0.0010] | 0.02 |
| c - cd (dup) | | +0.0188 [+0.0002, +0.0374] | +0.0057 [-0.0002, +0.0116] | 0.71 |
| c - cd (meta = union) | | +0.0285 [+0.0149, +0.0420] | +0.0099 [+0.0055, +0.0144] | 1.08 |
| c - cd (generic) | | +0.0244 [+0.0053, +0.0435] | +0.0086 [+0.0052, +0.0121] | 0.92 |
| r - rd (dup) | | -0.0086 [-0.0174, +0.0002] | -0.0030 [-0.0075, +0.0016] | |
| r - rd (meta = union) | | -0.0248 [-0.0410, -0.0086] | -0.0095 [-0.0162, -0.0028] | |
| r - rd (generic) | | -0.0183 [-0.0388, +0.0023] | -0.0056 [-0.0108, -0.0004] | |

Same, **run to convergence** (the 4B's (a) runs 53 iterations on average, ref 15):

| variant (flag) | ΔACC | ΔNMI | closes |
| --- | --- | --- | ---: |
| ref - a | +0.0138 [+0.0055, +0.0221] | -0.0026 [-0.0077, +0.0026] | |
| m - a | +0.0258 [+0.0137, +0.0380] | +0.0063 [+0.0005, +0.0121] | |
| b (dup) | -0.0052 [-0.0234, +0.0129] | -0.0074 [-0.0140, -0.0008] | -0.38 |
| b (meta = union) | +0.0060 [-0.0096, +0.0215] | -0.0060 [-0.0125, +0.0004] | 0.43 |
| b (generic) | +0.0092 [-0.0081, +0.0265] | -0.0039 [-0.0120, +0.0041] | 0.67 |
| c (dup) | +0.0153 [-0.0010, +0.0316] | +0.0030 [-0.0048, +0.0107] | 0.59 |
| c (meta = union) | +0.0273 [+0.0187, +0.0359] | +0.0074 [+0.0025, +0.0122] | 1.06 |
| c (generic) | +0.0258 [+0.0188, +0.0327] | +0.0080 [+0.0032, +0.0129] | 1.00 |
| d (dup) | -0.0152 [-0.0277, -0.0027] | -0.0087 [-0.0144, -0.0030] | |
| d (meta = union) | -0.0239 [-0.0337, -0.0140] | -0.0142 [-0.0212, -0.0073] | |
| d (generic) | -0.0266 [-0.0356, -0.0175] | -0.0123 [-0.0201, -0.0046] | |
| r - m (dup) | -0.0126 [-0.0255, +0.0003] | -0.0042 [-0.0086, +0.0002] | |
| r - m (meta = union) | -0.0240 [-0.0447, -0.0033] | -0.0088 [-0.0160, -0.0016] | |
| r - m (generic) | -0.0210 [-0.0377, -0.0042] | -0.0049 [-0.0098, -0.00004] | |
| rd - m (dup) | -0.0024 [-0.0107, +0.0058] | -0.0004 [-0.0023, +0.0016] | |
| rd - m (meta = union) | -0.0022 [-0.0088, +0.0044] | -0.0003 [-0.0014, +0.0008] | |
| rd - m (generic) | -0.0031 [-0.0142, +0.0081] | +0.0004 [-0.0020, +0.0028] | |
| cd (dup) | -0.0024 [-0.0104, +0.0055] | -0.0020 [-0.0074, +0.0034] | -0.09 |
| cd (meta = union) | -0.0013 [-0.0107, +0.0080] | -0.0022 [-0.0080, +0.0036] | -0.05 |
| cd (generic) | +0.0020 [-0.0075, +0.0114] | -0.0001 [-0.0055, +0.0054] | 0.08 |
| c - cd (dup) | +0.0177 [-0.0025, +0.0380] | +0.0050 [-0.0015, +0.0114] | 0.69 |
| c - cd (meta = union) | +0.0286 [+0.0170, +0.0403] | +0.0096 [+0.0045, +0.0147] | 1.11 |
| c - cd (generic) | +0.0238 [+0.0107, +0.0370] | +0.0081 [+0.0067, +0.0096] | 0.92 |
| r - rd (dup) | -0.0102 [-0.0184, -0.0020] | -0.0039 [-0.0078, +0.0001] | |
| r - rd (meta = union) | -0.0218 [-0.0402, -0.0033] | -0.0085 [-0.0159, -0.0011] | |
| r - rd (generic) | -0.0179 [-0.0383, +0.0025] | -0.0054 [-0.0105, -0.0002] | |

Fixed and converged agree in sign and size everywhere (the convergence confound of #33 section 6 does not change any conclusion here). The full set, including ARI, the relative objective, (b) minus (d), (c) minus (d), (r) minus (rd) and the seeds above zero, is in `counterfactual_swaps_tables.md` / `counterfactual_swaps_paired.csv`.

Observations (measured):

- **(c) is the clear effect.** Giving the flagged clusters the 1.7B's summaries lifts the 4B by +0.026 ACC (union) in 5/5 seeds, which is the whole 4B-to-1.7B gap (99% fixed, 106% converged), and +0.007 NMI (the gap is +0.0065). For the duplicate clusters alone (12 to 24 per seed), +0.015 (56% of the gap); that interval now includes zero (fixed [-0.0006, +0.0301], and converged), where it just excluded it with the CPU embeddings. The objective after the block falls by 1.3% (union).
- **Reverse:** putting the 4B's union-flagged summaries into the 1.7B costs -0.027 ACC (0/5 seeds above zero), against -0.002 [-0.010, +0.005] for the same number of random unflagged clusters. The 4B's summaries elsewhere cost the 1.7B nothing measurable.
- **(b) is not shown.** Keeping the k-means center in the flagged clusters changes ACC by +0.006 (union; +0.011 generic; -0.008 dup), every interval includes zero, and NMI goes down (-0.005 union, interval excludes zero when fixed; converged -0.006 [-0.0125, +0.0004]). It closes 40% (union) to 68% (generic) of an ACC gap to k-means that is itself only 0.0159.
- **The control (d) is not neutral, it hurts.** Reverting the k-means center of random unflagged clusters lowers ACC by 0.019 to 0.032 (0/5 seeds above zero) and NMI by 0.010 to 0.017. After the replacement it leaves 28 to 39 empty clusters, against 8 (b, union) and 14 (a): near-data k-means centers sitting beside far-away summary embeddings take points from them. The spread over the 20 draws within a seed is sd 0.006 to 0.011 ACC. So (b) minus (d) is large (+0.034 union [+0.025, +0.044]) mainly because (d) damages the partition, not because (b) repairs it; **(b) against (a) is the cleaner comparison and it is inconclusive.** Why the mixture hurts has not been tested.

## 3. Noise floor and what is not interpretable

- **(a) replayed against the logged step-1 partition** (`counterfactual_swaps_fidelity.csv`; the replay starts from the GPU-encoded summaries, as the logged run did). 0 of 22,500 points differ in every seed for both the 4B and the 1.7B, the number of iterations is the logged one (20), and the replayed ACC and NMI equal the logged ones. So the table's (a) is the logged 4B (mean ACC 0.5356). With the CPU-encoded summaries the 4B replay differed from the logged partition in 421, 293, 759, 1,937 and 0 points (seeds 0-4) and in ACC by up to 0.0040; that was the encoder noise measured in #33 section 2.
- **Noise floor** from #33 section 2 for the 4B under tie-preserving noise of the size of the CPU/GPU encoder difference (a re-encode, not a replay error): mean absolute change 0.0030 ACC (at most 0.0084), 0.0019 NMI. Differences below that are **uninterpretable**. Against the mean floor, (c) union (+0.026 ACC, about 9 times), (r) union (-0.027, about 9 times) and (d) (-0.019 to -0.032) are far larger, and c-dup (+0.015) is about 5 times; against the maximum floor, c-dup is about 1.8 times and the others 2.2 to 3.8 times. The NMI effects (+0.0069 for (c) union, against a mean floor of 0.0019) are about 4 times. (b) union (+0.006 ACC) is below the maximum of the floor, and its interval includes zero anyway; (rd) (-0.002 to -0.003) is inside the floor, and (r) dup (-0.011) is 1.4 times its maximum with an interval that includes zero (fixed and converged), so for those the interval is the guide.
- **Stability under perturbation** (`counterfactual_swaps_stability.csv`): the paired ACC and NMI differences of b, c and r were recomputed from three tie-preserving perturbations of both models' embeddings (`start * (1 + 1e-6 N(0, 1))`, identical texts get the same noise; the k-means centers are not perturbed, being exact). A single perturbed rerun moved the mean ACC difference by at most 0.0025 (the mean over the three reruns by at most 0.0014) (for example c union fixed +0.0261 against +0.0261 to +0.0287; b union +0.0064 against +0.0064 to +0.0089) and kept its sign in all 3 reruns for all 12 comparisons (and for NMI). The unperturbed reference is now the exact replay, so this measures the effect of a further 1e-6 re-encoding on top of the logged state.
- The `iid` noise rows of #33 (which break ties) were not used.

## 4. Reading against #47's criteria

Criterion: flagged summaries cause the drop if (b) and (c) close most of the gap to k-means or the 1.7B, (d) does not, and (for the reverse) they hurt the 1.7B; stage 2 must recover the final scores.

- **(c) holds, and it is specific to the flagged clusters.** The 1.7B's summaries in the flagged clusters close about all of the gap to the 1.7B (meta = union 99% fixed, 106% converged; generic 94%, 100%; dup 56%, 59%), 5/5 seeds, about 9 times the mean noise floor (0.0030 ACC) and 3.1 times its maximum (0.0084). The matched control `cd` (the same number of random unflagged 4B clusters taking the 1.7B's summaries) closes **-9%** of the gap (meta; ACC -0.0023 [-0.0120, +0.0074] fixed, -0.0013 converged; dup -15%, generic +2%): nothing. So the gain is not "1.7B summaries are better everywhere". **c - cd** (meta = union) is +0.0285 [+0.0149, +0.0420] ACC fixed and +0.0286 [+0.0170, +0.0403] converged, 5/5 seeds, +0.010 NMI [+0.006, +0.014] fixed; generic +0.0244 [+0.0053, +0.0435] fixed and +0.0238 [+0.0107, +0.0370] converged; dup +0.0188 [+0.0002, +0.0374] fixed and +0.0177 [-0.0025, +0.0380] converged (the duplicate clusters alone: the interval just excludes zero fixed and includes it converged). Shares of the gap closed by c - cd: meta 1.08 fixed, 1.11 converged; generic 0.92, 0.92; dup 0.71, 0.69 (`counterfactual_swaps_gap_closed.csv`, variant `c-cd`). These are ratios of means, and the gap they divide by (m - a, +0.0265 [+0.0133, +0.0397]) is itself uncertain, so a share above 1 means "about all of it", not more than all.
- **The reverse holds the same way.** The 4B's flagged summaries lower the 1.7B by 0.027 ACC (meta = union), the same number of random 4B summaries by 0.002; **r - rd** is -0.0248 [-0.0410, -0.0086] fixed and -0.0218 [-0.0402, -0.0033] converged (NMI -0.0095 [-0.0162, -0.0028] fixed); generic -0.0183 [-0.0388, +0.0023] fixed, -0.0179 [-0.0383, +0.0025] converged (both intervals include zero, where the converged one just excluded it with the CPU embeddings); dup -0.0086 [-0.0174, +0.0002] fixed (now includes zero) and -0.0102 [-0.0184, -0.0020] converged. The loss that r - rd opens is 93% fixed and 84% converged (meta = union), 69% and 69% (generic), 32% and 39% (dup) of the 1.7B's lead over the 4B (variant `r-rd`, same file). The two directions agree: the effect travels with the summary text of the flagged clusters, and is not something else in the 4B run or a property of every 1.7B (or 4B) summary. **(d) closes nothing**, but because it harms the partition, so it does not play its intended role of showing that (b) works only through replacing fewer centroids. meta and union are the same set (dup is a subset of meta in every seed), so their rows are one result, not two.
- **(b) does not hold at step 1.** Falling back to the k-means center in the flagged clusters does not recover a measurable amount of ACC and lowers NMI. In ACC the 4B at step 1 is below no replacement (ref - a = +0.0159 [+0.0065, +0.0253], 5/5 seeds, fixed; +0.0138 [+0.0055, +0.0221] converged) and below the 1.7B; in NMI it is **not** below no replacement (ref - a = -0.0011 [-0.0050, +0.0028]). (b) does not show that the flagged set explains the 4B's ACC deficit to the unreplaced k-means state: neither the flagged summaries as a set nor any flag recovers it. What the data do show is that the flagged 4B summaries are worse than the 1.7B's summaries of the same clusters.
- **So: partly.** This is a step-1 counterfactual on logged state, and it supports a step-1 effect of the flagged summaries: replacing them with the 1.7B's removes the 4B's deficit to the 1.7B (against a matched control that does nothing), and the reverse transplant creates a comparable deficit (against a control that does nothing), so at step 1 the flagged summaries (as a set, by meta/union or generic) carry the difference between the two models. It does not show that they explain the 4B's whole gap in whole runs, and, through (b), it does not show that the flagged set accounts for the 4B's ACC deficit to the unreplaced k-means state (+0.0159); neither is established here. The remaining question, whether the effect survives later steps, is stage 2.
- Flag by flag: dup alone (12 to 24 clusters) recovers about 56% of the 1.7B gap in (c) and costs the 1.7B -0.011 in (r), but both intervals now include zero (c - a, r - m) when fixed, so dup alone is suggestive, not shown; meta/union and generic recover about all of it. Because generic contains the union and the union is 64% to 80% of generic, the data cannot separate the flags' contributions beyond "more flagged clusters, more effect".

## 5. Limits

- **Flagged clusters are 29% to 37% (meta) and 41% to 50% (generic) of all clusters,** so "random unflagged clusters" are not a size-matched set of points; the controls match the number of clusters, not of points (flagged clusters hold 32% to 42% of the points for meta, 46% to 53% for generic).
- **One step.** Later steps need new summaries of the new partitions, which cannot be produced offline. The statement supported is: at step 1, replacing the flagged summaries with X changes the block's ACC by Y. It does not carry to whole runs; the final logged scores (#31) are not tested here. Stage 2 (GPU) does that.
- **Encoder noise.** The 4B's blocks are sensitive to 1e-7 changes in the start centers (Section 3). The (a) replay equals the logged 4B partition exactly with the GPU embeddings; with the CPU embeddings it differed in up to 1,937 points, and the paired differences moved by at most 0.0013 ACC. The perturbation reruns show the differences are stable to a further 1e-6 re-encode; they do not say how the results would look with another encoder or device beyond that.
- **Five seeds,** one dataset, one encoder, one model pair. Intervals are t-intervals with 4 degrees of freedom. About 100 comparisons are listed; no multiplicity correction. The dup, meta and generic rows share the same seeds and overlapping clusters, so they are not independent confirmations; union rows are identical to meta rows.
- **The flags describe clusters, which differ in other ways.** Flagged clusters may be harder clusters (mixed intents), and the replaced summary comes from a model that saw the same members. The swap is an intervention on the summary text of those clusters, but not on which clusters are hard.
- **The control (d) design** mixes near-data and far centers (Section 2); a control that replaced random unflagged clusters' summaries with something else of similar kind would be cleaner.

## 6. Reproduce

```
CUDA_VISIBLE_DEVICES="" python scripts/counterfactual_swaps.py \
    --own results/clinc150_qwen3-4b/20261027-221042 --other results/clinc150_qwen3-1.7b/20261027-202532 \
    --cache-dir cache --summary-embeddings results/replacement_diagnostic/summary_embeddings_cuda.npz --out-dir docs/results
```

`summary_embeddings_cuda.npz` is the GPU encode of the logged summaries (`scripts/embed_logged_summaries.py ... --device cuda`, about 20 s, see #33 section 9). Without a GPU the CPU encode `summary_embeddings_cpu.npz` can be passed instead; it is the fallback and costs exactness (the (a) replay then differs from the logged 4B partition).

CPU only. `--rebuild` recomputes the CSVs and tables from the saved rows in seconds (no replay). 20.5 minutes on 12 cores in the GPU-embedding rerun (run niced alongside other jobs; the CPU-embedding run took 18.8 minutes with the `cd` control, 14.1 without) (WSL2), 5 seeds in parallel with 2 OpenMP threads each (the thread count affects the summation order of the Lloyd step, so it is fixed). Committed: `counterfactual_swaps_{flags,fidelity,per_seed,draws,paired,gap_closed,stability}.csv` and `counterfactual_swaps_tables.md`. Not committed: every row including each control draw, in the gitignored `results/counterfactual_swaps/`.
