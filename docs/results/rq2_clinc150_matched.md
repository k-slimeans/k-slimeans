# RQ2 matched comparisons

All deltas are adaptive minus reference over the seeds actually paired (listed per table), shown as `mean (sd across seeds; min..max)`. The adaptive configs were compared and one was picked on seeds 0-2, so everything in the first part is in-sample and optimistically biased for the winner; the H2 verdict is judged on held-out seeds 3-9 in the last section.

# Part 1: selection seeds 0-2 (in-sample)

## vs fixed-n5 (default schedule)

| Config | paired seeds | ΔACC | ΔNMI | calls / fixed-n5 |
| --- | ---: | ---: | ---: | ---: |
| adaptive-d0.1-r0.15 | 0-2 | -0.024 (sd 0.013; -0.038..-0.013) | -0.012 (sd 0.002; -0.015..-0.011) | 0.70 |
| adaptive-d0.1-r0.4 | 0-2 | -0.035 (sd 0.010; -0.043..-0.024) | -0.013 (sd 0.003; -0.015..-0.009) | 0.65 |
| adaptive-d0.1-r0.7 | 0-2 | -0.035 (sd 0.010; -0.043..-0.024) | -0.013 (sd 0.003; -0.015..-0.009) | 0.65 |
| adaptive-d0.3-r0.15 | 0-2 | -0.026 (sd 0.020; -0.049..-0.012) | -0.012 (sd 0.005; -0.018..-0.008) | 0.69 |
| adaptive-d0.3-r0.4 | 0-2 | -0.019 (sd 0.019; -0.041..-0.007) | -0.010 (sd 0.009; -0.021..-0.003) | 0.48 |
| adaptive-d0.3-r0.7 | 0-2 | -0.016 (sd 0.008; -0.026..-0.012) | -0.007 (sd 0.002; -0.009..-0.006) | 0.42 |
| adaptive-d0.6-r0.15 | 0-2 | -0.026 (sd 0.020; -0.049..-0.012) | -0.012 (sd 0.005; -0.018..-0.008) | 0.69 |
| adaptive-d0.6-r0.4 | 0-2 | -0.017 (sd 0.016; -0.028..+0.001) | -0.009 (sd 0.005; -0.013..-0.004) | 0.46 |
| adaptive-d0.6-r0.7 | 0-2 | -0.012 (sd 0.018; -0.032..+0.002) | -0.004 (sd 0.007; -0.013..+0.000) | 0.34 |
| adaptive-d0.3-r0.4-s2 | 0-2 | -0.021 (sd 0.004; -0.025..-0.017) | -0.010 (sd 0.002; -0.013..-0.009) | 0.71 |
| adaptive-d0.6-r0.7-s2 | 0-2 | -0.017 (sd 0.023; -0.040..+0.005) | -0.008 (sd 0.008; -0.014..+0.000) | 0.67 |

## vs random control matched on calls

The control is matched on summarizer calls, not on summary resets: it is given the adaptive run's per-step calls and spends them on random clusters among those whose membership changed. The adaptive schedule may also re-apply cached summaries of unchanged clusters for free (drift- or `s_max`-triggered resets), which the control cannot, so `summarized` can differ while calls match. Calls ratios are computed per seed; the flag fires when any seed deviates from 1 by more than 5%.

| Config | paired seeds | ΔACC | ΔNMI | summarized (adaptive / random) | calls (adaptive / random) | random / adaptive calls | max per-seed deviation | budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive-d0.1-r0.15 | 0-2 | +0.000 (sd 0.005; -0.005..+0.005) | +0.003 (sd 0.001; +0.002..+0.004) | 497 / 496 | 496 / 491 | 0.989 | 1.3% | ok |
| adaptive-d0.1-r0.4 | 0-2 | -0.007 (sd 0.030; -0.041..+0.013) | -0.001 (sd 0.017; -0.021..+0.009) | 455 / 454 | 454 / 452 | 0.995 | 0.9% | ok |
| adaptive-d0.1-r0.7 | 0-2 | -0.007 (sd 0.030; -0.041..+0.013) | -0.001 (sd 0.017; -0.021..+0.009) | 455 / 454 | 454 / 452 | 0.995 | 0.9% | ok |
| adaptive-d0.3-r0.15 | 0-2 | +0.004 (sd 0.017; -0.016..+0.016) | -0.001 (sd 0.006; -0.007..+0.006) | 485 / 484 | 484 / 480 | 0.990 | 1.0% | ok |
| adaptive-d0.3-r0.4 | 0-2 | +0.004 (sd 0.019; -0.017..+0.022) | -0.002 (sd 0.007; -0.010..+0.004) | 336 / 335 | 335 / 334 | 0.998 | 0.3% | ok |
| adaptive-d0.3-r0.7 | 0-2 | +0.005 (sd 0.018; -0.014..+0.021) | +0.003 (sd 0.005; -0.002..+0.009) | 295 / 294 | 294 / 293 | 0.998 | 0.7% | ok |
| adaptive-d0.6-r0.15 | 0-2 | +0.004 (sd 0.017; -0.016..+0.016) | -0.001 (sd 0.006; -0.007..+0.006) | 485 / 484 | 484 / 480 | 0.990 | 1.0% | ok |
| adaptive-d0.6-r0.4 | 0-2 | +0.005 (sd 0.019; -0.016..+0.019) | +0.001 (sd 0.008; -0.008..+0.007) | 328 / 327 | 327 / 327 | 0.999 | 0.3% | ok |
| adaptive-d0.6-r0.7 | 0-2 | -0.005 (sd 0.010; -0.013..+0.006) | -0.002 (sd 0.004; -0.005..+0.002) | 241 / 240 | 240 / 239 | 0.994 | 1.2% | ok |
| adaptive-d0.3-r0.4-s2 | 0-2 | +0.000 (sd 0.014; -0.015..+0.012) | +0.000 (sd 0.006; -0.007..+0.004) | 518 / 501 | 501 / 497 | 0.992 | 1.2% | ok |
| adaptive-d0.6-r0.7-s2 | 0-2 | -0.005 (sd 0.021; -0.022..+0.018) | -0.005 (sd 0.007; -0.011..+0.003) | 488 / 473 | 473 / 468 | 0.988 | 1.3% | ok |

## vs the fixed family at matched calls

Interpolation: the two fixed-n points bracketing the adaptive config's mean calls are joined by a chord and evaluated at those calls (per seed). Caveat: accuracy vs calls is typically concave, so a chord lies below the true curve and the interpolated delta can favour the adaptive config; the deltas against the bracketing fixed-n points alone show the same comparison without interpolating. A bracketing point lacking any compared seed is n/a (no partial-seed averaging), and below fixed-n1's or above fixed-n5's calls the missing side is n/a. In the held-out section the bracketing points are the ones the planner extended to seeds 5-9, chosen from the config's calls on the selection seeds; if its held-out calls fall outside their range the chord is clamped to the nearer point.

| Config | paired seeds | mean calls | ΔACC interp. | ΔNMI interp. | ΔACC vs lower neighbour | ΔACC vs upper neighbour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive-d0.1-r0.15 | 0-2 | 496 | -0.019 | -0.009 | -0.018 (n=3) | -0.020 (n=4) |
| adaptive-d0.1-r0.4 | 0-2 | 454 | -0.029 | -0.010 | -0.029 (n=3) | -0.031 (n=4) |
| adaptive-d0.1-r0.7 | 0-2 | 454 | -0.029 | -0.010 | -0.029 (n=3) | -0.031 (n=4) |
| adaptive-d0.3-r0.15 | 0-2 | 484 | -0.021 | -0.010 | -0.020 (n=3) | -0.022 (n=4) |
| adaptive-d0.3-r0.4 | 0-2 | 335 | -0.015 | -0.009 | -0.016 (n=2) | -0.014 (n=3) |
| adaptive-d0.3-r0.7 | 0-2 | 294 | -0.013 | -0.007 | -0.013 (n=2) | -0.011 (n=3) |
| adaptive-d0.6-r0.15 | 0-2 | 484 | -0.021 | -0.010 | -0.020 (n=3) | -0.022 (n=4) |
| adaptive-d0.6-r0.4 | 0-2 | 327 | -0.013 | -0.008 | -0.013 (n=2) | -0.012 (n=3) |
| adaptive-d0.6-r0.7 | 0-2 | 240 | -0.010 | -0.004 | -0.012 (n=1) | -0.008 (n=2) |
| adaptive-d0.3-r0.4-s2 | 0-2 | 501 | -0.017 | -0.007 | -0.016 (n=3) | -0.018 (n=4) |
| adaptive-d0.6-r0.7-s2 | 0-2 | 473 | -0.012 | -0.005 | -0.011 (n=3) | -0.013 (n=4) |

## H2 per config, selection seeds 0-2 (in-sample)

H2 holds for a config if fixed-n5's ACC minus the adaptive ACC is at most 0.01 (within one ACC point, or better) and its calls are at most 50% of fixed-n5's.

| Config | paired seeds | ΔACC | calls / fixed-n5 | H2 |
| --- | ---: | ---: | ---: | ---: |
| adaptive-d0.1-r0.15 | 0-2 | -0.024 (sd 0.013; -0.038..-0.013) | 0.70 | no |
| adaptive-d0.1-r0.4 | 0-2 | -0.035 (sd 0.010; -0.043..-0.024) | 0.65 | no |
| adaptive-d0.1-r0.7 | 0-2 | -0.035 (sd 0.010; -0.043..-0.024) | 0.65 | no |
| adaptive-d0.3-r0.15 | 0-2 | -0.026 (sd 0.020; -0.049..-0.012) | 0.69 | no |
| adaptive-d0.3-r0.4 | 0-2 | -0.019 (sd 0.019; -0.041..-0.007) | 0.48 | no |
| adaptive-d0.3-r0.7 | 0-2 | -0.016 (sd 0.008; -0.026..-0.012) | 0.42 | no |
| adaptive-d0.6-r0.15 | 0-2 | -0.026 (sd 0.020; -0.049..-0.012) | 0.69 | no |
| adaptive-d0.6-r0.4 | 0-2 | -0.017 (sd 0.016; -0.028..+0.001) | 0.46 | no |
| adaptive-d0.6-r0.7 | 0-2 | -0.012 (sd 0.018; -0.032..+0.002) | 0.34 | no |
| adaptive-d0.3-r0.4-s2 | 0-2 | -0.021 (sd 0.004; -0.025..-0.017) | 0.71 | no |
| adaptive-d0.6-r0.7-s2 | 0-2 | -0.017 (sd 0.023; -0.040..+0.005) | 0.67 | no |

# Part 2: held-out seeds 3-9

Config `adaptive-d0.6-r0.7` was chosen on seeds 0-2: highest ACC among configs with calls <= 50% of fixed-n5.

**Overall H2 (held-out seeds 3-9, paired seeds 3-9): SUPPORTED.** ΔACC vs fixed-n5 -0.007 (sd 0.011; -0.028..+0.010; se 0.004; 95% CI [-0.017, +0.004]) at 0.33 of its calls (criterion: fixed-n5 ACC − adaptive ACC ≤ 0.01 and calls ≤ 50%).

Additional check (separate from the verdict above): the lower end of the 95% t-interval of ΔACC (adaptive − fixed-n5) is -0.017, which is not above -0.01, so "within one ACC point" is not supported at 95%.

vs the call-matched random control (paired seeds 3-9): ΔACC -0.013 (sd 0.021; -0.050..+0.013; se 0.008; 95% CI [-0.032, +0.007]), ΔNMI -0.006 (sd 0.007; -0.018..+0.004; se 0.003; 95% CI [-0.012, +0.000]), max per-seed calls deviation 0.4%.

### vs the fixed family at matched calls, held-out seeds 3-9

Interpolation: the two fixed-n points bracketing the adaptive config's mean calls are joined by a chord and evaluated at those calls (per seed). Caveat: accuracy vs calls is typically concave, so a chord lies below the true curve and the interpolated delta can favour the adaptive config; the deltas against the bracketing fixed-n points alone show the same comparison without interpolating. A bracketing point lacking any compared seed is n/a (no partial-seed averaging), and below fixed-n1's or above fixed-n5's calls the missing side is n/a. In the held-out section the bracketing points are the ones the planner extended to seeds 5-9, chosen from the config's calls on the selection seeds; if its held-out calls fall outside their range the chord is clamped to the nearer point.

- vs interpolated fixed family: ΔACC -0.024 (sd 0.014; -0.039..-0.005; se 0.005; 95% CI [-0.037, -0.012]); ΔNMI -0.010 (sd 0.006; -0.017..-0.001; se 0.002; 95% CI [-0.016, -0.005])
- vs fixed-n1 (lower neighbour): ΔACC -0.028 (sd 0.012; -0.042..-0.009; se 0.004; 95% CI [-0.039, -0.018]); ΔNMI -0.011 (sd 0.004; -0.017..-0.005; se 0.002; 95% CI [-0.015, -0.007])
- vs fixed-n2 (upper neighbour): ΔACC -0.022 (sd 0.016; -0.039..-0.003; se 0.006; 95% CI [-0.036, -0.007]); ΔNMI -0.010 (sd 0.007; -0.019..+0.001; se 0.003; 95% CI [-0.016, -0.003])

For reference, on the selection seeds 0-2 (in-sample, optimistically biased): ΔACC vs fixed-n5 -0.012 (sd 0.018; -0.032..+0.002) at 0.34 of its calls (paired seeds 0-2).

| Variant | seeds | calls | clusters summarized | % of fixed-n5 | ACC | NMI | ARI | s/seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed-n5 | 3-9 | 702 ± 4 | 750 | 100.0% | 0.529 ± 0.007 | 0.740 ± 0.004 | 0.429 ± 0.007 | 537.4 |
| adaptive-d0.6-r0.7 | 3-9 | 235 ± 19 | 236 | 33.5% | 0.522 ± 0.014 | 0.739 ± 0.004 | 0.421 ± 0.008 | 170.7 |
| random-d0.6-r0.7 | 3-9 | 235 ± 19 | 235 | 33.5% | 0.535 ± 0.015 | 0.745 ± 0.007 | 0.431 ± 0.014 | 167.1 |
