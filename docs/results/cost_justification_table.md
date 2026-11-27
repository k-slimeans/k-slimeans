Timing codes (column `timing`):

- `-`: CPU method, no card involved
- `A`: card idle-checked at the start and clean during the run (within about our own CUDA context)
- `B`: card not idle-checked (run predates #35); seconds inflated, step-1 inflation measured (CLINC150 October prompt)
- `C`: card not idle-checked (run predates #35); seconds probably inflated, inflation not measured
- `D`: card idle at the start but foreign memory seen during the run: not a clean timing
- `E`: run overlapped other jobs: an upper bound on idle-card seconds

## CLINC150 (intent)

### Cost

| Method | seeds | batch | calls | input tokens | output tokens | seconds | local $ per seed | API $ per seed (claude-sonnet-4-6) | API $ per seed (deepseek-v4-pro) | API $ per seed (gpt-4o) | idle-card s per call (batch) | timing | timing source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| k-means | 5 | n/a | n/a | 0 | 0 | 0 (no summarizer) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU method; total seconds as logged |
| k-NLPmeans (LSA) | 5 | n/a | 636 | 0 | 0 | 14.8 ± 1.8 | 0.0010 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU summarizer; seconds as logged (the card is not used for the summaries) |
| k-SLMmeans Qwen3-0.6B, October (#31) | 5 | 8 | 707 | 1,113,037 | 10,227 | 895.8 ± 29.7 | 0.0602 | 3.4925 | 1.5097 | 2.8849 | 0.63 (8) | B | logged, batch 8; card not idle-checked (run predates #35), so the seconds are probably inflated; step 1 of seed 0 measured 1.76x slower than on an idle card (whole-run inflation not measured) |
| k-SLMmeans Qwen3-1.7B, October (#31) | 5 | 4 | 702 | 1,101,755 | 18,288 | 1,238.6 ± 102.5 | 0.0833 | 3.5796 | 1.5267 | 2.9373 | 1.11 (4) | B | logged, batch 4; card not idle-checked (run predates #35), so the seconds are probably inflated; step 1 of seed 0 measured 1.72x slower than on an idle card (whole-run inflation not measured) |
| k-SLMmeans Qwen3-4B, October (#31) | 5 | 1 | 714 | 1,120,190 | 15,620 | 1,855.0 ± 188.6 | 0.1247 | 3.5949 | 1.5405 | 2.9567 | 2.10 (1) | B | logged, batch 1; card not idle-checked (run predates #35), so the seconds are probably inflated; step 1 of seed 0 measured 1.15x slower than on an idle card (whole-run inflation not measured) |
| k-SLMmeans Qwen3-0.6B, v2 (#46) | 5 | 8 | 689 | 1,120,788 | 7,806 | 373.4 ± 8.5 | 0.0251 | 3.4795 | 1.5104 | 2.8800 | n/a | A | logged, batch 8; card idle at start, +139 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |
| k-SLMmeans Qwen3-1.7B, v2 (#46) | 5 | 4 | 701 | 1,137,714 | 11,239 | 737.0 ± 198.1 | 0.0495 | 3.5817 | 1.5463 | 2.9567 | n/a | D | logged, batch 4; card idle at start but not clean during the run, +1,625 MiB non-torch memory above the pre-run baseline, not a clean timing |
| k-SLMmeans Qwen3-4B, v2 (#46) | 5 | 1 | 698 | 1,131,772 | 9,068 | 1,093.7 ± 20.7 | 0.0735 | 3.5313 | 1.5298 | 2.9201 | n/a | A | logged, batch 1; card idle at start, +139 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |

### Benefit (ACC, NMI, ARI; paired difference, 95% interval over seeds)

| Method | ACC | NMI | ARI | ACC vs LSA | NMI vs LSA | ARI vs LSA | ACC vs k-means | NMI vs k-means | ARI vs k-means |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.551 ± 0.007 | 0.748 ± 0.002 | 0.439 ± 0.005 | n/a | n/a | n/a | n/a | n/a | n/a |
| k-NLPmeans (LSA) | 0.560 ± 0.008 | 0.752 ± 0.004 | 0.449 ± 0.007 | n/a | n/a | n/a | +0.009 [+0.001, +0.017] | +0.004 [+0.001, +0.008] | +0.010 [+0.002, +0.018] |
| k-SLMmeans Qwen3-0.6B, October (#31) | 0.541 ± 0.013 | 0.747 ± 0.003 | 0.437 ± 0.008 | -0.019 [-0.032, -0.007] | -0.005 [-0.012, +0.001] | -0.012 [-0.023, -0.001] | -0.010 [-0.018, -0.002] | -0.001 [-0.004, +0.003] | -0.001 [-0.007, +0.004] |
| k-SLMmeans Qwen3-1.7B, October (#31) | 0.570 ± 0.011 | 0.760 ± 0.006 | 0.461 ± 0.012 | +0.010 [-0.011, +0.030] | +0.008 [-0.002, +0.018] | +0.012 [-0.007, +0.031] | +0.018 [+0.004, +0.033] | +0.012 [+0.005, +0.019] | +0.022 [+0.011, +0.034] |
| k-SLMmeans Qwen3-4B, October (#31) | 0.511 ± 0.015 | 0.740 ± 0.006 | 0.421 ± 0.010 | -0.049 [-0.072, -0.026] | -0.013 [-0.023, -0.002] | -0.028 [-0.043, -0.014] | -0.040 [-0.058, -0.022] | -0.009 [-0.017, -0.000] | -0.018 [-0.027, -0.009] |
| k-SLMmeans Qwen3-0.6B, v2 (#46) | 0.563 ± 0.008 | 0.755 ± 0.002 | 0.453 ± 0.006 | +0.002 [-0.006, +0.010] | +0.002 [-0.000, +0.005] | +0.004 [-0.004, +0.012] | +0.011 [+0.005, +0.017] | +0.007 [+0.005, +0.008] | +0.014 [+0.008, +0.021] |
| k-SLMmeans Qwen3-1.7B, v2 (#46) | 0.563 ± 0.007 | 0.754 ± 0.006 | 0.454 ± 0.010 | +0.003 [-0.003, +0.008] | +0.002 [-0.004, +0.007] | +0.004 [-0.006, +0.015] | +0.012 [+0.005, +0.018] | +0.006 [+0.001, +0.011] | +0.015 [+0.005, +0.025] |
| k-SLMmeans Qwen3-4B, v2 (#46) | 0.568 ± 0.010 | 0.760 ± 0.005 | 0.457 ± 0.012 | +0.007 [-0.001, +0.016] | +0.008 [+0.002, +0.014] | +0.008 [-0.002, +0.017] | +0.016 [+0.009, +0.023] | +0.012 [+0.007, +0.017] | +0.018 [+0.007, +0.029] |

### Cost per ACC point (only where the paired interval excludes 0)

| Method | ACC vs LSA | extra s vs LSA | s per ACC point vs LSA | local $ per ACC point vs LSA | API $ per ACC point vs LSA (claude-sonnet-4-6) | API $ per ACC point vs LSA (deepseek-v4-pro) | API $ per ACC point vs LSA (gpt-4o) | ACC vs k-means | extra s vs k-means | s per ACC point vs k-means | local $ per ACC point vs k-means | API $ per ACC point vs k-means (claude-sonnet-4-6) | API $ per ACC point vs k-means (deepseek-v4-pro) | API $ per ACC point vs k-means (gpt-4o) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a |  |  |  |  | n/a | n/a | n/a |  |  |  |  |
| k-NLPmeans (LSA) | n/a | n/a | n/a |  |  |  |  | +0.89 | +14.8 | 16.6 | 0.0011 | 0.0000 | 0.0000 | 0.0000 |
| k-SLMmeans Qwen3-0.6B, October (#31) | -1.94 | +881.0 | measured loss |  |  |  |  | -1.04 | +895.8 | measured loss |  |  |  |  |
| k-SLMmeans Qwen3-1.7B, October (#31) | +0.95 | +1,223.8 | no measured benefit |  |  |  |  | +1.84 | +1,238.6 | 672.5 | 0.0452 | 1.9435 | 0.8289 | 1.5948 |
| k-SLMmeans Qwen3-4B, October (#31) | -4.92 | +1,840.2 | measured loss |  |  |  |  | -4.02 | +1,855.0 | measured loss |  |  |  |  |
| k-SLMmeans Qwen3-0.6B, v2 (#46) | +0.22 | +358.6 | no measured benefit |  |  |  |  | +1.12 | +373.4 | 334.7 | 0.0225 | 3.1190 | 1.3539 | 2.5817 |
| k-SLMmeans Qwen3-1.7B, v2 (#46) | +0.27 | +722.2 | no measured benefit |  |  |  |  | +1.16 | +737.0 | 635.3 | 0.0427 | 3.0877 | 1.3330 | 2.5489 |
| k-SLMmeans Qwen3-4B, v2 (#46) | +0.72 | +1,078.9 | no measured benefit |  |  |  |  | +1.62 | +1,093.7 | 676.8 | 0.0455 | 2.1852 | 0.9467 | 1.8070 |

## BANKING77 (#20)

### Cost

| Method | seeds | batch | calls | input tokens | output tokens | seconds | local $ per seed | API $ per seed (claude-sonnet-4-6) | API $ per seed (deepseek-v4-pro) | API $ per seed (gpt-4o) | idle-card s per call (batch) | timing | timing source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| k-means | 5 | n/a | n/a | 0 | 0 | 0.3 ± 0.0 (run total) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU method; total seconds as logged |
| k-NLPmeans (LSA) | 5 | n/a | not logged | 0 | 0 | 1.6 ± 0.2 (run total) | 0.0001 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU summarizer; seconds as logged (the card is not used for the summaries) |
| k-SLMmeans Qwen3-0.6B, upstream BANKING77 | 5 | 8 | not logged | not logged | not logged | 102.7 ± 2.3 (run total) | 0.0069 |  |  |  | n/a | C | logged, batch 8; card not idle-checked (run predates #35), so the seconds are probably inflated; inflation not measured for this run (one BANKING77 check estimated about 1.7x) |
| k-SLMmeans Qwen3-1.7B, upstream BANKING77 | 5 | 4 | not logged | not logged | not logged | 229.0 ± 8.5 (run total) | 0.0154 |  |  |  | n/a | C | logged, batch 4; card not idle-checked (run predates #35), so the seconds are probably inflated; inflation not measured for this run (one BANKING77 check estimated about 1.7x) |
| k-SLMmeans Qwen3-4B, upstream BANKING77 | 5 | 2 | not logged | not logged | not logged | 381.9 ± 8.2 (run total) | 0.0257 |  |  |  | n/a | C | logged, batch 2; card not idle-checked (run predates #35), so the seconds are probably inflated; inflation not measured for this run (one BANKING77 check estimated about 1.7x) |

### Benefit (ACC, NMI, ARI; paired difference, 95% interval over seeds)

| Method | ACC | NMI | ARI | ACC vs LSA | NMI vs LSA | ARI vs LSA | ACC vs k-means | NMI vs k-means | ARI vs k-means |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.435 ± 0.005 | 0.643 ± 0.003 | 0.295 ± 0.006 | n/a | n/a | n/a | n/a | n/a | n/a |
| k-NLPmeans (LSA) | 0.449 ± 0.009 | 0.651 ± 0.002 | 0.305 ± 0.005 | n/a | n/a | n/a | +0.014 [+0.006, +0.022] | +0.008 [+0.003, +0.013] | +0.010 [+0.003, +0.018] |
| k-SLMmeans Qwen3-0.6B, upstream BANKING77 | 0.440 ± 0.010 | 0.645 ± 0.002 | 0.293 ± 0.006 | -0.009 [-0.025, +0.007] | -0.007 [-0.011, -0.003] | -0.012 [-0.018, -0.006] | +0.005 [-0.011, +0.021] | +0.002 [-0.002, +0.005] | -0.002 [-0.007, +0.003] |
| k-SLMmeans Qwen3-1.7B, upstream BANKING77 | 0.436 ± 0.010 | 0.648 ± 0.004 | 0.292 ± 0.008 | -0.012 [-0.033, +0.008] | -0.004 [-0.009, +0.002] | -0.013 [-0.023, -0.004] | +0.002 [-0.016, +0.020] | +0.005 [+0.001, +0.009] | -0.003 [-0.015, +0.010] |
| k-SLMmeans Qwen3-4B, upstream BANKING77 | 0.444 ± 0.016 | 0.648 ± 0.009 | 0.300 ± 0.014 | -0.005 [-0.033, +0.022] | -0.003 [-0.015, +0.010] | -0.005 [-0.029, +0.018] | +0.009 [-0.011, +0.029] | +0.005 [-0.008, +0.019] | +0.005 [-0.018, +0.028] |

### Cost per ACC point (only where the paired interval excludes 0)

| Method | ACC vs LSA | extra s vs LSA | s per ACC point vs LSA | local $ per ACC point vs LSA | API $ per ACC point vs LSA (claude-sonnet-4-6) | API $ per ACC point vs LSA (deepseek-v4-pro) | API $ per ACC point vs LSA (gpt-4o) | ACC vs k-means | extra s vs k-means | s per ACC point vs k-means | local $ per ACC point vs k-means | API $ per ACC point vs k-means (claude-sonnet-4-6) | API $ per ACC point vs k-means (deepseek-v4-pro) | API $ per ACC point vs k-means (gpt-4o) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a |  |  |  |  | n/a | n/a | n/a |  |  |  |  |
| k-NLPmeans (LSA) | n/a | n/a | n/a |  |  |  |  | +1.42 | +1.3 | 0.9 | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| k-SLMmeans Qwen3-0.6B, upstream BANKING77 | -0.93 | +101.2 | no measured benefit |  |  |  |  | +0.49 | +102.5 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-1.7B, upstream BANKING77 | -1.23 | +227.5 | no measured benefit |  |  |  |  | +0.18 | +228.8 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-4B, upstream BANKING77 | -0.53 | +380.3 | no measured benefit |  |  |  |  | +0.89 | +381.6 | no measured benefit |  |  |  |  |

## BBC, TF-IDF (#18)

### Cost

| Method | seeds | batch | calls | input tokens | output tokens | seconds | local $ per seed | API $ per seed (claude-sonnet-4-6) | API $ per seed (deepseek-v4-pro) | API $ per seed (gpt-4o) | idle-card s per call (batch) | timing | timing source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| k-means | 5 | n/a | n/a | 0 | 0 | 0 (no summarizer) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU method; total seconds as logged |
| k-NLPmeans (LSA) | 5 | n/a | 21 | 0 | 0 | 2.6 ± 0.7 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU summarizer; seconds as logged (the card is not used for the summaries) |
| k-SLMmeans Qwen3-0.6B, BBC prompt | 5 | 8 | 23 | 83,813 | 588 | 26.5 ± 1.4 | 0.0018 | 0.2603 | 0.1130 | 0.2154 | n/a | A | logged, batch 8; card idle at start, +137 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |
| k-SLMmeans Qwen3-1.7B, BBC prompt | 5 | 4 | 24 | 86,898 | 868 | 46.2 ± 3.3 | 0.0031 | 0.2737 | 0.1181 | 0.2259 | n/a | A | logged, batch 4; card idle at start, +139 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |
| k-SLMmeans Qwen3-4B, BBC prompt | 5 | 1 | 24 | 87,474 | 732 | 87.2 ± 4.5 | 0.0059 | 0.2734 | 0.1184 | 0.2260 | n/a | A | logged, batch 1; card idle at start, +137 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |

### Benefit (ACC, NMI, ARI; paired difference, 95% interval over seeds)

| Method | ACC | NMI | ARI | ACC vs LSA | NMI vs LSA | ARI vs LSA | ACC vs k-means | NMI vs k-means | ARI vs k-means |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.788 ± 0.080 | 0.659 ± 0.062 | 0.592 ± 0.098 | n/a | n/a | n/a | n/a | n/a | n/a |
| k-NLPmeans (LSA) | 0.839 ± 0.033 | 0.679 ± 0.033 | 0.641 ± 0.075 | n/a | n/a | n/a | +0.051 [-0.020, +0.121] | +0.020 [-0.037, +0.077] | +0.049 [-0.011, +0.109] |
| k-SLMmeans Qwen3-0.6B, BBC prompt | 0.851 ± 0.022 | 0.697 ± 0.025 | 0.668 ± 0.048 | +0.012 [-0.051, +0.075] | +0.018 [-0.046, +0.082] | +0.027 [-0.114, +0.167] | +0.063 [-0.053, +0.179] | +0.038 [-0.060, +0.136] | +0.076 [-0.092, +0.244] |
| k-SLMmeans Qwen3-1.7B, BBC prompt | 0.873 ± 0.021 | 0.732 ± 0.027 | 0.717 ± 0.043 | +0.034 [-0.014, +0.082] | +0.052 [+0.001, +0.103] | +0.076 [-0.030, +0.182] | +0.085 [-0.009, +0.178] | +0.072 [-0.001, +0.146] | +0.125 [+0.007, +0.243] |
| k-SLMmeans Qwen3-4B, BBC prompt | 0.854 ± 0.022 | 0.702 ± 0.024 | 0.670 ± 0.052 | +0.015 [-0.015, +0.045] | +0.023 [-0.002, +0.049] | +0.029 [-0.038, +0.095] | +0.066 [-0.027, +0.159] | +0.043 [-0.023, +0.110] | +0.078 [-0.036, +0.192] |

### Cost per ACC point (only where the paired interval excludes 0)

| Method | ACC vs LSA | extra s vs LSA | s per ACC point vs LSA | local $ per ACC point vs LSA | API $ per ACC point vs LSA (claude-sonnet-4-6) | API $ per ACC point vs LSA (deepseek-v4-pro) | API $ per ACC point vs LSA (gpt-4o) | ACC vs k-means | extra s vs k-means | s per ACC point vs k-means | local $ per ACC point vs k-means | API $ per ACC point vs k-means (claude-sonnet-4-6) | API $ per ACC point vs k-means (deepseek-v4-pro) | API $ per ACC point vs k-means (gpt-4o) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a |  |  |  |  | n/a | n/a | n/a |  |  |  |  |
| k-NLPmeans (LSA) | n/a | n/a | n/a |  |  |  |  | +5.08 | +2.6 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-0.6B, BBC prompt | +1.20 | +24.0 | no measured benefit |  |  |  |  | +6.28 | +26.5 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-1.7B, BBC prompt | +3.40 | +43.6 | no measured benefit |  |  |  |  | +8.48 | +46.2 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-4B, BBC prompt | +1.52 | +84.6 | no measured benefit |  |  |  |  | +6.60 | +87.2 | no measured benefit |  |  |  |  |

## BBC, dense (#18)

### Cost

| Method | seeds | batch | calls | input tokens | output tokens | seconds | local $ per seed | API $ per seed (claude-sonnet-4-6) | API $ per seed (deepseek-v4-pro) | API $ per seed (gpt-4o) | idle-card s per call (batch) | timing | timing source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| k-means | 5 | n/a | n/a | 0 | 0 | 0 (no summarizer) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU method; total seconds as logged |
| k-NLPmeans (LSA) | 5 | n/a | 13 | 0 | 0 | 0.8 ± 0.3 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU summarizer; seconds as logged (the card is not used for the summaries) |
| k-SLMmeans Qwen3-0.6B, BBC prompt | 5 | 8 | 15 | 59,061 | 351 | 17.3 ± 6.5 | 0.0012 | 0.1824 | 0.0793 | 0.1512 | n/a | A | logged, batch 8; card idle at start, +139 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |
| k-SLMmeans Qwen3-1.7B, BBC prompt | 5 | 4 | 13 | 47,923 | 425 | 24.3 ± 13.0 | 0.0016 | 0.1501 | 0.0649 | 0.1241 | n/a | A | logged, batch 4; card idle at start, +139 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |
| k-SLMmeans Qwen3-4B, BBC prompt | 5 | 1 | 11 | 41,438 | 311 | 38.8 ± 17.0 | 0.0026 | 0.1290 | 0.0559 | 0.1067 | n/a | A | logged, batch 1; card idle at start, +139 MiB non-torch memory above the pre-run baseline (about our own CUDA context) |

### Benefit (ACC, NMI, ARI; paired difference, 95% interval over seeds)

| Method | ACC | NMI | ARI | ACC vs LSA | NMI vs LSA | ARI vs LSA | ACC vs k-means | NMI vs k-means | ARI vs k-means |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.866 ± 0.129 | 0.823 ± 0.086 | 0.798 ± 0.139 | n/a | n/a | n/a | n/a | n/a | n/a |
| k-NLPmeans (LSA) | 0.897 ± 0.131 | 0.845 ± 0.073 | 0.837 ± 0.128 | n/a | n/a | n/a | +0.032 [-0.091, +0.154] | +0.022 [-0.090, +0.133] | +0.038 [-0.113, +0.190] |
| k-SLMmeans Qwen3-0.6B, BBC prompt | 0.960 ± 0.001 | 0.888 ± 0.004 | 0.904 ± 0.002 | +0.063 [-0.100, +0.226] | +0.044 [-0.046, +0.133] | +0.067 [-0.091, +0.225] | +0.095 [-0.064, +0.254] | +0.065 [-0.039, +0.170] | +0.105 [-0.066, +0.277] |
| k-SLMmeans Qwen3-1.7B, BBC prompt | 0.904 ± 0.127 | 0.854 ± 0.077 | 0.845 ± 0.132 | +0.006 [+0.001, +0.012] | +0.009 [+0.003, +0.016] | +0.008 [+0.003, +0.014] | +0.038 [-0.084, +0.160] | +0.031 [-0.087, +0.149] | +0.047 [-0.109, +0.202] |
| k-SLMmeans Qwen3-4B, BBC prompt | 0.902 ± 0.130 | 0.848 ± 0.076 | 0.843 ± 0.134 | +0.005 [+0.001, +0.009] | +0.003 [-0.006, +0.012] | +0.006 [-0.004, +0.016] | +0.036 [-0.083, +0.156] | +0.025 [-0.082, +0.131] | +0.045 [-0.104, +0.193] |

### Cost per ACC point (only where the paired interval excludes 0)

| Method | ACC vs LSA | extra s vs LSA | s per ACC point vs LSA | local $ per ACC point vs LSA | API $ per ACC point vs LSA (claude-sonnet-4-6) | API $ per ACC point vs LSA (deepseek-v4-pro) | API $ per ACC point vs LSA (gpt-4o) | ACC vs k-means | extra s vs k-means | s per ACC point vs k-means | local $ per ACC point vs k-means | API $ per ACC point vs k-means (claude-sonnet-4-6) | API $ per ACC point vs k-means (deepseek-v4-pro) | API $ per ACC point vs k-means (gpt-4o) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a |  |  |  |  | n/a | n/a | n/a |  |  |  |  |
| k-NLPmeans (LSA) | n/a | n/a | n/a |  |  |  |  | +3.16 | +0.8 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-0.6B, BBC prompt | +6.32 | +16.5 | no measured benefit |  |  |  |  | +9.48 | +17.3 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-1.7B, BBC prompt | +0.64 | +23.5 | 36.6 | 0.0025 | 0.2346 | 0.1015 | 0.1938 | +3.80 | +24.3 | no measured benefit |  |  |  |  |
| k-SLMmeans Qwen3-4B, BBC prompt | +0.48 | +38.0 | 79.1 | 0.0053 | 0.2687 | 0.1165 | 0.2223 | +3.64 | +38.8 | no measured benefit |  |  |  |  |

## arXiv 200k pilot (#48, seed 0)

One seed, descriptive only: no paired intervals, so no cost per benefit.

### Cost

| Method | seeds | batch | calls | input tokens | output tokens | seconds | local $ per seed | API $ per seed (claude-sonnet-4-6) | API $ per seed (deepseek-v4-pro) | API $ per seed (gpt-4o) | idle-card s per call (batch) | timing | timing source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| k-means | 1 | n/a | n/a | 0 | 0 | 118.4 ± n/a (run total) | 0.0080 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU method; total seconds as logged |
| k-NLPmeans (LSA) | 1 | n/a | not logged | 0 | 0 | 38.7 ± n/a (run total) | 0.0026 | 0.0000 | 0.0000 | 0.0000 | n/a | - | CPU summarizer; seconds as logged (the card is not used for the summaries); excludes 3,319 s of one-off spaCy sentence segmentation (not summarizer time, CPU shared) |
| k-SLMmeans Qwen3-1.7B, arXiv prompt | 1 | 4 | not logged | not logged | not logged | 5,884.7 ± n/a (run total) | 0.3956 |  |  |  | n/a | E | logged, batch 4; card idle at start but not clean during the run, +596 MiB non-torch memory above the pre-run baseline, not a clean timing; whole-method runtime incl. 8 s embedding and the k-means refits; ran next to other jobs (an upper bound on idle-card seconds, not clean); calls at most 3,360 scheduled, not logged |

### Benefit (ACC, NMI, ARI; paired difference, 95% interval over seeds)

| Method | ACC | NMI | ARI | ACC vs LSA | NMI vs LSA | ARI vs LSA | ACC vs k-means | NMI vs k-means | ARI vs k-means |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | 0.274 ± n/a | 0.542 ± n/a | 0.171 ± n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| k-NLPmeans (LSA) | 0.269 ± n/a | 0.538 ± n/a | 0.181 ± n/a | n/a | n/a | n/a | -0.004 (n=1, no interval) | -0.004 (n=1, no interval) | +0.010 (n=1, no interval) |
| k-SLMmeans Qwen3-1.7B, arXiv prompt | 0.266 ± n/a | 0.535 ± n/a | 0.175 ± n/a | -0.003 (n=1, no interval) | -0.003 (n=1, no interval) | -0.006 (n=1, no interval) | -0.007 (n=1, no interval) | -0.007 (n=1, no interval) | +0.003 (n=1, no interval) |

### Cost per ACC point (only where the paired interval excludes 0)

| Method | ACC vs LSA | extra s vs LSA | s per ACC point vs LSA | local $ per ACC point vs LSA | API $ per ACC point vs LSA (claude-sonnet-4-6) | API $ per ACC point vs LSA (deepseek-v4-pro) | API $ per ACC point vs LSA (gpt-4o) | ACC vs k-means | extra s vs k-means | s per ACC point vs k-means | local $ per ACC point vs k-means | API $ per ACC point vs k-means (claude-sonnet-4-6) | API $ per ACC point vs k-means (deepseek-v4-pro) | API $ per ACC point vs k-means (gpt-4o) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a |  |  |  |  | n/a | n/a | n/a |  |  |  |  |
| k-NLPmeans (LSA) | n/a | n/a | n/a |  |  |  |  | n/a | n/a | n/a |  |  |  |  |
| k-SLMmeans Qwen3-1.7B, arXiv prompt | n/a | n/a | n/a |  |  |  |  | n/a | n/a | n/a |  |  |  |  |

### Prices used

| item | usd | source | checked |
| --- | ---: | ---: | ---: |
| gpu | 0.242 | https://www.aquanode.io/pricing | 2026-11-26 |
| api_input:gpt-4o | 2.5 | https://developers.openai.com/api/docs/models/gpt-4o?site_locale=en | 2026-11-26 |
| api_output:gpt-4o | 10 | https://developers.openai.com/api/docs/models/gpt-4o?site_locale=en | 2026-11-26 |
| api_input:claude-sonnet-4-6 | 3 | https://platform.claude.com/docs/en/models/sonnet-4-6/overview | 2026-11-26 |
| api_output:claude-sonnet-4-6 | 15 | https://platform.claude.com/docs/en/models/sonnet-4-6/overview | 2026-11-26 |
| api_input:deepseek-v4-pro | 1.32 | https://api-docs.deepseek.com/quick_start/pricing/ | 2026-11-26 |
| api_output:deepseek-v4-pro | 3.96 | https://api-docs.deepseek.com/quick_start/pricing/ | 2026-11-26 |
