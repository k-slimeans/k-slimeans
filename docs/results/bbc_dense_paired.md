Paired difference vs k-means (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-NLPmeans (LSA) | ACC | +0.032 | [-0.091, +0.154] | 2/5 |
| k-NLPmeans (LSA) | NMI | +0.022 | [-0.090, +0.133] | 2/5 |
| k-NLPmeans (LSA) | ARI | +0.038 | [-0.113, +0.190] | 2/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.095 | [-0.064, +0.254] | 4/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.065 | [-0.039, +0.170] | 4/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | +0.105 | [-0.066, +0.277] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.038 | [-0.084, +0.160] | 2/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.031 | [-0.087, +0.149] | 2/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.047 | [-0.109, +0.202] | 2/5 |
| k-SLMmeans (Qwen3-4B) | ACC | +0.036 | [-0.083, +0.156] | 3/5 |
| k-SLMmeans (Qwen3-4B) | NMI | +0.025 | [-0.082, +0.131] | 2/5 |
| k-SLMmeans (Qwen3-4B) | ARI | +0.045 | [-0.104, +0.193] | 3/5 |

Paired difference vs k-NLPmeans (LSA) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.032 | [-0.154, +0.091] | 3/5 |
| k-means | NMI | -0.022 | [-0.133, +0.090] | 3/5 |
| k-means | ARI | -0.038 | [-0.190, +0.113] | 3/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.063 | [-0.100, +0.226] | 5/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.044 | [-0.046, +0.133] | 5/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | +0.067 | [-0.091, +0.225] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.006 | [+0.001, +0.012] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.009 | [+0.003, +0.016] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.008 | [+0.003, +0.014] | 5/5 |
| k-SLMmeans (Qwen3-4B) | ACC | +0.005 | [+0.001, +0.009] | 4/5 |
| k-SLMmeans (Qwen3-4B) | NMI | +0.003 | [-0.006, +0.012] | 3/5 |
| k-SLMmeans (Qwen3-4B) | ARI | +0.006 | [-0.004, +0.016] | 3/5 |

Paired difference vs k-SLMmeans (Qwen3-0.6B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.095 | [-0.254, +0.064] | 1/5 |
| k-means | NMI | -0.065 | [-0.170, +0.039] | 1/5 |
| k-means | ARI | -0.105 | [-0.277, +0.066] | 1/5 |
| k-NLPmeans (LSA) | ACC | -0.063 | [-0.226, +0.100] | 0/5 |
| k-NLPmeans (LSA) | NMI | -0.044 | [-0.133, +0.046] | 0/5 |
| k-NLPmeans (LSA) | ARI | -0.067 | [-0.225, +0.091] | 0/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | -0.057 | [-0.215, +0.101] | 1/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | -0.034 | [-0.129, +0.060] | 2/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | -0.059 | [-0.221, +0.104] | 2/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.058 | [-0.219, +0.102] | 1/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.041 | [-0.133, +0.052] | 1/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.061 | [-0.226, +0.104] | 1/5 |

Paired difference vs k-SLMmeans (Qwen3-1.7B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.038 | [-0.160, +0.084] | 2/5 |
| k-means | NMI | -0.031 | [-0.149, +0.087] | 3/5 |
| k-means | ARI | -0.047 | [-0.202, +0.109] | 3/5 |
| k-NLPmeans (LSA) | ACC | -0.006 | [-0.012, -0.001] | 0/5 |
| k-NLPmeans (LSA) | NMI | -0.009 | [-0.016, -0.003] | 0/5 |
| k-NLPmeans (LSA) | ARI | -0.008 | [-0.014, -0.003] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.057 | [-0.101, +0.215] | 2/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.034 | [-0.060, +0.129] | 3/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | +0.059 | [-0.104, +0.221] | 3/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.002 | [-0.007, +0.003] | 2/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.006 | [-0.019, +0.007] | 2/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.002 | [-0.012, +0.007] | 2/5 |

Paired difference vs k-SLMmeans (Qwen3-4B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.036 | [-0.156, +0.083] | 1/5 |
| k-means | NMI | -0.025 | [-0.131, +0.082] | 3/5 |
| k-means | ARI | -0.045 | [-0.193, +0.104] | 2/5 |
| k-NLPmeans (LSA) | ACC | -0.005 | [-0.009, -0.001] | 0/5 |
| k-NLPmeans (LSA) | NMI | -0.003 | [-0.012, +0.006] | 2/5 |
| k-NLPmeans (LSA) | ARI | -0.006 | [-0.016, +0.004] | 2/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.058 | [-0.102, +0.219] | 2/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.041 | [-0.052, +0.133] | 4/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | +0.061 | [-0.104, +0.226] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.002 | [-0.003, +0.007] | 2/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.006 | [-0.007, +0.019] | 3/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.002 | [-0.007, +0.012] | 3/5 |
