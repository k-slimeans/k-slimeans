Paired difference vs k-means (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-NLPmeans (LSA) | ACC | +0.051 | [-0.020, +0.121] | 4/5 |
| k-NLPmeans (LSA) | NMI | +0.020 | [-0.037, +0.077] | 3/5 |
| k-NLPmeans (LSA) | ARI | +0.049 | [-0.011, +0.109] | 4/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.063 | [-0.053, +0.179] | 3/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.038 | [-0.060, +0.136] | 3/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | +0.076 | [-0.092, +0.244] | 3/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.085 | [-0.009, +0.178] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.072 | [-0.001, +0.146] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.125 | [+0.007, +0.243] | 5/5 |
| k-SLMmeans (Qwen3-4B) | ACC | +0.066 | [-0.027, +0.159] | 4/5 |
| k-SLMmeans (Qwen3-4B) | NMI | +0.043 | [-0.023, +0.110] | 4/5 |
| k-SLMmeans (Qwen3-4B) | ARI | +0.078 | [-0.036, +0.192] | 4/5 |

Paired difference vs k-NLPmeans (LSA) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.051 | [-0.121, +0.020] | 1/5 |
| k-means | NMI | -0.020 | [-0.077, +0.037] | 2/5 |
| k-means | ARI | -0.049 | [-0.109, +0.011] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.012 | [-0.051, +0.075] | 2/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.018 | [-0.046, +0.082] | 3/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | +0.027 | [-0.114, +0.167] | 2/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.034 | [-0.014, +0.082] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.052 | [+0.001, +0.103] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.076 | [-0.030, +0.182] | 4/5 |
| k-SLMmeans (Qwen3-4B) | ACC | +0.015 | [-0.015, +0.045] | 3/5 |
| k-SLMmeans (Qwen3-4B) | NMI | +0.023 | [-0.002, +0.049] | 5/5 |
| k-SLMmeans (Qwen3-4B) | ARI | +0.029 | [-0.038, +0.095] | 3/5 |

Paired difference vs k-SLMmeans (Qwen3-0.6B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.063 | [-0.179, +0.053] | 2/5 |
| k-means | NMI | -0.038 | [-0.136, +0.060] | 2/5 |
| k-means | ARI | -0.076 | [-0.244, +0.092] | 2/5 |
| k-NLPmeans (LSA) | ACC | -0.012 | [-0.075, +0.051] | 3/5 |
| k-NLPmeans (LSA) | NMI | -0.018 | [-0.082, +0.046] | 2/5 |
| k-NLPmeans (LSA) | ARI | -0.027 | [-0.167, +0.114] | 3/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.022 | [-0.018, +0.062] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.034 | [-0.015, +0.083] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.049 | [-0.035, +0.134] | 4/5 |
| k-SLMmeans (Qwen3-4B) | ACC | +0.003 | [-0.037, +0.043] | 2/5 |
| k-SLMmeans (Qwen3-4B) | NMI | +0.005 | [-0.038, +0.048] | 3/5 |
| k-SLMmeans (Qwen3-4B) | ARI | +0.002 | [-0.087, +0.091] | 2/5 |

Paired difference vs k-SLMmeans (Qwen3-1.7B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.085 | [-0.178, +0.009] | 0/5 |
| k-means | NMI | -0.072 | [-0.146, +0.001] | 0/5 |
| k-means | ARI | -0.125 | [-0.243, -0.007] | 0/5 |
| k-NLPmeans (LSA) | ACC | -0.034 | [-0.082, +0.014] | 1/5 |
| k-NLPmeans (LSA) | NMI | -0.052 | [-0.103, -0.001] | 1/5 |
| k-NLPmeans (LSA) | ARI | -0.076 | [-0.182, +0.030] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.022 | [-0.062, +0.018] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.034 | [-0.083, +0.015] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.049 | [-0.134, +0.035] | 1/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.019 | [-0.054, +0.017] | 2/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.029 | [-0.070, +0.012] | 1/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.047 | [-0.128, +0.034] | 2/5 |

Paired difference vs k-SLMmeans (Qwen3-4B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.066 | [-0.159, +0.027] | 1/5 |
| k-means | NMI | -0.043 | [-0.110, +0.023] | 1/5 |
| k-means | ARI | -0.078 | [-0.192, +0.036] | 1/5 |
| k-NLPmeans (LSA) | ACC | -0.015 | [-0.045, +0.015] | 1/5 |
| k-NLPmeans (LSA) | NMI | -0.023 | [-0.049, +0.002] | 0/5 |
| k-NLPmeans (LSA) | ARI | -0.029 | [-0.095, +0.038] | 2/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.003 | [-0.043, +0.037] | 3/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.005 | [-0.048, +0.038] | 2/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.002 | [-0.091, +0.087] | 3/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.019 | [-0.017, +0.054] | 3/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.029 | [-0.012, +0.070] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.047 | [-0.034, +0.128] | 3/5 |
