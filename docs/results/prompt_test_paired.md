Paired difference vs k-SLMmeans (Qwen3-4B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | +0.040 | [+0.022, +0.058] | 5/5 |
| k-means | NMI | +0.009 | [+0.000, +0.017] | 5/5 |
| k-means | ARI | +0.018 | [+0.009, +0.027] | 5/5 |
| k-NLPmeans (LSA) | ACC | +0.049 | [+0.026, +0.072] | 5/5 |
| k-NLPmeans (LSA) | NMI | +0.013 | [+0.002, +0.023] | 5/5 |
| k-NLPmeans (LSA) | ARI | +0.028 | [+0.014, +0.043] | 5/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | +0.030 | [+0.008, +0.051] | 5/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | +0.008 | [+0.001, +0.015] | 4/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | +0.017 | [+0.004, +0.029] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.059 | [+0.045, +0.072] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.021 | [+0.015, +0.027] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.040 | [+0.028, +0.053] | 5/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ACC | +0.056 | [+0.039, +0.074] | 5/5 |
| k-SLMmeans-v2 (Qwen3-4B) | NMI | +0.021 | [+0.013, +0.029] | 5/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ARI | +0.036 | [+0.019, +0.053] | 5/5 |

Paired difference vs k-means (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-NLPmeans (LSA) | ACC | +0.009 | [+0.001, +0.017] | 4/5 |
| k-NLPmeans (LSA) | NMI | +0.004 | [+0.001, +0.008] | 5/5 |
| k-NLPmeans (LSA) | ARI | +0.010 | [+0.002, +0.018] | 5/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.010 | [-0.018, -0.002] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.001 | [-0.004, +0.003] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.001 | [-0.007, +0.004] | 2/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.018 | [+0.004, +0.033] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.012 | [+0.005, +0.019] | 5/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.022 | [+0.011, +0.034] | 5/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.040 | [-0.058, -0.022] | 0/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.009 | [-0.017, -0.000] | 0/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.018 | [-0.027, -0.009] | 0/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ACC | +0.016 | [+0.009, +0.023] | 5/5 |
| k-SLMmeans-v2 (Qwen3-4B) | NMI | +0.012 | [+0.007, +0.017] | 5/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ARI | +0.018 | [+0.007, +0.029] | 5/5 |

Paired difference vs k-NLPmeans (LSA) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.009 | [-0.017, -0.001] | 1/5 |
| k-means | NMI | -0.004 | [-0.008, -0.001] | 0/5 |
| k-means | ARI | -0.010 | [-0.018, -0.002] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.019 | [-0.032, -0.007] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.005 | [-0.012, +0.001] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.012 | [-0.023, -0.001] | 1/5 |
| k-SLMmeans (Qwen3-1.7B) | ACC | +0.010 | [-0.011, +0.030] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | NMI | +0.008 | [-0.002, +0.018] | 4/5 |
| k-SLMmeans (Qwen3-1.7B) | ARI | +0.012 | [-0.007, +0.031] | 4/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.049 | [-0.072, -0.026] | 0/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.013 | [-0.023, -0.002] | 0/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.028 | [-0.043, -0.014] | 0/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ACC | +0.007 | [-0.001, +0.016] | 5/5 |
| k-SLMmeans-v2 (Qwen3-4B) | NMI | +0.008 | [+0.002, +0.014] | 5/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ARI | +0.008 | [-0.002, +0.017] | 4/5 |

Paired difference vs k-SLMmeans (Qwen3-1.7B) (method minus reference):

| Method | Metric | mean diff | 95% CI | wins/n |
| --- | ---: | ---: | ---: | ---: |
| k-means | ACC | -0.018 | [-0.033, -0.004] | 1/5 |
| k-means | NMI | -0.012 | [-0.019, -0.005] | 0/5 |
| k-means | ARI | -0.022 | [-0.034, -0.011] | 0/5 |
| k-NLPmeans (LSA) | ACC | -0.010 | [-0.030, +0.011] | 1/5 |
| k-NLPmeans (LSA) | NMI | -0.008 | [-0.018, +0.002] | 1/5 |
| k-NLPmeans (LSA) | ARI | -0.012 | [-0.031, +0.007] | 1/5 |
| k-SLMmeans (Qwen3-0.6B) | ACC | -0.029 | [-0.048, -0.009] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | NMI | -0.013 | [-0.017, -0.009] | 0/5 |
| k-SLMmeans (Qwen3-0.6B) | ARI | -0.024 | [-0.033, -0.015] | 0/5 |
| k-SLMmeans (Qwen3-4B) | ACC | -0.059 | [-0.072, -0.045] | 0/5 |
| k-SLMmeans (Qwen3-4B) | NMI | -0.021 | [-0.027, -0.015] | 0/5 |
| k-SLMmeans (Qwen3-4B) | ARI | -0.040 | [-0.053, -0.028] | 0/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ACC | -0.002 | [-0.021, +0.017] | 2/5 |
| k-SLMmeans-v2 (Qwen3-4B) | NMI | -0.000 | [-0.007, +0.007] | 2/5 |
| k-SLMmeans-v2 (Qwen3-4B) | ARI | -0.005 | [-0.023, +0.014] | 2/5 |
