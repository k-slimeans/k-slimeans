| Method | calls | input tokens | output tokens | summarizer s | total s | empty summaries | truncated clusters | dropped texts | peak GPU GB | reserved GPU GB | batch size | dtype | GPU | GPU idle | GPU MiB above pre-run baseline (non-torch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a | n/a | 0.0 ± 0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |
| k-NLPmeans (LSA) | 13 ± 6 | 0 ± 0 | 0 ± 0 | 0.8 ± 0.3 | 1.0 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.11 ± 0.00 | 0.32 ± 0.00 | n/a | n/a | NVIDIA GeForce RTX 3060 | yes | 131 |
| k-SLMmeans (Qwen3-0.6B) | 15 ± 6 | 59,061 ± 21,200 | 351 ± 136 | 17.3 ± 6.5 | 17.5 ± 6.6 | 0.0 ± 0.0 | 4.4 ± 1.1 | 51.2 ± 70.1 | 4.45 ± 0.00 | 4.59 ± 0.00 | 8 | torch.bfloat16 | NVIDIA GeForce RTX 3060 | yes | 139 |
| k-SLMmeans (Qwen3-1.7B) | 13 ± 7 | 47,923 ± 21,493 | 425 ± 234 | 24.3 ± 13.0 | 24.5 ± 13.2 | 0.0 ± 0.0 | 3.8 ± 1.1 | 114.0 ± 205.7 | 6.37 ± 0.00 | 6.52 ± 0.00 | 4 | torch.bfloat16 | NVIDIA GeForce RTX 3060 | yes | 139 |
| k-SLMmeans (Qwen3-4B) | 11 ± 5 | 41,438 ± 15,096 | 311 ± 149 | 38.8 ± 17.0 | 38.9 ± 17.2 | 0.0 ± 0.0 | 3.4 ± 0.9 | 104.6 ± 194.9 | 9.07 ± 0.00 | 9.19 ± 0.00 | 1 | torch.bfloat16 | NVIDIA GeForce RTX 3060 | yes | 139 |

step0_sha1 shared across all methods for seeds 0-4: yes
