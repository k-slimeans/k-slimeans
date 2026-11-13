| Method | calls | input tokens | output tokens | summarizer s | total s | empty summaries | truncated clusters | dropped texts | peak GPU GB | reserved GPU GB | batch size | dtype | GPU | GPU idle | GPU MiB above pre-run baseline (non-torch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a | n/a | 5.3 ± 1.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |
| k-NLPmeans (LSA) | 636 ± 8 | 0 ± 0 | 0 ± 0 | 14.8 ± 1.8 | 23.7 ± 2.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.37 ± 0.00 | 0.53 ± 0.08 | n/a | n/a | NVIDIA GeForce RTX 3060 |  |  |
| k-SLMmeans (Qwen3-0.6B) | 707 ± 7 | 1,113,037 ± 8,773 | 10,227 ± 211 | 895.8 ± 29.7 | 903.9 ± 29.6 | 0.0 ± 0.0 | 6.6 ± 3.5 | 173.0 ± 147.1 | 6.51 ± 0.00 | 6.70 ± 0.00 | 8 | torch.bfloat16 | NVIDIA GeForce RTX 3060 |  |  |
| k-SLMmeans (Qwen3-1.7B) | 702 ± 6 | 1,101,755 ± 11,449 | 18,288 ± 387 | 1238.6 ± 102.5 | 1247.5 ± 103.4 | 0.0 ± 0.0 | 6.8 ± 2.4 | 304.0 ± 42.8 | 6.54 ± 0.00 | 6.68 ± 0.00 | 4 | torch.bfloat16 | NVIDIA GeForce RTX 3060 |  |  |
| k-SLMmeans (Qwen3-4B) | 714 ± 6 | 1,120,190 ± 7,604 | 15,620 ± 463 | 1855.0 ± 188.6 | 1861.7 ± 189.8 | 0.0 ± 0.0 | 10.2 ± 3.4 | 502.4 ± 264.2 | 9.25 ± 0.00 | 9.42 ± 0.00 | 1 | torch.bfloat16 | NVIDIA GeForce RTX 3060 |  |  |
| k-SLMmeans-v2 (Qwen3-4B) | 698 ± 10 | 1,131,772 ± 11,868 | 9,068 ± 185 | 1093.7 ± 20.7 | 1098.8 ± 20.6 | 0.0 ± 0.0 | 3.8 ± 2.2 | 86.2 ± 69.0 | 9.25 ± 0.00 | 9.41 ± 0.00 | 1 | torch.bfloat16 | NVIDIA GeForce RTX 3060 | yes | 139 |

step0_sha1 shared across all methods for seeds 0-4: yes
