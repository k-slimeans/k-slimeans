| Method | calls | input tokens | output tokens | summarizer s | total s | empty summaries | truncated clusters | dropped texts | peak GPU GB | reserved GPU GB | batch size | dtype | GPU | GPU idle | GPU MiB above pre-run baseline (non-torch) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-means | n/a | n/a | n/a | n/a | 0.0 ± 0.0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |  |
| k-NLPmeans (LSA) | 21 ± 5 | 0 ± 0 | 0 ± 0 | 2.6 ± 0.7 | 2.9 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | n/a | n/a | NVIDIA GeForce RTX 3060 | yes | 0 |
| k-SLMmeans (Qwen3-0.6B) | 23 ± 1 | 83,813 ± 3,752 | 588 ± 41 | 26.5 ± 1.4 | 26.6 ± 1.5 | 0.0 ± 0.0 | 10.6 ± 0.9 | 302.6 ± 38.1 | 4.35 ± 0.00 | 4.48 ± 0.01 | 8 | torch.bfloat16 | NVIDIA GeForce RTX 3060 | yes | 137 |
| k-SLMmeans (Qwen3-1.7B) | 24 ± 1 | 86,898 ± 1,916 | 868 ± 30 | 46.2 ± 3.3 | 46.2 ± 3.3 | 0.0 ± 0.0 | 11.6 ± 1.9 | 266.8 ± 54.2 | 6.27 ± 0.00 | 6.42 ± 0.00 | 4 | torch.bfloat16 | NVIDIA GeForce RTX 3060 | yes | 139 |
| k-SLMmeans (Qwen3-4B) | 24 ± 1 | 87,474 ± 4,390 | 732 ± 31 | 87.2 ± 4.5 | 87.3 ± 4.5 | 0.0 ± 0.0 | 10.6 ± 0.9 | 270.2 ± 67.0 | 8.98 ± 0.00 | 9.08 ± 0.01 | 1 | torch.bfloat16 | NVIDIA GeForce RTX 3060 | yes | 137 |

step0_sha1 shared across all methods for seeds 0-4: yes
