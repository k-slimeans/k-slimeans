# Qwen3 checkpoint differences and the missing size trend

## Purpose

These notes examine whether the three Qwen3 checkpoints used in k-SLiMeans differ in ways other than parameter count, and whether those differences provide a plausible explanation for their prompt-dependent clustering performance. The primary source is the original [Qwen3 Technical Report, arXiv:2505.09388v1 (14 May 2025)](https://arxiv.org/html/2505.09388v1). Section and table numbers refer to that version.

## 1. Architecture and context length

Source: technical report §2, Table 1.

| Property | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|---|---:|---:|---:|
| Parameters (nominal) | 0.6B | 1.7B | 4B |
| Architecture | Dense | Dense | Dense |
| Layers | 28 | 28 | 36 |
| Query attention heads | 16 | 16 | 32 |
| Key/value heads | 8 | 8 | 8 |
| Tied embeddings | Yes | Yes | Yes |
| Context capacity (report) | 32K | 32K | 128K |

All three use the same general architectural family, including grouped-query attention and a common tokenizer. The 4B model differs in depth and query-head count, not merely weight count. The context capacities in Table 1 are not the input lengths used in our experiments: the project limits input to 4,096 tokens. Consequently, the larger advertised context window is not evidence of an advantage or disadvantage in our setting.

## 2. Pre-training data and procedure

Source: §§3.1–3.2.

The report describes a shared corpus of approximately 36 trillion tokens spanning 119 languages and dialects. Its training curriculum progresses from general language learning to reasoning-oriented material and then 32,768-token sequences. Learning-rate and batch-size strategies are selected for individual architectures using scaling laws.

The report does not provide separate corpus manifests, exact checkpoint-specific token exposures, or data-mixture proportions for these three models. It therefore supports a shared broad training recipe, not a claim that every training variable was held identical. It also does not support an assertion that the 4B checkpoint saw more summarization or question-answering data than the smaller models.

## 3. Post-training and knowledge distillation

Source: §4, Figure 1; §§4.3–4.5.

All three checkpoints belong to the lightweight group trained through strong-to-weak distillation (§4.5). The two phases are:

- Off-policy: students learn from teacher responses in both reasoning modes.
- On-policy: students generate responses and learn to match teacher output distributions using KL divergence.

The report identifies Qwen3-32B and Qwen3-235B-A22B as teacher possibilities, but does not map a particular teacher, training mixture or distillation budget to each of the three students. The four-stage flagship post-training description should not be mistaken for evidence that every student independently underwent the same four stages.

The common distillation strategy makes it plausible that smaller checkpoints inherit useful capabilities from larger models. Different student capacities may affect how those capabilities are expressed, but the report does not establish that this explains our observed size trend. In particular, there is no documented distinction showing that only the 0.6B or 1.7B model was distilled while the 4B was not.

## 4. Instruction following and non-thinking mode

Source: §4.3, Table 9; §§4.4, 4.6, Tables 18 and 20.

Qwen3 integrates thinking and non-thinking behavior; its chat template supports disabling thinking with `enable_thinking=False`. Our experiments use non-thinking generation, so the relevant published comparison is the non-thinking benchmark, not reasoning-mode performance.

| Non-thinking benchmark | 0.6B | 1.7B | 4B |
|---|---:|---:|---:|
| IFEval (strict prompt accuracy, %) | 54.5 | 68.2 | 81.2 |
| MMLU-Redux (%) | 44.6 | 64.4 | 77.3 |
| WritingBench (reported score) | 5.13 | 6.54 | 6.85 |

The benchmarks favor the larger checkpoint on these measures. They do not evaluate summary embeddings as replacement centroids, however. Following an instruction successfully in a general evaluation is not equivalent to preserving the semantic distinctions needed for clustering.

## 5. In k-SLiMeans

The project's CLINC150 results (five-seed mean accuracy, November 16 progress report, Table 1) are:

| Model | Original prompt | Clearer prompt (v2) | Change (percentage points) |
|---|---:|---:|---:|
| 0.6B | 54.1% | 56.3% | +2.2 |
| 1.7B | 57.0% | 56.3% | -0.7 |
| 4B | 51.1% | 56.8% | +5.6 |

The original 4B model has the highest published IFEval score of the three but the lowest clustering ACC under the original prompt. The clearer prompt improves its ACC without changing the checkpoint. The 1.7B change is uncertain; the revised means do not establish a reliable size ranking.

The [failure-mode analysis](results/size_failure_modes.md) provides a complementary, preliminary review of the same 25 initial clusters across all six configurations. For example, in cluster 80 (dice rolling), old 4B asks for a general summary of the cluster, whereas v2 requests a dice roll. The latter is more directly about the underlying intent even though it lacks a question mark. In cluster 108 (air travel), v2 4B instead asks about an American Express card, showing that the clearer prompt does not guarantee fidelity.

A plausible explanation is a mismatch between general-purpose instruction following and the specific role of generated text in our algorithm. A task-level question can be a grammatical response to an ambiguous instruction but a poor representation of the actual cluster. If similar task-level summaries occur across distinct clusters, their embeddings can become indistinguishable, leading to centroid collisions and reassignment. The clearer prompt requests a representative user question rather than a description of the summarization task. This interpretation is consistent with the project's prompt intervention and replacement diagnostics, but the prompt also changes other properties, such as specificity and copying; it does not isolate any one mechanism.

## 6. Limits and takeaway for the size slides

- **Documented:** the checkpoints differ in architectural dimensions and supported context capacity; they share a broad pre-training recipe and distillation approach. General instruction-following scores rise across these sizes.
- **Observed in the project:** clustering ACC does not follow that same ordering under the original prompt, and changing the prompt improves the 4B and 0.6B results.
- **Not established:** which particular training data, teacher, hyperparameter or architectural component causes the observed failure modes. Different inference batch sizes are another potential confound in the project; its diagnostics show that changing batch size can change greedy outputs. No controlled experiment isolates parameter count from these factors.

**Suggested manuscript/slide wording:** Across these three Qwen3 checkpoints, general instruction-following ability did not translate monotonically into clustering accuracy. The prompt intervention indicates that the suitability of the generated text as a centroid matters more to interpreting these results than a simple model-size ordering. The comparison is checkpoint-specific, not a controlled test of parameter count alone.

## References

- Qwen Team (2025). *Qwen3 Technical Report*. arXiv:2505.09388v1. https://doi.org/10.48550/arXiv.2505.09388. Relevant locations: §2 Table 1; §§3.1–3.2; §4 Figure 1; §§4.3–4.6, Tables 9, 18 and 20.
- k-SLiMeans, November 16 progress report, Table 1 and replacement/cost diagnostics; [progress index](results/progress_nov16.md).
- k-SLiMeans, [prompt test](results/prompt_test.md) and [failure-mode analysis](results/size_failure_modes.md).
