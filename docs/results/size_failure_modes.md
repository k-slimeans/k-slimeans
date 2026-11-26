# Failure modes by Qwen3 size and prompt

## Method

The review packet contains 25 CLINC150 clusters from seed 0, summary replacement step 1, with five clusters sampled from each of five purity bins (sampling seed 20261119). Each cluster has the same initial membership across all six model–prompt configurations. Ten sampled members per cluster and the six generated summaries were examined using the eight predefined binary tags and the optional `other` field. No new model runs were performed.

## Results

Counts are out of 25 summaries per configuration. Tags can overlap; `other` counts nonempty entries.

| Failure mode | 0.6B old | 0.6B v2 | 1.7B old | 1.7B v2 | 4B old | 4B v2 |
|---|---:|---:|---:|---:|---:|---:|
| Echoes instruction | 1 | 0 | 1 | 0 | 8 | 0 |
| Mentions cluster or questions | 1 | 0 | 2 | 0 | 9 | 0 |
| Quote marks or markup | 1 | 1 | 0 | 0 | 0 | 2 |
| Copies displayed member verbatim | 0 | 0 | 0 | 0 | 0 | 0 |
| No question mark | 0 | 2 | 0 | 0 | 0 | 10 |
| Answers instead of asking | 0 | 0 | 0 | 0 | 2 | 0 |
| Covers only one subgroup | 13 | 17 | 3 | 12 | 1 | 10 |
| Adds detail absent from displayed members | 11 | 11 | 10 | 6 | 3 | 2 |
| Other | 4 | 2 | 3 | 3 | 2 | 2 |

## Observations by model size

**0.6B.** Narrow coverage is the recurring issue. In cluster 59, the old output focuses on date calculation and v2 on a five-minute timer, although the members also include reservations, alarms and meetings. The clearer prompt removes the observed instruction echo but does not resolve narrow coverage or sample-absent details.

**1.7B.** The old prompt often produces broader summaries spanning multiple intents. In cluster 149, it addresses lost and damaged cards, whereas v2 focuses on damaged cards alone. Subgroup-only tags rise from 3 to 12, while sample-absent details fall from 10 to 6. Shorter, more natural wording can therefore come at the expense of breadth.

**4B.** The old prompt produces the most task-level questions and explanatory answers. In cluster 80, it asks what the cluster's general summary is; v2 instead requests a dice roll, matching the dominant intent. The clearer prompt removes all observed instruction echoes and cluster mentions, but ten outputs lack a question mark. This often reflects the informal style of the members rather than a semantic failure. Not all v2 outputs are faithful: cluster 108 concerns air travel, but its v2 output asks about applying for an American Express card.

## Relation to the missing size trend

The observations agree with the full-run surface checks in [clinc150_summarizers.md, §2.4](clinc150_summarizers.md) and [prompt_test.md, §§5–6, 8.3](prompt_test.md): old 4B cluster mentions occurred in 34% of summaries versus 8% for each smaller model, disappearing under v2. Repeated final summaries also fell from 16–21 affected clusters per seed for old 4B to 0–2 under v2. The paired ACC gains for 0.6B and 4B were +0.022 and +0.056; the 1.7B change of −0.007 remains uncertain (see [prompt_test.md, §8.2](prompt_test.md)). The findings suggest prompt-dependent representational behavior rather than a simple parameter-count advantage. Generic task-level summaries can collapse distinct centroids; removing them is helpful, but punctuation and narrower coverage alone do not establish centroid quality. The prompt changes more than one summary characteristic, so these observations do not isolate the cause of the accuracy gains.

## Limitations

The counts are descriptive, unweighted and limited to a single seed and replacement step. A task echo may cover no subgroup and thus escape the subgroup-only tag. Only ten members per cluster are visible, so sample-absent details are not confirmed hallucinations and zero displayed-member copies does not establish zero copying across the full clusters.

**Annotation source:** `size_failure_modes_tags_ai_assisted.csv`; review packet: [size_failure_modes_step1_seed0.md](../review_packets/size_failure_modes_step1_seed0.md). Update the counts and annotation source if the reviewer changes the tags.
