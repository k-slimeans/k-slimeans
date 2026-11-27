# Cost and pricing assumptions (#10)

Pricing snapshot checked: **2026-11-26**. All amounts are in USD. These assumptions support the cost–quality comparison in #82 and are inputs to `scripts/cost_table.py`. The machine-readable values are in [`cost_assumptions.csv`](cost_assumptions.csv).

## GPU rental reference

| Experimental GPU | Reference provider and hardware | USD per GPU-hour | Source | Checked |
| --- | --- | ---: | --- | --- |
| NVIDIA RTX 3060, 12 GB | AquaNode, RTX 3060, 12 GB (Poland) | 0.242 | [GPU rental pricing](https://www.aquanode.io/pricing) | 2026-11-26 |

AquaNode lists this rate as its lowest on-demand offer for the RTX 3060 12 GB, rather than a preemptible rate. The reference matches the experimental GPU model and VRAM capacity, so no adjustment for a different GPU architecture or memory size is needed. Host CPU, RAM, storage, power limits, software environment and utilization can still differ. This is a dated marketplace quote, not a guaranteed future rental price or a market average.

Use USD 0.242/hour as a cloud-equivalent proxy for the recorded local execution time. It is not the actual cost of owning or operating the local GPU and does not measure electricity use, energy consumption or hardware depreciation.

## Large-model API reference prices

| Model (API identifier) | Input USD / 1M tokens | Output USD / 1M tokens | Source | Checked |
| --- | ---: | ---: | --- | --- |
| GPT-4o (`gpt-4o`) | 2.50 | 10.00 | [Official provider pricing](https://developers.openai.com/api/docs/models/gpt-4o?site_locale=en) | 2026-11-26 |
| Claude Sonnet 4.6 (`claude-sonnet-4-6`) | 3.00 | 15.00 | [Official provider pricing](https://platform.claude.com/docs/en/models/sonnet-4-6/overview) | 2026-11-26 |
| DeepSeek V4 Pro (V4-Pro-0813) (`deepseek-v4-pro`) | 1.32 | 3.96 | [Official provider pricing](https://api-docs.deepseek.com/quick_start/pricing/) | 2026-11-26 |

Use ordinary uncached text input and output rates without batch or caching discounts. For DeepSeek V4 Pro, use the peak cache-miss input and peak output rates; its off-peak rates are lower ($0.66 input and $1.98 output per million tokens). The peak rate is a conservative reference, not a time-weighted estimate. DeepSeek V4 Pro is selected instead of V3 because the current official pricing table lists V4 Pro rather than V3. The GPT-4o and Claude Sonnet 4.6 model-specific documentation supplies the listed rates; the latter is marked active (legacy). These are illustrative large-model price references, not a claim that these are the newest or best models. They are cost figures only: they say nothing about how good a large model's summaries would be (#38).

## Cost calculation and interpretation

For a run with recorded GPU-associated execution time `t_seconds`, the cloud-equivalent estimate is:

`local_usd = (t_seconds / 3600) * 0.242`

Use the timing definition and provenance reported in #35 and #82. Wall-clock runtime is not necessarily active GPU kernel time; retain the existing timing labels and do not present contended timings as clean measurements. If a CPU-only baseline is costed using the same rate, label it as equivalent rental-time cost, not actual GPU consumption.

For a run with logged input tokens `n_input` and output tokens `n_output`, the hypothetical API estimate for model `m` is:

`api_usd(m) = (n_input * input_rate(m) + n_output * output_rate(m)) / 1000000`

The API prices are supposed to be used for a hypothetical cost estimates based on the recorded token volumes of the local summarizers. Actual API costs may be different due to differences in tokenization and generated o/p lengths. The prices here exclude taxes, additional service fees and may be other overhead as well. All pricing information is collected from publicly available sources without making API requests or renting GPU resources.

