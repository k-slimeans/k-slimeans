# When is a summary worth its cost? Cost table and cost per unit of benefit (#82)

This is the cost half of the main question (#82, part of #84), built only from runs that already exist: what each method costs on each dataset, what it buys in ACC, NMI and ARI over the extractive LSA summarizer and over plain k-means, and the extra seconds it costs per ACC point **where the benefit is clear of seed noise**. Nothing was re-run and nothing was re-timed. The overall synthesis is #50's job; this document only reads the cost against the benefit.

Scope, in one place: **Qwen3-0.6B / 1.7B / 4B, greedy decoding, one RTX 3060 12 GB under WSL2, 5 seeds (arXiv: 1 seed), the prompts named in each row.** Every interval is the paired 95% Student-t interval over seeds (4 degrees of freedom), uncorrected for the many comparisons, taken from the repo's `aggregate.paired_differences` (the CLINC150 and BBC intervals equal the committed `prompt_test_paired.csv`, `bbc_tfidf_paired.csv` and `bbc_dense_paired.csv`; the BANKING77 ones agree with `prelim_bank77.md`). Dollars use the #10 price list (`docs/cost_assumptions.md`, checked 2026-11-26; Section 4).

Files: [`cost_justification_table.csv`](cost_justification_table.csv) (every column), [`cost_justification_table.md`](cost_justification_table.md) (cost, benefit and cost-per-benefit tables for every dataset, with the full timing text of each row), [`cost_justification_pareto.png`](cost_justification_pareto.png).

## 1. What the table is

One row per (dataset, method[, prompt]): seeds, summarizer calls, input and output tokens, seconds, the timing source and its caveat, ACC / NMI / ARI (mean ± sd over seeds), and the paired difference against LSA and against k-means with its interval. All counts and seconds are per seed (mean over seeds), as logged.

**Seconds.** The logged `summarizer_seconds` where the run has it (CLINC150 and BBC). BANKING77 and the arXiv runs predate the cost fields (#30), so their column is the logged seconds of the whole method (`runtime_s`, or `runtime` for arXiv), marked `*` below; for BANKING77 that is the summarizer plus the k-means fits and embedding of the summaries (5 to 10 s per seed on CLINC150, `cost_verification.md` section 2), and for arXiv it also includes the work listed in the table notes. Calls and tokens are "not logged" for those runs, and the arXiv call count is at most 3,360 scheduled calls (`arxiv_status.md`).

**Timing codes.** Every timing in the table names its batch size and whether the card was verified idle (full text per row in `cost_justification_table.md`):

| Code | Meaning |
| --- | --- |
| `-` | CPU method (k-means; LSA summarizer), no card involved |
| `A` | card idle-checked at the start and clean during the run (non-torch memory within about our own CUDA context, +137 to +139 MiB). For BBC, light low-priority CPU work for other issues ran alongside, so the CPU-side parts of those seconds may be slightly inflated; the GPU part is clean (`bbc.md`, "Time") |
| `B` | card not idle-checked (the run predates #35). CLINC150 October prompt only: at step 1 of seed 0 the logged seconds were **1.76 / 1.72 / 1.15 times** the idle-card seconds for 0.6B / 1.7B / 4B (`cost_timing.md` section 4, one step of one seed per size). **The whole-run inflation was not measured**; the full-seed re-time was dropped. Step 1 was the least inflated step of the 0.6B and 4B runs (later steps ran slower per output token, `cost_timing.md` section 4), so 1.76 / 1.72 / 1.15 are probably lower bounds for the whole run; they are not a whole-run correction and are not applied |
| `C` | card not idle-checked, inflation not measured (BANKING77; one BANKING77 check had estimated about 1.7x, from a single run) |
| `D` | idle at start, but 1,625 MiB of foreign memory seen during the run: not a clean timing (CLINC150 v2 1.7B; its seed-to-seed sd is 198 s against 9 and 21 s for the other two v2 runs) |
| `E` | the run overlapped other jobs, so the seconds are an upper bound on idle-card seconds (the arXiv pilot) |

Batch sizes differ by model (CLINC150 and BBC 8 / 4 / 1 for 0.6B / 1.7B / 4B, BANKING77 8 / 4 / 2, arXiv 4), and batch size changes both the seconds and, under greedy decoding, some of the summaries (`cost_timing.md` section 5). So seconds are not a clean comparison between sizes. The idle-card fixed workload (150 step-1 prompts, `cost_timing.md` section 2) gives 0.63 / 1.11 / 2.10 s per call at the October-prompt runs' batch sizes (8 / 4 / 1); it is in the full table's "idle-card s per call" column for those three rows only, because it used that prompt. It is a rate for one step, not a corrected whole-run time, and the doc does not use it to correct the logged seconds.

**What the seconds do not include.** LSA's seconds exclude the spaCy sentence segmentation, which is cached (BANKING77, CLINC150) or a one-off preprocessing step (arXiv: 3,319 s on a shared CPU). The corpus embeddings are cached for every method.

## 2. The summary view

ACC columns are in ACC units (0.01 = one ACC point); the intervals are paired differences over seeds. "s per ACC point" is (row seconds minus the reference's seconds) divided by the ACC difference in points, shown **only when the interval's lower end is above 0**; "measured loss" means the interval is entirely below 0, and "no measured benefit" that it contains 0. arXiv has one seed, so no interval: its cells are `n/a`, and its extra seconds are left empty in the CSV because its k-means, LSA and pilot seconds come from different shared-machine conditions and bases. k-means has no summarizer, so its seconds are 0 where the dataset's seconds are summarizer seconds. `*`: seconds of the whole method, not summarizer seconds. NMI and ARI differences are in the full table; the rule is applied to ACC only.

| Dataset | Method | ACC | seconds | timing | ACC vs LSA | s per ACC point vs LSA | ACC vs k-means | s per ACC point vs k-means |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| CLINC150 (intent) | k-means | 0.551 | 0 | - | n/a | n/a | n/a | n/a |
| CLINC150 (intent) | k-NLPmeans (LSA) | 0.560 | 14.8 | - | n/a | n/a | +0.009 [+0.001, +0.017] | 16.6 |
| CLINC150 (intent) | k-SLMmeans Qwen3-0.6B, October (#31) | 0.541 | 895.8 | B | -0.019 [-0.032, -0.007] | measured loss | -0.010 [-0.018, -0.002] | measured loss |
| CLINC150 (intent) | k-SLMmeans Qwen3-1.7B, October (#31) | 0.570 | 1,238.6 | B | +0.010 [-0.011, +0.030] | no measured benefit | +0.018 [+0.004, +0.033] | 672.5 |
| CLINC150 (intent) | k-SLMmeans Qwen3-4B, October (#31) | 0.511 | 1,855.0 | B | -0.049 [-0.072, -0.026] | measured loss | -0.040 [-0.058, -0.022] | measured loss |
| CLINC150 (intent) | k-SLMmeans Qwen3-0.6B, v2 (#46) | 0.563 | 373.4 | A | +0.002 [-0.006, +0.010] | no measured benefit | +0.011 [+0.005, +0.017] | 334.7 |
| CLINC150 (intent) | k-SLMmeans Qwen3-1.7B, v2 (#46) | 0.563 | 737.0 | D | +0.003 [-0.003, +0.008] | no measured benefit | +0.012 [+0.005, +0.018] | 635.3 |
| CLINC150 (intent) | k-SLMmeans Qwen3-4B, v2 (#46) | 0.568 | 1,093.7 | A | +0.007 [-0.001, +0.016] | no measured benefit | +0.016 [+0.009, +0.023] | 676.8 |
| BANKING77 (#20) | k-means | 0.435 | 0.3* | - | n/a | n/a | n/a | n/a |
| BANKING77 (#20) | k-NLPmeans (LSA) | 0.449 | 1.6* | - | n/a | n/a | +0.014 [+0.006, +0.022] | 0.9 |
| BANKING77 (#20) | k-SLMmeans Qwen3-0.6B, upstream BANKING77 | 0.440 | 102.7* | C | -0.009 [-0.025, +0.007] | no measured benefit | +0.005 [-0.011, +0.021] | no measured benefit |
| BANKING77 (#20) | k-SLMmeans Qwen3-1.7B, upstream BANKING77 | 0.436 | 229.0* | C | -0.012 [-0.033, +0.008] | no measured benefit | +0.002 [-0.016, +0.020] | no measured benefit |
| BANKING77 (#20) | k-SLMmeans Qwen3-4B, upstream BANKING77 | 0.444 | 381.9* | C | -0.005 [-0.033, +0.022] | no measured benefit | +0.009 [-0.011, +0.029] | no measured benefit |
| BBC, TF-IDF (#18) | k-means | 0.788 | 0 | - | n/a | n/a | n/a | n/a |
| BBC, TF-IDF (#18) | k-NLPmeans (LSA) | 0.839 | 2.6 | - | n/a | n/a | +0.051 [-0.020, +0.121] | no measured benefit |
| BBC, TF-IDF (#18) | k-SLMmeans Qwen3-0.6B, BBC prompt | 0.851 | 26.5 | A | +0.012 [-0.051, +0.075] | no measured benefit | +0.063 [-0.053, +0.179] | no measured benefit |
| BBC, TF-IDF (#18) | k-SLMmeans Qwen3-1.7B, BBC prompt | 0.873 | 46.2 | A | +0.034 [-0.014, +0.082] | no measured benefit | +0.085 [-0.009, +0.178] | no measured benefit |
| BBC, TF-IDF (#18) | k-SLMmeans Qwen3-4B, BBC prompt | 0.854 | 87.2 | A | +0.015 [-0.015, +0.045] | no measured benefit | +0.066 [-0.027, +0.159] | no measured benefit |
| BBC, dense (#18) | k-means | 0.866 | 0 | - | n/a | n/a | n/a | n/a |
| BBC, dense (#18) | k-NLPmeans (LSA) | 0.897 | 0.8 | - | n/a | n/a | +0.032 [-0.091, +0.154] | no measured benefit |
| BBC, dense (#18) | k-SLMmeans Qwen3-0.6B, BBC prompt | 0.960 | 17.3 | A | +0.063 [-0.100, +0.226] | no measured benefit | +0.095 [-0.064, +0.254] | no measured benefit |
| BBC, dense (#18) | k-SLMmeans Qwen3-1.7B, BBC prompt | 0.904 | 24.3 | A | +0.006 [+0.001, +0.012] | 36.6 | +0.038 [-0.084, +0.160] | no measured benefit |
| BBC, dense (#18) | k-SLMmeans Qwen3-4B, BBC prompt | 0.902 | 38.8 | A | +0.005 [+0.001, +0.009] | 79.1 | +0.036 [-0.083, +0.156] | no measured benefit |
| arXiv 200k pilot (#48, seed 0) | k-means | 0.274 | 118.4* | - | n/a | n/a | n/a | n/a |
| arXiv 200k pilot (#48, seed 0) | k-NLPmeans (LSA) | 0.269 | 38.7* | - | n/a | n/a | -0.004 (n=1, no interval) | n/a |
| arXiv 200k pilot (#48, seed 0) | k-SLMmeans Qwen3-1.7B, arXiv prompt | 0.266 | 5,884.7* | E | -0.003 (n=1, no interval) | n/a | -0.007 (n=1, no interval) | n/a |

![ACC against summarizer seconds](cost_justification_pareto.png)

Mean ACC ± sd over seeds against seconds (log x), one panel per dataset, k-means as the dashed line. The error bars are across-seed standard deviations, **not** the paired interval used for the gate (a bar that overlaps another point does not decide the gate either way). Points with timing code B, C, D or E are marked at the point. The x positions of the CLINC150 October-prompt points (and all BANKING77 points) carry the inflation described above; the arXiv panel is one seed.

## 3. Reading it, dataset by dataset

Scoped to what ran. With 5 seeds the intervals are wide, so "no measured benefit" means the data cannot show a benefit, not that there is none.

**CLINC150 (5 seeds, both prompts).**
- LSA gains +0.009 ACC over k-means (interval excludes 0) for 14.8 s of summarizer time, 16.6 s per ACC point.
- October prompt: the 1.7B is the only SLM clearly above k-means (+0.018 [+0.004, +0.033]) and is not distinguishable from LSA (+0.010 [-0.011, +0.030]), at 84 times LSA's summarizer seconds (logged, with the step-1 inflation above). The 0.6B and 4B are clearly below LSA and below k-means: more cost bought less ACC.
- v2 prompt: all three sizes are above k-means (+0.011 to +0.016, intervals exclude 0), and none is distinguishable from LSA on ACC (+0.002 to +0.007, all intervals contain 0); the exception is the 4B's NMI against LSA, +0.008 [+0.002, +0.014], which is not gated. Against k-means the clean-timing runs cost about 335 s (0.6B) and 677 s (4B) per ACC point, against 16.6 s for LSA; the 1.7B's 635 s is not a clean timing (`D`).
- Among the rows that gain over k-means (LSA, the October 1.7B and the three v2 sizes), LSA's summarizer seconds are 1/25 to 1/84 of an SLM's (logged; the October 4B, at 1/125, is a loss and is left out). The SLM means that are higher than LSA's are not distinguishable from it on ACC, so an SLM is not shown to be worth its extra seconds here; whether the v2 SLMs add anything over LSA cannot be told from 5 seeds.

**BANKING77 (5 seeds, upstream prompt, timings of code `C`).**
- LSA gains +0.014 [+0.006, +0.022] ACC over k-means for 1.3 extra seconds. No SLM size has an ACC interval excluding 0 against either reference.
- The 0.6B is clearly below LSA on NMI and ARI, and the 1.7B on ARI (`cost_justification_table.md`). LSA is the better choice on this evidence; the SLMs cost 65 to 241 times LSA's seconds as logged.

**BBC, TF-IDF and dense (500 articles, 5 seeds, #18).**
- TF-IDF: no ACC interval excludes 0, against LSA or k-means, for LSA or any SLM (k-means ACC sd 0.080 over seeds). The 1.7B's NMI against LSA is +0.052 [+0.001, +0.103] and its ARI against k-means +0.125 [+0.007, +0.243], but these are other metrics, not gated, and weak (lower ends at 0.001 and 0.007).
- Dense: the 1.7B and 4B are above LSA on ACC by +0.006 [+0.001, +0.012] and +0.005 [+0.001, +0.009], which costs 24 and 38 extra seconds (code `A`, clean timings): 36.6 and 79.1 s per ACC point over LSA. These are 0.5 to 0.6 ACC points. Against k-means nothing is clear. The k-means and LSA ACC spreads over seeds are large (sd 0.129 and 0.131), the 0.6B has the highest mean (0.960, sd 0.001) but its interval against LSA is [-0.100, +0.226], and these intervals are uncorrected; treat the dense gap as small and suggestive. Per-point seconds also depend on step counts: the dense runs stopped before the fifth summary step in several seeds (`bbc.md`, Section 6), so calls per seed differ (15.2 / 13.0 / 11.0 for 0.6B / 1.7B / 4B against 13.0 for LSA) and the seconds per ACC point are not at equal work. It is the only measured ACC gain over LSA in this table, and `bbc.md` (Section 2) does not read it as a method difference: in those seeds LSA, the 1.7B and the 4B end in the same basin and differ by a few articles.
- On the 500-article corpus the whole cost is small in absolute terms (under 90 s per seed for any SLM).

**arXiv 200k, seed 0 (descriptive only).**
- One seed, so no interval and no cost per benefit. ACC is 0.274 (k-means), 0.269 (LSA) and 0.266 (k-SLMmeans, Qwen3-1.7B); ARI is 0.171, 0.181 and 0.175. These gaps cannot be told from seed variation.
- The pilot took 5,885 s (code `E`: concurrent jobs, upper bound), 1.75 s per scheduled call, against 39 s for the LSA clustering plus 3,319 s of one-off segmentation, and 118 s for k-means. No comparison is claimed (`arxiv_status.md`).

**When the extractive summarizer is the better choice, and when an SLM is worth it (what ran).**
- Where the paired ACC interval against LSA contains 0 (CLINC150: the October 1.7B and the three v2 sizes; BANKING77; BBC TF-IDF), the SLMs show no measured benefit over LSA, and LSA costs less: 25 to 84 times fewer summarizer seconds on CLINC150 and 10 to 34 on BBC TF-IDF as logged, and 65 to 241 times fewer whole-method seconds on BANKING77 (a different seconds basis, so not pooled). Where the interval is clearly below 0 (CLINC150 October 0.6B and 4B), the SLM costs more and does worse. On this evidence LSA is the better-value choice in all of these; whether an SLM adds a small gain there cannot be told from 5 seeds.
- The only measured ACC gain of an SLM over LSA is BBC with the dense embedder: +0.5 to +0.6 ACC points for the 1.7B and 4B, uncorrected for the many paired comparisons made (`bbc.md` Section 2: 4 of 30 dense intervals exclude 0, all for differences of at most 0.009, where LSA, the 1.7B and the 4B end in the same basin and differ by a few articles, so `bbc.md` does not read it as a method difference), lower interval ends at +0.001, 24 to 38 more seconds per seed on a 500-article corpus at unequal step counts, which is $0.002 and $0.005 per ACC point at the GPU rental rate (Section 4).
- Where the data cannot tell at all: arXiv (1 seed, no interval).
- One model family, one prompt per dataset except CLINC150, greedy decoding, one GPU, batch sizes that differ by model: the table does not separate model size from batching, and says nothing about larger models or about summary quality (#38, #50).

## 4. Prices

The prices are #10's, in `docs/cost_assumptions.md` and `docs/cost_assumptions.csv` (all checked 2026-11-26):
- **GPU:** an RTX 3060 12 GB at **$0.242 per GPU-hour**. That is an on-demand rental quote for the same card, used as a cloud-equivalent price for the logged local seconds. It is not the cost of owning the card and does not include electricity.
- **API list prices** per 1M input / output tokens: GPT-4o $2.50 / $10.00, Claude Sonnet 4.6 $3.00 / $15.00, and DeepSeek V4 Pro $1.32 / $3.96 (its peak rate).

The table was rerun with `--prices docs/cost_assumptions.csv`. Every dollar column of `cost_justification_table.csv` and `.md` comes from these prices. Local dollars are the seconds of Section 1 at the GPU rate, so they carry the same timing codes.

**What the dollars add, on CLINC150 (per seed, as logged):**
- **Local cost.**
  - LSA costs $0.001. That is an upper bound: its summarizer runs on the CPU but is priced at the GPU rate.
  - The SLMs cost $0.025 (v2 0.6B) to $0.125 (October 4B).
  - Per ACC point over k-means, where the gate passes: LSA $0.001, the SLMs $0.02 to $0.05.
  - Every method costs cents per seed on 22,500 queries. Seconds and dollars use one rate, so the ratios to LSA are the same as for seconds: 25 to 84 times for the rows that gain over k-means.
- **Hypothetical API cost.** Sending the same prompts (about 1.1M input and 8k to 18k output tokens per seed) to a hosted large model would cost, per seed:
  - $1.51 to $1.55 (DeepSeek V4 Pro);
  - $2.88 to $2.96 (GPT-4o);
  - $3.48 to $3.59 (Claude Sonnet 4.6).

  That is 12 to 139 times the local rental cost of the same calls, almost all of it input tokens. A large model would tokenize and answer differently, so this is the price of our token counts at those rates, nothing more. It says nothing about how good a large model's summaries would be (#38).

**Elsewhere:**
- BBC: $0.001 to $0.006 per seed locally and $0.06 to $0.27 by API. The one SLM gain over LSA, BBC dense (Section 3), costs $0.002 (1.7B) and $0.005 (4B) per ACC point at the GPU rate.
- BANKING77: $0.007 to $0.026 per seed, on whole-method seconds. No API dollars, because tokens were not logged.
- arXiv pilot: $0.40 for the one-seed SLM run (code `E`, an upper bound).

**Reading.** In absolute terms everything here is cheap, because the datasets are small and the model runs locally. Dollars grow with the number of calls and tokens. The cost question on these datasets is therefore a relative one. Where the data can tell, no SLM shows an ACC gain over LSA, at 25 to 84 times LSA's summarizer cost (CLINC150) and 65 to 241 times its whole-method seconds (BANKING77). A hosted large model would add about 1.5 to 3.6 dollars per seed on CLINC150 for summaries whose quality we did not measure.

**Format.** The script reads `--prices`, a CSV with `item,unit,usd,source,checked` rows:
- `gpu`, unit `hour`, is the local GPU price.
- `api_input:<model>` and `api_output:<model>`, unit `1M tokens`, are one API model's list prices. Hypothetical API dollars are 0 for k-means and LSA (no API call) and empty where tokens were not logged (BANKING77, arXiv).
- Units are compared loosely: `per hour` and `per 1M tokens` are accepted, case and spacing are ignored.
- Every row needs a source and a check date.
- The dollars per ACC point use the same interval gate as the seconds.

A small example file with made-up numbers, for the unit tests only, is `tests/fixtures/cost_prices_example.csv`.

## 5. Reproduce

Run from the repo root of the checkout that holds `results/` (the runs are not in git; the run directories are listed in `kslimeans.cost_table.DATASETS`). `--results` takes that checkout or its `results/` directory (default: the current directory). CPU only, a few seconds:

```
python scripts/cost_table.py --out-dir docs/results
# from another checkout (a worktree, say), pointing at the one that holds the runs:
python scripts/cost_table.py --results <checkout or its results/ directory> --out-dir docs/results
# with prices, once docs/cost_assumptions.md exists (#10):
python scripts/cost_table.py --out-dir docs/results --prices prices.csv
```

It writes `cost_justification_table.csv`, `cost_justification_table.md` and `cost_justification_pareto.png`. The summary table of section 2 is the same data, reduced to the ACC columns. Tests: `tests/test_cost_table.py` (set `KSLIMEANS_RESULTS` to also check the intervals against the committed paired CSVs).
