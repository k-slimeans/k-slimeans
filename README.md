# k-SLiMeans

**Text clustering with small-language-model summaries as centroids.**

This is our semester project repo. It starts from a clean re-implementation of
k-NLPmeans / k-LLMmeans ([arXiv:2502.09667](https://arxiv.org/abs/2502.09667),
upstream code: [jairoadiazr/summaryCentroids](https://github.com/jairoadiazr/summaryCentroids)).
The method is k-means on text embeddings where, every so often, each cluster is summarized in text, the
summary is embedded, and that embedding becomes the cluster's new centroid. The code is organized so that
the two things we plan to research are each a single, swappable piece:

- **the summarizer**: how a cluster becomes text. Extractive methods (LSA / centroid / TextRank) and any
  Hugging Face instruct model (the "SLM") are built in.
- **the schedule**: *when* the loop summarizes instead of doing plain k-means updates. This is where the
  scheduled summarizer goes.

The paper's method is reproduced exactly. The same settings give the same assignments as the original code
(see [Reproduction status](#reproduction-status)).

---

## Contents

1. [Quickstart](#quickstart)
2. [How the algorithm maps to the code](#how-the-algorithm-maps-to-the-code)
3. [Repo map](#repo-map)
4. [Running experiments](#running-experiments)
5. [Swapping the SLM](#swapping-the-slm)
6. [Adding a summarizer](#adding-a-summarizer)
7. [Adding a schedule (scheduled summarizer)](#adding-a-schedule-scheduled-summarizer)
8. [Adding a dataset or embedder](#adding-a-dataset-or-embedder)
9. [Tests](#tests)
10. [Reproduction status](#reproduction-status)
11. [Differences from upstream](#differences-from-upstream)
12. [Working together](#working-together)
13. [Citation](#citation)

---

## Quickstart

Requires Python ≥ 3.10. A CUDA GPU is strongly recommended for SLM runs; extractive runs are fine on CPU.

```bash
git clone git@github.com:k-slimeans/k-slimeans.git
cd k-slimeans
python -m venv .venv && source .venv/bin/activate

# Install the PyTorch build that matches your machine first:
#   https://pytorch.org/get-started/locally/   (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128)
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

python scripts/smoke_bank77.py      # ~1 min on a GPU; downloads the distilbert embedder on first run
```

The smoke test clusters Bank77 with plain k-means and with k-NLPmeans (LSA summaries). On our reference
machine it prints:

```
k-means      NMI=0.6541  ARI=0.3138
k-NLPmeans   NMI=0.6518  ARI=0.2945
```

Different hardware (CPU vs GPU, a different GPU) can change embeddings in the last few bits and move a few
points, so numbers within about 0.01 mean your environment is good.

---

## How the algorithm maps to the code

```
texts ──► Embedder ──► X (n × d)
                        │
                        ▼
        KMeans(init="k-means++", max_iter = schedule.first_iters())
                        │
              ┌─────────▼──────────┐
       ┌─────►│ schedule.next_step │── None ──► return Result
       │      └─────────┬──────────┘
       │                │ Step(summarize, kmeans_iters)
       │     summarize? ├── no ──► start = current k-means centers
       │                │
       │               yes
       │                ▼
       │   per cluster: pick member docs (all, or `force_context_length` via k-means++)
       │                ▼
       │   Summarizer.summarize_batch(clusters) ──► one text per cluster
       │                ▼
       │   Embedder.encode(summaries) ──► start = summary embeddings
       │   (stop early if they moved less than `tol` since the last summary step)
       │                ▼
       └── KMeans(init=start, max_iter=kmeans_iters) ◄──┘
```

All of this lives in [`summary_kmeans()`](src/kslimeans/core.py), about 100 lines. The three
pluggable objects it takes are:

| Object | Interface | Built-ins | Where |
|---|---|---|---|
| `Embedder` | `encode(texts) -> float32 array` | `distilbert`, `sbert`, `e5-large`, any sentence-transformers id | `embeddings.py` |
| `Summarizer` | `summarize(cluster) -> str` (or `summarize_batch`) | `lsa`, `centroid`, `textrank`, `hf` (any HF chat model) | `summarizers/` |
| `SummarySchedule` | `first_iters()`, `next_step(state) -> Step \| None` | `FixedSchedule` (the paper's) | `schedules.py` |

`kNLPmeans(...)` and `kLLMmeans(...)` are thin wrappers with the paper's argument names (`max_llm_iter`,
`max_iter`, `force_context_length`, `nlp`, `top_k`, ...). They return a `Result` dataclass; call
`result.as_tuple()` if you want the original 6-tuple.

```python
import kslimeans as ks

ds = ks.load_dataset("bank77")
res = ks.kNLPmeans(ds.texts, ds.n_clusters, nlp="lsa", max_llm_iter=5, random_state=0)
res.assignments, res.summaries[:3]
```

The mini-batch / streaming variant (`miniBatchKNLPmeans`, `miniBatchKLLMmeans`,
`mini_batch_summary_kmeans`) runs the same loop batch by batch and keeps a count-weighted running average
of the centroids. See `minibatch.py`.

---

## Repo map

```
src/kslimeans/
  core.py          summary_kmeans() loop, Result, kNLPmeans / kLLMmeans
  schedules.py     State, Step, SummarySchedule, FixedSchedule, build_schedule()
  summarizers/
    base.py        Cluster, Summarizer
    extractive.py  lsa / centroid / textrank (+ MMR reranking), ported verbatim
    hf.py          HFSummarizer: any Hugging Face instruct model
    __init__.py    build_summarizer(): config dict -> Summarizer
  minibatch.py     mini-batch / streaming variant + baselines
  embeddings.py    Embedder, get_embedder()
  sentences.py     spaCy sentence splitting, SentenceCorpus (for extractive summarizers)
  data.py          dataset loaders (bank77, clinic, massive_I, massive_D, goemo, stackexchange)
  metrics.py       ACC (Hungarian), NMI, ARI, centroid/summary distances
  experiment.py    the offline experiment runner behind scripts/run_offline.py
  config.py        YAML loading + --set overrides
  cache.py         on-disk cache of document / sentence embeddings (cache/)
  results.py       results/<name>/<timestamp>/ writer
configs/           experiment configs (YAML)
scripts/           smoke_bank77.py, run_offline.py, run_streaming.py, prepare_stackexchange.py
data/              all datasets, checked in (see data/README.md)
tests/             pytest suite, incl. exact-equivalence fixtures generated from the original code
```

---

## Running experiments

Experiments are YAML configs run by a script. Any key can be overridden from the command line with
`--set dotted.key=value`. Values are parsed as YAML, so `--set seeds=[0,1,2]` and
`--set 'dataset={name: stackexchange, year: 2023}'` both work. Unknown keys are rejected, so a typo fails
loudly instead of being silently ignored.

```bash
# Offline benchmark (one dataset per run)
python scripts/run_offline.py configs/bank77_knlpmeans.yaml
python scripts/run_offline.py configs/bank77_slm.yaml --set summarizer.model_id=Qwen/Qwen2.5-1.5B-Instruct
python scripts/run_offline.py configs/offline_knlpmeans.yaml --set dataset=clinic --set seeds=[0,1,2]
```

An offline config looks like this:

```yaml
name: bank77_knlpmeans            # results go to results/<name>/<timestamp>/
dataset: bank77                   # bank77 | clinic | massive_I | massive_D | goemo
embedder: distilbert              # distilbert | sbert | e5-large | any sentence-transformers id
seeds: [0]
baselines:
  kmeans: {n_init: 10}            # sklearn kwargs; omit the block for no baselines
method_name: k-NLPmeans           # label for the method's rows in metrics/tables
force_context_length: 0           # 0 = summarize all members; N = pick N diverse members via k-means++
tol: 1.0e-4
schedule: {type: fixed, n_summaries: 5, max_iter: 100}
summarizer: {type: lsa, top_k: 5}
```

Each run writes `results/<name>/<timestamp>/`:

- `config.yaml`: the resolved config, after overrides
- `metrics.jsonl`: one row per (seed, method) with `acc`, `nmi`, `ari`, the four distances, and runtime
- `summaries.jsonl`: final summaries and how they evolved, per (seed, method)
- `summary.csv`: mean and std over seeds (also printed at the end)

Document and sentence embeddings are cached in `cache/`, keyed by dataset and embedder. Each entry is checked
against a fingerprint of the texts, so editing a data file invalidates it automatically. Only the first run of
a dataset pays for embedding. `cache/` and `results/` are git-ignored.

### Streaming (StackExchange)

```bash
python scripts/prepare_stackexchange.py     # one-time: extracts data/stackexchange/clean_stackexchange.csv (110 MB)
python scripts/run_streaming.py configs/stackexchange_stream.yaml
# quick check on a small slice:
python scripts/run_streaming.py configs/stackexchange_stream.yaml --set years=[2023] --set dataset.limit=5000 --set max_batch_size=1000
```

Posts are streamed in `CreationDate` order in batches of `max_batch_size`, and the method is compared against
MiniBatchKMeans, sequential MiniBatchKMeans, and full k-means. The benchmark has 35 communities and
39k–69k posts per year, so full runs are long. Try ideas on a `limit`ed slice first; the 5,000-post slice
above takes about 1–2 minutes. On an RTX 3060, the first full run over all four years spends about 8 minutes on
document embeddings and about 30 minutes on spaCy sentence splitting (only extractive summarizers need
sentences). Both are cached, so later runs skip them. `configs/stackexchange_stream_slm.yaml` is the same
benchmark with the SLM summarizer and `force_context_length: 10`.

---

## Swapping the SLM

The `hf` summarizer loads any Hugging Face causal LM that has a chat template. Switching models is one line:

```yaml
summarizer:
  type: hf
  model_id: Qwen/Qwen2.5-0.5B-Instruct   # ← change this
  max_new_tokens: 64       # summaries are one sentence/question; keep this small for speed
  batch_size: 16           # clusters per generate() call; lower it if you run out of VRAM
  max_input_tokens: 4096   # trailing texts are dropped to fit (a warning tells you when)
  dtype: auto              # auto | bfloat16 | float16 | float32
  # device: auto           # auto (cuda if available) | cuda | cpu
  # prompt: "..."          # defaults to the dataset's prompt (see data.py)
  # text_type: "..."       # assistant prefill, e.g. "Question:"; defaults to the dataset's
```

or from the command line: `--set summarizer.model_id=HuggingFaceTB/SmolLM2-1.7B-Instruct`.

From Python:

```python
from kslimeans.summarizers.hf import HFSummarizer

slm = HFSummarizer("Qwen/Qwen2.5-1.5B-Instruct", prompt=ds.prompt, text_type=ds.text_type)
res = ks.kLLMmeans(ds.texts, ds.n_clusters, summarizer=slm, random_state=0)
```

Things to know:

- **Cost:** `configs/bank77_slm.yaml` with Qwen2.5-0.5B-Instruct takes about 85 s per seed on an RTX 3060
  (77 clusters × 5 summary steps), peaks at about 3.5 GB of VRAM, and reaches NMI ≈ 0.64.
- **VRAM:** weights need about 2 bytes per parameter in bf16/fp16, plus activations. 0.5–1.5B models fit
  easily on an 8–12 GB card, and 3–4B models fit with `batch_size` lowered. On CPU, stick to ≤ 0.5B.
- **Output quirks:** small models often wrap the answer in quotes, repeat a member text, or add a second
  line. We store the raw output unchanged, as upstream did. Prompting and post-processing are fair game for
  experiments. Look at `summaries.jsonl` in the run directory to see what the model actually wrote.
- **Gated models** (Llama, Gemma): accept the license on the model page, then `huggingface-cli login`.
- **Prompt format:** we send the same conversation the paper sent to GPT-4o and friends: a system prompt, then
  the dataset prompt plus the cluster's texts, then an assistant turn *prefilled* with `text_type` (e.g.
  `Sentence:`) that the model continues. Templates that reject a system role (e.g. Gemma) are handled by
  folding the system prompt into the user turn.
- **Decoding is greedy**, so runs are deterministic on a given machine. The loop memoizes summaries by cluster
  membership, so an unchanged cluster is never re-summarized. That is also what lets the `tol` early stop
  trigger.
- **`force_context_length`** caps how many member texts each prompt contains. For big clusters (StackExchange)
  use something like 10–50. Members are chosen by k-means++ seeding, so they're diverse.
- Starter models to compare: `Qwen/Qwen2.5-{0.5,1.5,3}B-Instruct`, `HuggingFaceTB/SmolLM2-{360M,1.7B}-Instruct`,
  `meta-llama/Llama-3.2-{1,3}B-Instruct`, `google/gemma-2-2b-it`, `microsoft/Phi-3.5-mini-instruct`.

---

## Adding a summarizer

1. Subclass `Summarizer` (see `summarizers/base.py`). Each call gets a `Cluster` with `texts` (the selected
   member documents), their global `doc_ids`, and, if you set `needs_sentences = True`, the members'
   `sentences` and `sentence_embeddings`.

   ```python
   # src/kslimeans/summarizers/my_summarizer.py
   from .base import Cluster, Summarizer

   class FirstTextSummarizer(Summarizer):
       """Toy example: the cluster's first member text is its summary."""

       def summarize(self, cluster: Cluster) -> str:
           return cluster.texts[0]
   ```

   Override `summarize_batch(clusters)` instead if your backend is faster batched (the `hf` one is).
2. Add a branch to `build_summarizer()` in `summarizers/__init__.py`, and import your class inside the branch
   so heavy dependencies only load when used.
3. Use it: `summarizer: {type: first_text}` in a config.

Rules the loop relies on: return exactly one string per cluster, in order, and don't mutate the inputs.
Empty clusters never reach your summarizer (they get `""`).

---

## Adding a schedule (scheduled summarizer)

A schedule decides, after every k-means fit, whether the next step is a **summary step** (replace centroids
with summary embeddings, then run `kmeans_iters` Lloyd iterations) or a **plain k-means continuation**
(`summarize=False`), or whether to **stop** (`None`).

It sees only a frozen `State` snapshot:

| field | meaning |
|---|---|
| `step` | steps taken so far (0 before the first `next_step` call) |
| `n_summaries` | summary updates applied so far |
| `lloyd_iters_used` | total Lloyd iterations run so far (sum of `KMeans.n_iter_`) |
| `labels`, `prev_labels`, `frac_changed` | current / previous assignments, fraction of points that moved |
| `centers`, `inertia`, `prev_inertia` | current k-means centers and objective |
| `last_n_iter`, `last_max_iter` | how many iterations the last fit took vs. was allowed (early convergence) |
| `summary_shift` | how far the centroids moved at the last summary step |

The paper's schedule is `FixedSchedule(n_summaries, max_iter, initial_iter, final_iter)`. It splits the
`max_iter` budget into equal blocks and summarizes between every pair of blocks.

A sketch of a new schedule (illustrative only, not included in the package):

```python
# src/kslimeans/schedules.py
class SummarizeWhenStable(SummarySchedule):
    """Plain k-means until assignments settle, then summarize; repeat up to n_summaries times."""

    def __init__(self, n_summaries=5, block=10, settle=0.01, max_steps=50):
        self.n_summaries, self.block, self.settle, self.max_steps = n_summaries, block, settle, max_steps

    def first_iters(self) -> int:
        return self.block

    def next_step(self, state: State) -> Step | None:
        if state.n_summaries >= self.n_summaries or state.step >= self.max_steps:
            return None
        settled = state.frac_changed is not None and state.frac_changed < self.settle
        return Step(summarize=settled, kmeans_iters=self.block)
```

Register it with a branch in `build_schedule()` (bottom of `schedules.py`), then use
`schedule: {type: when_stable, settle: 0.02}` in a config.

Guidelines:

- **Keep schedules stateless.** Decide only from `state` and your constructor arguments, so one instance can
  be reused across seeds and mini-batches.
- **Always terminate.** Bound the number of steps, as `max_steps` does above.
- The `tol` early stop applies to summary steps no matter which schedule you use (`tol: -1` disables it).
- Restarting `KMeans` from its own centers (`summarize=False`) is not bit-identical to one longer Lloyd run,
  because sklearn re-centers the data internally. Two schedules with the same total iterations can differ
  slightly. That's expected.

---

## Adding a dataset or embedder

- **Dataset:** write a loader in `data.py` that returns a `Dataset(name, texts, labels, n_clusters, prompt,
  text_type)` and add it to `_LOADERS`. `prompt` and `text_type` are what SLM summarizers use for that
  dataset. Put the data file in `data/` and document its source in `data/README.md`. If the loader takes
  options, fold them into `name` (e.g. `stackexchange_2023_limit5000`), because caches are keyed by name.
- **Embedder:** any sentence-transformers model id already works (`embedder: BAAI/bge-small-en-v1.5`). For
  anything else, subclass `Embedder` in `embeddings.py` and hook it into `get_embedder()`. Always encode a whole
  list in one call and return float32.

---

## Tests

```bash
pytest                    # fast suite (seconds, no model downloads)
pytest -m slow            # real-data + model tests, ~2 min (downloads distilbert / SmolLM2-135M; GPU recommended)
```

`tests/fixtures/*.npz` hold outputs of the **original** summaryCentroids code, produced by
`tests/fixtures/make_fixtures.py`. `tests/test_core_equivalence.py` and the slow Bank77 tests assert that our
loop reproduces them exactly. If you intentionally change the core loop's behavior, those tests are supposed
to fail. Put new behavior behind a new schedule or summarizer instead of changing `FixedSchedule` or
`summary_kmeans` semantics.

---

## Reproduction status

Checked on an RTX 3060 against the fork of the original code:

- **Smoke test** (Bank77, distilbert, seed 0): k-means NMI 0.6541 and k-NLPmeans (LSA) NMI 0.6518, identical to
  the original (`scripts/smoke_bank77.py`, `tests/test_bank77.py`).
- **k-NLPmeans grid** (Bank77, LSA / centroid / TextRank × top_k 3, 5 × 1 or 5 summary steps, max_iter 120):
  identical assignments, summaries, and summary embeddings in all 12 settings (`tests/test_bank77_grid.py`).
- **Synthetic fixtures:** all `initial_iter` / `final_iter` combinations, the `tol` early stop, and
  `force_context_length` sampling match exactly (`tests/test_core_equivalence.py`).
- **Mini-batch** (Bank77, batches of 1,000, LSA): the same assignments and NMI as the original
  (`tests/test_minibatch.py`).
- The final `cluster_centers_` can differ in the last float32 bit between *any* two runs, including two runs
  of the original code, because of sklearn's multithreaded Lloyd. Assignments are unaffected.

---

## Differences from upstream

**Removed:** hosted LLM APIs (OpenAI, Anthropic, DeepSeek, Llama API) and OpenAI / INSTRUCTOR embeddings,
since we use local models only. Also removed: `kLLMmedoids` and the k-medoids baseline (`scikit-learn-extra`
doesn't install on current Python), the tweet loader (no data), the notebooks (replaced by scripts and
configs), and the StackExchange download notebooks (the cleaned archive is checked in; provenance is in
`data/README.md`).

**Fixed:**

- With extractive summarizers and `force_context_length > 0`, upstream looked up sentences using the sampled
  members' *within-cluster* positions as *global* document ids. It summarized the wrong documents. The
  paper's reported runs used `force_context_length = 0` for extractive methods, so they are unaffected.
- In the streaming experiments, upstream built per-document sentences before sorting posts by date, so
  sentences and documents were misaligned. Here sentences are always built from the sorted stream.
- `miniBatchKLLMmeans` crashed (it called `.items()` on `text_sentences=None`). The mini-batch centroid
  average divided by zero for clusters that were still empty. The GoEmotions loader read from the working
  directory. `load_dataset("massive_I")` crashed without an extra argument. The streaming notebook's k-means
  baseline stored the wrong assignments.

**Changed:** Bank77 and CLINC are read from checked-in CSVs, because `datasets` dropped the script loaders
upstream relied on. Dataset prompts, label mappings, and all numerics are unchanged. `max_iter` and baseline
`n_init` are explicit config keys, because upstream's notebooks (120, sklearn default) and smoke test (100,
10) used different values.

---

## Working together

- Branch per feature (`slm-sweep`, `scheduled-summarizer`, ...). Open a PR into `main`, and run `pytest` plus
  the smoke test before asking for review.
- New methods go in new files: a summarizer in `summarizers/`, a schedule next to `FixedSchedule`. Add a
  config in `configs/` that runs them, so others can reproduce your numbers with one command.
- Don't commit `cache/` or `results/`. Share result tables in the PR description or a notebook instead.

---

## Citation

This repo builds on the method and data of:

> *Summaries as Centroids for Scalable and Interpretable Text Clustering* (k-NLPmeans / k-LLMmeans),
> [arXiv:2502.09667](https://arxiv.org/abs/2502.09667). Code: [jairoadiazr/summaryCentroids](https://github.com/jairoadiazr/summaryCentroids).

If you use the StackExchange benchmark, also cite the [Internet Archive StackExchange dump](https://archive.org/download/stackexchange).
