# Data

Everything the experiments need is checked in here, so nothing depends on dataset hubs at run time.

| File | Dataset | Split / notes | Source |
|---|---|---|---|
| `bank77_test.csv` | BANKING77 (77 intents) | test split, 3,080 rows (`text`, `category`) | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) (CC BY 4.0) |
| `clinc_small_test.csv` | CLINC150 (150 intents) | "small" config, test split, 5,500 rows (`text`, `intent`, `intent_name`); intent `42` is `oos` and is dropped by the loader | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) (CC BY 3.0) |
| `clinc150_full.csv` | CLINC150 full (150 intents / 10 domains) | "full" data, train + validation + test, in-scope only, 22,500 rows (`text`, `intent`, `intent_name`); `oos` already removed. `intent` ids match the Hugging Face ids in `clinc_small_test.csv`. Loaders: `clinc150_intent`, `clinc150_domain` | `data/data_full.json` from [clinc/oos-eval](https://github.com/clinc/oos-eval) (CC BY 3.0), the same queries as the `full`/`plus` configs of [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos); exported with `python scripts/export_clinc150.py` |
| `clinc150_domains.csv` | CLINC150 intent → domain mapping | 150 rows (`intent_name`, `domain`), 15 intents per domain | `data/domains.json` from [clinc/oos-eval](https://github.com/clinc/oos-eval) (CC BY 3.0) |
| `massive.jsonl` | MASSIVE en-US (59 intents / 18 scenarios) | all partitions; the loader keeps `partition == "test"` | [alexa/massive](https://github.com/alexa/massive) (CC BY 4.0) |
| `goemo.tsv` | GoEmotions (27 emotions) | test split; the loader keeps single-label rows and drops `neutral` | [google-research/goemotions](https://github.com/google-research/google-research/tree/master/goemotions) |
| `stackexchange/clean_stackexchange.7z.00{1,2,3}` | StackExchange streaming benchmark (35 communities, 2020–2023, 205,943 posts) | split 7z archive of `clean_stackexchange.csv` (`Text`, `CreationDate`, `Year`, `Label`); extract with `python scripts/prepare_stackexchange.py` | built by the summaryCentroids authors from the [Internet Archive StackExchange dump](https://archive.org/download/stackexchange) (CC BY-SA) |

The first four files and the StackExchange archive are copied from the upstream
[summaryCentroids](https://github.com/jairoadiazr/summaryCentroids) repository, except BANKING77 and CLINC,
which were exported once from the sources above. Upstream loaded those two from the Hugging Face hub at run
time, and that path broke when `datasets` dropped script-based loaders.

CLINC150 is from Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope
Prediction", EMNLP-IJCNLP 2019 ([D19-1131](https://aclanthology.org/D19-1131/)); cite it if you use either CLINC file.

If you use the StackExchange benchmark, cite both the summaryCentroids paper
([arXiv:2502.09667](https://arxiv.org/abs/2502.09667)) and the original dump.

## arXiv abstracts (`data/arxiv/`, generated, git-ignored)

The L4 / RQ2 scale test clusters arXiv abstracts. Nothing here is checked in; build it with

    python scripts/prepare_arxiv.py [--stages download filter embed subsample] [--force {filter,embed,subsample}]

Each stage is resumable and skipped when its output is complete (`--force STAGE` redoes `filter`, `embed` or
`subsample`, and forcing `filter` also discards the embeddings and subsample; delete `raw/` to re-download).
The embed stage checkpoints to `progress.json` after every chunk and resumes after an interruption.
`manifest.json` records the counts below for the actual run.

- **Source:** the weekly arXiv OAI metadata snapshot, via the Hugging Face parquet mirror
  [librarian-bots/arxiv-metadata-snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot),
  license CC0-1.0, pinned to revision `5733e66fde7e840bf1dcb2d9cffa0c64f4c5654d` (3,205,432 rows, 11 shards,
  2.7 GB) and downloaded to `data/arxiv/raw/`.
- **Filter rules,** applied in this order, each counted in the manifest (`filter.dropped_*`):
  1. withdrawn: the abstract matches `(?i)\b(?:has been|is|was|been) withdrawn`, or the comments match
     `^\W*(this )?(paper|article)?\s*(has been |is |was )?withdrawn` (case-insensitive);
  2. short: fewer than 50 whitespace-separated words;
  3. duplicate: the normalized, lower-cased abstract was already seen (first occurrence wins).

  Whitespace in the abstract is normalized (`" ".join(abstract.split())`) before any rule is applied.

  Known misses: a probe of shard 10 found about 5 real withdrawals among ~250k kept rows that the regexes
  do not catch ("has been temporarily withdrawn", "is hereby withdrawn", "-paper withdrawn"). They are left in
  deliberately, at about 0.002% of rows.
- **Labels:** `label` is the primary category (first token of `categories`, e.g. `cs.LG`, `hep-th`); `archive` is
  the part before the dot (`cs`, `hep-th`, `math`).
- **Files:** `arxiv.parquet` (`id`, `text`, `label`, `archive`, `update_date`; row i is row i of the embeddings),
  `embeddings_minilm_fp16.npy` (N x 384 float16, L2-normalized `sentence-transformers/all-MiniLM-L6-v2` on `text`,
  encoded in fp32), `subsample_200k.npy` (sorted int64 row indices, seed 0) and `arxiv_sub200k.parquet` (those
  rows plus a `row` column). Parquet row groups hold at most 50,000 rows (one per 50k-row input batch,
  fewer after filtering). Counts after the filter, the label vocabulary and the embedding throughput are in
  `manifest.json`.
- **Memory:** the raw shards are read with explicit `columns=` in batches and the embeddings are written through
  a memmap, so neither stage holds the corpus in RAM.
- **Result at the pinned revision:** 3,205,432 rows in; 5,587 withdrawn, 163,016 short and 1,185 duplicate rows
  dropped; **3,035,644 rows out**, 171 primary categories in 38 archives (168 categories occur in the 200k
  subsample). On an RTX 3060 the filter took 2.4 min (peak RSS 2.0 GB) and the embed + subsample stages took
  70.5 min at 725 rows/s (peak RSS 5.8 GB); the embeddings file is 2.3 GB.
