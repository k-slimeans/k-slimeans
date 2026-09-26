# Data

Everything the experiments need is checked in here, so nothing depends on dataset hubs at run time.

| File | Dataset | Split / notes | Source |
|---|---|---|---|
| `bank77_test.csv` | BANKING77 (77 intents) | test split, 3,080 rows (`text`, `category`) | [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets) (CC BY 4.0) |
| `clinc_small_test.csv` | CLINC150 (150 intents) | "small" config, test split, 5,500 rows (`text`, `intent`, `intent_name`); intent `42` is `oos` and is dropped by the loader | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) (CC BY 3.0) |
| `massive.jsonl` | MASSIVE en-US (59 intents / 18 scenarios) | all partitions; the loader keeps `partition == "test"` | [alexa/massive](https://github.com/alexa/massive) (CC BY 4.0) |
| `goemo.tsv` | GoEmotions (27 emotions) | test split; the loader keeps single-label rows and drops `neutral` | [google-research/goemotions](https://github.com/google-research/google-research/tree/master/goemotions) |
| `stackexchange/clean_stackexchange.7z.00{1,2,3}` | StackExchange streaming benchmark (35 communities, 2020–2023, 205,943 posts) | split 7z archive of `clean_stackexchange.csv` (`Text`, `CreationDate`, `Year`, `Label`); extract with `python scripts/prepare_stackexchange.py` | built by the summaryCentroids authors from the [Internet Archive StackExchange dump](https://archive.org/download/stackexchange) (CC BY-SA) |

The first four files and the StackExchange archive are copied from the upstream
[summaryCentroids](https://github.com/jairoadiazr/summaryCentroids) repository, except BANKING77 and CLINC,
which were exported once from the sources above. Upstream loaded those two from the Hugging Face hub at run
time, and that path broke when `datasets` dropped script-based loaders.

If you use the StackExchange benchmark, cite both the summaryCentroids paper
([arXiv:2502.09667](https://arxiv.org/abs/2502.09667)) and the original dump.
