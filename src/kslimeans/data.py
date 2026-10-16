"""Dataset loaders.

Each loader returns a `Dataset`: texts, integer ground-truth labels, the
number of clusters, and the prompt/text_type strings the paper used when
summarizing that dataset with an LLM (kept around for every loader, not
just LLM runs, since `load_dataset` is the one place that knows the
paper's exact wording).

Paths resolve against `DATA_DIR`, which defaults to the repo's `data/`
directory and can be overridden with the `KSLIMEANS_DATA_DIR` environment
variable (e.g. to point at a copy on a different disk).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(
    os.environ.get("KSLIMEANS_DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data"))
)


@dataclass
class Dataset:
    name: str
    texts: list[str]
    labels: np.ndarray
    n_clusters: int
    prompt: str
    text_type: str
    label_names: list[str] | None = None  # class name per label id, when the dataset has names
    # Precomputed vectors for datasets too big to embed through `cached_embeddings`: a (N, dim) .npy
    # memmap, the rows of it belonging to this dataset (None = all rows in order), and the embedder
    # name they came from. See `cache.dataset_embeddings`.
    embeddings_path: Path | None = None
    embeddings_rows: np.ndarray | None = None
    embeddings_embedder: str | None = None


def _load_bank77() -> Dataset:
    df = pd.read_csv(DATA_DIR / "bank77_test.csv")
    categories = sorted(df["category"].unique())
    mapping = {c: i for i, c in enumerate(categories)}
    labels = np.array([mapping[c] for c in df["category"]], dtype=np.int64)
    prompt = (
        "The following is a cluster of online banking questions. Write a single "
        "question that represents the following cluster concisely:"
    )
    text_type = "Sentence:"
    return Dataset("bank77", list(df["text"]), labels, 77, prompt, text_type, list(categories))


_CLINC_PROMPT = "The following is cluster of questions. Write a question that summarizes the following cluster:"
_CLINC_TEXT_TYPE = "Question:"


def _load_clinic() -> Dataset:
    df = pd.read_csv(DATA_DIR / "clinc_small_test.csv")
    kept = df[df["intent"] != 42]

    intent_mapping: dict[int, int] = {}
    remapped = []
    for intent in kept["intent"]:
        if intent not in intent_mapping:
            intent_mapping[intent] = len(intent_mapping)
        remapped.append(intent_mapping[intent])
    labels = np.array(remapped, dtype=np.int64)

    return Dataset("clinic", list(kept["text"]), labels, 150, _CLINC_PROMPT, _CLINC_TEXT_TYPE)


def _load_clinc150(level: str) -> Dataset:
    """Full CLINC150: 22,500 in-scope queries (train + val + test, `oos` dropped).

    `level="intent"` labels each query with one of 150 intents; `level="domain"`
    labels the same queries with one of 10 domains via clinc150_domains.csv.
    Label ids are LabelEncoder ids over the sorted intent/domain names (not the
    Hugging Face ids in the CSV's `intent` column), so `label_names[i]` names label `i`.
    """
    df = pd.read_csv(DATA_DIR / "clinc150_full.csv")
    if level == "intent":
        raw_labels, n_clusters = df["intent_name"], 150
    elif level == "domain":
        mapping = pd.read_csv(DATA_DIR / "clinc150_domains.csv")
        raw_labels = df["intent_name"].map(dict(zip(mapping["intent_name"], mapping["domain"])))
        if raw_labels.isna().any():
            raise ValueError("clinc150_domains.csv is missing intents found in clinc150_full.csv")
        n_clusters = 10
    else:
        raise ValueError(f"Unknown clinc150 level '{level}', expected 'intent' or 'domain'")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(raw_labels).astype(np.int64)
    return Dataset(
        f"clinc150_{level}",
        list(df["text"]),
        labels,
        n_clusters,
        _CLINC_PROMPT,
        _CLINC_TEXT_TYPE,
        [str(name) for name in encoder.classes_],
    )


def _load_massive_raw() -> pd.DataFrame:
    rows = []
    with open(DATA_DIR / "massive.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    df = pd.DataFrame(rows)
    return df[df["partition"] == "test"]


def _load_massive(which: str) -> Dataset:
    df = _load_massive_raw()
    if which == "I":
        prompt = (
            "The following is a cluster of virtual assistant utterances. Write a short "
            "utterance that represents the following cluster:"
        )
        text_type = "Utterance:"
        n_clusters = 59
        raw_labels = df["intent"]
    elif which == "D":
        prompt = (
            "The following is a cluster of virtual assistant utterances. Write a summary "
            "that represents the following cluster:"
        )
        text_type = "Summary:"
        n_clusters = 18
        raw_labels = df["scenario"]
    else:
        raise ValueError(f"Unknown massive variant '{which}', expected 'I' or 'D'")

    labels = LabelEncoder().fit_transform(raw_labels).astype(np.int64)
    return Dataset(f"massive_{which}", list(df["utt"]), labels, n_clusters, prompt, text_type)


def _load_goemo() -> Dataset:
    df = pd.read_csv(
        DATA_DIR / "goemo.tsv", sep="\t", header=None, names=["text", "emo", "user"], dtype={"emo": str}
    )
    df["emo"] = df["emo"].str.strip()
    single_label = ~df["emo"].str.contains(",")
    df = df[single_label & (df["emo"] != "27")]
    labels = df["emo"].astype(np.int64).to_numpy()

    prompt = (
        "The following is cluster of comments expressing the same emotion. Concisely "
        "classify the emotion(s) that summarize the following cluster:"
    )
    text_type = "Emotion(s):"
    return Dataset("goemo", list(df["text"]), labels, 27, prompt, text_type)


def _load_stackexchange_csv() -> pd.DataFrame:
    path = DATA_DIR / "stackexchange" / "clean_stackexchange.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/prepare_stackexchange.py` first to "
            "extract it from the byte-split archive under data/stackexchange/."
        )
    return pd.read_csv(path)


def _stackexchange_name(year: int | None, limit: int | None) -> str:
    parts = ["stackexchange"]
    if year is not None:
        parts.append(str(year))
    if limit is not None:
        parts.append(f"limit{limit}")
    return "_".join(parts)


def load_stackexchange(year: int | None = None, limit: int | None = None) -> Dataset:
    """StackExchange streaming benchmark (35 communities, 2020-2023).

    Labels come from a `LabelEncoder` fit on every community in the full
    (year-unfiltered) CSV, sorted -- so a community's label id is the same
    whether or not `year` happens to filter it out, and is stable across
    different `year`/`limit` calls. Posts are optionally filtered to
    `year`, then STABLE-sorted by `CreationDate` (an ISO-8601 string, so
    lexicographic order already is chronological order) to recover the
    paper's streaming order; `limit`, if given, then keeps only the first
    `limit` posts of that stream. `n_clusters` is the number of distinct
    labels actually present in the resulting slice.

    Sentence corpora for this dataset must always be built from
    `Dataset.texts` -- i.e. from these already-filtered, already-sorted
    texts -- never from the raw CSV order. Upstream built its per-document
    sentence pickles from the raw (unsorted) per-year documents and only
    sorted by `CreationDate` afterwards, so its streaming experiments'
    sentences ended up silently misaligned with the (re-sorted) documents.
    `SentenceCorpus.build`/`cache.cached_sentences` are always called with
    this `Dataset`'s (post-sort) `texts`, which avoids that bug by
    construction.
    """
    df = _load_stackexchange_csv()
    encoder = LabelEncoder().fit(df["Label"])

    if year is not None:
        df = df[df["Year"] == year]

    df = df.sort_values("CreationDate", kind="stable")

    if limit is not None:
        df = df.iloc[:limit]

    texts = list(df["Text"])
    labels = encoder.transform(df["Label"]).astype(np.int64)
    n_clusters = int(len(np.unique(labels)))

    prompt = (
        "The following is a cluster of questions from the same community. Write a "
        "summary that represents the cluster:"
    )
    text_type = "Summary:"
    return Dataset(_stackexchange_name(year, limit), texts, labels, n_clusters, prompt, text_type)


_ARXIV_PROMPT = (
    "The following is a cluster of arXiv paper abstracts. Write a single sentence that "
    "summarizes the cluster's topic:"
)
_ARXIV_TEXT_TYPE = "Topic:"
_ARXIV_EMBEDDER = "minilm"
_ARXIV_PREPARE_HINT = "Run `python scripts/prepare_arxiv.py` first."


def _read_parquet_prefix(path: Path, columns: list[str], year: int | None, limit: int | None) -> pd.DataFrame:
    """Rows of `path` in file order, optionally only those updated in `year`, stopping after `limit`.

    The result has an extra `file_row` column: each row's position in `path`.

    Streams record batches so a small `limit` never materializes the multi-GB full table.
    `year` filters on the leading four characters of the `update_date` string column.
    """
    read_columns = columns + (["update_date"] if year is not None else [])
    frames, count, offset = [], 0, 0
    for batch in pq.ParquetFile(path).iter_batches(columns=read_columns):
        df = batch.to_pandas()
        df["file_row"] = np.arange(offset, offset + len(df), dtype=np.int64)
        offset += len(df)
        if year is not None:
            df = df[df["update_date"].str[:4] == str(year)].drop(columns="update_date")
        frames.append(df)
        count += len(df)
        if limit is not None and count >= limit:
            break
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns + ["file_row"])
    return out.iloc[:limit] if limit is not None else out


def _load_arxiv(parquet_name: str, name: str, year: int | None, limit: int | None) -> Dataset:
    arxiv_dir = DATA_DIR / "arxiv"
    parquet_path = arxiv_dir / parquet_name
    full_path = arxiv_dir / "arxiv.parquet"
    embeddings_path = arxiv_dir / "embeddings_minilm_fp16.npy"
    manifest_path = arxiv_dir / "manifest.json"
    for path in (parquet_path, full_path, embeddings_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. {_ARXIV_PREPARE_HINT}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("embedding", {}).get("complete"):
        raise ValueError(f"The arXiv embeddings in {arxiv_dir} are incomplete. {_ARXIV_PREPARE_HINT}")
    n_rows = pq.ParquetFile(full_path).metadata.num_rows
    shape = np.load(embeddings_path, mmap_mode="r").shape
    if len(shape) != 2 or shape[0] != n_rows:
        raise ValueError(
            f"{embeddings_path} has shape {shape} but arxiv.parquet has {n_rows} rows. {_ARXIV_PREPARE_HINT}"
        )

    subsample = parquet_name != "arxiv.parquet"
    df = _read_parquet_prefix(parquet_path, ["text", "label"] + (["row"] if subsample else []), year, limit)
    if df.empty:
        raise ValueError(f"{name} has no rows for year={year}, limit={limit}")
    if subsample:
        rows = df["row"].to_numpy(dtype=np.int64)
    elif year is None:
        rows = None if limit is None else df["file_row"].to_numpy(dtype=np.int64)
    else:
        rows = df["file_row"].to_numpy(dtype=np.int64)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["label"]).astype(np.int64)
    label_names = [str(c) for c in encoder.classes_]
    if year is not None:
        name += f"_{year}"
    if limit is not None:
        name += f"_limit{limit}"
    return Dataset(
        name,
        df["text"].tolist(),
        labels,
        len(label_names),
        _ARXIV_PROMPT,
        _ARXIV_TEXT_TYPE,
        label_names,
        embeddings_path=embeddings_path,
        embeddings_rows=rows,
        embeddings_embedder=_ARXIV_EMBEDDER,
    )


def load_arxiv(year: int | None = None, limit: int | None = None) -> Dataset:
    """The full arXiv abstracts corpus (~3.0M, primary category as label), in file order.

    `year` keeps papers whose `update_date` falls in that year; `limit` then keeps the first
    `limit` rows. Label ids are contiguous over the categories present in the returned rows.
    Vectors are precomputed (MiniLM), see `Dataset.embeddings_path`. Unlimited, this holds ~3.0M
    texts in RAM and the runners copy the vectors as float32 (~4.7 GB); prefer `arxiv_sub200k`.
    """
    return _load_arxiv("arxiv.parquet", "arxiv", year, limit)


def load_arxiv_sub200k(year: int | None = None, limit: int | None = None) -> Dataset:
    """The 200k random subsample of arXiv abstracts; same semantics as `load_arxiv`."""
    return _load_arxiv("arxiv_sub200k.parquet", "arxiv_sub200k", year, limit)


# Registry of loaders by dataset name.
_LOADERS = {
    "bank77": _load_bank77,
    "clinic": _load_clinic,
    "clinc150_intent": lambda: _load_clinc150("intent"),
    "clinc150_domain": lambda: _load_clinc150("domain"),
    "massive_I": lambda: _load_massive("I"),
    "massive_D": lambda: _load_massive("D"),
    "goemo": _load_goemo,
    "stackexchange": load_stackexchange,
    "arxiv": load_arxiv,
    "arxiv_sub200k": load_arxiv_sub200k,
}


def load_dataset(name: str, **kwargs) -> Dataset:
    """Load a `Dataset` by name. See `_LOADERS` for the valid names.

    Extra keyword arguments are forwarded to the loader, e.g.
    `load_dataset("stackexchange", year=2023, limit=5000)`; loaders that
    take no arguments (all but `stackexchange`, currently) just ignore an
    empty `kwargs`.
    """
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}', expected one of {sorted(_LOADERS)}") from None
    return loader(**kwargs)
