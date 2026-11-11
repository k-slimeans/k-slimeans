"""On-disk caches for embeddings and sentence corpora.

Keyed by dataset name + embedder name (with '/' sanitized out of model
ids, since model ids often look like `org/name`), so switching embedders
or datasets never silently reuses a mismatched cache entry. Both
functions also recompute from scratch whenever the cache file is
missing, or its stored content fingerprint -- a sha256 over `texts`,
computed by `_fingerprint` -- doesn't match: a document-count match
alone isn't enough, since editing a data file in place without changing
its row count would otherwise silently reuse stale embeddings. A cache
file from before fingerprints were added has none stored, and is always
treated as stale (recomputed) rather than trusted on count alone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .data import Dataset
from .embeddings import Embedder, get_embedder
from .sentences import SentenceCorpus


def _sanitize(name: str) -> str:
    return name.replace("/", "__")


def _embedder_name(embedder: Embedder | str) -> str:
    return embedder if isinstance(embedder, str) else embedder.name


def _fingerprint(texts: list[str]) -> str:
    """Sha256 over `texts`, each length-prefixed so concatenation can never be
    ambiguous (no separator string could collide with one chosen inside a text)."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def embeddings_cache_path(dataset_name: str, embedder: Embedder | str, cache_dir: str = "cache") -> Path:
    """Where `cached_embeddings` keeps the embeddings of `dataset_name` under `embedder`."""
    return Path(cache_dir) / f"{_sanitize(f'{dataset_name}__{_embedder_name(embedder)}')}_embeddings.npz"


def _read_embeddings(path: Path, texts: list[str]) -> np.ndarray | None:
    """The cached embeddings at `path` if they belong to `texts` (fingerprint and row count), else `None`."""
    if not path.exists():
        return None
    data = np.load(path)
    embeddings = data["embeddings"]
    cached_fingerprint = str(data["fingerprint"]) if "fingerprint" in data.files else None
    if cached_fingerprint == _fingerprint(texts) and embeddings.shape[0] == len(texts):
        return embeddings
    return None


def has_cached_embeddings(
    dataset_name: str, texts: list[str], embedder: Embedder | str, cache_dir: str = "cache"
) -> bool:
    """Whether `cached_embeddings` would return a valid cache entry instead of encoding."""
    return _read_embeddings(embeddings_cache_path(dataset_name, embedder, cache_dir), texts) is not None


def cached_embeddings(
    dataset_name: str, texts: list[str], embedder: Embedder | str, cache_dir: str = "cache"
) -> np.ndarray:
    """Embed `texts` with `embedder`, caching under `<cache_dir>/<dataset>__<embedder>_embeddings.npz`."""
    path = embeddings_cache_path(dataset_name, embedder, cache_dir)
    cached = _read_embeddings(path, texts)
    if cached is not None:
        return cached

    if isinstance(embedder, str):
        embedder = get_embedder(embedder)
    embeddings = embedder.encode(texts)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, fingerprint=_fingerprint(texts))
    return embeddings


def dataset_embeddings(ds: Dataset, embedder: Embedder | str, cache_dir: str = "cache") -> np.ndarray:
    """The embedding matrix for `ds`: its precomputed vectors if it has any, else `cached_embeddings`.

    Precomputed vectors (`ds.embeddings_path`) must come from the same embedder the caller asks
    for. They are memory-mapped, the dataset's rows selected, and returned as a float32 copy.
    """
    if ds.embeddings_path is None:
        return cached_embeddings(ds.name, ds.texts, embedder, cache_dir)

    name = _embedder_name(embedder)
    if name != ds.embeddings_embedder:
        raise ValueError(
            f"Dataset '{ds.name}' has precomputed '{ds.embeddings_embedder}' embeddings, "
            f"but embedder '{name}' was requested"
        )
    mmap = np.load(ds.embeddings_path, mmap_mode="r")
    rows = slice(None) if ds.embeddings_rows is None else ds.embeddings_rows
    embeddings = np.asarray(mmap[rows], dtype=np.float32)
    if embeddings.shape[0] != len(ds.texts):
        raise ValueError(f"{ds.embeddings_path} yields {embeddings.shape[0]} rows for {len(ds.texts)} texts")
    return embeddings


def sentences_cache_base(dataset_name: str, embedder: Embedder | str, cache_dir: str = "cache") -> str:
    """Path prefix (`<base>.npz`, `<base>.json`) where `cached_sentences` keeps the corpus."""
    return str(Path(cache_dir) / f"{_sanitize(f'{dataset_name}__{_embedder_name(embedder)}')}_sentences")


def _read_sentences(base: str, texts: list[str]) -> SentenceCorpus | None:
    """The cached corpus at `base` if it belongs to `texts`, else `None`."""
    if Path(f"{base}.npz").exists() and Path(f"{base}.json").exists():
        if SentenceCorpus.load_fingerprint(base) == _fingerprint(texts):
            corpus = SentenceCorpus.load(base)
            if len(corpus.doc_ptr) - 1 == len(texts):
                return corpus
    return None


def has_cached_sentences(
    dataset_name: str, texts: list[str], embedder: Embedder | str, cache_dir: str = "cache"
) -> bool:
    """Whether `cached_sentences` would return a valid cache entry instead of building the corpus."""
    return _read_sentences(sentences_cache_base(dataset_name, embedder, cache_dir), texts) is not None


def cached_sentences(
    dataset_name: str, texts: list[str], embedder: Embedder | str, cache_dir: str = "cache"
) -> SentenceCorpus:
    """Build a `SentenceCorpus`, caching under `<cache_dir>/<dataset>__<embedder>_sentences.*`."""
    base = sentences_cache_base(dataset_name, embedder, cache_dir)
    corpus = _read_sentences(base, texts)
    if corpus is not None:
        return corpus

    if isinstance(embedder, str):
        embedder = get_embedder(embedder)
    corpus = SentenceCorpus.build(texts, embedder)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    corpus.save(base, fingerprint=_fingerprint(texts))
    return corpus
