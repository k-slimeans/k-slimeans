"""Sentence splitting and a per-document sentence/embedding corpus.

Ports upstream's `split_into_sentences` / `build_sentence_corpus`: a
sentence is kept if it has at least 5 whitespace-separated tokens, and a
document that yields zero kept sentences falls back to using the whole
document text as its one "sentence" (so every document always contributes
at least one entry).

`build_sentence_corpus` uses `nlp.pipe(..., disable=["ner", "lemmatizer"])`
instead of calling `nlp(text)` once per document. Sentence boundaries come
from the parser/senter, which stays enabled, so this only skips work that
sentence splitting never used; a direct check against per-document calls
on all of bank77 (3080 docs) found zero differing outputs and a ~5x
speedup, so it is safe to use for speed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .embeddings import Embedder

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run:\n"
                "    python -m spacy download en_core_web_sm"
            ) from exc
    return _NLP


def split_into_sentences(text: str) -> list[str]:
    """Split `text` into sentences, keeping only those with >= 5 tokens."""
    doc = _get_nlp()(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 5]


def build_sentence_corpus(docs: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Split every document into kept sentences.

    Returns the flat list of sentences and a parallel `index_map` where
    `index_map[i] = (doc_index, sentence_index_within_doc)` for
    `sentences[i]`. A document with no sentences >= 5 tokens contributes
    the whole document text as its single sentence.
    """
    nlp = _get_nlp()
    sentences: list[str] = []
    index_map: list[tuple[int, int]] = []
    for di, doc in enumerate(nlp.pipe(docs, disable=["ner", "lemmatizer"])):
        ss = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) >= 5]
        if len(ss) == 0:
            ss = [docs[di]]
        sentences.extend(ss)
        index_map.extend([(di, si) for si in range(len(ss))])
    return sentences, index_map


@dataclass
class SentenceCorpus:
    """Every document's kept sentences, plus their embeddings, in one flat array.

    `doc_ptr` is a CSR-style offset array: document `d`'s sentences are
    `sentences[doc_ptr[d]:doc_ptr[d + 1]]` (and the same slice into
    `embeddings`).
    """

    sentences: list[str]
    embeddings: np.ndarray  # (n_sentences, d) float32
    doc_ptr: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))

    @classmethod
    def build(cls, docs: list[str], embedder: Embedder) -> "SentenceCorpus":
        """Split `docs` into sentences and embed them all in one `encode` call."""
        sentences, index_map = build_sentence_corpus(docs)
        counts = np.zeros(len(docs), dtype=np.int64)
        for doc_idx, _ in index_map:
            counts[doc_idx] += 1
        doc_ptr = np.zeros(len(docs) + 1, dtype=np.int64)
        np.cumsum(counts, out=doc_ptr[1:])
        embeddings = embedder.encode(sentences)
        return cls(sentences=sentences, embeddings=embeddings, doc_ptr=doc_ptr)

    def subset(self, start: int, stop: int) -> "SentenceCorpus":
        """Sentences of documents `[start, stop)`, re-based so document 0 of the
        result is global document `start`.

        Used by `minibatch.mini_batch_summary_kmeans` to slice a whole-corpus
        `SentenceCorpus` into per-batch corpora without re-splitting/re-embedding
        sentences for every batch.
        """
        s0, s1 = int(self.doc_ptr[start]), int(self.doc_ptr[stop])
        doc_ptr = self.doc_ptr[start : stop + 1] - self.doc_ptr[start]
        return SentenceCorpus(sentences=self.sentences[s0:s1], embeddings=self.embeddings[s0:s1], doc_ptr=doc_ptr)

    def gather(self, doc_ids) -> tuple[list[str], np.ndarray]:
        """Sentences (and their embeddings) of `doc_ids`, in doc order then sentence order."""
        sentences: list[str] = []
        positions: list[int] = []
        for d in doc_ids:
            start, end = int(self.doc_ptr[d]), int(self.doc_ptr[d + 1])
            sentences.extend(self.sentences[start:end])
            positions.extend(range(start, end))
        embeddings = self.embeddings[positions] if positions else self.embeddings[:0]
        return sentences, embeddings

    def save(self, path: str, fingerprint: str | None = None) -> None:
        """Write `<path>.npz` (embeddings, doc_ptr, and `fingerprint` if given) and
        `<path>.json` (sentences).

        `fingerprint` is opaque caller metadata (see `cache.py`'s content-fingerprint
        cache invalidation) stored alongside the arrays so a later `load_fingerprint`
        can tell a cache hit from stale data without re-reading every sentence.
        """
        extra = {} if fingerprint is None else {"fingerprint": fingerprint}
        np.savez(f"{path}.npz", embeddings=self.embeddings, doc_ptr=self.doc_ptr, **extra)
        with open(f"{path}.json", "w") as f:
            json.dump(self.sentences, f)

    @classmethod
    def load(cls, path: str) -> "SentenceCorpus":
        data = np.load(f"{path}.npz")
        with open(f"{path}.json") as f:
            sentences = json.load(f)
        return cls(sentences=sentences, embeddings=data["embeddings"], doc_ptr=data["doc_ptr"])

    @staticmethod
    def load_fingerprint(path: str) -> str | None:
        """The fingerprint stored by `save(path, fingerprint=...)`, or `None` if the
        `<path>.npz` file has none (saved without one, or from before fingerprints
        were added)."""
        data = np.load(f"{path}.npz")
        return str(data["fingerprint"]) if "fingerprint" in data.files else None
