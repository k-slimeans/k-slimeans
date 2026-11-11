"""Text-to-vector embedders.

Ports the `emb_type` branches of upstream's `get_embeddings`, dropping the
OpenAI and INSTRUCTOR options (hosted APIs, out of scope here). Each named
alias below reproduces the upstream numerics exactly, including the manual
L2 normalization for `distilbert` (as opposed to `normalize_embeddings=True`
for the others -- these are *not* interchangeable at the bit level, so keep
them as written).

Callers should always pass a whole batch of texts to `encode` in one call:
sentence-transformers' output depends on how texts are grouped into
batches internally, so encoding the same text alone vs. alongside others
can produce different floats.

`TfidfEmbedder` is the one embedder that is fit on a corpus (the dataset's own texts) instead of
loaded: see its docstring and `build_corpus_embedder`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import sklearn
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

_ST_MODEL_IDS = {
    "distilbert": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "e5-large": "intfloat/e5-large",
    "sbert": "all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


class Embedder(ABC):
    """A named text encoder producing float32 vectors."""

    name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into a (len(texts), d) float32 array."""

    def stats(self) -> dict[str, int]:
        """Cumulative counters over this instance's lifetime, like `Summarizer.stats()`.

        `summary_kmeans` logs the change across each of its summary-encoding calls next to the
        summarizer's cost counters. Embedders with nothing to report return `{}`, and then nothing
        is added to any output.
        """
        return {}


class SentenceTransformerEmbedder(Embedder):
    """Wraps a `sentence_transformers.SentenceTransformer` model."""

    def __init__(self, name: str) -> None:
        self.name = name
        model_id = _ST_MODEL_IDS.get(name, name)
        self._model = SentenceTransformer(model_id)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.name == "distilbert":
            embeddings = self._model.encode(texts)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        elif self.name == "e5-large":
            prefixed = ["query: " + text for text in texts]
            embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        else:
            # 'sbert' and any other sentence-transformers model id.
            embeddings = self._model.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        assert embeddings.dtype == np.float32
        return embeddings


def _df_label(value: int | float) -> str:
    """A `min_df`/`max_df` for `TfidfEmbedder.name`: an int is a document count ("20"), a float a proportion
    ("1p" for 1.0, "0.5p"), so 1.0 and 1.00 give one name and a count never collides with a proportion."""
    return str(value) if isinstance(value, int) else f"{value:g}p"


class TfidfEmbedder(Embedder):
    """Sparse TF-IDF vectors (returned dense) from a vectorizer fit once on a fixed corpus.

    The vocabulary and IDF weights come from `corpus` alone (English stop words removed, terms in fewer
    than `min_df` documents or more than `max_df` of them dropped, optionally the `max_features` most
    frequent kept); the rows are L2-normalized. `encode` only ever transforms, so summaries are
    embedded in the corpus's space and the fit never sees them. A text with no in-vocabulary term gets an
    all-zero row.

    Fitting raises if fewer than `min_terms` terms survive the filters, so a corpus or setting that
    shrinks the space below the intended dimensionality fails instead of running at a different size.
    The fit is deterministic. `name` encodes every setting, so it can key the embedding cache.
    """

    def __init__(
        self,
        corpus: list[str],
        *,
        min_df: int = 20,
        max_df: float = 1.0,
        max_features: int | None = None,
        sublinear_tf: bool = False,
        min_terms: int = 700,
    ) -> None:
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.min_terms = min_terms
        # The scikit-learn version is in the name because its tokenizer and English stop-word list are part
        # of the fit: an upgrade that changes either must not reuse a cached matrix.
        self.name = (
            f"tfidf-mindf{_df_label(min_df)}-maxdf{_df_label(max_df)}-maxf{max_features or 'all'}"
            f"-sub{int(sublinear_tf)}-en-sk{sklearn.__version__}"
        )
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        self._vectorizer.fit(corpus)
        self._vocabulary = self._vectorizer.vocabulary_
        self._analyze = self._vectorizer.build_analyzer()
        self._stats = {"embed_tokens": 0, "embed_oov_tokens": 0, "embed_zero_vectors": 0}
        if self.vocab_size < min_terms:
            raise ValueError(
                f"TF-IDF vocabulary has {self.vocab_size} terms on {len(corpus)} texts, fewer than the "
                f"required {min_terms} ({self.name}); lower min_df or raise max_df/max_features"
            )

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    def encode(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.vocab_size), dtype=np.float32)
        for text in texts:
            tokens = self._analyze(text)
            self._stats["embed_tokens"] += len(tokens)
            self._stats["embed_oov_tokens"] += sum(1 for token in tokens if token not in self._vocabulary)
        matrix = self._vectorizer.transform(texts)
        self._stats["embed_zero_vectors"] += int(np.count_nonzero(matrix.getnnz(axis=1) == 0))
        return matrix.toarray().astype(np.float32, copy=False)

    def stats(self) -> dict[str, int]:
        """Cumulative over every `encode` call: tokens (after stop-word removal), tokens outside the
        vocabulary, and texts with no in-vocabulary term (all-zero rows)."""
        return dict(self._stats)


def build_corpus_embedder(spec: dict, corpus: list[str]) -> Embedder:
    """Build a corpus-fitted embedder from a config dict `{name: tfidf, **TfidfEmbedder kwargs}`."""
    spec = dict(spec)
    name = spec.pop("name", None)
    if name != "tfidf":
        raise ValueError(f"Unknown corpus-fitted embedder {name!r}, expected 'tfidf'")
    return TfidfEmbedder(corpus, **spec)


_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder(name: str) -> Embedder:
    """Return a cached `Embedder` for `name`, constructing it on first use.

    Upstream reloaded the `SentenceTransformer` model on every call to
    `get_embeddings`; caching the loaded model here is purely a speed
    optimization and must not (and does not) change the numbers `encode`
    produces.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"get_embedder takes an embedder name, got {type(name).__name__}: corpus-fitted embedder specs "
            "need the dataset texts; resolve them via experiment._resolve_embedder"
        )
    if name == "tfidf":
        raise ValueError(
            "'tfidf' is fit on the dataset's texts, so it needs a dict spec: embedder: {name: tfidf, min_df: 20}"
        )
    embedder = _EMBEDDER_CACHE.get(name)
    if embedder is None:
        embedder = SentenceTransformerEmbedder(name)
        _EMBEDDER_CACHE[name] = embedder
    return embedder
