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
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

_ST_MODEL_IDS = {
    "distilbert": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "e5-large": "intfloat/e5-large",
    "sbert": "all-mpnet-base-v2",
}


class Embedder(ABC):
    """A named text encoder producing float32 vectors."""

    name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into a (len(texts), d) float32 array."""


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


_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder(name: str) -> Embedder:
    """Return a cached `Embedder` for `name`, constructing it on first use.

    Upstream reloaded the `SentenceTransformer` model on every call to
    `get_embeddings`; caching the loaded model here is purely a speed
    optimization and must not (and does not) change the numbers `encode`
    produces.
    """
    embedder = _EMBEDDER_CACHE.get(name)
    if embedder is None:
        embedder = SentenceTransformerEmbedder(name)
        _EMBEDDER_CACHE[name] = embedder
    return embedder
