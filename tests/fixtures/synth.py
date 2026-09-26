"""Shared synthetic data + fakes for the fixture generator and the equivalence tests.

Both `make_fixtures.py` (run once, against the ORIGINAL `summaryCentroids.py`)
and `tests/test_core_equivalence.py` (run every time, against the new
`kslimeans.summary_kmeans`) import this module, so the synthetic corpus
and the "embedder"/"summarizer" fakes are defined exactly once and can
never drift between the two sides of the comparison.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

N_DOCS = 300
N_BLOBS = 6
DIM = 16


def make_synthetic_data(seed: int = 0) -> tuple[list[str], np.ndarray]:
    """6 Gaussian blobs in 16-D, 300 points, float32 features; texts doc0..doc299."""
    rng = np.random.default_rng(seed)
    per_blob = N_DOCS // N_BLOBS
    centers = rng.normal(scale=5.0, size=(N_BLOBS, DIM))
    X = np.empty((N_DOCS, DIM), dtype=np.float32)
    for b in range(N_BLOBS):
        blob = centers[b] + rng.normal(scale=1.0, size=(per_blob, DIM))
        X[b * per_blob : (b + 1) * per_blob] = blob.astype(np.float32)
    texts = [f"doc{i}" for i in range(N_DOCS)]
    return texts, X


def make_synthetic_minibatch_data(
    seed: int = 0, n_docs: int = 320, n_blobs: int = N_BLOBS, dim: int = DIM
) -> tuple[list[str], np.ndarray]:
    """A bigger synthetic corpus for multi-batch minibatch fixtures.

    Same style as `make_synthetic_data` (Gaussian blobs, float32 features,
    texts `doc0..doc{n_docs-1}`), but `n_docs` need not be a multiple of
    `n_blobs`: blob sizes are as equal as `divmod` allows so every row of
    `X` is filled, letting a caller pick `n_docs`/`max_batch_size`
    combinations that produce uneven batch splits (`make_synthetic_data`'s
    300 points split evenly into any divisor of 300, so it can't).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(n_blobs, dim))
    base, extra = divmod(n_docs, n_blobs)
    sizes = [base + 1 if b < extra else base for b in range(n_blobs)]
    X = np.empty((n_docs, dim), dtype=np.float32)
    pos = 0
    for b, size in enumerate(sizes):
        blob = centers[b] + rng.normal(scale=1.0, size=(size, dim))
        X[pos : pos + size] = blob.astype(np.float32)
        pos += size
    texts = [f"doc{i}" for i in range(n_docs)]
    return texts, X


def make_lookup_embedder(texts: list[str], X: np.ndarray) -> Callable[[list[str]], np.ndarray]:
    """A fake 'embedder': looks text up in a fixed table; "" -> zeros; stacked float32."""
    lookup = {t: X[i] for i, t in enumerate(texts)}
    dim = X.shape[1]

    def embed(query_texts: list[str]) -> np.ndarray:
        if not query_texts:
            return np.zeros((0, dim), dtype=np.float32)
        rows = [lookup[t] if t != "" else np.zeros(dim, dtype=np.float32) for t in query_texts]
        return np.stack(rows).astype(np.float32)

    return embed


def closest_to_mean(member_texts: list[str], embed: Callable[[list[str]], np.ndarray]) -> str:
    """The member text whose vector is closest to the members' mean (ties -> first)."""
    if not member_texts:
        return ""
    vecs = embed(member_texts)
    mean = vecs.mean(axis=0)
    j = int(np.argmin(np.linalg.norm(vecs - mean, axis=1)))
    return member_texts[j]
