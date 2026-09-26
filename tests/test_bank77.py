"""Real-data reproduction check: kNLPmeans(lsa) on Bank77 vs. the original code.

Marked slow because it downloads/loads the distilbert sentence-transformer
model and spaCy's parser. Run with `pytest -m slow tests/test_bank77.py`
after regenerating fixtures with `tests/fixtures/make_fixtures.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import normalized_mutual_info_score

from kslimeans.core import kNLPmeans
from kslimeans.data import load_dataset
from kslimeans.embeddings import get_embedder

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bank77_lsa_seed0.npz"


@pytest.mark.slow
def test_bank77_lsa_matches_upstream_fixture():
    dataset = load_dataset("bank77")
    embedder = get_embedder("distilbert")
    X = embedder.encode(dataset.texts)

    result = kNLPmeans(
        dataset.texts,
        dataset.n_clusters,
        emb_type="distilbert",
        text_features=X,
        nlp="lsa",
        max_llm_iter=5,
        random_state=0,
    )

    fixture = np.load(FIXTURE)
    assert np.array_equal(result.assignments, fixture["assignments"])

    nmi = normalized_mutual_info_score(dataset.labels, result.assignments)
    assert round(nmi, 4) == 0.6518
