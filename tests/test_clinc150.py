"""Fast checks for the full CLINC150 loaders (reads two small CSVs, no models)."""

import numpy as np
import pandas as pd
import pytest

from kslimeans.data import DATA_DIR, load_dataset


@pytest.mark.parametrize(("name", "n_labels"), [("clinc150_intent", 150), ("clinc150_domain", 10)])
def test_clinc150_shape(name, n_labels):
    ds = load_dataset(name)
    assert ds.name == name
    assert len(ds.texts) == 22_500
    assert len(ds.labels) == 22_500
    assert len(np.unique(ds.labels)) == n_labels
    assert ds.n_clusters == n_labels


def test_clinc150_has_no_oos():
    df = pd.read_csv(DATA_DIR / "clinc150_full.csv")
    assert "oos" not in set(df["intent_name"])
    assert 42 not in set(df["intent"])  # 42 is oos in the Hugging Face label ids


def test_clinc150_reuses_clinc_prompt():
    small = load_dataset("clinic")
    for name in ("clinc150_intent", "clinc150_domain"):
        ds = load_dataset(name)
        assert (ds.prompt, ds.text_type) == (small.prompt, small.text_type)
