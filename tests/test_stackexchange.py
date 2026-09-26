"""`load_dataset("stackexchange", ...)` semantics, and the sentence-alignment
regression this loader is designed to avoid (see `data.py`'s docstring).

Marked slow when a test needs the extracted CSV (`data/stackexchange/clean_stackexchange.csv`),
which isn't checked in directly -- run `python scripts/prepare_stackexchange.py` first. Those
tests skip cleanly (rather than failing) if the CSV is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kslimeans.data import DATA_DIR, load_dataset
from kslimeans.sentences import build_sentence_corpus, split_into_sentences

CSV_PATH = Path(DATA_DIR) / "stackexchange" / "clean_stackexchange.csv"

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason=f"{CSV_PATH} missing -- run `python scripts/prepare_stackexchange.py` first",
)


@pytest.mark.slow
def test_load_stackexchange_full_dataset_matches_known_counts():
    ds = load_dataset("stackexchange")
    assert ds.name == "stackexchange"
    assert len(ds.texts) == 205_943
    assert ds.n_clusters == 35
    assert ds.prompt.startswith("The following is a cluster of questions from the same community")
    assert ds.text_type == "Summary:"


@pytest.mark.slow
@pytest.mark.parametrize(
    "year,expected", [(2020, 69_147), (2021, 54_322), (2022, 43_521), (2023, 38_953)]
)
def test_load_stackexchange_year_filter_matches_known_counts(year, expected):
    ds = load_dataset("stackexchange", year=year)
    assert len(ds.texts) == expected
    assert ds.name == f"stackexchange_{year}"


@pytest.mark.slow
def test_load_stackexchange_limit_keeps_the_first_n_of_the_stream():
    full = load_dataset("stackexchange", year=2023)
    limited = load_dataset("stackexchange", year=2023, limit=5000)

    assert len(limited.texts) == 5000
    assert limited.texts == full.texts[:5000]
    assert np.array_equal(limited.labels, full.labels[:5000])
    assert limited.name == "stackexchange_2023_limit5000"


@pytest.mark.slow
def test_load_stackexchange_limit_without_year_encodes_only_limit_in_the_name():
    ds = load_dataset("stackexchange", limit=1000)
    assert ds.name == "stackexchange_limit1000"
    assert len(ds.texts) == 1000


@pytest.mark.slow
def test_load_stackexchange_stream_order_is_chronological():
    ds = load_dataset("stackexchange", year=2023)
    # CreationDate isn't exposed on Dataset, but the loader promises a stable sort by
    # it; re-derive the sorted CSV and check texts line up in the same order.
    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    df = df[df["Year"] == 2023].sort_values("CreationDate", kind="stable")
    assert ds.texts == list(df["Text"])


@pytest.mark.slow
def test_load_stackexchange_label_ids_are_stable_across_years():
    """A community's label id must not depend on which `year` slice is loaded --
    the LabelEncoder is fit on every community in the (year-unfiltered) CSV."""
    full = load_dataset("stackexchange")
    ds_2020 = load_dataset("stackexchange", year=2020)

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(CSV_PATH)
    encoder = LabelEncoder().fit(df["Label"])
    df_2020 = df[df["Year"] == 2020].sort_values("CreationDate", kind="stable")
    expected_2020_labels = encoder.transform(df_2020["Label"])

    assert np.array_equal(ds_2020.labels, expected_2020_labels)
    # every label id used in 2020 must mean the same community in the full dataset
    encoder_full_check = LabelEncoder().fit(df["Label"])
    assert list(encoder_full_check.classes_) == list(encoder.classes_)
    assert full.n_clusters == 35


@pytest.mark.slow
def test_load_stackexchange_missing_csv_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr("kslimeans.data.DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare_stackexchange"):
        load_dataset("stackexchange")


# --- sentence/document alignment regression -----------------------------------------
#
# Upstream built its per-document sentence pickles from the RAW per-year CSV order and
# only sorted by CreationDate afterwards (see data_loaders/preprocess_stackexchange.ipynb
# + sequential_experiments.ipynb), so its streaming sentence corpora were silently
# misaligned with the (re-sorted) documents. Our sentence corpora are always built from
# `Dataset.texts` -- i.e. from the already-sorted stream -- so document i's sentences
# always come from text i.


@pytest.mark.slow
def test_sentence_corpus_doc_i_sentences_come_from_text_i():
    ds = load_dataset("stackexchange", year=2023, limit=200)
    sentences, index_map = build_sentence_corpus(ds.texts)

    for flat_idx, (doc_idx, sent_idx_in_doc) in enumerate(index_map):
        expected_sentences = split_into_sentences(ds.texts[doc_idx])
        if not expected_sentences:
            expected_sentences = [ds.texts[doc_idx]]
        assert sentences[flat_idx] == expected_sentences[sent_idx_in_doc]
